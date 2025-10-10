"""
Training Job Management System for QLoRA Training API

This module provides a comprehensive job management system for running QLoRA training
tasks in the background, tracking progress, and managing training lifecycle.
"""

import threading
import queue
import uuid
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

from .qlora.trainer import QLoRATrainer
from .qlora.config import TrainingConfig
from ..models_config import get_model_id

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Training job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Training job data structure"""
    job_id: str
    model_name: str
    status: JobStatus
    progress: float = 0.0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Training configuration
    lora_config: Optional[Dict] = None
    training_config: Optional[Dict] = None
    
    # Training data
    train_texts: Optional[List[str]] = None
    output_dir: Optional[str] = None
    
    # Results and metrics
    metrics: Optional[Dict] = None
    error_message: Optional[str] = None
    adapter_path: Optional[str] = None
    
    # Progress tracking
    current_step: int = 0
    total_steps: int = 0
    current_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization"""
        result = asdict(self)
        
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'started_at', 'completed_at']:
            if result[field]:
                result[field] = result[field].isoformat()
        
        # Convert enum to string
        result['status'] = result['status'].value
        
        return result


class TrainingJobManager:
    """
    Manages QLoRA training jobs with background execution and progress tracking.
    
    Features:
    - Job queuing and execution
    - Progress monitoring
    - Background threading
    - Job persistence
    - Error handling and recovery
    """
    
    def __init__(self, max_concurrent_jobs: int = 1, jobs_dir: str = "./training_jobs"):
        """
        Initialize the training job manager.
        
        Args:
            max_concurrent_jobs: Maximum number of concurrent training jobs
            jobs_dir: Directory to store job data and results
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # Job storage
        self.jobs: Dict[str, TrainingJob] = {}
        self.job_queue = queue.Queue()
        self.running_jobs: Dict[str, threading.Thread] = {}
        
        # Threading
        self.worker_threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()
        
        # Start worker threads
        self._start_workers()
        
        # Load existing jobs
        self._load_jobs()
        
        logger.info(f"TrainingJobManager initialized with {max_concurrent_jobs} max concurrent jobs")
    
    def _start_workers(self):
        """Start worker threads for job execution"""
        for i in range(self.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TrainingWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
            logger.info(f"Started training worker thread: {worker.name}")
    
    def _worker_loop(self):
        """Main worker loop for processing training jobs"""
        while not self.shutdown_event.is_set():
            try:
                # Wait for a job with timeout
                try:
                    job_id = self.job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if job_id is None:  # Shutdown signal
                    break
                
                # Execute the job
                self._execute_job(job_id)
                
                # Mark task as done
                self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker thread error: {e}", exc_info=True)
    
    def _execute_job(self, job_id: str):
        """Execute a training job"""
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        try:
            logger.info(f"Starting execution of job {job_id}")
            
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self._save_job(job)
            
            # Create trainer with resolved model name
            resolved_model_name = get_model_id(job.model_name)
            logger.info(f"Resolved model name: {job.model_name} -> {resolved_model_name}")
            
            trainer = QLoRATrainer(
                model_name=resolved_model_name,
                model_cache_dir=str(self.jobs_dir / "model_cache")
            )
            
            # Prepare model with LoRA config
            if job.lora_config:
                # Extract parameters for prepare_model method
                trainer.prepare_model(
                    lora_rank=job.lora_config.get('r', 4),
                    lora_alpha=job.lora_config.get('lora_alpha', 8),
                    lora_dropout=job.lora_config.get('lora_dropout', 0.05),
                    target_modules=job.lora_config.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"])
                )
            else:
                trainer.prepare_model(lora_rank=4, lora_alpha=8, lora_dropout=0.05)
            
            # Create progress callback
            def progress_callback(step: int, total_steps: int, metrics: Dict):
                job.current_step = step
                job.total_steps = total_steps
                job.progress = (step / total_steps) * 100 if total_steps > 0 else 0
                
                if 'loss' in metrics:
                    job.current_loss = metrics['loss']
                if 'learning_rate' in metrics:
                    job.learning_rate = metrics['learning_rate']
                
                # Save progress
                self._save_job(job)
                logger.info(f"Job {job_id} progress: {job.progress:.1f}% (step {step}/{total_steps})")
            
            # Execute training
            output_dir = job.output_dir or str(self.jobs_dir / f"job_{job_id}")
            training_kwargs = job.training_config or {}
            
            results = trainer.train(
                train_texts=job.train_texts,
                output_dir=output_dir,
                progress_callback=progress_callback,
                **training_kwargs
            )
            
            # Update job with results
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 100.0
            job.metrics = results.metrics if hasattr(results, 'metrics') else {}
            job.adapter_path = results.adapter_path if hasattr(results, 'adapter_path') else None
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            # Handle job failure
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error_message = str(e)
            
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        
        finally:
            # Clean up
            self._save_job(job)
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
    
    def submit_job(
        self,
        model_name: str,
        train_texts: List[str],
        lora_config: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Submit a new training job.
        
        Args:
            model_name: Name of the model to train
            train_texts: List of training texts
            lora_config: LoRA configuration parameters
            training_config: Training configuration parameters
            output_dir: Output directory for training results
            
        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            status=JobStatus.PENDING,
            train_texts=train_texts,
            lora_config=lora_config,
            training_config=training_config,
            output_dir=output_dir
        )
        
        # Store job
        self.jobs[job_id] = job
        self._save_job(job)
        
        # Queue for execution
        self.job_queue.put(job_id)
        
        logger.info(f"Submitted training job {job_id} for model {model_name}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[TrainingJob]:
        """Get all jobs"""
        return list(self.jobs.values())
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a training job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            bool: True if job was cancelled, False otherwise
        """
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status == JobStatus.PENDING:
            # Remove from queue (this is a simplified approach)
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            self._save_job(job)
            logger.info(f"Cancelled pending job {job_id}")
            return True
        
        elif job.status == JobStatus.RUNNING:
            # For running jobs, we would need more sophisticated cancellation
            # This is a placeholder for now
            logger.warning(f"Cannot cancel running job {job_id} - not implemented")
            return False
        
        return False
    
    def _save_job(self, job: TrainingJob):
        """Save job data to disk"""
        try:
            job_file = self.jobs_dir / f"job_{job.job_id}.json"
            with open(job_file, 'w') as f:
                json.dump(job.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save job {job.job_id}: {e}")
    
    def _load_jobs(self):
        """Load existing jobs from disk"""
        try:
            for job_file in self.jobs_dir.glob("job_*.json"):
                try:
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                    
                    # Convert datetime strings back to datetime objects
                    for field in ['created_at', 'started_at', 'completed_at']:
                        if job_data.get(field):
                            job_data[field] = datetime.fromisoformat(job_data[field])
                    
                    # Convert status string to enum
                    job_data['status'] = JobStatus(job_data['status'])
                    
                    # Create job object
                    job = TrainingJob(**job_data)
                    self.jobs[job.job_id] = job
                    
                except Exception as e:
                    logger.error(f"Failed to load job from {job_file}: {e}")
            
            logger.info(f"Loaded {len(self.jobs)} existing jobs")
            
        except Exception as e:
            logger.error(f"Failed to load jobs: {e}")
    
    def cleanup_old_jobs(self, max_age_days: int = 30):
        """Clean up old completed/failed jobs"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                job.completed_at and job.completed_at < cutoff_date):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            # Remove job file
            job_file = self.jobs_dir / f"job_{job_id}.json"
            if job_file.exists():
                job_file.unlink()
            
            # Remove from memory
            del self.jobs[job_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    def shutdown(self):
        """Shutdown the job manager"""
        logger.info("Shutting down TrainingJobManager...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Add shutdown signals to queue for each worker
        for _ in self.worker_threads:
            self.job_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        logger.info("TrainingJobManager shutdown complete")


# Global job manager instance
job_manager: Optional[TrainingJobManager] = None


def get_job_manager() -> TrainingJobManager:
    """Get the global job manager instance"""
    global job_manager
    if job_manager is None:
        job_manager = TrainingJobManager()
    return job_manager


def initialize_job_manager(**kwargs) -> TrainingJobManager:
    """Initialize the global job manager with custom settings"""
    global job_manager
    job_manager = TrainingJobManager(**kwargs)
    return job_manager