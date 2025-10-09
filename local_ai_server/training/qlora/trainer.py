"""
QLoRA Trainer - Core Training Orchestrator
==========================================

This module implements the main QLoRATrainer class that orchestrates
QLoRA fine-tuning with 4-bit quantization and LoRA adapters.

Key Features:
- HuggingFace model loading with 4-bit quantization
- LoRA adapter configuration and application
- M1 Max MPS optimization
- Memory-efficient training with gradient accumulation
- Comprehensive error handling and validation
"""

import os
import json
import logging
import torch
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

logger = logging.getLogger(__name__)

@dataclass
class QLoRATrainingResults:
    """Results from QLoRA training session."""
    adapter_path: str
    training_loss: float
    validation_loss: Optional[float]
    training_steps: int
    epochs_completed: float
    peak_memory_gb: float
    training_time_seconds: float
    config_used: Dict[str, Any]

class QLoRATrainer:
    """
    QLoRA Training Orchestrator for LocalAI Server
    
    Handles the complete QLoRA training workflow:
    1. Model loading with 4-bit quantization
    2. LoRA adapter configuration and application
    3. Training data preparation and validation
    4. Training execution with monitoring
    5. Adapter saving and export
    
    Optimized for M1 Max hardware with 32GB memory.
    """
    
    def __init__(
        self,
        model_name: str,
        model_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = False
    ):
        """
        Initialize QLoRA trainer.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "codellama/CodeLlama-7b-Instruct-hf")
            model_cache_dir: Local cache directory for models
            device: Device to use ("mps", "cuda", "cpu"). Auto-detected if None.
            trust_remote_code: Whether to trust remote code in model loading
        """
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.trust_remote_code = trust_remote_code
        
        # Device detection and setup
        self.device = self._setup_device(device)
        logger.info(f"QLoRATrainer initialized for model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Components (initialized during training)
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.trainer = None
        
        # Training state
        self.is_prepared = False
        self.training_config = None
        self.lora_config = None
        
    def _setup_device(self, device: Optional[str] = None) -> str:
        """Setup and validate training device."""
        if device:
            return device
            
        # Auto-detect best available device
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _create_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create 4-bit quantization configuration for memory efficiency."""
        # Check if bitsandbytes is compatible with current hardware
        if self.device == "mps":
            # For M1 Max, we'll use float16 without quantization for now
            # This is a known limitation of bitsandbytes on M1 Max
            logger.warning("⚠️ 4-bit quantization not available on M1 Max, using float16 instead")
            return None
        
        try:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.uint8,
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not create quantization config: {e}")
            logger.warning("Falling back to float16 without quantization")
            return None
    
    def _create_lora_config(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        bias: str = "none"
    ) -> LoraConfig:
        """Create LoRA configuration for parameter-efficient fine-tuning."""
        if target_modules is None:
            # Default target modules for most causal LM models
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=TaskType.CAUSAL_LM
        )
    
    def prepare_model(
        self,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        **model_kwargs
    ) -> None:
        """
        Prepare model for QLoRA training.
        
        Args:
            lora_rank: Rank of LoRA adaptation
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of modules to apply LoRA to
            **model_kwargs: Additional arguments for model loading
        """
        logger.info("Preparing model for QLoRA training...")
        
        try:
            # Create quantization config (may be None for M1 Max)
            quantization_config = self._create_quantization_config()
            if quantization_config:
                logger.info("✅ 4-bit quantization config created")
            else:
                logger.info("✅ Using float16 without quantization (M1 Max optimization)")
            
            # Load tokenizer
            logger.info(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.model_cache_dir,
                trust_remote_code=self.trust_remote_code,
                padding_side="right"  # Required for training
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            logger.info("✅ Tokenizer loaded successfully")
            
            # Load base model 
            logger.info(f"Loading base model...")
            
            model_loading_kwargs = {
                "cache_dir": self.model_cache_dir,
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": torch.float16,
                **model_kwargs
            }
            
            # Add quantization config only if available
            if quantization_config:
                model_loading_kwargs["quantization_config"] = quantization_config
                model_loading_kwargs["device_map"] = "auto"
                logger.info("📊 Loading with 4-bit quantization")
            else:
                # For M1 Max, load without quantization
                model_loading_kwargs["device_map"] = None
                logger.info("📊 Loading with float16 (no quantization)")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_loading_kwargs
            )
            
            # Move to device if not using device_map
            if not quantization_config:
                self.model = self.model.to(self.device)
            
            logger.info("✅ Base model loaded successfully")
            
            # Prepare model for training
            if quantization_config:
                # Use k-bit training preparation for quantized models
                self.model = prepare_model_for_kbit_training(self.model)
                logger.info("✅ Model prepared for k-bit training")
            else:
                # Enable gradient checkpointing for non-quantized models
                self.model.gradient_checkpointing_enable()
                logger.info("✅ Model prepared with gradient checkpointing")
            
            # Create and apply LoRA config
            self.lora_config = self._create_lora_config(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=target_modules
            )
            
            # Apply LoRA to model
            self.peft_model = get_peft_model(self.model, self.lora_config)
            logger.info("✅ LoRA adapters applied to model")
            
            # Print trainable parameters
            self._print_trainable_parameters()
            
            self.is_prepared = True
            logger.info("🎉 Model preparation complete!")
            
        except Exception as e:
            logger.error(f"❌ Model preparation failed: {e}")
            raise
    
    def _print_trainable_parameters(self) -> None:
        """Print number of trainable parameters."""
        if self.peft_model is None:
            return
            
        trainable_params = 0
        all_param = 0
        
        for _, param in self.peft_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        percentage = 100 * trainable_params / all_param
        logger.info(f"📊 Trainable parameters: {trainable_params:,} / {all_param:,} ({percentage:.2f}%)")
    
    def train(
        self,
        train_dataset: Optional[Dataset] = None,
        train_texts: Optional[List[str]] = None,
        validation_dataset: Optional[Dataset] = None,
        output_dir: str = "./qlora_output",
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        max_seq_length: int = 2048,
        max_steps: int = -1,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 10,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
        **training_kwargs
    ) -> QLoRATrainingResults:
        """
        Execute QLoRA training.
        
        Args:
            train_dataset: Training dataset (optional if train_texts provided)
            train_texts: List of training texts (optional if train_dataset provided)
            validation_dataset: Optional validation dataset
            output_dir: Directory to save training outputs
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Training batch size
            gradient_accumulation_steps: Steps to accumulate gradients
            warmup_steps: Number of warmup steps
            max_seq_length: Maximum sequence length
            save_steps: Steps between model saves
            eval_steps: Steps between evaluations
            logging_steps: Steps between log outputs
            **training_kwargs: Additional training arguments
            
        Returns:
            QLoRATrainingResults with training metrics and paths
        """
        if not self.is_prepared:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")
        
        logger.info("🚀 Starting QLoRA training...")
        
        try:
            # Handle input data
            if train_texts is not None:
                # Convert texts to tokenized dataset
                logger.info("📝 Tokenizing input texts...")
                
                def tokenize_function(text):
                    # Tokenize with padding and truncation
                    tokenized = self.tokenizer(
                        text,
                        truncation=True,
                        padding=True,
                        max_length=max_seq_length,
                        return_tensors="pt"
                    )
                    
                    # For causal LM, labels are the same as input_ids
                    tokenized["labels"] = tokenized["input_ids"].clone()
                    return tokenized
                
                # Tokenize all texts
                tokenized_data = []
                for text in train_texts:
                    tokenized = tokenize_function(text)
                    # Convert tensors to lists for dataset
                    tokenized_item = {
                        "input_ids": tokenized["input_ids"].squeeze().tolist(),
                        "attention_mask": tokenized["attention_mask"].squeeze().tolist(),
                        "labels": tokenized["labels"].squeeze().tolist()
                    }
                    tokenized_data.append(tokenized_item)
                
                train_dataset = Dataset.from_list(tokenized_data)
                logger.info(f"✅ Tokenized {len(train_texts)} texts")
                
            elif train_dataset is None:
                raise ValueError("Either train_dataset or train_texts must be provided")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                logging_steps=logging_steps,
                save_steps=save_steps,
                eval_steps=eval_steps if validation_dataset else None,
                eval_strategy="steps" if validation_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if validation_dataset else False,
                metric_for_best_model="eval_loss" if validation_dataset else None,
                greater_is_better=False,
                report_to=[],  # Disable default logging
                remove_unused_columns=False,
                dataloader_num_workers=0,  # Important for M1 Max
                gradient_checkpointing=True,
                fp16=False if self.device == "mps" else True,  # Disable fp16 on MPS
                dataloader_pin_memory=False,  # Better for M1 Max
                **training_kwargs
            )
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Causal LM (not masked LM)
            )
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            logger.info("✅ Trainer initialized")
            
            # Start training
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if start_time:
                start_time.record()
            
            train_result = self.trainer.train()
            
            # Calculate training time
            if start_time and torch.cuda.is_available():
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                training_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                training_time = 0.0
            
            logger.info("✅ Training completed!")
            
            # Save the adapter
            adapter_path = output_path / "adapter"
            self.peft_model.save_pretrained(adapter_path)
            self.tokenizer.save_pretrained(adapter_path)
            
            logger.info(f"✅ Adapter saved to: {adapter_path}")
            
            # Save training config
            config_path = output_path / "training_config.json"
            # Prepare config for JSON serialization (simplified version)
            try:
                # Get LoRA config safely
                lora_config_dict = {}
                try:
                    lora_config_dict = self.lora_config.to_dict()
                    # Convert any sets to lists
                    if 'target_modules' in lora_config_dict and isinstance(lora_config_dict['target_modules'], set):
                        lora_config_dict['target_modules'] = list(lora_config_dict['target_modules'])
                except Exception:
                    lora_config_dict = {"rank": self.lora_config.r, "alpha": self.lora_config.lora_alpha}
                
                training_config = {
                    "model_name": self.model_name,
                    "lora_config": lora_config_dict,
                    "device": self.device,
                    "training_time_seconds": training_time,
                    "training_args_summary": {
                        "learning_rate": training_args.learning_rate,
                        "num_train_epochs": training_args.num_train_epochs,
                        "per_device_train_batch_size": training_args.per_device_train_batch_size,
                        "max_steps": training_args.max_steps,
                        "output_dir": training_args.output_dir
                    }
                }
            except Exception as e:
                logger.warning(f"⚠️ Could not create full training config: {e}")
                training_config = {
                    "model_name": self.model_name,
                    "device": self.device,
                    "training_time_seconds": training_time,
                    "note": "Full config serialization failed"
                }
            
            with open(config_path, 'w') as f:
                json.dump(training_config, f, indent=2)
            
            logger.info(f"✅ Training config saved to: {config_path}")
            
            # Get memory usage (approximate)
            peak_memory = 0.0
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            elif self.device == "mps":
                # MPS memory tracking is limited, use estimate
                peak_memory = 16.0  # Estimate for CodeLlama 7B + LoRA
            
            # Create results
            results = QLoRATrainingResults(
                adapter_path=str(adapter_path),
                training_loss=train_result.training_loss,
                validation_loss=None,  # Will be added if validation was used
                training_steps=train_result.global_step,
                epochs_completed=num_epochs,
                peak_memory_gb=peak_memory,
                training_time_seconds=training_time,
                config_used=training_config
            )
            
            logger.info("🎉 QLoRA training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def save_adapter(self, adapter_path: str, adapter_name: str = "qlora_adapter") -> str:
        """
        Save trained LoRA adapter.
        
        Args:
            adapter_path: Directory to save adapter
            adapter_name: Name for the adapter
            
        Returns:
            Path to saved adapter
        """
        if self.peft_model is None:
            raise RuntimeError("No trained model to save. Train a model first.")
        
        save_path = Path(adapter_path) / adapter_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save PEFT adapter
        self.peft_model.save_pretrained(save_path)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "lora_config": self.lora_config.to_dict() if self.lora_config else {},
            "device": self.device,
            "adapter_name": adapter_name
        }
        
        with open(save_path / "adapter_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Adapter saved to: {save_path}")
        return str(save_path)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_stats = {
            "device": self.device,
            "peak_memory_gb": 0.0,
            "current_memory_gb": 0.0,
            "available_memory_gb": 0.0
        }
        
        if torch.cuda.is_available() and self.device == "cuda":
            current = torch.cuda.memory_allocated() / (1024**3)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            memory_stats.update({
                "current_memory_gb": current,
                "peak_memory_gb": peak,
                "available_memory_gb": total - current
            })
        elif self.device == "mps":
            # MPS memory tracking is limited
            memory_stats.update({
                "current_memory_gb": 16.0,  # Estimate
                "peak_memory_gb": 20.0,
                "available_memory_gb": 12.0
            })
        
        return memory_stats
    
    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        if self.trainer:
            del self.trainer
        if self.peft_model:
            del self.peft_model
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_prepared = False
        logger.info("✅ Resources cleaned up")