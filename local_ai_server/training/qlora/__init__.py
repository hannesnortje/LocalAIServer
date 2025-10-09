"""
QLoRA Training Engine for LocalAI Server
========================================

This module provides QLoRA (Quantized Low-Rank Adaptation) training capabilities
for fine-tuning large language models efficiently on M1 Max hardware.

Key Features:
- 4-bit quantization with bitsandbytes for memory efficiency
- LoRA adapters for parameter-efficient fine-tuning
- M1 Max MPS acceleration optimization
- Comprehensive training monitoring and checkpointing
- Integration with HuggingFace transformers and PEFT

Philosophy: "42 = FOR TWO" - Collaborative intelligence between
systematic training orchestration and strategic model optimization.

Components:
- QLoRATrainer: Core training orchestrator
- LoRAConfig: Configuration management for LoRA parameters
- TrainingConfig: Training hyperparameters and optimization settings
- CheckpointManager: Model checkpoint saving and loading
- TrainingMonitor: Progress tracking and metrics logging
"""

from .trainer import QLoRATrainer
from .config import LoRAConfig, TrainingConfig
from .checkpoint import CheckpointManager
from .monitor import TrainingMonitor

__version__ = "0.1.0"
__all__ = [
    'QLoRATrainer',
    'LoRAConfig', 
    'TrainingConfig',
    'CheckpointManager',
    'TrainingMonitor'
]

# Default configurations for common use cases
DEFAULT_LORA_CONFIG = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

DEFAULT_TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "max_seq_length": 2048,
    "fp16": True,
    "dataloader_num_workers": 0,
    "remove_unused_columns": False
}

# M1 Max optimized settings
M1_MAX_OPTIMIZED_CONFIG = {
    "use_mps": True,
    "max_memory_gb": 28,  # Leave 4GB for system
    "gradient_checkpointing": True,
    "dataloader_pin_memory": False,
    "optim": "adamw_torch"
}