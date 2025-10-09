"""
QLoRA Configuration Classes
==========================

This module provides configuration classes for QLoRA training,
including LoRA parameters and training hyperparameters.

Key Features:
- Predefined configurations for different model types
- Validation and optimization for M1 Max hardware
- Easy serialization and loading of configurations
- Template configurations for common use cases
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation) parameters.
    
    Args:
        rank: Rank of adaptation (higher = more parameters but better adaptation)
        alpha: LoRA scaling parameter (typically 2x rank)
        dropout: Dropout probability for LoRA layers
        target_modules: List of modules to apply LoRA to
        bias: How to handle bias parameters ("none", "all", "lora_only")
        modules_to_save: Additional modules to save with adapter
        task_type: Type of task (CAUSAL_LM for language modeling)
    """
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"
    modules_to_save: Optional[List[str]] = None
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("LoRA dropout must be between 0 and 1")
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError("LoRA bias must be 'none', 'all', or 'lora_only'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoRAConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def for_model_size(cls, model_size: str) -> 'LoRAConfig':
        """Create optimized LoRA config based on model size."""
        configs = {
            "7b": cls(rank=16, alpha=32, dropout=0.1),
            "13b": cls(rank=8, alpha=16, dropout=0.05),
            "30b": cls(rank=4, alpha=8, dropout=0.05),
            "65b": cls(rank=4, alpha=8, dropout=0.05)
        }
        
        size_key = model_size.lower().replace("-", "").replace("_", "")
        for key in configs:
            if key in size_key:
                return configs[key]
        
        # Default to 7B config
        return configs["7b"]

@dataclass 
class TrainingConfig:
    """
    Configuration for QLoRA training parameters.
    
    Args:
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        batch_size: Training batch size per device
        gradient_accumulation_steps: Steps to accumulate gradients
        warmup_steps: Number of warmup steps
        max_seq_length: Maximum sequence length for training
        weight_decay: Weight decay for optimizer
        lr_scheduler_type: Type of learning rate scheduler
        save_steps: Steps between model saves
        eval_steps: Steps between evaluations
        logging_steps: Steps between log outputs
        max_grad_norm: Maximum gradient norm for clipping
        fp16: Whether to use 16-bit floating point
        gradient_checkpointing: Whether to use gradient checkpointing
        dataloader_num_workers: Number of dataloader workers
        remove_unused_columns: Whether to remove unused dataset columns
        report_to: List of logging services to report to
        seed: Random seed for reproducibility
    """
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_seq_length: int = 2048
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    max_grad_norm: float = 1.0
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    report_to: List[str] = field(default_factory=list)
    seed: int = 42
    
    # M1 Max specific optimizations
    use_mps: bool = True
    dataloader_pin_memory: bool = False
    optim: str = "adamw_torch"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("Max sequence length must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def for_hardware(cls, hardware_type: str = "m1_max") -> 'TrainingConfig':
        """Create optimized training config for specific hardware."""
        if hardware_type.lower() in ["m1_max", "m1", "mps"]:
            return cls(
                batch_size=1,
                gradient_accumulation_steps=4,
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
                use_mps=True,
                fp16=True,
                gradient_checkpointing=True,
                optim="adamw_torch"
            )
        elif hardware_type.lower() in ["cuda", "gpu"]:
            return cls(
                batch_size=2,
                gradient_accumulation_steps=2,
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
                use_mps=False,
                fp16=True,
                gradient_checkpointing=True,
                optim="adamw_torch"
            )
        else:  # CPU
            return cls(
                batch_size=1,
                gradient_accumulation_steps=8,
                dataloader_num_workers=2,
                dataloader_pin_memory=False,
                use_mps=False,
                fp16=False,
                gradient_checkpointing=True,
                optim="adamw_torch"
            )
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size considering gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

@dataclass
class QLoRAConfig:
    """
    Combined configuration for QLoRA training including both LoRA and training parameters.
    """
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model_name: str = ""
    output_dir: str = "./qlora_output"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lora": self.lora.to_dict(),
            "training": self.training.to_dict(),
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QLoRAConfig':
        """Create from dictionary."""
        return cls(
            lora=LoRAConfig.from_dict(config_dict.get("lora", {})),
            training=TrainingConfig.from_dict(config_dict.get("training", {})),
            model_name=config_dict.get("model_name", ""),
            output_dir=config_dict.get("output_dir", "./qlora_output"),
            description=config_dict.get("description", "")
        )
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Configuration saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'QLoRAConfig':
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        logger.info(f"Configuration loaded from: {filepath}")
        return cls.from_dict(config_dict)

# Predefined configuration templates
class ConfigTemplates:
    """Predefined configuration templates for common use cases."""
    
    @staticmethod
    def coding_assistant() -> QLoRAConfig:
        """Configuration optimized for coding assistant training."""
        return QLoRAConfig(
            lora=LoRAConfig(
                rank=16,
                alpha=32,
                dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            ),
            training=TrainingConfig(
                learning_rate=2e-4,
                num_epochs=3,
                batch_size=1,
                gradient_accumulation_steps=4,
                max_seq_length=2048,
                warmup_steps=100
            ),
            description="Optimized for coding assistant fine-tuning with comprehensive LoRA coverage"
        )
    
    @staticmethod
    def methodology_trainer() -> QLoRAConfig:
        """Configuration optimized for methodology/philosophy training."""
        return QLoRAConfig(
            lora=LoRAConfig(
                rank=8,
                alpha=16,
                dropout=0.05,
                target_modules=["q_proj", "v_proj"]  # Focused on attention for reasoning
            ),
            training=TrainingConfig(
                learning_rate=1e-4,
                num_epochs=5,
                batch_size=1,
                gradient_accumulation_steps=8,
                max_seq_length=1024,
                warmup_steps=50
            ),
            description="Optimized for methodology and philosophy training with careful learning"
        )
    
    @staticmethod
    def fast_experiment() -> QLoRAConfig:
        """Configuration for fast experimentation and testing."""
        return QLoRAConfig(
            lora=LoRAConfig(
                rank=4,
                alpha=8,
                dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            ),
            training=TrainingConfig(
                learning_rate=5e-4,
                num_epochs=1,
                batch_size=1,
                gradient_accumulation_steps=2,
                max_seq_length=512,
                warmup_steps=10,
                save_steps=100,
                eval_steps=100,
                logging_steps=5
            ),
            description="Fast configuration for experimentation and testing"
        )
    
    @staticmethod
    def memory_efficient() -> QLoRAConfig:
        """Configuration optimized for maximum memory efficiency."""
        return QLoRAConfig(
            lora=LoRAConfig(
                rank=4,
                alpha=8,
                dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            ),
            training=TrainingConfig(
                learning_rate=2e-4,
                num_epochs=3,
                batch_size=1,
                gradient_accumulation_steps=16,  # Large accumulation to compensate for small batch
                max_seq_length=1024,
                gradient_checkpointing=True,
                fp16=True,
                dataloader_num_workers=0
            ),
            description="Maximum memory efficiency for large models or limited memory"
        )
    
    @staticmethod
    def get_all_templates() -> Dict[str, QLoRAConfig]:
        """Get all available configuration templates."""
        return {
            "coding_assistant": ConfigTemplates.coding_assistant(),
            "methodology_trainer": ConfigTemplates.methodology_trainer(),
            "fast_experiment": ConfigTemplates.fast_experiment(),
            "memory_efficient": ConfigTemplates.memory_efficient()
        }
    
    @staticmethod
    def list_templates() -> List[str]:
        """List all available template names."""
        return list(ConfigTemplates.get_all_templates().keys())

def create_config_for_model(
    model_name: str,
    use_case: str = "coding_assistant",
    hardware: str = "m1_max"
) -> QLoRAConfig:
    """
    Create an optimized configuration for a specific model and use case.
    
    Args:
        model_name: HuggingFace model identifier
        use_case: Use case template to base config on
        hardware: Hardware type for optimization
        
    Returns:
        Optimized QLoRAConfig
    """
    # Get base template
    templates = ConfigTemplates.get_all_templates()
    if use_case not in templates:
        logger.warning(f"Unknown use case '{use_case}', using 'coding_assistant'")
        use_case = "coding_assistant"
    
    config = templates[use_case]
    
    # Optimize LoRA for model size
    model_size = "7b"  # Default
    for size in ["7b", "13b", "30b", "65b"]:
        if size in model_name.lower():
            model_size = size
            break
    
    config.lora = LoRAConfig.for_model_size(model_size)
    
    # Optimize training for hardware
    config.training = TrainingConfig.for_hardware(hardware)
    
    # Set model name
    config.model_name = model_name
    
    return config