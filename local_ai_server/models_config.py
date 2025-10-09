"""Configuration for available models and their metadata."""

# Add the EMBEDDING_MODEL constant - importing from endpoints.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# M1 Max HuggingFace Models - Optimized for LoRA Training
# All models support training via LoRA (not QLoRA due to bitsandbytes M1 limitations)
AVAILABLE_MODELS = {
    # Tier 1: Primary Coding Models (LoRA Training ~14-16GB RAM)
    "codellama-7b-instruct": {
        "name": "CodeLlama 7B Instruct",
        "description": (
            "Meta's specialized coding model - BEST for LoRA training on M1 Max"
        ),
        "type": "huggingface",
        "model_id": "codellama/CodeLlama-7b-Instruct-hf", 
        "size": "~13GB",
        "quantized_size": "~6.5GB",  # 4-bit quantization
        "context_window": 8192,
        "category": "coding",
        "tier": "1",
        "trainable": True,
        "training_methods": ["LoRA"],  # QLoRA not available on M1 Max
        "training_memory": "~16GB",
        "inference_memory": "~14GB", 
        "m1_optimized": True,
        "supports_mps": True,
        "recommended": True,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    },
    "mistral-7b-instruct-v0.2": {
        "name": "Mistral 7B Instruct v0.2",
        "description": (
            "Excellent for LoRA training - Great coding + reasoning abilities"
        ),
        "type": "huggingface",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "size": "~13GB", 
        "quantized_size": "~6.5GB",
        "context_window": 8192,
        "category": "coding",
        "tier": "1",
        "trainable": True,
        "training_methods": ["LoRA"],
        "training_memory": "~16GB",
        "inference_memory": "~14GB",
        "m1_optimized": True,
        "supports_mps": True,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none", 
            "task_type": "CAUSAL_LM"
        }
    },
    "llama-2-7b-chat": {
        "name": "Llama 2 7B Chat",
        "description": (
            "Proven trainable base with extensive LoRA community support"
        ),
        "type": "huggingface", 
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "size": "~13GB",
        "quantized_size": "~6.5GB",
        "context_window": 4096,
        "category": "coding",
        "tier": "1",
        "trainable": True,
        "training_methods": ["LoRA"],
        "training_memory": "~16GB",
        "inference_memory": "~14GB", 
        "m1_optimized": True,
        "supports_mps": True,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    },
    
    # Tier 2: Advanced Models (Higher Memory Requirements)
    "codellama-13b-instruct": {
        "name": "CodeLlama 13B Instruct", 
        "description": (
            "Advanced coding model - Requires more memory but higher quality"
        ),
        "type": "huggingface",
        "model_id": "codellama/CodeLlama-13b-Instruct-hf",
        "size": "~25GB",
        "quantized_size": "~12GB",
        "context_window": 8192,
        "category": "coding",
        "tier": "2",
        "trainable": True,
        "training_methods": ["LoRA"],
        "training_memory": "~28GB",  # Close to M1 Max limit
        "inference_memory": "~26GB",
        "m1_optimized": True,
        "supports_mps": True,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    },
    
    # Tier 3: Lightweight Models (Fast Experimentation ~8-10GB RAM)
    "phi-2": {
        "name": "Phi-2 (Coding Optimized)",
        "description": (
            "Fast training cycles - Perfect for testing principle injection"
        ),
        "type": "huggingface",
        "model_id": "microsoft/phi-2",
        "size": "~5GB",
        "quantized_size": "~2.5GB", 
        "context_window": 2048,
        "category": "coding",
        "tier": "lightweight",
        "trainable": True,
        "training_methods": ["LoRA"],
        "training_memory": "~8GB",
        "inference_memory": "~6GB",
        "m1_optimized": True,
        "supports_mps": True,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    }
}

# M1 Max Optimized defaults for HuggingFace models
MODEL_DEFAULTS = {
    "huggingface": {
        "max_tokens": 2048,
        "temperature": 0.1,  # Lower temperature for more deterministic code
        "top_p": 0.9,  # Slightly more focused for coding
        "do_sample": True,
        "pad_token_id": None,  # Will be set based on tokenizer
        "eos_token_id": None,  # Will be set based on tokenizer
        "repetition_penalty": 1.1,  # Reduce repetition in code
        "length_penalty": 1.0,
        # M1 Max MPS optimizations
        "torch_dtype": "float16",  # Use FP16 for better performance
        "device_map": "auto",  # Auto device mapping
        "low_cpu_mem_usage": True,  # Optimize for M1 Max
        "trust_remote_code": False,  # Security
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16", 
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_quant_storage": "uint8"
    },
    "lora_defaults": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "inference_mode": False
    },
    # Legacy GGUF defaults (for backward compatibility)
    "gguf": {
        "max_tokens": 4096,  # Increased for longer code generation
        "temperature": 0.1,  # Lower temperature for more deterministic code
        "top_p": 0.9,  # Slightly more focused for coding
        "frequency_penalty": 0.1,  # Reduce repetition in code
        "presence_penalty": 0.0,
        "stop": ["User:", "Assistant:", "```\n\n", "# End of code"],
        # M1 Max specific optimizations
        "n_threads": 8,  # Optimize for M1 Max cores
        "use_mmap": True,  # Memory mapping for efficiency
        "use_mlock": True,  # Lock memory for performance
        "f16_kv": True,  # Use 16-bit for better performance
    }
}


# Helper functions for model management
def get_trainable_models() -> dict:
    """Get only models that can be trained on M1 Max"""
    return {
        k: v for k, v in AVAILABLE_MODELS.items()
        if v.get("trainable", False)
    }


def get_models_by_tier(tier: str) -> dict:
    """Get models filtered by performance tier"""
    return {
        k: v for k, v in AVAILABLE_MODELS.items()
        if v.get("tier") == tier
    }


def get_lora_trainable_models() -> dict:
    """Get models that support LoRA training (all M1 Max models)"""
    result = {}
    for k, v in AVAILABLE_MODELS.items():
        training_methods = v.get("training_methods", [])
        if isinstance(training_methods, list) and "LoRA" in training_methods:
            result[k] = v
    return result


def get_coding_models() -> dict:
    """Get models optimized for coding tasks"""
    return {
        k: v for k, v in AVAILABLE_MODELS.items()
        if v.get("category") == "coding"
    }


def get_m1_optimized_models() -> dict:
    """Get models specifically optimized for M1 Max"""
    return {
        k: v for k, v in AVAILABLE_MODELS.items()
        if v.get("m1_optimized", False)
    }


def get_huggingface_models() -> dict:
    """Get all HuggingFace models"""
    return {
        k: v for k, v in AVAILABLE_MODELS.items()
        if v.get("type") == "huggingface"
    }


def get_recommended_coding_model() -> str:
    """Get the recommended coding model for M1 Max training"""
    # Return Code Llama 7B as the primary recommendation
    return "codellama-7b-instruct"


def get_training_info(model_id: str) -> dict:
    """Get training information for a specific model"""
    model = AVAILABLE_MODELS.get(model_id)
    if not model:
        return {}
    
    return {
        "trainable": model.get("trainable", False),
        "methods": model.get("training_methods", []),
        "memory": model.get("training_memory", "unknown"),
        "inference_memory": model.get("inference_memory", "unknown"),
        "recommended_for_m1": model.get("trainable", False) and model.get("m1_optimized", False),
        "lora_config": model.get("lora_config", {})
    }


def get_model_config(model_id: str) -> dict:
    """Get complete model configuration"""
    model = AVAILABLE_MODELS.get(model_id)
    if not model:
        raise ValueError(f"Model {model_id} not found in configuration")
    return model


def get_model_id(model_name: str) -> str:
    """Get HuggingFace model ID for a model"""
    model = AVAILABLE_MODELS.get(model_name)
    if not model:
        raise ValueError(f"Model {model_name} not found")
    return model.get("model_id", model_name)


def get_lora_config(model_id: str) -> dict:
    """Get LoRA configuration for a specific model"""
    model = AVAILABLE_MODELS.get(model_id)
    if not model:
        return MODEL_DEFAULTS["lora_defaults"].copy()
    
    lora_config = model.get("lora_config", {})
    # Merge with defaults
    defaults = MODEL_DEFAULTS["lora_defaults"].copy()
    defaults.update(lora_config)
    return defaults


def is_model_available_for_training(model_id: str) -> bool:
    """Check if a model is available and can be trained"""
    model = AVAILABLE_MODELS.get(model_id)
    return model is not None and model.get("trainable", False)


def get_memory_requirements(model_id: str, include_training: bool = False) -> dict:
    """Get memory requirements for a model"""
    model = AVAILABLE_MODELS.get(model_id)
    if not model:
        return {"error": "Model not found"}
    
    result = {
        "model_size": model.get("size", "unknown"),
        "quantized_size": model.get("quantized_size", "unknown"),
        "inference_memory": model.get("inference_memory", "unknown")
    }
    
    if include_training:
        result["training_memory"] = model.get("training_memory", "unknown")
    
    return result
