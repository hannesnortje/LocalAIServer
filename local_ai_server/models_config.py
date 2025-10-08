"""Configuration for available models and their metadata."""

# Add the EMBEDDING_MODEL constant - importing from endpoints.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# M1 Max Trainable Coding Models - Only models suitable for fine-tuning
# Focused on models that can be trained with LoRA/QLoRA on M1 Max
AVAILABLE_MODELS = {
    # Tier 1: Best Trainable Models for M1 Max (LoRA Training ~6-7GB RAM)
    "codellama-7b-instruct.Q4_K_M.gguf": {
        "name": "Code Llama 7B Instruct",
        "description": (
            "Meta's specialized coding model - BEST for LoRA training on M1 Max"
        ),
        "size": "4.1GB",
        "type": "gguf",
        "context_window": 8192,
        "category": "coding",
        "tier": "1",
        "trainable": "excellent",
        "training_methods": ["LoRA", "QLoRA"],
        "training_memory": "~6GB",
        "m1_optimized": True,
        "url": (
            "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/"
            "resolve/main/codellama-7b-instruct.Q4_K_M.gguf"
        ),
    },
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf": {
        "name": "Mistral 7B Instruct v0.2",
        "description": (
            "Excellent for LoRA training - Great coding + reasoning abilities"
        ),
        "size": "4.4GB",
        "type": "gguf",
        "context_window": 8192,
        "category": "coding",
        "tier": "1",
        "trainable": "excellent",
        "training_methods": ["LoRA", "QLoRA"],
        "training_memory": "~7GB",
        "m1_optimized": True,
        "url": (
            "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/"
            "resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        ),
    },
    "llama-2-7b-chat.Q4_K_M.gguf": {
        "name": "Llama 2 7B Chat",
        "description": (
            "Proven trainable base with extensive LoRA community support"
        ),
        "size": "4.1GB",
        "type": "gguf",
        "context_window": 4096,
        "category": "coding",
        "tier": "1",
        "trainable": "excellent",
        "training_methods": ["LoRA", "QLoRA", "Full FT"],
        "training_memory": "~6GB",
        "m1_optimized": True,
        "url": (
            "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/"
            "resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
        ),
    },
    
    # Tier 2: Advanced Trainable Models (QLoRA Training ~20-25GB RAM)
    "codellama-13b-instruct.Q4_K_M.gguf": {
        "name": "Code Llama 13B Instruct",
        "description": (
            "Advanced coding model - Trainable with QLoRA on M1 Max"
        ),
        "size": "7.4GB",
        "type": "gguf",
        "context_window": 8192,
        "category": "coding",
        "tier": "2",
        "trainable": "qlora_only",
        "training_methods": ["QLoRA"],
        "training_memory": "~22GB",
        "m1_optimized": True,
        "url": (
            "https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/"
            "resolve/main/codellama-13b-instruct.Q4_K_M.gguf"
        ),
    },
    
    # Tier 3: Lightweight Trainable Models (Fast Experimentation ~3-4GB RAM)
    "phi-2.Q4_K_M.gguf": {
        "name": "Phi-2 (Coding Optimized)",
        "description": (
            "Fast training cycles - Perfect for testing principle injection"
        ),
        "size": "2.3GB",
        "type": "gguf",
        "context_window": 2048,
        "category": "coding",
        "tier": "lightweight",
        "trainable": "excellent",
        "training_methods": ["LoRA", "QLoRA", "Full FT"],
        "training_memory": "~4GB",
        "m1_optimized": True,
        "url": (
            "https://huggingface.co/TheBloke/phi-2-GGUF/"
            "resolve/main/phi-2.Q4_K_M.gguf"
        ),
    }
}

# M1 Max Optimized defaults for coding tasks
MODEL_DEFAULTS = {
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
    },
    "hf": {
        "max_tokens": 2048,
        "temperature": 0.1,  # Lower for coding
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": 50256,  # Common pad token
    }
}


# Helper functions for model management
def get_trainable_models() -> dict:
    """Get only models that can be trained on M1 Max"""
    return {
        k: v for k, v in AVAILABLE_MODELS.items()
        if v.get("trainable") in ["excellent", "qlora_only"]
    }


def get_models_by_tier(tier: str) -> dict:
    """Get models filtered by performance tier"""
    return {
        k: v for k, v in AVAILABLE_MODELS.items()
        if v.get("tier") == tier
    }


def get_lora_trainable_models() -> dict:
    """Get models that support LoRA training (best for M1 Max)"""
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


def get_recommended_coding_model() -> str:
    """Get the recommended coding model for M1 Max training"""
    # Return Code Llama 7B as the primary recommendation
    return "codellama-7b-instruct.Q4_K_M.gguf"


def get_training_info(model_id: str) -> dict:
    """Get training information for a specific model"""
    model = AVAILABLE_MODELS.get(model_id)
    if not model:
        return {}
    
    return {
        "trainable": model.get("trainable", "unknown"),
        "methods": model.get("training_methods", []),
        "memory": model.get("training_memory", "unknown"),
        "recommended_for_m1": model.get("trainable") == "excellent"
    }
