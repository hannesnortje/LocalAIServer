import logging
import torch
from pathlib import Path
from typing import List, Dict, Optional, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel
from huggingface_hub import snapshot_download
import os

from .models_config import (
    AVAILABLE_MODELS, 
    MODEL_DEFAULTS, 
    get_model_config,
    get_model_id,
    get_lora_config
)

logger = logging.getLogger(__name__)

class ModelStatus:
    def __init__(self, loaded: bool, model_type: Optional[str] = None, 
                 context_window: Optional[int] = None, description: Optional[str] = None):
        self.loaded = loaded
        self.model_type = model_type
        self.context_window = context_window
        self.description = description

class ModelManager:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.model_type = None
        self.context_window = 2048  # default context window
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.adapters_dir = models_dir / "adapters"
        self.adapters_dir.mkdir(exist_ok=True)
        self.current_adapter = None
        
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration for M1 Max"""
        try:
            # Try to create quantization config
            # Note: This may not work on M1 Max due to bitsandbytes limitations
            quant_config = MODEL_DEFAULTS["quantization"]
            return BitsAndBytesConfig(
                load_in_4bit=quant_config["load_in_4bit"],
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
                bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"]
            )
        except Exception as e:
            logger.warning(f"Quantization not available: {e}")
            return None

    def load_model(self, model_name: str):
        """Load a HuggingFace model"""
        if self.current_model_name == model_name and self.model is not None:
            logger.info(f"Model {model_name} already loaded")
            return
        
        # Get model configuration
        try:
            model_config = get_model_config(model_name)
            model_id = get_model_id(model_name)
        except ValueError as e:
            logger.error(f"Model {model_name} not found in configuration: {e}")
            raise
        
        logger.info(f"Loading model: {model_name} ({model_id})")
        
        # Clear previous model
        self._unload_model()
        
        try:
            # Check if model is downloaded locally
            local_model_path = self.models_dir / model_name
            
            if local_model_path.exists() and (local_model_path / "config.json").exists():
                logger.info(f"Loading from local path: {local_model_path}")
                model_path = str(local_model_path)
            else:
                logger.info(f"Using HuggingFace Hub: {model_id}")
                model_path = model_id
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=False,
                use_fast=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try quantization first, fall back to FP16
            quantization_config = self._get_quantization_config()
            
            logger.info("Loading model...")
            try:
                # Try with quantization
                if quantization_config is not None:
                    logger.info("Attempting to load with 4-bit quantization...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=False,
                        low_cpu_mem_usage=True
                    )
                else:
                    raise Exception("Quantization not available, using FP16")
                    
            except Exception as quant_error:
                logger.warning(f"Quantization failed: {quant_error}")
                logger.info("Loading with FP16...")
                # Fall back to FP16 without quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto" if self.device == "mps" else None,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True
                )
                
                # Move to MPS if available and not using device_map
                if self.device == "mps" and not hasattr(self.model, "hf_device_map"):
                    self.model = self.model.to(self.device)
            
            # Set model properties
            self.current_model_name = model_name
            self.model_type = "huggingface"
            self.context_window = model_config.get("context_window", 2048)
            
            # Enable eval mode for inference
            self.model.eval()
            
            logger.info(f"Successfully loaded {model_name}")
            logger.info(f"Model device: {self.model.device}")
            logger.info(f"Context window: {self.context_window}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self._unload_model()
            raise
            
    def _unload_model(self):
        """Unload current model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer  
            self.tokenizer = None
        self.current_model_name = None
        self.current_adapter = None
        
        # Clear GPU cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_text(self, prompt: str, max_tokens: int = 512, 
                     temperature: float = 0.1, top_p: float = 0.9, 
                     stop_sequences: Optional[List[str]] = None) -> str:
        """Generate text using the loaded model"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=self.context_window - max_tokens)
        
        # Move to device
        if self.device == "mps":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )
                
                # Decode only the new tokens
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Apply stop sequences
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]
                
                return generated_text
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise

    def get_model_status(self) -> ModelStatus:
        """Get current model status"""
        if self.model is None:
            return ModelStatus(loaded=False)
        
        return ModelStatus(
            loaded=True,
            model_type=self.model_type,
            context_window=self.context_window,
            description=f"{self.current_model_name} (HuggingFace)"
        )

    def list_available_models(self) -> List[str]:
        """List all available models from configuration"""
        return list(AVAILABLE_MODELS.keys())

    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model"""
        try:
            return get_model_config(model_name)
        except ValueError:
            return {}

    # Legacy methods for backward compatibility with existing endpoints
    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> Dict:
        """Legacy generate method for compatibility"""
        try:
            generated_text = self.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=kwargs.get('temperature', 0.1),
                top_p=kwargs.get('top_p', 0.9),
                stop_sequences=kwargs.get('stop', [])
            )
            
            return {
                "choices": [{
                    "text": generated_text,
                    "finish_reason": "stop"
                }]
            }
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "choices": [{
                    "text": f"Error: {str(e)}",
                    "finish_reason": "error"
                }]
            }

    def create_chat_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """Create chat completion (simplified implementation)"""
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Generate response
        response = self.generate(prompt, **kwargs)
        
        # Format as chat completion
        if response["choices"]:
            content = response["choices"][0]["text"].strip()
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": response["choices"][0]["finish_reason"]
                }]
            }
        else:
            return {"choices": []}

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert chat messages to a prompt string"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant: ")
        return "\n".join(prompt_parts)

    # Legacy methods for backward compatibility with existing endpoints
    def get_status(self) -> Dict[str, ModelStatus]:
        """Get status of all models (legacy compatibility method)"""
        models = {}
        
        # Add actually downloaded models from the models directory
        for model_name in AVAILABLE_MODELS.keys():
            is_loaded = self.current_model_name == model_name
            
            # Check if model is actually downloaded and complete
            model_path = self.models_dir / model_name
            is_downloaded = False
            
            if model_path.exists() and (model_path / "config.json").exists():
                # Check if model weight files are complete
                # Look for safetensors or bin files
                safetensors_files = list(model_path.glob("*.safetensors"))
                bin_files = list(model_path.glob("*.bin"))
                
                if safetensors_files or bin_files:
                    # For safetensors models, check if index file exists and all referenced files are present
                    index_file = model_path / "model.safetensors.index.json"
                    if index_file.exists():
                        try:
                            import json
                            with open(index_file, 'r') as f:
                                index_data = json.load(f)
                            
                            # Get all required files from the weight_map
                            required_files = set(index_data.get("weight_map", {}).values())
                            existing_files = {f.name for f in safetensors_files}
                            
                            # Check if all required files are present
                            is_downloaded = required_files.issubset(existing_files)
                        except (json.JSONDecodeError, FileNotFoundError, KeyError):
                            # If we can't parse the index, fall back to checking if files exist
                            is_downloaded = len(safetensors_files) > 0
                    else:
                        # No index file, assume single file model
                        is_downloaded = len(safetensors_files) > 0 or len(bin_files) > 0
                
            # Only include models that are completely downloaded
            if is_downloaded:
                models[model_name] = ModelStatus(
                    loaded=is_loaded,
                    model_type="huggingface",
                    context_window=self.context_window if is_loaded else None,
                    description=AVAILABLE_MODELS[model_name].get('description')
                )
        
        # Also check for custom uploaded models (files in models directory not in AVAILABLE_MODELS)
        if self.models_dir.exists():
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir() and model_dir.name not in AVAILABLE_MODELS:
                    # Check if it's a valid model directory
                    if (model_dir / "config.json").exists() or any(model_dir.glob("*.gguf")):
                        is_loaded = self.current_model_name == model_dir.name
                        models[model_dir.name] = ModelStatus(
                            loaded=is_loaded,
                            model_type="custom",
                            context_window=self.context_window if is_loaded else None,
                            description="Custom uploaded model"
                        )
        
        return models

    def list_models(self) -> List[Dict[str, str]]:
        """List models for v1/models endpoint (legacy compatibility)"""
        models = []
        
        for model_name, config in AVAILABLE_MODELS.items():
            models.append({
                "id": model_name,
                "object": "model",
                "owned_by": "local",
                "type": "huggingface"
            })
        
        return models

    def update_model_info(self, model_name: str, model_type: Optional[str] = None, 
                         context_window: Optional[int] = None):
        """Update internal model info cache for dynamic model additions (legacy compatibility)"""
        logger.info(f"Updated model info for {model_name}: type={model_type}, context={context_window}")

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response to the given prompt using the loaded model (legacy compatibility)"""
        return self.generate_text(prompt, **kwargs)
# Create global instance
model_manager = ModelManager(Path(__file__).parent / 'models')
