import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
import logging
import shutil
import time
import ssl

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from OpenSSL import crypto
except ImportError:
    logger.error("OpenSSL not found. Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "pyOpenSSL", "cryptography"])
    from OpenSSL import crypto

# Configuration
HTTP_PORT = int(os.getenv('LOCAL_AI_HTTP_PORT', 5000))
HTTPS_PORT = int(os.getenv('LOCAL_AI_HTTPS_PORT', 5443))

# Get package directory and models path
PACKAGE_DIR = Path(__file__).parent
MODELS_DIR = Path(os.getenv('LOCAL_AI_MODELS_DIR', PACKAGE_DIR / 'models'))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Package directory: {PACKAGE_DIR}")
logger.info(f"Models directory: {MODELS_DIR}")
logger.info(f"Models directory absolute: {MODELS_DIR.absolute()}")

def verify_model_directory():
    logger.debug(f"Verifying models directory: {MODELS_DIR}")
    
    # Check both package directory and parent directory for models
    search_paths = [PACKAGE_DIR, PACKAGE_DIR.parent]
    for search_path in search_paths:
        project_models = list(search_path.glob('*.gguf'))
        logger.debug(f"Looking for models in {search_path}: found {project_models}")
        
        for model in project_models:
            target = MODELS_DIR / model.name
            if not target.exists():
                logger.info(f"Moving model {model} to {target}")
                shutil.copy2(model, target)

app = FastAPI(
    title="Local AI Server",
    description="A local server that provides OpenAI-compatible endpoints for Hugging Face models",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/health", tags=["Health"])
async def health_check():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "healthy", "version": "0.1.0"}
    )

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 100

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "gpt2",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "temperature": 0.7,
                "max_tokens": 100
            }
        }
    }

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]

class ModelListResponse(BaseModel):
    data: List[Dict[str, str]]

class ModelStatus(BaseModel):
    loaded: bool
    model_type: Optional[str] = None
    context_window: Optional[int] = None

class ModelStatusResponse(BaseModel):
    models: Dict[str, ModelStatus]
    server_status: str = "ready"

class AlternateModelInfo(BaseModel):
    name: str
    type: str
    loaded: bool
    context_size: Optional[int] = None

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.model_type = None
        self.context_window = 512  # default context window
        logger.debug(f"Models directory: {MODELS_DIR}")
        verify_model_directory()
    
    def load_model(self, model_name: str):
        if self.current_model_name == model_name:
            return
        
        model_path = MODELS_DIR / model_name
        
        if model_path.exists():
            print(f"Loading model from {model_path}")
            if str(model_path).endswith('.gguf'):
                self.model = Llama(
                    model_path=str(model_path),
                    n_ctx=2048,  # Increase context window
                    verbose=False
                )
                self.model_type = 'gguf'
                self.tokenizer = None
                self.context_window = self.model.n_ctx()
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                self.model = AutoModelForCausalLM.from_pretrained(str(model_path))
                self.model_type = 'hf'
        else:
            # Only download for HF models
            print(f"Downloading model {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model_type = 'hf'
            
            print(f"Saving model to {model_path}")
            self.tokenizer.save_pretrained(str(model_path))
            self.model.save_pretrained(str(model_path))
        
        self.current_model_name = model_name

    def list_models(self) -> List[Dict[str, str]]:
        models = []
        logger.debug(f"Scanning models directory: {MODELS_DIR}")
        
        try:
            # Print absolute path for debugging
            abs_path = MODELS_DIR.absolute()
            logger.debug(f"Absolute path: {abs_path}")
            logger.debug(f"Directory exists: {abs_path.exists()}")
            logger.debug(f"Is directory: {abs_path.is_dir()}")
            
            # List all files and their details
            if abs_path.exists() and abs_path.is_dir():
                for item in abs_path.iterdir():
                    logger.debug(f"Found item: {item}, is_file: {item.is_file()}")
                    if item.is_file():
                        model_type = 'gguf' if item.name.endswith('.gguf') else 'hf'
                        models.append({
                            "id": item.name,
                            "object": "model",
                            "owned_by": "local",
                            "type": model_type
                        })
            
        except Exception as e:
            logger.error(f"Error scanning models directory: {e}", exc_info=True)
        
        logger.debug(f"Found models: {models}")
        return models

    def get_status(self) -> Dict[str, ModelStatus]:
        models = {}
        for model_file in MODELS_DIR.glob('*'):
            if model_file.is_file():
                model_type = 'gguf' if model_file.name.endswith('.gguf') else 'hf'
                is_loaded = self.current_model_name == model_file.name
                models[model_file.name] = ModelStatus(
                    loaded=is_loaded,
                    model_type=model_type,
                    context_window=self.context_window if is_loaded else None
                )
        return models

    async def generate_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            if self.model_type == 'gguf':
                # Calculate available tokens
                estimated_prompt_tokens = len(prompt.split())  # Rough estimation
                if estimated_prompt_tokens + max_tokens > self.context_window:
                    max_tokens = max(0, self.context_window - estimated_prompt_tokens)
                    logger.warning(f"Adjusted max_tokens to {max_tokens} due to context window limits")
                
                response = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    echo=False
                )
                return response['choices'][0]['text']
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + max_tokens,
                    temperature=temperature,
                )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
        except ValueError as e:
            if "context window" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Input too long. Maximum context window is {self.context_window} tokens."
                )
            raise e
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

model_manager = ModelManager()

@app.post("/v1/chat/completions", 
    tags=["Chat"],
    summary="Create a chat completion",
    response_model=ChatCompletionResponse,
    response_description="The generated chat completion response")
async def chat_completion(request: ChatCompletionRequest):
    try:
        if model_manager.model is None:
            model_manager.load_model(request.model)
        
        prompt = ""
        for msg in request.messages:
            prompt += f"{msg['role']}: {msg['content']}\n"
        
        response_text = await model_manager.generate_response(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatCompletionResponse(
            id="chat_completion",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }]
        )
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.options("/v1/models")
async def models_options():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"methods": ["GET", "OPTIONS"]},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

@app.get("/v1/models",
    tags=["Models"],
    summary="List available models",
    response_model=ModelListResponse,
    response_description="List of available models")
async def list_models():
    response = ModelListResponse(data=model_manager.list_models())
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response.model_dump(),
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

@app.get("/api/models/all",
    tags=["Models"],
    summary="Get all available models",
    response_model=List[AlternateModelInfo])
async def list_all_models():
    models = model_manager.get_status()
    return [
        AlternateModelInfo(
            name=name,
            type=info.model_type or "unknown",
            loaded=info.loaded,
            context_size=info.context_window
        )
        for name, info in models.items()
    ]

@app.get("/api/status/models", 
    tags=["Status"],
    summary="Get models status",
    response_model=ModelStatusResponse)
async def models_status():
    return ModelStatusResponse(
        models=model_manager.get_status(),
        server_status="ready" if model_manager.model is not None else "idle"
    )

def get_ssl_context():
    try:
        ssl_dir = PACKAGE_DIR / 'ssl'
        ssl_dir.mkdir(exist_ok=True)
        cert_path = ssl_dir / 'cert.pem'
        key_path = ssl_dir / 'key.pem'
        
        if not (cert_path.exists() and key_path.exists()):
            logger.info("Generating self-signed SSL certificate...")
            k = crypto.PKey()
            k.generate_key(crypto.TYPE_RSA, 2048)
            cert = crypto.X509()
            cert.get_subject().CN = "localhost"
            cert.set_serial_number(1000)
            cert.gmtime_adj_notBefore(0)
            cert.gmtime_adj_notAfter(10*365*24*60*60)  # 10 years
            cert.set_issuer(cert.get_subject())
            cert.set_pubkey(k)
            cert.sign(k, 'sha256')
            
            with open(cert_path, "wb") as f:
                f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
            with open(key_path, "wb") as f:
                f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
            
            logger.info(f"SSL certificate generated at {ssl_dir}")
        
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(cert_path, key_path)
        return context, str(cert_path), str(key_path)
    except Exception as e:
        logger.error(f"Error setting up SSL: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    try:
        ssl_context, cert_file, key_file = get_ssl_context()
        print(f"Starting server at https://localhost:{HTTPS_PORT}")
        print(f"API documentation available at https://localhost:{HTTPS_PORT}/docs")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=HTTPS_PORT,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
