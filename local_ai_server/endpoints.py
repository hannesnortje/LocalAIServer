"""API endpoints for the Local AI Server."""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
import json
import logging
import aiohttp
from typing import List, Dict, Optional, Union
from pydantic import BaseModel
from .models_config import AVAILABLE_MODELS
from .model_manager import model_manager
from .config import MODELS_DIR
from . import __version__
import time

logger = logging.getLogger(__name__)
router = APIRouter()

class ModelListResponse(BaseModel):
    data: List[Dict[str, str]]

class AlternateModelInfo(BaseModel):
    name: str
    type: str
    loaded: bool
    context_size: Optional[int] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    
    # model_config = {
    #     "json_schema_extra": {
    #         "example": {
    #             "model": "phi-2.Q4_K_M.gguf",
    #             "messages": [{"role": "user", "content": "Hello, how are you?"}],
    #             "temperature": 0.7,
    #             "max_tokens": 100
    #         }
    #     }
    # }

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]

@router.get("/api/available-models")
async def get_available_models():
    """Get list of available models for download"""
    return AVAILABLE_MODELS

@router.post("/api/download-model/{model_id}")
async def download_model(model_id: str):
    """Download a model with progress streaming"""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = AVAILABLE_MODELS[model_id]
    target_path = MODELS_DIR / model_id
    temp_path = target_path.with_suffix('.tmp')

    async def download_stream():
        if target_path.exists():
            yield json.dumps({
                "status": "exists",
                "progress": 100,
                "message": "Model already downloaded"
            }) + "\n"
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(model_info["url"]) as response:
                    if response.status != 200:
                        if temp_path.exists():
                            temp_path.unlink()
                        raise HTTPException(status_code=500, detail="Download failed")
                    
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(temp_path, 'wb') as f:
                        downloaded = 0
                        async for data in response.content.iter_chunked(1024*1024):
                            f.write(data)
                            downloaded += len(data)
                            progress = int((downloaded / total_size) * 100) if total_size else 0
                            
                            yield json.dumps({
                                "status": "downloading",
                                "progress": progress,
                                "downloaded": downloaded,
                                "total": total_size
                            }) + "\n"
                    
                    temp_path.rename(target_path)
                    
                    yield json.dumps({
                        "status": "success",
                        "progress": 100,
                        "message": "Model downloaded successfully",
                        "model_id": model_id
                    }) + "\n"
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Download error: {e}")
            raise

    return StreamingResponse(
        download_stream(),
        media_type="application/x-ndjson"
    )

@router.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    model_path = MODELS_DIR / model_id
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model_path.unlink()
        return {"status": "success", "message": f"Model {model_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete model")

@router.get("/api/models/all",
    tags=["Models"],
    summary="Get all available models",
    response_model=List[AlternateModelInfo])
async def list_all_models():
    """List all installed models"""
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

@router.get("/v1/models",
    tags=["Models"],
    summary="List available models",
    response_model=ModelListResponse,
    response_description="List of available models")
async def list_models():
    """List installed models"""
    response = ModelListResponse(data=model_manager.list_models())
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response.model_dump(),
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS, POST",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )

@router.options("/v1/models", include_in_schema=False)
async def models_options():
    """Handle OPTIONS request for models endpoint"""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={},
        headers={
            "Access-Control-Allow-Origin": "*", 
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
            "Content-Length": "0"
        }
    )

@router.post("/v1/chat/completions", 
    tags=["Chat"],
    summary="Create a chat completion",
    response_model=ChatCompletionResponse,
    response_description="The generated chat completion response")
async def chat_completion(request: ChatCompletionRequest):
    """Generate a chat completion response"""
    try:
        if model_manager.model is None or model_manager.current_model_name != request.model:
            model_manager.load_model(request.model)
        
        # Build the prompt from messages
        prompt = ""
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n"
        
        # Pass only specified parameters to the model
        params = {}
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        
        # Let the model manager use its defaults for unspecified parameters
        response_text = await model_manager.generate_response(prompt=prompt, **params)
        
        return ChatCompletionResponse(
            id=f"chat_{int(time.time())}",
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

@router.get("/health", tags=["Health"])
async def health_check():
    """Check server health status"""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "healthy", "version": __version__}
    )
