from flask import jsonify, request, Response, stream_with_context
import json
import logging
import requests
from typing import List, Dict, Optional, Union
import time
from . import __version__
from .models_config import AVAILABLE_MODELS
from .model_manager import model_manager
from .config import MODELS_DIR

logger = logging.getLogger(__name__)

# List of valid parameters that can be passed to the model
VALID_MODEL_PARAMS = {
    'temperature', 'max_tokens', 'stream', 'top_p', 
    'frequency_penalty', 'presence_penalty', 'stop'
}

def setup_routes(app):
    @app.route("/api/available-models", methods=['GET'])
    def get_available_models():
        """Get list of available models for download"""
        return jsonify(AVAILABLE_MODELS)

    @app.route("/api/models/all", methods=['GET'])
    def list_all_models():
        """List all installed models"""
        models = model_manager.get_status()
        return jsonify([{
            "name": name,
            "type": info.model_type or "unknown",
            "loaded": info.loaded,
            "context_size": info.context_window
        } for name, info in models.items()])

    @app.route("/v1/models", methods=['GET'])
    def list_models():
        """List installed models"""
        return jsonify({"data": model_manager.list_models()})

    @app.route("/v1/chat/completions", methods=['POST'])
    def chat_completion():
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            model_name = data.get('model')
            if not model_name:
                return jsonify({"error": "Model name is required"}), 400

            messages = data.get('messages', [])
            if not messages:
                return jsonify({"error": "Messages array is required"}), 400

            # Extract only valid parameters that are explicitly provided
            params = {k: v for k, v in data.items() if k in VALID_MODEL_PARAMS and v is not None}

            # Load model if needed
            if model_manager.model is None or model_manager.current_model_name != model_name:
                try:
                    model_manager.load_model(model_name)
                except Exception as e:
                    return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

            # Handle streaming response
            if params.get('stream', False):
                def generate():
                    try:
                        response = model_manager.create_chat_completion(messages, **params)
                        yield f"data: {json.dumps({
                            'id': f'chat_{int(time.time())}',
                            'object': 'chat.completion.chunk',
                            'created': int(time.time()),
                            'model': model_name,
                            'choices': [{
                                'index': 0,
                                'delta': response,
                                'finish_reason': 'stop'
                            }]
                        })}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logger.error(f"Streaming error: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"

                return Response(
                    stream_with_context(generate()), 
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
                )

            # Handle regular response
            response = model_manager.create_chat_completion(messages, **params)
            return jsonify({
                "id": f"chat_{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": response,
                    "finish_reason": "stop"
                }]
            })

        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/health")
    def health_check():
        """Check server health status"""
        return jsonify({
            "status": "healthy",
            "version": __version__
        })

    @app.route("/api/download-model/<model_id>", methods=['POST'])
    def download_model(model_id: str):
        """Download a model with progress streaming"""
        if model_id not in AVAILABLE_MODELS:
            return jsonify({"error": "Model not found"}), 404
        
        model_info = AVAILABLE_MODELS[model_id]
        target_path = MODELS_DIR / model_id
        temp_path = target_path.with_suffix('.tmp')

        def download_stream():
            if target_path.exists():
                yield json.dumps({
                    "status": "exists",
                    "progress": 100,
                    "message": "Model already downloaded"
                }) + "\n"
                return

            try:
                response = requests.get(model_info["url"], stream=True)
                if response.status_code != 200:
                    if temp_path.exists():
                        temp_path.unlink()
                    yield json.dumps({
                        "status": "error",
                        "message": "Download failed"
                    }) + "\n"
                    return

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
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
                yield json.dumps({
                    "status": "error",
                    "message": str(e)
                }) + "\n"

        return Response(stream_with_context(download_stream()), 
                      mimetype='application/x-ndjson')

    @app.route("/api/models/<model_id>", methods=['DELETE'])
    def delete_model(model_id: str):
        """Delete a model"""
        model_path = MODELS_DIR / model_id
        if not model_path.exists():
            return jsonify({"error": "Model not found"}), 404
        
        try:
            model_path.unlink()
            return jsonify({"status": "success", "message": f"Model {model_id} deleted successfully"})
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return jsonify({"error": "Failed to delete model"}), 500
