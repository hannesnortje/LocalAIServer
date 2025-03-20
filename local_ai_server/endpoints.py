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
from .vector_store import get_vector_store
from .rag import RAG
from .history_manager import get_response_history  # Use factory function
from .config import ENABLE_RESPONSE_HISTORY

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
            
            # Check if RAG should be used
            use_retrieval = data.get('use_retrieval', False)
            
            if use_retrieval:
                # Get the last message as the query
                if messages[-1]['role'] != 'user':
                    return jsonify({"error": "Last message must be from the user when using retrieval"}), 400
                
                query = messages[-1]['content']
                
                # Extract search parameters
                search_params = data.get('search_params', {})
                
                # Handle streaming for RAG
                if params.get('stream', False):
                    def generate():
                        try:
                            # Generate RAG response
                            rag_params = params.copy()
                            rag_params.pop('stream', None)
                            
                            response = RAG.generate_rag_response(
                                query=query,
                                model_name=model_name,
                                search_params=search_params,
                                generation_params=rag_params
                            )
                            
                            doc_count = len(response['retrieved_documents'])
                            rag_answer = response['answer']
                            
                            yield f"data: {json.dumps({
                                'id': f'chatrag_{int(time.time())}',
                                'object': 'chat.completion.chunk',
                                'created': int(time.time()),
                                'model': model_name,
                                'choices': [{
                                    'index': 0,
                                    'delta': {
                                        'role': 'assistant',
                                        'content': rag_answer
                                    },
                                    'finish_reason': 'stop'
                                }],
                                'usage': {
                                    'prompt_tokens': 0,  # Estimated
                                    'completion_tokens': len(rag_answer.split()),
                                    'total_tokens': len(rag_answer.split())
                                }
                            })}\n\n"
                            yield "data: [DONE]\n\n"
                        except Exception as e:
                            logger.error(f"RAG streaming error: {e}")
                            yield f"data: {json.dumps({'error': str(e)})}\n\n"

                    return Response(
                        stream_with_context(generate()), 
                        mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
                    )
                
                # Regular RAG response
                response = RAG.generate_rag_response(
                    query=query,
                    model_name=model_name,
                    search_params=search_params,
                    generation_params=params
                )
                
                return jsonify({
                    "id": f"chatrag_{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response["answer"]
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": sum(len(m.get('content', '').split()) for m in messages),
                        "completion_tokens": len(response["answer"].split()),
                        "total_tokens": sum(len(m.get('content', '').split()) for m in messages) + len(response["answer"].split())
                    }
                })
            
            # Regular chat completion (non-RAG) - continue with existing code
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

    @app.route("/api/documents", methods=['POST'])
    def add_documents():
        """Add documents to the vector store"""
        try:
            vector_store = get_vector_store()
            data = request.get_json()
            if not data or 'texts' not in data:
                return jsonify({"error": "Missing texts in request"}), 400

            texts = data['texts']
            metadata = data.get('metadata', [{}] * len(texts))

            if len(metadata) != len(texts):
                return jsonify({"error": "Metadata length must match texts length"}), 400

            ids = vector_store.add_texts(texts, metadata)
            return jsonify({
                "status": "success",
                "ids": ids,
                "count": len(ids)
            })
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/search", methods=['POST'])
    def search_documents():
        """Search for similar documents"""
        try:
            vector_store = get_vector_store()
            data = request.get_json()
            if not data or 'query' not in data:
                return jsonify({"error": "Missing query in request"}), 400

            query = data['query']
            k = data.get('limit', 4)
            filter_params = data.get('filter')

            results = vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_params
            )
            return jsonify({
                "status": "success",
                "results": results
            })
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/documents", methods=['DELETE'])
    def delete_documents():
        """Delete documents from the vector store"""
        try:
            vector_store = get_vector_store()
            data = request.get_json()
            if not data or 'ids' not in data:
                return jsonify({"error": "Missing ids in request"}), 400

            vector_store.delete_texts(data['ids'])
            return jsonify({
                "status": "success",
                "message": f"Deleted {len(data['ids'])} documents"
            })
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/v1/embeddings", methods=['POST'])
    def create_embeddings():
        """Create embeddings using the OpenAI-compatible format"""
        try:
            vector_store = get_vector_store()
            data = request.get_json()
            if not data or 'input' not in data:
                return jsonify({"error": "Missing input in request"}), 400

            input_texts = data['input']
            if isinstance(input_texts, str):
                input_texts = [input_texts]

            embeddings = vector_store.model.encode(input_texts, convert_to_numpy=True)
            
            return jsonify({
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": embedding.tolist(),
                        "index": i
                    } for i, embedding in enumerate(embeddings)
                ],
                "model": EMBEDDING_MODEL,
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in input_texts),
                    "total_tokens": sum(len(text.split()) for text in input_texts)
                }
            })
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/v1/completions", methods=['POST'])
    def create_completion():
        """Create completion using the OpenAI-compatible format"""
        try:
            data = request.get_json()
            if not data or 'prompt' not in data:
                return jsonify({"error": "Missing prompt in request"}), 400

            model_name = data.get('model')
            if not model_name:
                return jsonify({"error": "Model name is required"}), 400

            # Extract parameters
            params = {k: v for k, v in data.items() if k in VALID_MODEL_PARAMS and v is not None}
            
            # Load model if needed
            if model_manager.model is None or model_manager.current_model_name != model_name:
                model_manager.load_model(model_name)

            # Handle streaming
            if params.get('stream', False):
                def generate():
                    response = model_manager.generate(data['prompt'], **params)
                    chunk = {
                        "id": f"cmpl-{int(time.time())}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "text": response,
                            "index": 0,
                            "finish_reason": "stop",
                            "logprobs": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return Response(
                    stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache'}
                )

            # Handle regular response
            response = model_manager.generate(data['prompt'], **params)
            return jsonify({
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "text": response,
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None
                }],
                "usage": {
                    "prompt_tokens": len(data['prompt'].split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(data['prompt'].split()) + len(response.split())
                }
            })

        except Exception as e:
            logger.error(f"Error in completion: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/rag", methods=['POST'])
    def rag_completion():
        """Generate response using RAG (Retrieval-Augmented Generation)"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            query = data.get('query')
            if not query:
                return jsonify({"error": "Query is required"}), 400
                
            model_name = data.get('model')
            if not model_name:
                return jsonify({"error": "Model name is required"}), 400
            
            # Get history preference (default to global setting)
            use_history = data.get('use_history', ENABLE_RESPONSE_HISTORY)
                
            # Extract parameters for search and generation
            search_params = data.get('search_params', {})
            generation_params = {k: v for k, v in data.items() if k in VALID_MODEL_PARAMS and v is not None}
            
            # Handle streaming
            if generation_params.get('stream', False):
                def generate():
                    try:
                        # Generate response using RAG
                        generation_params_copy = generation_params.copy()
                        generation_params_copy.pop('stream', None)
                        response = RAG.generate_rag_response(
                            query=query,
                            model_name=model_name,
                            search_params=search_params,
                            generation_params=generation_params_copy,
                            use_history=use_history
                        )
                        
                        yield f"data: {json.dumps({
                            'id': f'rag_{int(time.time())}',
                            'object': 'rag.completion.chunk',
                            'created': int(time.time()),
                            'model': model_name,
                            'answer': response['answer'],
                            'retrieved_document_count': len(response['retrieved_documents']),
                            'history_count': len(response.get('history_items', [])),
                            'finish_reason': 'stop'
                        })}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logger.error(f"RAG streaming error: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
                return Response(
                    stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
                )
            
            # Generate response using RAG
            response = RAG.generate_rag_response(
                query=query,
                model_name=model_name,
                search_params=search_params,
                generation_params=generation_params,
                use_history=use_history
            )
            
            return jsonify({
                "id": f"rag_{int(time.time())}",
                "object": "rag.completion",
                "created": int(time.time()),
                "model": model_name,
                "answer": response["answer"],
                "retrieved_documents": response["retrieved_documents"],
                "history_items": response.get("history_items", []),
                "metadata": response["metadata"]
            })
            
        except Exception as e:
            logger.error(f"Error in RAG completion: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Add history management endpoints
    @app.route("/api/history", methods=['GET'])
    def search_history():
        """Search response history"""
        try:
            if not ENABLE_RESPONSE_HISTORY:
                return jsonify({"error": "Response history is disabled"}), 400
                
            history_manager = get_response_history()  # Use factory function
            query = request.args.get('query', '')
            limit = int(request.args.get('limit', 10))
            min_score = float(request.args.get('min_score', 0.7))
            
            # Parse filter parameters (format: filter.key=value)
            filter_params = {}
            for key, value in request.args.items():
                if (key.startswith('filter.')):
                    filter_key = key[7:]  # Remove 'filter.' prefix
                    filter_params[filter_key] = value
            
            results = history_manager.find_similar_responses(
                query=query,
                limit=limit,
                min_score=min_score,
                filter_params=filter_params if filter_params else None
            )
            
            return jsonify({
                "status": "success",
                "results": results,
                "count": len(results)
            })
            
        except Exception as e:
            logger.error(f"Error searching history: {e}")
            return jsonify({"error": str(e)}), 500
            
    @app.route("/api/history/clean", methods=['POST'])
    def clean_history():
        """Clean old history entries"""
        try:
            if not ENABLE_RESPONSE_HISTORY:
                return jsonify({"error": "Response history is disabled"}), 400
                
            history_manager = get_response_history()  # Use factory function
            data = request.get_json() or {}
            days = int(data.get('days', 30))
            
            count = history_manager.clean_old_entries(days=days)
            
            return jsonify({
                "status": "success",
                "message": f"Cleaned {count} old history entries",
                "count": count
            })
            
        except Exception as e:
            logger.error(f"Error cleaning history: {e}")
            return jsonify({"error": str(e)}), 500
            
    @app.route("/api/history/clear", methods=['POST'])
    def clear_history():
        """Clear all history"""
        try:
            if not ENABLE_RESPONSE_HISTORY:
                return jsonify({"error": "Response history is disabled"}), 400
                
            history_manager = get_response_history()  # Use factory function
            success = history_manager.delete_all_history()
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": "All history cleared"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to clear history"
                }), 500
                
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return jsonify({"error": str(e)}), 500
            
    @app.route("/api/history/status", methods=['GET'])
    def history_status():
        """Get history status"""
        return jsonify({
            "enabled": ENABLE_RESPONSE_HISTORY
        })

    # ...rest of existing routes...
