import os
from pathlib import Path
from flask import Flask, redirect, jsonify, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint
import logging
import ssl
from OpenSSL import crypto

from .models_config import AVAILABLE_MODELS
from .endpoints import setup_routes
from .model_manager import model_manager
from .config import (
    PACKAGE_DIR, 
    MODELS_DIR, STATIC_DIR, SSL_DIR,
    HTTP_PORT, HTTPS_PORT
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create static directory if it doesn't exist
static_dir = Path(__file__).parent / 'static'
static_dir.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(static_dir), static_url_path='/static')

# Configure Swagger UI
SWAGGER_URL = '/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Local AI Server"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Set up all routes from endpoints.py
setup_routes(app)

@app.route('/')
def index():
    """Serve the index.html page"""
    try:
        return send_from_directory(str(static_dir), 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return "Error loading page", 500

@app.route('/static/swagger.json')
def serve_swagger():
    """Serve the Swagger JSON configuration"""
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "Local AI Server",
            "description": "A local server that provides OpenAI-compatible endpoints for language models",
            "version": "1.0.0"
        },
        "paths": {
            "/api/available-models": {
                "get": {
                    "summary": "Get list of available models for download",
                    "responses": {"200": {"description": "List of available models"}}
                }
            },
            "/api/models/all": {
                "get": {
                    "summary": "List all installed models",
                    "responses": {"200": {"description": "List of all models"}}
                }
            },
            "/api/download-model/{model_id}": {
                "post": {
                    "summary": "Download a model",
                    "parameters": [
                        {
                            "name": "model_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {
                        "200": {"description": "Download stream"},
                        "404": {"description": "Model not found"}
                    }
                }
            },
            "/api/models/{model_id}": {
                "delete": {
                    "summary": "Delete a model",
                    "parameters": [
                        {
                            "name": "model_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {
                        "200": {"description": "Model deleted"},
                        "404": {"description": "Model not found"}
                    }
                }
            },
            "/v1/models": {
                "get": {
                    "summary": "List installed models",
                    "responses": {"200": {"description": "List of models"}}
                }
            },
            "/v1/chat/completions": {
                "post": {
                    "summary": "Create chat completion",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model": {"type": "string"},
                                        "messages": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "role": {"type": "string"},
                                                    "content": {"type": "string"}
                                                }
                                            }
                                        },
                                        "stream": {"type": "boolean"},
                                        "temperature": {"type": "number"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Chat completion response"},
                        "500": {"description": "Error"}
                    }
                }
            },
            "/v1/embeddings": {
                "post": {
                    "summary": "Create embeddings",
                    "description": "Create embeddings for the given texts (OpenAI-compatible)",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "input": {
                                            "oneOf": [
                                                {"type": "string"},
                                                {"type": "array", "items": {"type": "string"}}
                                            ]
                                        },
                                        "model": {"type": "string"}
                                    },
                                    "required": ["input"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Embeddings generated successfully"},
                        "400": {"description": "Invalid request"}
                    }
                }
            },
            "/v1/completions": {
                "post": {
                    "summary": "Create completion",
                    "description": "Create text completion (OpenAI-compatible)",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model": {"type": "string"},
                                        "prompt": {"type": "string"},
                                        "temperature": {"type": "number"},
                                        "max_tokens": {"type": "integer"},
                                        "stream": {"type": "boolean"}
                                    },
                                    "required": ["model", "prompt"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Completion generated successfully"},
                        "400": {"description": "Invalid request"}
                    }
                }
            },
            "/api/documents": {
                "post": {
                    "summary": "Add documents",
                    "description": "Add documents to the vector store",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "texts": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "metadata": {
                                            "type": "array",
                                            "items": {"type": "object"}
                                        }
                                    },
                                    "required": ["texts"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Documents added successfully"},
                        "400": {"description": "Invalid request"}
                    }
                },
                "delete": {
                    "summary": "Delete documents",
                    "description": "Delete documents from the vector store",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "ids": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["ids"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Documents deleted successfully"},
                        "404": {"description": "Documents not found"}
                    }
                }
            },
            "/api/search": {
                "post": {
                    "summary": "Search documents",
                    "description": "Search for similar documents in the vector store",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"},
                                        "limit": {"type": "integer", "default": 4},
                                        "filter": {"type": "object"}
                                    },
                                    "required": ["query"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Search results"},
                        "400": {"description": "Invalid request"}
                    }
                }
            }
        }
    })

# CORS configuration
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    response.headers.add('Access-Control-Expose-Headers', '*')
    return response

def get_ssl_context():
    """Create SSL context with proper certificate for HTTPS"""
    try:
        cert_path = SSL_DIR / 'cert.pem'
        key_path = SSL_DIR / 'key.pem'
        
        # Generate certificates if they don't exist
        if not (cert_path.exists() and key_path.exists()):
            logger.info("Generating self-signed SSL certificate...")
            try:
                # Configure the certificate
                k = crypto.PKey()
                k.generate_key(crypto.TYPE_RSA, 2048)
                
                cert = crypto.X509()
                cert.get_subject().C = "US"
                cert.get_subject().ST = "State"
                cert.get_subject().L = "City"
                cert.get_subject().O = "Local AI Server"
                cert.get_subject().OU = "Development"
                cert.get_subject().CN = "localhost"
                
                cert.set_serial_number(1000)
                cert.gmtime_adj_notBefore(0)
                cert.gmtime_adj_notAfter(10*365*24*60*60)
                cert.set_issuer(cert.get_subject())
                cert.set_pubkey(k)
                
                # Add Subject Alternative Names for local development
                san_list = b"DNS:localhost,DNS:127.0.0.1,IP:127.0.0.1,DNS:0.0.0.0,IP:0.0.0.0"
                cert.add_extensions([
                    crypto.X509Extension(b"subjectAltName", False, san_list),
                    crypto.X509Extension(b"basicConstraints", True, b"CA:FALSE"),
                    crypto.X509Extension(b"keyUsage", True, b"digitalSignature,keyEncipherment"),
                    crypto.X509Extension(b"extendedKeyUsage", True, b"serverAuth"),
                ])
                
                cert.sign(k, 'sha256')
                
                # Save certificate
                with open(cert_path, "wb") as f:
                    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
                with open(key_path, "wb") as f:
                    f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
                
                logger.info(f"SSL certificate generated at {SSL_DIR}")
            except Exception as e:
                logger.error(f"Error generating certificate: {e}")
                raise
                
        # Create and configure context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # Disable old TLS versions
        # Remove HTTP/2 specific configuration
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
        
        return context, str(cert_path), str(key_path)
    except Exception as e:
        logger.error(f"Error setting up SSL: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    from waitress import serve
    try:
        # Note: only run HTTP server here for simplicity
        print(f"Starting server at http://localhost:{HTTP_PORT}")
        print(f"API documentation available at http://localhost:{HTTP_PORT}/docs")
        serve(
            app, 
            host="0.0.0.0", 
            port=HTTP_PORT
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
