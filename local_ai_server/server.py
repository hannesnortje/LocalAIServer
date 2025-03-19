import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
import logging
import shutil
import time
import ssl
import aiohttp
import asyncio
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from OpenSSL import crypto
except ImportError:
    logger.error("OpenSSL not found. Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "pyOpenSSL", "cryptography"])
    from OpenSSL import crypto

from .models_config import AVAILABLE_MODELS
from .endpoints import router
from .model_manager import model_manager
from .config import (
    HTTP_PORT, HTTPS_PORT, PACKAGE_DIR, 
    MODELS_DIR, STATIC_DIR, SSL_DIR
)

# Configuration has been moved to config.py
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
    allow_origins=["*"],  # Allow all origins for simplicity during development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Mount static files directory
static_dir = PACKAGE_DIR / 'static'
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount API routes
app.include_router(router)

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the index.html page"""
    if not (static_dir / "index.html").exists():
        return RedirectResponse(url="/docs", status_code=status.HTTP_303_SEE_OTHER)
    return FileResponse(static_dir / "index.html")

@app.options("/{rest_of_path:path}")
async def options_handler(rest_of_path: str):
    """Global handler for OPTIONS requests to support CORS preflight requests"""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
            "Access-Control-Max-Age": "86400",
        }
    )

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
        # Follow modern security best practices
        context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # Disable old TLS versions
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
        context.set_alpn_protocols(['h2', 'http/1.1'])  # Support HTTP/2
        
        return context, str(cert_path), str(key_path)
    except Exception as e:
        logger.error(f"Error setting up SSL: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import uvicorn
    try:
        # Note: only run HTTP server here for simplicity
        print(f"Starting server at http://localhost:{HTTP_PORT}")
        print(f"API documentation available at http://localhost:{HTTP_PORT}/docs")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=HTTP_PORT
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
