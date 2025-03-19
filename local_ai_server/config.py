"""Shared configuration settings."""
import os
from pathlib import Path

# Configuration
HTTP_PORT = int(os.getenv('LOCAL_AI_HTTP_PORT', 5000))
HTTPS_PORT = int(os.getenv('LOCAL_AI_HTTPS_PORT', 5443))

# Package directories
PACKAGE_DIR = Path(__file__).parent
MODELS_DIR = Path(os.getenv('LOCAL_AI_MODELS_DIR', PACKAGE_DIR / 'models'))
STATIC_DIR = PACKAGE_DIR / 'static'
SSL_DIR = PACKAGE_DIR / 'ssl'

# Create required directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
SSL_DIR.mkdir(exist_ok=True)
