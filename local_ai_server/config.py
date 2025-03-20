"""Shared configuration settings."""
import os
from pathlib import Path

# Base directories
PACKAGE_DIR = Path(__file__).parent
PROJECT_DIR = PACKAGE_DIR.parent

# Server configuration
HTTP_PORT = int(os.getenv('HTTP_PORT', 5000))
HTTPS_PORT = int(os.getenv('HTTPS_PORT', 5443))

# Directory structure
MODELS_DIR = PACKAGE_DIR / 'models'
STATIC_DIR = PACKAGE_DIR / 'static'
SSL_DIR = PACKAGE_DIR / 'ssl'

# Ensure directories exist
for directory in [MODELS_DIR, STATIC_DIR, SSL_DIR]:
    directory.mkdir(exist_ok=True)

# Model configuration
DEFAULT_CONTEXT_WINDOW = 2048
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7

# SSL configuration
SSL_CERT_DURATION = 365 * 10  # 10 years
SSL_COUNTRY = "US"
SSL_STATE = "State"
SSL_CITY = "City"
SSL_ORG = "Local AI Server"
SSL_ORG_UNIT = "Development"
SSL_COMMON_NAME = "localhost"

# API configuration
API_VERSION = "1.0"
API_TITLE = "Local AI Server"
API_DESCRIPTION = "A local server that provides OpenAI-compatible endpoints for language models"
