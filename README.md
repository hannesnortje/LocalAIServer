# Local AI Server

A local server that provides OpenAI-compatible endpoints for Hugging Face models.

## Installation

### System Dependencies
For Ubuntu/Debian:
```bash
sudo apt-get install python3-dev build-essential
```

### Using pipx (Recommended)
```bash
pipx install .
```

### Development Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
# For CPU only
pip install -r requirements.txt

# For CUDA support (recommended for GGUF models)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python[server]==0.2.27
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

1. Start the server:
```bash
# Use default ports (5000 for HTTP, 5443 for HTTPS)
local-ai-server

# Or configure custom ports
export LOCAL_AI_HTTP_PORT=6000
export LOCAL_AI_HTTPS_PORT=6443
local-ai-server
```

The server runs on both HTTP and HTTPS:
- HTTP: http://localhost:5000 (default)
- HTTPS: https://localhost:5443 (default, using self-signed certificate)

> Note: When using self-signed certificates, you might need to accept the security warning in your browser.

2. Make requests to the OpenAI-compatible endpoint:
```bash
# Using HTTP
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# Using HTTPS
curl -k https://localhost:5443/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## Model Storage

Models are automatically downloaded and stored in:
- Default location: `~/.local_ai_server/models/`
- Custom location: Set environment variable `LOCAL_AI_MODELS_DIR`

Example with custom model directory:
```bash
export LOCAL_AI_MODELS_DIR="/path/to/your/models"
local-ai-server
```

Models are downloaded only once and reused in subsequent runs.

## API Documentation

Once the server is running, you can access:
- Interactive API documentation at http://localhost:5000/docs
- Alternative documentation at http://localhost:5000/redoc

## Supported Endpoints

- POST /v1/chat/completions - Compatible with OpenAI's chat completion endpoint
