# Local AI Server

A self-hosted server that provides OpenAI-compatible APIs for running local language models with RAG capabilities.

## Features

- ✅ OpenAI-compatible API endpoints (drop-in replacement for applications)
- ✅ Support for GGUF format models (Llama, Mistral, Phi, etc.)
- ✅ Document storage with vector embeddings
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Response history and conversational memory
- ✅ Swagger UI documentation
- ✅ Easy installation with pipx
- ✅ HTTP and HTTPS support with automatic certificate generation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or pipx (recommended)

### Install with pipx (recommended)

```bash
pipx install git+https://github.com/hannesnortje/LocalAIServer.git
```

This will install the package in an isolated environment and make the CLI command available globally.

### Install with pip

```bash
pip install git+https://github.com/hannesnortje/LocalAIServer.git
```

### GPU Support (optional)

For GPU acceleration, install with CUDA support:

```bash
pip install "git+https://github.com/hannesnortje/LocalAIServer.git#egg=local_ai_server[cuda]"
```

## Quick Start

### Start the server

```bash
local-ai-server start
```

The server will start on:
- HTTP: http://localhost:5000
- HTTPS: https://localhost:5443
- API Documentation: http://localhost:5000/docs

### Download a model

1. Open http://localhost:5000 in your browser
2. Browse available models
3. Click "Download" on a model of your choice (e.g., Phi-2)

Alternatively, use the API:

```bash
curl -X POST http://localhost:5000/api/download-model/phi-2.Q4_K_M.gguf
```

### Add Documents

Add documents to the vector store for RAG:

```bash
curl -X POST http://localhost:5000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This is a sample document about artificial intelligence.", "Another document about machine learning."],
    "metadata": [{"source": "Sample 1"}, {"source": "Sample 2"}]
  }'
```

### Run a chat completion

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-2.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

### Run RAG (Retrieval-Augmented Generation)

Use the RAG endpoint to answer questions based on your documents:

```bash
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "model": "phi-2.Q4_K_M.gguf",
    "use_history": true
  }'
```

## Core Features

### Vector Document Storage

Store and retrieve documents with metadata:

```bash
# Add documents
curl -X POST http://localhost:5000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Document content goes here"],
    "metadata": [{"source": "Book", "author": "John Doe"}]
  }'

# Search documents
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search term",
    "limit": 5,
    "filter": {"author": "John Doe"}
  }'
```

### Response History

Response history is enabled by default. All queries and responses are automatically stored for future context. 

To disable response history:

```bash
# Disable temporarily for a single session
ENABLE_RESPONSE_HISTORY=false local-ai-server start

# Or disable permanently by adding to your shell configuration:
echo 'export ENABLE_RESPONSE_HISTORY=false' >> ~/.bashrc  # For bash
echo 'export ENABLE_RESPONSE_HISTORY=false' >> ~/.zshrc   # For zsh
```

Available history endpoints:
```bash
# Search history
curl "http://localhost:5000/api/history?query=previous%20search&limit=5"

# Clean old history
curl -X POST http://localhost:5000/api/history/clean \
  -H "Content-Type: application/json" \
  -d '{"days": 30}'

# Clear all history
curl -X POST http://localhost:5000/api/history/clear

# Check history status
curl http://localhost:5000/api/history/status
```

### Retrieval-Augmented Generation (RAG)

LocalAIServer provides powerful RAG capabilities that combine document retrieval with language models:

```bash
# Basic RAG query
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "model": "phi-2.Q4_K_M.gguf"
  }'
```

#### Automatic RAG Integration

You can enable automatic RAG for all chat completions, which ensures all queries benefit from document retrieval:

```bash
# Use OpenAI-compatible endpoint with retrieval
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-2.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "use_retrieval": true,
    "search_params": {
      "limit": 5,
      "filter": {"category": "technology"}
    }
  }'
```

To enable automatic RAG by default for all chat completions:

1. Edit `endpoints.py` to change the default value:
   ```python
   use_retrieval = data.get('use_retrieval', True)  # Change from False to True
   ```

2. This seamlessly integrates both document knowledge and conversation history into every response, providing more informative and contextual answers.

The LocalAIServer codebase is already well-structured for this integration:

1. The `/v1/chat/completions` endpoint supports RAG through the `use_retrieval` parameter
2. Changing `use_retrieval` to default to `true` enables automatic RAG for all queries
3. The history management is robust with ChromaDB as the backend, ensuring reliable conversation context

### OpenAI Compatibility

LocalAIServer provides drop-in replacement for OpenAI's API:

```python
import openai

# Configure to use local server
openai.api_base = "http://localhost:5000/v1"
openai.api_key = "not-needed"

# Use just like OpenAI's SDK
response = openai.ChatCompletion.create(
    model="phi-2.Q4_K_M.gguf",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print(response.choices[0].message.content)
```

## API Reference

All API endpoints are documented via Swagger UI at http://localhost:5000/docs when the server is running.

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completion |
| `/v1/completions` | POST | OpenAI-compatible text completion |
| `/v1/embeddings` | POST | OpenAI-compatible text embeddings |
| `/v1/models` | GET | List available models |
| `/api/rag` | POST | Retrieval-Augmented Generation |
| `/api/documents` | POST | Add documents to vector store |
| `/api/search` | POST | Search documents |
| `/api/history` | GET | Search response history |

## Configuration

LocalAIServer can be configured through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_PORT` | 5000 | HTTP server port |
| `HTTPS_PORT` | 5443 | HTTPS server port |
| `VECTOR_DB_TYPE` | chroma | Vector database type (chroma or qdrant) |
| `QDRANT_PATH` | ./storage/vectors | Path for Qdrant vector storage |
| `CHROMA_PATH` | ./storage/chroma | Path for Chroma vector storage |
| `QDRANT_COLLECTION` | documents | Collection name in Qdrant |
| `CHROMA_COLLECTION` | documents | Collection name in Chroma |
| `ENABLE_RESPONSE_HISTORY` | true | Enable/disable response history |
| `MAX_HISTORY_ITEMS` | 5 | Max history items per query |
| `HISTORY_RETENTION_DAYS` | 30 | Days to retain history |

### Vector Database Options

LocalAIServer supports two vector database backends:

1. **ChromaDB** (default): A vector database with excellent concurrency handling
   - Better for multi-user deployments
   - Resilient against concurrent access issues
   - Simple architecture and good performance
   - Recommended for most use cases

2. **Qdrant**: A high-performance vector database
   - Very fast search performance
   - More advanced filtering capabilities
   - Can experience locking issues with concurrent access
   - Consider for single-user deployments with complex queries

To use Qdrant instead of ChromaDB:

```bash
# Set environment variable
export VECTOR_DB_TYPE=qdrant
local-ai-server start

# Or specify on the command line
local-ai-server start --vector-db qdrant
```

## Advanced Usage

### Running with Docker

```bash
docker run -p 5000:5000 -p 5443:5443 -v ./models:/app/models -v ./storage:/app/storage hannesnortje/local-ai-server
```

### Using Custom Models

To use your own models, place the model files in the `models` directory:

```bash
cp path/to/your/model.gguf ~/.local/share/local_ai_server/models/
```

### Integration with LangChain

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="phi-2.Q4_K_M.gguf",
    openai_api_base="http://localhost:5000/v1",
    openai_api_key="not-needed"
)

messages = [HumanMessage(content="Hello, how are you?")]
response = chat(messages)
print(response.content)
```

## Development

### Setting Up Development Environment

```bash
git clone https://github.com/hannesnortje/LocalAIServer.git
cd LocalAIServer
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Running Tests

```bash
python -m pytest
```

### Project Structure

- `local_ai_server/`: Main package
  - `__main__.py`: Entry point
  - `server.py`: Flask server setup
  - `endpoints.py`: API routes
  - `model_manager.py`: Model loading and inference
  - `vector_store.py`: Vector database (Qdrant)
  - `rag.py`: Retrieval-Augmented Generation
  - `history_manager.py`: Response history

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Use different ports: `HTTP_PORT=8000 HTTPS_PORT=8443 local-ai-server start`

2. **Memory issues with large models**
   - Use smaller quantized models (e.g., Q4_K_M variants)
   - Increase system swap space
   - Add CUDA support for GPU acceleration

3. **SSL Certificate Warnings**
   - The server generates self-signed certificates
   - Add exception in your browser or use HTTP for local development

### Logs

Logs are stored in `local_ai_server.log` in the current directory.

## License

MIT
