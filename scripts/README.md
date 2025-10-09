# Scripts Directory

This directory contains utility scripts for the LocalAI QLoRA training system.

## Available Scripts

### `download_utility.py`
Downloads and configures HuggingFace models for QLoRA training.

**Usage:**
```bash
# From project root
python scripts/download_utility.py <model_id>
python scripts/download_utility.py codellama-7b-instruct --auto-load

# Examples
python scripts/download_utility.py codellama/CodeLlama-7b-Instruct-hf
python scripts/download_utility.py microsoft/DialoGPT-medium
```

**Features:**
- Downloads models with HuggingFace authentication
- Validates model compatibility with QLoRA training
- Automatic model configuration updates
- Progress tracking and error handling

**Requirements:**
- HuggingFace account and token
- Sufficient disk space for model files
- Virtual environment activated

## Script Development Guidelines

When adding new scripts:

1. **Documentation**: Include comprehensive docstrings and usage examples
2. **Error Handling**: Implement robust error handling and user feedback
3. **Logging**: Use the project's logging configuration
4. **Dependencies**: Keep external dependencies minimal
5. **Testing**: Add corresponding test files in `/tests/`

## Integration

Scripts can be imported and used by other parts of the system:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.download_utility import download_model
```