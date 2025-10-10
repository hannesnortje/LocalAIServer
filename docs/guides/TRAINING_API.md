# Training API Endpoints Documentation
*Step 7: Training API Endpoints - Complete Reference*

## Overview

The Training API provides comprehensive endpoints for managing QLoRA training jobs and LoRA adapters. This API enables users to start training jobs, monitor progress, and manage trained adapters through RESTful HTTP endpoints.

## Base URL
```
http://localhost:5001/api
```

---

## Training Endpoints

### 1. Start Training Job
**Endpoint:** `POST /training/start`

Start a new QLoRA training job with specified parameters.

**Request Body:**
```json
{
  "model_name": "codellama-7b-instruct",
  "train_texts": [
    "### Instruction:\nWrite a Python function\n\n### Response:\ndef example():\n    pass"
  ],
  "lora_config": {
    "r": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"]
  },
  "training_config": {
    "num_epochs": 3,
    "batch_size": 1,
    "learning_rate": 2e-4,
    "max_steps": -1,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500
  },
  "output_dir": "./custom_output"
}
```

**Required Fields:**
- `model_name` (string): Name of the base model to train
- `train_texts` (array): List of training text examples

**Optional Fields:**
- `lora_config` (object): LoRA configuration parameters
- `training_config` (object): Training hyperparameters
- `output_dir` (string): Custom output directory

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "submitted",
  "message": "Training job uuid-string submitted successfully"
}
```

**Status Codes:**
- `200`: Job submitted successfully
- `400`: Invalid request data
- `500`: Server error

---

### 2. Get Training Status
**Endpoint:** `GET /training/status/<job_id>`

Get detailed status and progress information for a training job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "model_name": "codellama-7b-instruct",
  "status": "running",
  "progress": 45.0,
  "created_at": "2025-10-09T12:00:00",
  "started_at": "2025-10-09T12:01:00",
  "completed_at": null,
  "current_step": 450,
  "total_steps": 1000,
  "current_loss": 1.234,
  "learning_rate": 0.0002,
  "metrics": {
    "train_loss": 1.234,
    "eval_loss": 1.456
  },
  "adapter_path": "./training_jobs/job_uuid/adapter",
  "error_message": null
}
```

**Status Values:**
- `pending`: Job is queued for execution
- `running`: Job is currently training
- `completed`: Job finished successfully
- `failed`: Job encountered an error
- `cancelled`: Job was cancelled by user

**Status Codes:**
- `200`: Status retrieved successfully
- `404`: Job not found
- `500`: Server error

---

### 3. List Training Jobs
**Endpoint:** `GET /training/jobs`

Get a list of all training jobs, sorted by creation time (newest first).

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "uuid-1",
      "model_name": "codellama-7b-instruct",
      "status": "completed",
      "progress": 100.0,
      "created_at": "2025-10-09T12:00:00",
      "started_at": "2025-10-09T12:01:00",
      "completed_at": "2025-10-09T12:30:00"
    }
  ],
  "total": 1
}
```

**Status Codes:**
- `200`: Jobs listed successfully
- `500`: Server error

---

### 4. Stop Training Job
**Endpoint:** `POST /training/stop/<job_id>`

Cancel a pending or running training job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "cancelled",
  "message": "Training job uuid-string cancelled successfully"
}
```

**Status Codes:**
- `200`: Job cancelled successfully
- `400`: Job cannot be cancelled (already completed/failed)
- `404`: Job not found
- `500`: Server error

---

### 5. Upload Training Data
**Endpoint:** `POST /training/data/upload`

Upload and validate training data for use in training jobs.

**Request Body:**
```json
{
  "dataset_name": "my_dataset",
  "train_texts": [
    "### Instruction:\nExample instruction\n\n### Response:\nExample response",
    "### Instruction:\nAnother instruction\n\n### Response:\nAnother response"
  ]
}
```

**Required Fields:**
- `train_texts` (array): List of training text strings

**Optional Fields:**
- `dataset_name` (string): Name for the dataset

**Response:**
```json
{
  "dataset_name": "my_dataset",
  "num_texts": 2,
  "status": "uploaded",
  "message": "Training data uploaded successfully with 2 texts"
}
```

**Status Codes:**
- `200`: Data uploaded successfully
- `400`: Invalid data format
- `500`: Server error

---

## Adapter Management Endpoints

### 1. List Adapters
**Endpoint:** `GET /adapters`

Get a list of all available trained LoRA adapters.

**Response:**
```json
{
  "adapters": [
    {
      "name": "coding_assistant_v1",
      "path": "./adapters/coding_assistant_v1",
      "model_name": "codellama-7b-instruct",
      "created_at": "2025-10-09T12:30:00",
      "size_mb": 12.5,
      "description": "Fine-tuned for Python coding tasks",
      "is_loaded": true,
      "metrics": {
        "final_loss": 0.845,
        "training_steps": 1000
      }
    }
  ],
  "total": 1,
  "current_adapter": "coding_assistant_v1"
}
```

**Status Codes:**
- `200`: Adapters listed successfully
- `500`: Server error

---

### 2. Get Adapter Info
**Endpoint:** `GET /adapters/<adapter_name>`

Get detailed information about a specific adapter.

**Response:**
```json
{
  "name": "coding_assistant_v1",
  "path": "./adapters/coding_assistant_v1",
  "model_name": "codellama-7b-instruct",
  "created_at": "2025-10-09T12:30:00",
  "size_mb": 12.5,
  "description": "Fine-tuned for Python coding tasks",
  "is_loaded": true,
  "training_config": {
    "r": 4,
    "lora_alpha": 8,
    "learning_rate": 2e-4
  },
  "metrics": {
    "final_loss": 0.845,
    "training_steps": 1000,
    "training_time_minutes": 15.5
  }
}
```

**Status Codes:**
- `200`: Adapter info retrieved successfully
- `404`: Adapter not found
- `500`: Server error

---

### 3. Load Adapter
**Endpoint:** `POST /adapters/<adapter_name>/load`

Load an adapter for inference use.

**Response:**
```json
{
  "adapter_name": "coding_assistant_v1",
  "status": "loaded",
  "message": "Adapter coding_assistant_v1 loaded successfully"
}
```

**Status Codes:**
- `200`: Adapter loaded successfully
- `400`: Failed to load adapter
- `404`: Adapter not found
- `500`: Server error

---

### 4. Unload Adapter
**Endpoint:** `POST /adapters/unload`

Unload the currently loaded adapter.

**Response:**
```json
{
  "adapter_name": "coding_assistant_v1",
  "status": "unloaded", 
  "message": "Adapter coding_assistant_v1 unloaded successfully"
}
```

**Status Codes:**
- `200`: Adapter unloaded successfully
- `400`: No adapter currently loaded
- `500`: Server error

---

### 5. Delete Adapter
**Endpoint:** `DELETE /adapters/<adapter_name>`

Permanently delete an adapter from the system.

**Response:**
```json
{
  "adapter_name": "coding_assistant_v1",
  "status": "deleted",
  "message": "Adapter coding_assistant_v1 deleted successfully"
}
```

**Status Codes:**
- `200`: Adapter deleted successfully
- `400`: Failed to delete adapter
- `404`: Adapter not found
- `500`: Server error

---

## Configuration Reference

### LoRA Configuration
```json
{
  "r": 4,                          // LoRA rank (4, 8, 16, 32)
  "lora_alpha": 8,                 // LoRA alpha parameter
  "lora_dropout": 0.05,            // LoRA dropout rate
  "target_modules": [              // Target modules for LoRA
    "q_proj", "v_proj"
  ],
  "bias": "none",                  // Bias training ("none", "all", "lora_only")
  "task_type": "CAUSAL_LM"         // Task type
}
```

### Training Configuration
```json
{
  "num_epochs": 3,                 // Number of training epochs
  "batch_size": 1,                 // Per-device batch size
  "learning_rate": 2e-4,           // Learning rate
  "max_steps": -1,                 // Max training steps (-1 = use epochs)
  "gradient_accumulation_steps": 4, // Gradient accumulation
  "warmup_steps": 100,             // Learning rate warmup steps
  "logging_steps": 10,             // Steps between progress logs
  "save_steps": 500,               // Steps between checkpoints
  "max_seq_length": 2048           // Maximum sequence length
}
```

---

## Error Handling

All endpoints return JSON error responses with the following format:

```json
{
  "error": "Descriptive error message"
}
```

### Common Error Scenarios

1. **Invalid Model Name**
   ```json
   {
     "error": "Model model_name not available"
   }
   ```

2. **Missing Required Fields**
   ```json
   {
     "error": "model_name is required"
   }
   ```

3. **Invalid Data Format**
   ```json
   {
     "error": "train_texts must be a list"
   }
   ```

4. **Resource Not Found**
   ```json
   {
     "error": "Job job_id not found"
   }
   ```

5. **Server Errors**
   ```json
   {
     "error": "Failed to start training: Internal server error"
   }
   ```

---

## Usage Examples

### Example 1: Complete Training Workflow

```bash
# 1. Upload training data
curl -X POST http://localhost:5001/api/training/data/upload \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "python_functions",
    "train_texts": [
      "### Instruction:\nWrite a function to add two numbers\n\n### Response:\ndef add(a, b):\n    return a + b"
    ]
  }'

# 2. Start training job
curl -X POST http://localhost:5001/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "codellama-7b-instruct",
    "train_texts": [
      "### Instruction:\nWrite a function to add two numbers\n\n### Response:\ndef add(a, b):\n    return a + b"
    ],
    "training_config": {
      "num_epochs": 1,
      "batch_size": 1,
      "max_steps": 10
    }
  }'

# 3. Monitor training progress
curl http://localhost:5001/api/training/status/JOB_ID

# 4. List available adapters
curl http://localhost:5001/api/adapters

# 5. Load trained adapter
curl -X POST http://localhost:5001/api/adapters/ADAPTER_NAME/load
```

### Example 2: Python Usage

```python
import requests
import json

# Start training
response = requests.post('http://localhost:5001/api/training/start', json={
    "model_name": "codellama-7b-instruct",
    "train_texts": [
        "### Instruction:\nWrite a Python function\n\n### Response:\ndef example():\n    pass"
    ],
    "lora_config": {"r": 4, "lora_alpha": 8},
    "training_config": {"num_epochs": 1, "max_steps": 5}
})

job_data = response.json()
job_id = job_data['job_id']

# Monitor progress
status_response = requests.get(f'http://localhost:5001/api/training/status/{job_id}')
status = status_response.json()
print(f"Training progress: {status['progress']}%")
```

---

## Integration Notes

### Model Manager Integration
- Training jobs automatically use available models from the model manager
- Model loading and caching is handled transparently
- M1 Max optimizations are applied automatically

### Vector Store Integration
- Training can potentially integrate with existing vector store data
- Future enhancement for RAG-based training data generation

### Security Considerations
- All endpoints require the server to be running locally
- No authentication implemented (local development use)
- File system access is limited to configured directories

---

## Next Steps

This API provides the foundation for:
1. **Web UI Integration**: Build a training dashboard
2. **Batch Processing**: Handle multiple training jobs
3. **Advanced Monitoring**: Add metrics visualization
4. **Model Serving**: Integrate trained adapters with inference endpoints
5. **Data Management**: Enhanced dataset management features

The Training API is now ready for production use in Step 8: Adapter Inference System integration!