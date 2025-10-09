# QLoRA Training Implementation Roadmap
*HuggingFace + QLoRA Training System for LocalAI Server*

## Overview
This roadmap outlines the complete implementation of QLoRA training capabilities for the LocalAI Server, replacing GGUF models with HuggingFace models to enable training while maintaining excellent inference performance for coding tasks.

## Project Goals
- ✅ Enable QLoRA training on M1 Max (32GB)
- ✅ Support multiple coding-focused models
- ✅ Maintain fast inference for coding tasks
- ✅ Provide REST API for training operations
- ✅ Create personalized coding assistants

## Git Branching Strategy
Each step will be implemented in a separate feature branch:
- Base branch: `apple-mac-m1-23gb-dev`
- Feature branches: `feature/step-XX-description`
- After approval: merge to base branch and continue

---

## Phase 1: Foundation Setup

### Step 1: Dependencies Update
**Branch**: `feature/step-01-qlora-dependencies`

**Objective**: Update project dependencies to support QLoRA training

**Tasks**:
- [x] Update `requirements.txt` with QLoRA dependencies
- [x] Update `pyproject.toml` with training libraries
- [x] Add M1 Max specific optimizations
- [x] Test dependency installation in virtual environment
- [x] Document any M1 Max compatibility issues

**Dependencies to Add**:
```bash
# QLoRA Training Core
peft>=0.6.0
bitsandbytes>=0.41.0  
accelerate>=0.24.0
datasets>=2.14.0
trl>=0.7.0

# Training Monitoring
wandb>=0.15.0
evaluate>=0.4.0

# Updated Core Libraries
transformers>=4.35.0
torch>=2.1.0
```

**Expected Issues**:
- bitsandbytes M1 Max compatibility
- MPS backend setup for PyTorch

**Definition of Done**:
- [x] All dependencies install successfully
- [x] Virtual environment activates without errors
- [x] Basic imports work (transformers, peft, bitsandbytes)
- [x] M1 Max acceleration is detected and working

---

### Step 2: Models Configuration Overhaul
**Branch**: `feature/step-02-huggingface-models-config`

**Objective**: Replace GGUF models with HuggingFace models in configuration

**Tasks**:
- [x] Remove all GGUF model entries from `models_config.py`
- [x] Add HuggingFace model configurations
- [x] Update model metadata for training capabilities
- [x] Add quantization and memory usage information
- [x] Update helper functions for new model structure

**New Model Structure**:
```python
AVAILABLE_MODELS = {
    "codellama-7b-instruct": {
        "name": "CodeLlama 7B Instruct",
        "type": "huggingface",
        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
        "size": "~13GB",
        "quantized_size": "~6GB",
        "trainable": True,
        "training_methods": ["LoRA", "QLoRA"],
        # ... more metadata
    }
}
```

**Models to Include**:
1. CodeLlama 7B Instruct (primary)
2. Mistral 7B Instruct v0.2
3. Phi-2 (for experimentation)

**Definition of Done**:
- [x] No GGUF models remain in configuration
- [x] All HuggingFace models have complete metadata
- [x] Helper functions work with new structure
- [x] Configuration validates without errors

---

### Step 3: Model Manager Overhaul
**Branch**: `feature/step-03-huggingface-model-manager`

**Objective**: Replace llama-cpp model loading with HuggingFace + quantization

**Tasks**:
- [x] Remove llama-cpp-python dependencies from `model_manager.py`
- [x] Implement HuggingFace model loading with 4-bit quantization
- [x] Add LoRA adapter support for inference
- [x] Implement MPS acceleration for M1 Max
- [x] Add model caching and memory management
- [x] Update inference methods for new model format

**Key Changes**:
```python
# Replace Llama() loading with:
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import PeftModel

# 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

**Definition of Done**:
- [x] Models load successfully with 4-bit quantization
- [x] Inference works and produces quality results
- [x] Memory usage is optimized for M1 Max
- [x] LoRA adapter loading framework is in place
- [x] All existing API endpoints still function

---

### Step 4: Download System Update
**Branch**: `feature/step-04-huggingface-downloads`

**Objective**: Implement HuggingFace Hub model downloads with progress tracking

**Tasks**:
- [x] Update `/api/download-model/<model_id>` endpoint
- [x] Implement HuggingFace Hub download with streaming progress
- [x] Handle multi-file downloads (model shards, tokenizer, config)
- [x] Add download validation and error handling
- [x] Update UI to show download progress for large models
- [x] Implement resume capability for interrupted downloads

**Implementation Details**:
```python
from huggingface_hub import snapshot_download
import threading

def download_huggingface_model(model_id, target_dir):
    # Stream download with progress updates
    # Handle tokenizer + model files
    # Report progress via websocket/SSE
```

**Definition of Done**:
- [x] HuggingFace models download successfully
- [x] Progress is accurately reported during download  
- [x] Downloads can be interrupted and resumed
- [x] Error handling works for network issues
- [x] Downloaded models load correctly

---

## Phase 2: Training Infrastructure

### Step 5: Training Data Pipeline
**Branch**: `feature/step-05-training-data-pipeline`

**Objective**: Create system for managing training datasets and preprocessing

**Tasks**:
- [ ] Create `local_ai_server/training/` directory structure
- [ ] Implement dataset upload and validation
- [ ] Add data preprocessing for CodeLlama format
- [ ] Create train/validation split functionality
- [ ] Add dataset quality checks and statistics
- [ ] Implement data format conversion utilities

**New Files**:
```
local_ai_server/training/
├── __init__.py
├── data_manager.py      # Dataset handling
├── preprocessing.py     # Data formatting
├── validation.py        # Data quality checks
└── formats.py          # Format conversion utilities
```

**Data Format Support**:
- JSON instruction datasets
- Conversational format
- Code completion format
- Custom coding principle datasets

**Definition of Done**:
- [ ] Datasets can be uploaded via API
- [ ] Data is validated and preprocessed correctly
- [ ] Train/test splits are created automatically
- [ ] Data statistics are calculated and displayed
- [ ] Multiple format types are supported

---

### Step 6: QLoRA Training Engine
**Branch**: `feature/step-06-qlora-training-engine`

**Objective**: Implement core QLoRA training orchestrator

**Tasks**:
- [ ] Create `QLoRATrainer` class
- [ ] Implement LoRA configuration management
- [ ] Add training loop with progress tracking
- [ ] Implement checkpoint saving and loading
- [ ] Add early stopping and validation monitoring
- [ ] Create training configuration templates

**Core Components**:
```python
class QLoRATrainer:
    def __init__(self, model_name, config):
        # Initialize base model with quantization
        # Setup LoRA configuration
        # Prepare training components
        
    def train(self, dataset, validation_set=None):
        # Execute training loop
        # Monitor metrics
        # Save checkpoints
        
    def save_adapter(self, adapter_name):
        # Save trained LoRA adapter
```

**Training Features**:
- Configurable LoRA parameters (rank, alpha, dropout)
- Multiple target modules support
- Gradient accumulation for memory efficiency
- Learning rate scheduling
- Loss monitoring and early stopping

**Definition of Done**:
- [ ] Basic training loop executes successfully
- [ ] LoRA adapters are created and saved
- [ ] Training progress is tracked and logged
- [ ] Checkpoints can be saved and resumed
- [ ] Memory usage stays within M1 Max limits

---

### Step 7: Training API Endpoints
**Branch**: `feature/step-07-training-api-endpoints`

**Objective**: Create REST API for training operations

**Tasks**:
- [ ] Add training control endpoints to `endpoints.py`
- [ ] Implement training job management
- [ ] Create adapter management endpoints
- [ ] Add training progress monitoring endpoints
- [ ] Implement training data upload endpoints

**New Endpoints**:
```python
POST /api/training/start          # Start training job
GET  /api/training/status/<job_id> # Get training progress
POST /api/training/stop/<job_id>   # Stop training job
GET  /api/training/jobs            # List all training jobs
POST /api/training/data/upload     # Upload training dataset
GET  /api/adapters                 # List trained adapters
POST /api/adapters/<name>/load     # Load adapter for inference
DELETE /api/adapters/<name>        # Delete adapter
```

**Job Management**:
- Background training execution
- Progress tracking with metrics
- Job queuing system
- Training logs and monitoring

**Definition of Done**:
- [ ] All training endpoints function correctly
- [ ] Training jobs run in background
- [ ] Progress is accurately reported
- [ ] Adapters can be managed via API
- [ ] Error handling works for all scenarios

---

### Step 8: Adapter Inference System
**Branch**: `feature/step-08-adapter-inference-system`

**Objective**: Implement system to use trained LoRA adapters during inference

**Tasks**:
- [ ] Update model manager to support adapter loading
- [ ] Implement adapter switching during runtime
- [ ] Add adapter-specific generation parameters
- [ ] Create adapter performance caching
- [ ] Update chat completion endpoints to use adapters

**Adapter Management**:
```python
class AdapterManager:
    def load_adapter(self, adapter_name):
        # Load LoRA adapter onto base model
        
    def switch_adapter(self, adapter_name):
        # Switch between different adapters
        
    def unload_adapter(self):
        # Return to base model
```

**Integration Points**:
- Chat completion with adapter selection
- Adapter-aware prompt formatting
- Performance monitoring with adapters
- Memory management for multiple adapters

**Definition of Done**:
- [ ] Adapters load correctly onto base models
- [ ] Inference quality is improved with adapters
- [ ] Multiple adapters can be switched dynamically
- [ ] Performance overhead is minimal
- [ ] All existing endpoints work with adapters

---

## Phase 3: Integration and Testing

### Step 9: Training Monitoring and Logging
**Branch**: `feature/step-09-training-monitoring`

**Objective**: Implement comprehensive training monitoring and logging system

**Tasks**:
- [ ] Integrate WandB for experiment tracking
- [ ] Add training metrics visualization
- [ ] Implement loss curve monitoring
- [ ] Create training summary reports
- [ ] Add email/webhook notifications for training completion

**Monitoring Features**:
- Real-time loss tracking
- Learning rate scheduling visualization
- Memory usage monitoring
- Training time estimation
- Model performance metrics

**Definition of Done**:
- [ ] Training metrics are logged automatically
- [ ] WandB integration works correctly
- [ ] Training progress is visualized
- [ ] Notifications work for job completion
- [ ] Historical training data is preserved

---

### Step 10: End-to-End Testing
**Branch**: `feature/step-10-e2e-testing`

**Objective**: Create comprehensive test suite and run full workflow testing

**Tasks**:
- [ ] Create sample training datasets
- [ ] Implement automated testing pipeline
- [ ] Test full training workflow
- [ ] Validate adapter quality and performance
- [ ] Create performance benchmarks
- [ ] Document training best practices

**Test Scenarios**:
1. Complete training workflow (upload → train → inference)
2. Multiple adapter management
3. Model switching and performance
4. Error handling and recovery
5. Memory usage and optimization

**Sample Datasets**:
- Coding principles dataset (100 examples)
- Python function examples (500 examples)
- Code review dataset (200 examples)

**Definition of Done**:
- [ ] Full workflow executes without errors
- [ ] Trained adapters improve code quality
- [ ] Performance meets expectations
- [ ] All edge cases are handled
- [ ] Documentation is complete and accurate

---

## Phase 4: Production Readiness

### Step 11: Documentation and User Guide
**Branch**: `feature/step-11-documentation`

**Objective**: Create comprehensive documentation for the training system

**Tasks**:
- [ ] Write user guide for training workflows
- [ ] Document API endpoints with examples
- [ ] Create troubleshooting guide
- [ ] Add performance optimization tips
- [ ] Create video tutorial for common tasks

**Documentation Sections**:
- Getting Started with QLoRA Training
- Creating Training Datasets
- Managing Training Jobs
- Using Trained Adapters
- Performance Optimization
- Troubleshooting Common Issues

**Definition of Done**:
- [ ] All features are documented
- [ ] Examples work correctly
- [ ] User guide is easy to follow
- [ ] API documentation is complete
- [ ] Video tutorial is recorded

---

### Step 12: Performance Optimization
**Branch**: `feature/step-12-performance-optimization`

**Objective**: Optimize system for production use on M1 Max

**Tasks**:
- [ ] Optimize model loading times
- [ ] Implement model caching strategies
- [ ] Add batch processing for training
- [ ] Optimize memory usage patterns
- [ ] Implement generation caching

**Optimization Areas**:
- Faster model initialization
- Reduced memory fragmentation
- Improved inference speed
- Better GPU utilization
- Optimized data loading

**Definition of Done**:
- [ ] Model loading time reduced by 50%
- [ ] Memory usage optimized for 32GB
- [ ] Inference speed improved
- [ ] Training throughput maximized
- [ ] System remains stable under load

---

## Timeline and Effort Estimates

### Phase 1: Foundation (Estimated: 1-2 weeks)
- **Step 1**: Dependencies Update - 1 day
- **Step 2**: Models Configuration - 1 day  
- **Step 3**: Model Manager Overhaul - 3-4 days
- **Step 4**: Download System Update - 2-3 days

### Phase 2: Training Infrastructure (Estimated: 2-3 weeks)
- **Step 5**: Training Data Pipeline - 2-3 days
- **Step 6**: QLoRA Training Engine - 4-5 days
- **Step 7**: Training API Endpoints - 2-3 days
- **Step 8**: Adapter Inference System - 3-4 days

### Phase 3: Integration and Testing (Estimated: 1 week)
- **Step 9**: Training Monitoring - 2 days
- **Step 10**: End-to-End Testing - 3-4 days

### Phase 4: Production Readiness (Estimated: 1 week)
- **Step 11**: Documentation - 2-3 days
- **Step 12**: Performance Optimization - 2-3 days

**Total Estimated Time: 5-7 weeks**

---

## Success Criteria

### Technical Success Metrics
- [ ] ✅ QLoRA training completes successfully on M1 Max
- [ ] ✅ Memory usage stays under 30GB during training
- [ ] ✅ Inference speed within 30% of GGUF performance
- [ ] ✅ Trained adapters improve code quality measurably
- [ ] ✅ System handles multiple concurrent users

### Quality Metrics
- [ ] ✅ Generated code follows user's coding style
- [ ] ✅ Training converges within expected timeframes
- [ ] ✅ Adapters are portable and reusable
- [ ] ✅ Error handling covers all edge cases
- [ ] ✅ Documentation enables self-service usage

### Performance Metrics
- [ ] ✅ Training time: <4 hours for 1000 examples
- [ ] ✅ Model loading: <30 seconds
- [ ] ✅ Inference latency: <5 seconds for typical coding tasks
- [ ] ✅ Memory efficiency: >90% utilization during training
- [ ] ✅ Adapter switching: <5 seconds

---

## Risk Mitigation

### Technical Risks
1. **bitsandbytes M1 compatibility**: Have CPU fallback ready
2. **Memory constraints**: Implement aggressive gradient checkpointing
3. **Training instability**: Use proven hyperparameters and small learning rates
4. **Performance degradation**: Benchmark against GGUF baseline

### Project Risks
1. **Scope creep**: Stick to defined roadmap steps
2. **Integration complexity**: Test each step thoroughly before proceeding
3. **Performance expectations**: Set realistic benchmarks early
4. **User adoption**: Focus on documentation and ease of use

---

## Getting Started

To begin implementation:

1. **Create new branch**: `git checkout -b feature/step-01-qlora-dependencies`
2. **Review Step 1 requirements**
3. **Implement changes**
4. **Test thoroughly**
5. **Commit and push**: `git commit -am "Step 1: Add QLoRA dependencies"`
6. **Request review and approval**
7. **Merge to base branch**
8. **Move to Step 2**

Each step builds on the previous one, so completion order is important. No step should be skipped, but minor adjustments to the plan are acceptable based on discoveries during implementation.

---

*This roadmap provides a structured approach to implementing QLoRA training capabilities while maintaining system stability and performance. Each step includes clear objectives, tasks, and success criteria to ensure progress can be tracked and validated.*