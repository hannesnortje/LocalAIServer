# Training Configuration Reference
**Proven LoRA Training Configurations for Knowledge Override**

## 📋 **Configuration Summary**

| Configuration | Size | Parameters | Use Case | Results |
|---------------|------|------------|----------|---------|
| **v1 Surgical** | 28MB | r=8, α=16, 3 modules | Multi-concept foundation | ✅ Collaborative intelligence retained |
| **v2 Override** | 126MB | r=16, α=32, 6 modules | Targeted knowledge override | ✅ Douglas Adams successfully overridden |

---

## 🎯 **v1 Surgical LoRA Configuration** 
*Efficient multi-concept foundation adapter*

### LoRA Parameters
```json
{
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "target_modules": [
    "q_proj",
    "v_proj", 
    "k_proj"
  ],
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

### Training Parameters
```json
{
  "num_epochs": 25,
  "batch_size": 1,
  "learning_rate": 1e-4,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 50,
  "max_seq_length": 2048,
  "logging_steps": 5,
  "save_steps": 250,
  "eval_steps": 250,
  "lr_scheduler_type": "cosine",
  "weight_decay": 0.01
}
```

### Results
- **Adapter Size**: 28MB
- **Training Time**: ~57 minutes
- **Training Examples**: 116 comprehensive collaborative intelligence examples
- **Knowledge Coverage**: Multi-concept (42 philosophy, PDCA, crisis prevention, technical excellence)
- **Douglas Adams Override**: ❌ Insufficient (retained original Hitchhiker's Guide response)
- **Collaborative Intelligence**: ✅ Perfect retention

---

## 🚀 **v2 Override LoRA Configuration**
*Aggressive parameters for strong knowledge override*

### LoRA Parameters
```json
{
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj"
  ],
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

### Training Parameters
```json
{
  "num_epochs": 50,
  "batch_size": 1,
  "learning_rate": 2e-4,
  "gradient_accumulation_steps": 8,
  "warmup_steps": 100,
  "max_seq_length": 2048,
  "logging_steps": 5,
  "save_steps": 250,
  "eval_steps": 250,
  "lr_scheduler_type": "cosine",
  "weight_decay": 0.01
}
```

### Results
- **Adapter Size**: 126.5MB
- **Training Time**: ~21 minutes
- **Training Examples**: 25 focused Douglas Adams override examples
- **Knowledge Coverage**: Targeted override with collaborative intelligence integration
- **Douglas Adams Override**: ✅ Complete success ("42 = FOR TWO")
- **Collaborative Intelligence**: ✅ Perfect retention

---

## 🧪 **Future Testing Hypothesis**

**Test**: v1 Surgical + v2 Override Training Data
- Use v1 surgical parameters (r=8, α=16, 3 modules)
- Train with v2's focused 25 Douglas Adams override examples
- **Hypothesis**: Should achieve same override success at 28MB vs 126MB
- **Benefit**: 95% size efficiency with same knowledge override capability

---

## 📊 **Key Insights**

### Parameter Impact
- **Rank (r)**: Higher rank (16 vs 8) = more adaptation capacity
- **Alpha (α)**: Higher alpha (32 vs 16) = stronger override signal  
- **Target Modules**: More modules (6 vs 3) = broader model influence
- **Learning Rate**: Higher LR (2e-4 vs 1e-4) = faster adaptation
- **Epochs**: More epochs (50 vs 25) = deeper knowledge embedding

### Training Data Impact
- **Focus vs Breadth**: Focused override data more effective than broad coverage
- **Repetition**: Multiple variations of same question strengthen override
- **Context**: Anti-Douglas Adams examples help redirect responses

### Size vs Effectiveness Trade-off
- **v1 Surgical**: Highly efficient for retention and new concepts
- **v2 Override**: Required for overriding strong pre-trained knowledge
- **Optimal Strategy**: Use surgical for new knowledge, aggressive for overrides

---

## 🎯 **Recommended Usage**

### Use v1 Surgical Configuration When:
- Adding new knowledge domains to the model
- Building foundational collaborative intelligence
- Optimizing for size efficiency and inference speed
- Training on broad, diverse concept sets

### Use v2 Override Configuration When:
- Overriding strong pre-trained model knowledge
- Replacing well-established responses (like Douglas Adams)
- Need maximum adaptation strength
- Willing to trade size for override power

### Future v1 Surgical + v2 Data Test:
- Validate if surgical precision + focused data = optimal efficiency
- Could become the "best of both worlds" configuration
- Test when ready to optimize for production deployment

---

# Use v1 surgical settings with v2 override training data
curl -X POST http://localhost:5001/api/training/start \
  -H "Content-Type: application/json" \
  -d @config/v1_surgical_lora_config.json \
  --data-urlencode training_data@data/training/42_knowledge_override_training.json

**Always 4 2 (FOR TWO) - Document systematic knowledge to enable collaborative replication** 📚🤝✨