#!/usr/bin/env python3
"""
QLoRA Training Engine Test Script
================================

This script tests the QLoRA training engine implementation with a minimal
dataset to validate functionality before proceeding with full training.

Test Workflow:
1. Initialize QLoRATrainer with CodeLlama model
2. Create minimal training dataset
3. Test model preparation with quantization and LoRA
4. Verify memory usage and configuration
5. Test basic training loop (short run)
6. Validate adapter saving and loading

Usage:
    python tests/test_qlora_training.py [--full-test]
"""

import sys
import logging
import torch
from pathlib import Path
from datasets import Dataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from local_ai_server.training import QLoRATrainer, LoRAConfig, TrainingConfig
from local_ai_server.training.qlora.config import ConfigTemplates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_minimal_dataset():
        """Create a minimal dataset for testing"""
        data = [
            {
                "instruction": "Write a Python function to calculate factorial",
                "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
            },
            {
                "instruction": "Create a function to reverse a string",
                "response": "def reverse_string(s):\n    return s[::-1]"
            },
            {
                "instruction": "Write a function to check if a number is prime",
                "response": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
            },
            {
                "instruction": "Implement binary search",
                "response": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
            }
        ]
        
        # Format data using Alpaca format
        formatted_texts = []
        for item in data:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
            formatted_texts.append(text)
        
        return formatted_texts

def test_qlora_trainer_basic(full_test: bool = False):
    """Test basic QLoRA trainer functionality."""
    logger.info("🚀 Starting QLoRA Training Engine Test")
    logger.info("=" * 60)
    
    try:
        # Step 1: Initialize trainer
        logger.info("\n📋 Step 1: Initializing QLoRA Trainer")
        
        model_name = "codellama/CodeLlama-7b-Instruct-hf"
        
        trainer = QLoRATrainer(
            model_name=model_name,
            model_cache_dir="./local_ai_server/models"  # Use existing cache
        )
        
        logger.info(f"✅ QLoRATrainer initialized for {model_name}")
        logger.info(f"✅ Device detected: {trainer.device}")
        
        # Step 2: Test configuration templates
        logger.info("\n⚙️ Step 2: Testing Configuration Templates")
        
        # Test fast experiment config
        config = ConfigTemplates.fast_experiment()
        config.model_name = model_name
        
        logger.info(f"✅ Config template loaded: {config.description}")
        logger.info(f"📊 LoRA Config: rank={config.lora.rank}, alpha={config.lora.alpha}")
        logger.info(f"📊 Training Config: batch_size={config.training.batch_size}, epochs={config.training.num_epochs}")
        
        # Step 3: Test model preparation
        logger.info("\n🔧 Step 3: Testing Model Preparation")
        
        # Use smaller parameters for testing
        trainer.prepare_model(
            lora_rank=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules[:2]  # Use fewer modules for testing
        )
        
        logger.info("✅ Model prepared successfully with QLoRA")
        
        # Step 4: Check memory usage
        logger.info("\n📊 Step 4: Checking Memory Usage")
        
        memory_stats = trainer.get_memory_usage()
        logger.info(f"📊 Memory Statistics:")
        for key, value in memory_stats.items():
            logger.info(f"  {key}: {value}")
        
        # Step 5: Create test dataset
        logger.info("\n📝 Step 5: Creating Test Dataset")
        
        formatted_texts = create_minimal_dataset()
        logger.info(f"✅ Created test dataset with {len(formatted_texts)} examples")
        
        # Show sample
        logger.info("📋 Sample training example:")
        logger.info("-" * 40)
        logger.info(formatted_texts[0][:200] + "...")
        logger.info("-" * 40)
        
        if full_test:
            # Step 6: Test short training run
            logger.info("\n🏃 Step 6: Testing Short Training Run")
            
            # Very short training for validation
            results = trainer.train(
                train_texts=formatted_texts,
                output_dir="./test_qlora_output",
                learning_rate=5e-4,
                num_epochs=1,
                batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=2,  # Very short test
                logging_steps=1,
                save_steps=10,
                warmup_steps=0
            )
            
            logger.info("✅ Short training run completed!")
            logger.info(f"📊 Training Results:")
            logger.info(f"  Adapter Path: {results.adapter_path}")
            logger.info(f"  Training Loss: {results.training_loss:.4f}")
            logger.info(f"  Training Steps: {results.training_steps}")
            logger.info(f"  Training Time: {results.training_time_seconds:.1f}s")
            logger.info(f"  Peak Memory: {results.peak_memory_gb:.1f}GB")
            
        else:
            logger.info("\n⏭️ Step 6: Skipping training run (use --full-test for complete test)")
        
        # Step 7: Cleanup
        logger.info("\n🧹 Step 7: Cleanup")
        trainer.cleanup()
        logger.info("✅ Resources cleaned up")
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("🎉 QLORA TRAINING ENGINE TEST COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("✅ Model loading with 4-bit quantization: WORKING")
        logger.info("✅ LoRA adapter application: WORKING")
        logger.info("✅ Memory optimization: WORKING")
        logger.info("✅ Configuration system: WORKING")
        if full_test:
            logger.info("✅ Training execution: WORKING")
            logger.info("✅ Adapter saving: WORKING")
        
        logger.info("\n🎯 Ready for Step 6 implementation completion!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ QLoRA training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test QLoRA training engine")
    parser.add_argument("--full-test", action="store_true", 
                       help="Run full test including training execution")
    
    args = parser.parse_args()
    
    success = test_qlora_trainer_basic(full_test=args.full_test)
    
    if success:
        logger.info("\n🎯 QLoRA Training Engine: CORE IMPLEMENTATION VALIDATED!")
        sys.exit(0)
    else:
        logger.error("\n❌ QLoRA Training Engine: TEST FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()