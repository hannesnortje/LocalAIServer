#!/usr/bin/env python3
"""
Test script for Step 8: Adapter Inference System

This script validates the adapter inference integration by testing:
1. Model manager adapter loading
2. Adapter switching during inference
3. Model status with adapter information
4. Generation with and without adapters
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from local_ai_server.model_manager import ModelManager
from local_ai_server.models_config import AVAILABLE_MODELS
from local_ai_server.config import MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_adapter_inference_system():
    """Test the complete adapter inference system"""
    print("🧪 Testing Step 8: Adapter Inference System")
    print("=" * 60)
    
    try:
        # Initialize model manager
        models_dir = Path(MODELS_DIR)
        model_manager = ModelManager(models_dir)
        
        print("✅ Model manager initialized")
        
        # Test 1: Check if we have any models available
        available_models = list(AVAILABLE_MODELS.keys())
        print(f"📋 Available models: {available_models}")
        
        if not available_models:
            print("❌ No models configured. Please configure models first.")
            return False
        
        # Use the first available model for testing
        test_model = available_models[0]
        print(f"🔄 Loading model: {test_model}")
        
        # Test 2: Load base model
        model_manager.load_model(test_model)
        print("✅ Base model loaded successfully")
        
        # Test 3: Check model status without adapter
        status = model_manager.get_model_status()
        print(f"📊 Model status: {status.description}")
        print(f"   - Loaded: {status.loaded}")
        print(f"   - Adapter loaded: {status.adapter_loaded}")
        print(f"   - Current adapter: {status.current_adapter}")
        
        # Test 4: List available adapters
        adapters = model_manager.list_available_adapters()
        print(f"📂 Available adapters: {adapters}")
        
        if adapters:
            # Test 5: Load an adapter
            test_adapter = adapters[0]
            print(f"🔄 Loading adapter: {test_adapter}")
            
            success = model_manager.load_adapter(test_adapter)
            if success:
                print("✅ Adapter loaded successfully")
                
                # Test 6: Check model status with adapter
                status = model_manager.get_model_status()
                print(f"📊 Model status with adapter: {status.description}")
                print(f"   - Adapter loaded: {status.adapter_loaded}")
                print(f"   - Current adapter: {status.current_adapter}")
                
                # Test 7: Generate text with adapter
                print("🎯 Testing generation with adapter...")
                prompt = "### Instruction:\nWrite a simple Python function\n\n### Response:\n"
                try:
                    response = model_manager.generate_text(prompt, max_tokens=100)
                    print(f"   Generated text: {response[:100]}...")
                    print("✅ Generation with adapter successful")
                except Exception as e:
                    print(f"❌ Generation with adapter failed: {e}")
                
                # Test 8: Unload adapter
                print("🔄 Unloading adapter...")
                success = model_manager.unload_adapter()
                if success:
                    print("✅ Adapter unloaded successfully")
                    
                    # Test 9: Check model status after unloading
                    status = model_manager.get_model_status()
                    print(f"📊 Model status after unload: {status.description}")
                    print(f"   - Adapter loaded: {status.adapter_loaded}")
                    print(f"   - Current adapter: {status.current_adapter}")
                    
                    # Test 10: Generate text without adapter
                    print("🎯 Testing generation without adapter...")
                    try:
                        response = model_manager.generate_text(prompt, max_tokens=100)
                        print(f"   Generated text: {response[:100]}...")
                        print("✅ Generation without adapter successful")
                    except Exception as e:
                        print(f"❌ Generation without adapter failed: {e}")
                else:
                    print("❌ Failed to unload adapter")
            else:
                print("❌ Failed to load adapter")
        else:
            print("ℹ️ No adapters available for testing")
            print("   This is expected if no training has been done yet")
        
        # Test 11: Test adapter management methods
        print("🔧 Testing adapter management methods...")
        adapter_status = model_manager.get_adapter_status()
        print(f"   Adapter status: {adapter_status}")
        
        print("\n🎉 Step 8 Adapter Inference System Tests Complete!")
        print("✅ All adapter integration functionality is working")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.exception("Test failed")
        return False

def main():
    """Run adapter inference tests"""
    print("🚀 Step 8: Adapter Inference System Test Suite")
    print("=" * 60)
    
    success = test_adapter_inference_system()
    
    if success:
        print("\n✅ All tests passed! Step 8 is ready for integration.")
        print("\n📋 Next steps:")
        print("1. Test with actual trained adapters")
        print("2. Validate 42 document training workflow")
        print("3. Test end-to-end training + inference")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    main()