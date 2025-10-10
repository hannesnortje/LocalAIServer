#!/usr/bin/env python3
"""
Adapter Structure Fix Utility

This script fixes the broken adapter structure by:
1. Moving nested adapter files to proper locations
2. Validating adapter configurations
3. Creating missing metadata files
4. Ensuring consistent structure
"""

import sys
import shutil
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from local_ai_server.config import ADAPTERS_DIR

def fix_adapter_structure(adapter_name: str) -> bool:
    """Fix the structure of a specific adapter"""
    adapter_path = ADAPTERS_DIR / adapter_name
    
    if not adapter_path.exists():
        print(f"❌ Adapter {adapter_name} not found at {adapter_path}")
        return False
    
    print(f"🔧 Fixing adapter structure: {adapter_name}")
    
    # Check if there's a nested adapter directory
    nested_adapter_path = adapter_path / "adapter"
    if nested_adapter_path.exists():
        print(f"   📁 Found nested adapter directory, flattening structure...")
        
        # Move all files from nested directory to parent
        for item in nested_adapter_path.iterdir():
            dest = adapter_path / item.name
            if dest.exists():
                print(f"   ⚠️  File already exists, skipping: {item.name}")
                continue
            
            shutil.move(str(item), str(dest))
            print(f"   ✅ Moved: {item.name}")
        
        # Remove empty nested directory
        if not any(nested_adapter_path.iterdir()):
            nested_adapter_path.rmdir()
            print(f"   🗑️  Removed empty nested directory")
    
    # Validate required files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = []
    
    for file_name in required_files:
        file_path = adapter_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"   ❌ Missing required files: {missing_files}")
        return False
    
    print(f"   ✅ All required files present")
    
    # Ensure metadata file exists
    metadata_path = adapter_path / "training_metadata.json"
    if not metadata_path.exists():
        print(f"   📝 Creating missing metadata file...")
        
        # Try to extract info from training_config.json if available
        training_config_path = adapter_path / "training_config.json"
        metadata = {
            "model_name": "codellama-7b-instruct",
            "description": f"LoRA adapter: {adapter_name}",
            "created_at": "2025-10-10T11:25:24"
        }
        
        if training_config_path.exists():
            with open(training_config_path, 'r') as f:
                training_config = json.load(f)
                metadata["training_config"] = training_config
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Created metadata file")
    
    print(f"✅ Fixed adapter structure: {adapter_name}")
    return True

def list_adapters() -> list:
    """List all adapters in the adapters directory"""
    adapters = []
    
    if not ADAPTERS_DIR.exists():
        return adapters
    
    for item in ADAPTERS_DIR.iterdir():
        if item.is_dir():
            adapters.append(item.name)
    
    return adapters

def main():
    """Main execution function"""
    print("🔧 Adapter Structure Fix Utility")
    print("=" * 50)
    print(f"📁 Adapters directory: {ADAPTERS_DIR}")
    
    # List all adapters
    adapters = list_adapters()
    
    if not adapters:
        print("❌ No adapters found")
        return
    
    print(f"📋 Found {len(adapters)} adapters:")
    for adapter in adapters:
        print(f"   - {adapter}")
    
    print("\n🔧 Fixing adapter structures...")
    
    success_count = 0
    for adapter in adapters:
        if fix_adapter_structure(adapter):
            success_count += 1
        print()  # Empty line for readability
    
    print(f"✅ Successfully fixed {success_count}/{len(adapters)} adapters")
    
    if success_count == len(adapters):
        print("🎉 All adapters are now properly structured!")
    else:
        print("⚠️  Some adapters need manual attention")

if __name__ == "__main__":
    main()