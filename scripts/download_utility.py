#!/usr/bin/env python3
"""
Model Download Utility for LocalAI Server

This script tests and demonstrates the model download functionality.
Use this to download HuggingFace models to your local system.
"""

import requests
import json
import time
import sys
import argparse

def check_server():
    """Check if server is running"""
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def list_models():
    """List available models"""
    try:
        response = requests.get("http://localhost:5001/api/available-models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("📦 Available models for download:\n")
            for model_id, info in models.items():
                size = info.get('size', 'Unknown size')
                description = info.get('description', '')
                print(f"  🔹 {model_id}")
                print(f"     Name: {info.get('name', 'Unknown')}")
                print(f"     Size: {size}")
                print(f"     Type: {info.get('type', 'Unknown')}")
                if description:
                    print(f"     Info: {description}")
                print()
            return models
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting models: {e}")
        return None

def download_model(model_id, auto_load=False, full_download=False):
    """Download a specific model"""
    print(f"🚀 Starting download of {model_id}...")
    
    try:
        download_response = requests.post(
            f"http://localhost:5001/api/download-model/{model_id}",
            json={"auto_load": auto_load},
            stream=True,
            timeout=300  # 5 minute timeout
        )
        
        if download_response.status_code != 200:
            print(f"❌ Download failed: {download_response.status_code} - {download_response.text}")
            return False
        
        print("📥 Download progress:")
        for line in download_response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    status = data.get('status')
                    progress = data.get('progress', 0)
                    message = data.get('message', '')
                    
                    if status == 'starting':
                        print(f"  🔄 {message}")
                    elif status == 'downloading':
                        print(f"  ⬇️  {message} ({progress}%)")
                    elif status == 'processing':
                        print(f"  🔄 {message} ({progress}%)")
                    elif status == 'success':
                        print(f"  ✅ {message}")
                        print(f"  📁 Model saved to: {data.get('path', 'Unknown path')}")
                        return True
                    elif status == 'exists':
                        print(f"  ℹ️  {message}")
                        if auto_load:
                            continue  # Wait for loading status
                        return True
                    elif status == 'loading':
                        print(f"  🔄 {message}")
                    elif status == 'loaded':
                        print(f"  ✅ {message}")
                        return True
                    elif status == 'error' or status == 'load_error':
                        print(f"  ❌ {message}")
                        return False
                    
                    # Stop early if not doing full download
                    if not full_download and status in ['downloading'] and progress > 10:
                        print(f"  ⏸️  Stopping test download early (reached {progress}%)")
                        return True
                        
                except json.JSONDecodeError:
                    print(f"  📝 Raw response: {line.decode('utf-8')}")
                    
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False
    
    return False

def main():
    parser = argparse.ArgumentParser(description='Download models for LocalAI Server')
    parser.add_argument('model_id', nargs='?', help='Model ID to download')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--auto-load', action='store_true', help='Automatically load model after download')
    parser.add_argument('--full', action='store_true', help='Complete full download (default: test mode)')
    
    args = parser.parse_args()
    
    print("🤖 LocalAI Server - Model Download Utility\n")
    
    # Check server
    if not check_server():
        print("❌ Server not running. Please start the server first:")
        print("   cd /path/to/LocalAIServer")
        print("   source .venv/bin/activate") 
        print("   nohup python -m local_ai_server > server.log 2>&1 &")
        return 1
    
    print("✅ Server is running on http://localhost:5001")
    
    # List models if requested
    if args.list or not args.model_id:
        models = list_models()
        if not models:
            return 1
        
        if not args.model_id:
            print("💡 To download a model, run:")
            print("   python download_utility.py <model_id>")
            print("   python download_utility.py codellama-7b-instruct --auto-load")
            return 0
    
    # Download specific model
    if args.model_id:
        success = download_model(args.model_id, args.auto_load, args.full)
        if success:
            print(f"\n🎉 Model {args.model_id} download completed!")
            if args.auto_load:
                print("🚀 Model is now loaded and ready for use!")
        else:
            print(f"\n❌ Failed to download {args.model_id}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())