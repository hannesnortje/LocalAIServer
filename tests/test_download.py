#!/usr/bin/env python3
"""Test script for model download functionality"""

import requests
import json
import time

def test_download():
    """Test the download endpoint"""
    
    # First check if server is running
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        print(f"Server health check: {response.status_code}")
    except Exception as e:
        print(f"Server not accessible: {e}")
        return False
    
    # Get available models
    try:
        response = requests.get("http://localhost:5001/api/available-models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("Available models:")
            for model_id, info in models.items():
                print(f"  - {model_id}: {info.get('name', 'Unknown')}")
            
            # Test with the first available model
            first_model = list(models.keys())[0]
            print(f"\nTesting download with model: {first_model}")
            
            # Test download endpoint
            download_response = requests.post(
                f"http://localhost:5001/api/download-model/{first_model}",
                json={"auto_load": False},
                stream=True,
                timeout=30
            )
            
            print(f"Download response status: {download_response.status_code}")
            
            if download_response.status_code == 200:
                print("Download progress:")
                for line in download_response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            print(f"  Status: {data.get('status')}, Progress: {data.get('progress', 0)}%, Message: {data.get('message', '')}")
                            
                            # Stop after first few updates to avoid long download
                            if data.get('status') in ['downloading', 'success', 'exists', 'error']:
                                break
                        except json.JSONDecodeError:
                            print(f"  Raw line: {line}")
            else:
                print(f"Download failed: {download_response.text}")
                
        else:
            print(f"Failed to get models: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error testing download: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_download()