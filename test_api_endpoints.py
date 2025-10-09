#!/usr/bin/env python3
"""
Test script to debug dashboard API issues
"""
import requests
import json

def test_api_endpoints():
    base_url = "http://localhost:5001"
    
    print("🔍 Testing API endpoints...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Available models
    try:
        response = requests.get(f"{base_url}/api/available-models", timeout=5)
        print(f"✅ Available models: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"   Found {len(models)} models")
            first_model = list(models.keys())[0] if models else None
            print(f"   First model: {first_model}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Available models failed: {e}")
        return
    
    # Test 3: Installed models
    try:
        response = requests.get(f"{base_url}/api/models/all", timeout=5)
        print(f"✅ Installed models: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"   Found {len(models)} installed models")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Installed models failed: {e}")
    
    # Test 4: Try a download request (but cancel early)
    if first_model:
        try:
            print(f"\n🧪 Testing download endpoint for {first_model}...")
            response = requests.post(
                f"{base_url}/api/download-model/{first_model}",
                json={"auto_load": False},
                stream=True,
                timeout=10
            )
            print(f"   Download response: {response.status_code}")
            if response.status_code == 200:
                # Read just the first few lines
                lines_read = 0
                for line in response.iter_lines():
                    if line and lines_read < 3:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            print(f"   Progress: {data}")
                            lines_read += 1
                        except:
                            print(f"   Raw line: {line}")
                            lines_read += 1
                    else:
                        break
                print("   ✅ Download endpoint working")
            else:
                print(f"   ❌ Download failed: {response.text}")
        except Exception as e:
            print(f"   ❌ Download test failed: {e}")
    
    # Test 5: Test delete endpoint (with non-existent model)
    try:
        print(f"\n🧪 Testing delete endpoint...")
        response = requests.delete(f"{base_url}/api/models/nonexistent-model", timeout=5)
        print(f"   Delete response: {response.status_code}")
        if response.status_code == 404:
            print("   ✅ Delete endpoint working (returned 404 for non-existent model)")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Delete test failed: {e}")

if __name__ == "__main__":
    test_api_endpoints()