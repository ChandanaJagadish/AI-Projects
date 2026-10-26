#!/usr/bin/env python3
"""
Test the API endpoint directly
"""

import requests
import json
import time

def test_api():
    """Test the animation generation API"""
    try:
        # Test the generate endpoint
        print("Testing /api/generate endpoint...")
        
        url = "http://127.0.0.1:5000/api/generate"
        data = {
            "prompt": "A happy cartoon dog playing in a sunny garden",
            "duration": 10
        }
        
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            print(f"Job ID: {job_id}")
            
            # Check status
            if job_id:
                status_url = f"http://127.0.0.1:5000/api/status/{job_id}"
                
                for i in range(10):  # Check status 10 times
                    print(f"\nChecking status (attempt {i+1})...")
                    status_response = requests.get(status_url)
                    print(f"Status Response: {status_response.text}")
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if status_data.get('status') in ['completed', 'failed']:
                            break
                    
                    time.sleep(2)
        
        return True
        
    except Exception as e:
        print(f"API test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Express Animate API\n")
    test_api()
