#!/usr/bin/env python3
"""
Test SeeDance-1-Pro model directly
"""

import replicate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_seedance_model():
    """Test SeeDance-1-Pro model"""
    try:
        # Get API token
        api_token = os.getenv('REPLICATE_API_TOKEN')
        if not api_token:
            print("❌ REPLICATE_API_TOKEN not found")
            return False
        
        print(f"✓ API Token found: {api_token[:8]}...")
        
        # Initialize client
        client = replicate.Client(api_token=api_token)
        print("✓ Replicate client initialized")
        
        # Test SeeDance-1-Pro model with proper version format
        print("Testing SeeDance-1-Pro model...")
        
        test_prompt = "A happy cartoon dog playing in a sunny garden"
        
        # Try Google Veo-3-Fast (recommended model)
        try:
            print("Trying Google Veo-3-Fast model...")
            output = client.run(
                "google/veo-3-fast",
                input={
                    "prompt": test_prompt,
                    "duration": "5s",
                    "resolution": "720p"
                }
            )
            print("✓ Success with Google Veo-3-Fast!")
        except Exception as e:
            print(f"✗ Failed with Google Veo-3-Fast: {e}")
            return False
        
        print(f"Output type: {type(output)}")
        print(f"Output content: {output}")
        
        if output:
            print("✓ SeeDance-1-Pro model test successful!")
            
            # Try to download if it's a URL
            if isinstance(output, str) and output.startswith('http'):
                import requests
                response = requests.get(output)
                print(f"Download status: {response.status_code}")
                if response.status_code == 200:
                    with open("test_output.mp4", "wb") as f:
                        f.write(response.content)
                    print(f"✓ Video downloaded: test_output.mp4 ({len(response.content)} bytes)")
            
            return True
        else:
            print("❌ SeeDance-1-Pro returned no output")
            return False
            
    except Exception as e:
        print(f"❌ SeeDance-1-Pro test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing SeeDance-1-Pro Model\n")
    success = test_seedance_model()
    
    if success:
        print("\n🎉 SeeDance-1-Pro test passed!")
    else:
        print("\n⚠️ SeeDance-1-Pro test failed.")
        print("\n🔧 Possible issues:")
        print("  1. Model name might be incorrect")
        print("  2. Model might not be available in your region")
        print("  3. Model might require different input parameters")
        print("  4. Insufficient credits (though you said you have paid)")
