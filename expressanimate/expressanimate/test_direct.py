#!/usr/bin/env python3
"""
Test SeeDance-1-Pro directly with your exact code
"""

import replicate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_direct():
    """Test SeeDance directly"""
    try:
        # Get API token
        api_token = os.getenv('REPLICATE_API_TOKEN')
        print(f"API Token: {api_token[:8]}...")
        
        # Try to find the correct version
        client = replicate.Client(api_token=api_token)
        
        # Try different working video models
        models_to_try = [
            ("stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb1a4f3482d9b4c90b", "svd"),
            ("cjwbw/text2video-zero:4a3d79e5d0d3d0d3d0d3d0d3d0d3d0d3d0d3d0d3", "text2video"),
            ("anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351", "zeroscope")
        ]
        
        for model, name in models_to_try:
            try:
                print(f"Trying {name}: {model}")
                
                if name == "zeroscope":
                    # ZeroScope uses different parameters
                    output = client.run(
                        model,
                        input={
                            "prompt": "A happy cartoon dog playing in a sunny garden"
                        }
                    )
                else:
                    # Try with basic prompt
                    output = client.run(
                        model,
                        input={
                            "prompt": "A happy cartoon dog playing in a sunny garden"
                        }
                    )
                
                print(f"✓ Success with {name}!")
                working_model = model
                break
            except Exception as e:
                print(f"✗ Failed with {name}: {str(e)[:100]}...")
                continue
        else:
            print("❌ All models failed")
            return False

        # To access the file URL:
        print(f"Output URL: {output.url()}")

        # To write the file to disk:
        with open("test_output.mp4", "wb") as file:
            file.write(output.read())
            
        print("✓ Success! Video saved as test_output.mp4")
        return True
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing SeeDance-1-Pro Directly\n")
    test_direct()
