#!/usr/bin/env python3
"""
Test your exact working code
"""

import replicate
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_working_code():
    """Test your exact working code"""
    try:
        API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
        client = replicate.Client(api_token=API_TOKEN)

        video_input = {
            "fps": 24,
            "prompt": "A happy cartoon dog playing in a sunny garden",
            "duration": 3,
            "resolution": "480p",
            "aspect_ratio": "16:9",
            "camera_fixed": False
        }

        print("Creating prediction...")
        # Create the prediction
        prediction = client.predictions.create(
            version="bytedance/seedance-1-pro",
            input=video_input
        )

        print("Waiting for completion...")
        # Wait for completion (updates prediction in place)
        prediction.wait()

        print(f"Prediction status: {prediction.status}")
        print(f"Prediction output: {prediction.output}")

        # Ensure the output exists and is valid
        if not prediction.output or not isinstance(prediction.output[0], str):
            raise Exception("Prediction output is invalid or empty.")

        video_url = prediction.output[0]
        print("Video URL:", video_url)
        
        # Fix URL if it's not complete
        if not video_url.startswith('http'):
            video_url = f"https://replicate.delivery/{video_url}"
            print("Fixed Video URL:", video_url)

        # Download the video safely
        response = requests.get(video_url)
        output_file = "test_working_output.mp4"
        
        with open(output_file, "wb") as f:
            f.write(response.content)

        print(f"✓ Video saved as: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Your Exact Working Code\n")
    success = test_working_code()
    
    if success:
        print("\n🎉 Your code works perfectly!")
    else:
        print("\n⚠️ There's an issue with the SeeDance model.")
