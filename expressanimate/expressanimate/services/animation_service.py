import os
import time
import uuid
import requests
import tempfile
import replicate
from replicate.exceptions import ModelError

class AnimationService:
    def __init__(self, api_token=None, timeout=600):
        """
        Initialize Replicate client for WAN 2.5 Text-to-Video.
        Uses Replicate v1.1.0b1 with proper timeout handling.
        """
        if api_token:
            os.environ["REPLICATE_API_TOKEN"] = api_token

        self.temp_dir = tempfile.mkdtemp()
        self.max_prediction_wait = timeout  # Prediction timeout (10 minutes)
        print("✅ AI Video Generation Service initialized")
    
    def generate_video(self, prompt, duration=10, resolution="720p"):
        """
        Generate a text-to-video animation using the official WAN 2.5 model.
        :param prompt: The text prompt describing the scene.
        :param duration: Desired video duration in seconds (up to model limit).
        :param resolution: Video resolution (e.g., 480p, 720p, 1080p).
        :return: Path to the downloaded video.
        """
        try:
            print(f"\n🎬 Processing your video request...")

            input_data = {
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution
            }

            # Use AI video generation service
            print("🚀 Starting video creation...")
            
            try:
                # Try with standard processing time
                print("⏱️ Processing video (this may take a few minutes)...")
                output = replicate.run(
                    "wan-video/wan-2.5-t2v",
                    input=input_data,
                    wait=60  # Maximum wait time
                )
                print("✅ Video processing completed!")
                
            except Exception as e:
                if "timeout" in str(e).lower() or "wait" in str(e).lower():
                    # If timeout, use extended processing for complex videos
                    print("⏱️ Complex video detected, using extended processing...")
                    
                    # Create prediction manually using extended processing
                    prediction = replicate.predictions.create(
                        model="wan-video/wan-2.5-t2v",
                        input=input_data
                    )
                    
                    # Wait manually with extended timeout
                    start_time = time.time()
                    
                    while prediction.status not in ['succeeded', 'failed', 'canceled']:
                        elapsed = time.time() - start_time
                        if elapsed > self.max_prediction_wait:
                            raise Exception(f"Video processing timed out after {self.max_prediction_wait} seconds")
                        
                        time.sleep(3)
                        prediction.reload()
                        print(f"⏳ Processing... (elapsed: {int(elapsed)}s)")
                    
                    if prediction.status != 'succeeded':
                        raise Exception(f"Video processing failed")
                    
                    output = prediction.output
                    print("✅ Extended processing completed!")
                else:
                    raise e

            if not output:
                raise Exception("No video output received.")

            # Handle FileOutput objects (get URL for download)
            if hasattr(output, 'url'):
                video_url = output.url
            elif isinstance(output, list) and len(output) > 0:
                video_url = output[0].url if hasattr(output[0], 'url') else str(output[0])
            else:
                video_url = str(output)
                
            print("✅ Video generated successfully!")

            # Download video
            return self.download_video(video_url)

        except ModelError as e:
            print(f"❌ Video generation failed. Please try again.")
            return None
        except Exception as e:
            print(f"❌ Video processing failed. Please try again.")
            return None

    def download_video(self, video_url):
        """Download a video from AI service."""
        try:
            file_path = os.path.join(self.temp_dir, f"video_{uuid.uuid4().hex}.mp4")
            print("⬇️ Preparing your video...")
            response = requests.get(video_url, stream=True)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print("✅ Video ready for download!")
            return file_path
        except Exception as e:
            print("❌ Failed to prepare video. Please try again.")
            return None
    
    def create_animated_sequence(self, prompt, duration=10, resolution="720p", **kwargs):
        """Main workflow entrypoint."""
        print("\n🎞️ Starting AI video creation...")
        video_path = self.generate_video(prompt, duration, resolution)
        if video_path:
            print("🎉 Video creation completed successfully!")
        else:
            print("⚠️ Video creation failed.")
        return video_path
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print("✓ Temporary files cleaned up")
        except Exception as e:
            print(f"Error cleaning up: {e}")
