import replicate
import os
import tempfile
import uuid

class AnimationService:
    def __init__(self, api_token=None):
        """Initialize AnimationService with WAN 2.5 T2V model"""
        self.temp_dir = tempfile.mkdtemp()
        # Set API token if provided
        if api_token:
            os.environ["REPLICATE_API_TOKEN"] = api_token
    
    def generate_video(self, prompt, duration=10):
        """Generate video using WAN 2.5 T2V model"""
        try:
            print(f"=== Generating video with WAN 2.5 ===")
            print(f"Prompt: {prompt}")
            print(f"Duration: {duration} seconds")
            
            # WAN 2.5 T2V input
            input_data = {
                "prompt": prompt,
                "duration": duration
            }
            
            print("Running WAN 2.5 T2V model...")
            output = replicate.run(
                "wan-video/wan-2.5-t2v",
                input=input_data
            )
            
            print("WAN 2.5 generation completed successfully!")
            
            # Save video to disk
            video_filename = f"wan_video_{uuid.uuid4().hex}.mp4"
            video_path = os.path.join(self.temp_dir, video_filename)
            
            print(f"Saving video to: {video_path}")
            with open(video_path, "wb") as file:
                file.write(output.read())
            
            print(f"✓ Video saved successfully: {video_path}")
            
            # Also get the URL for reference
            video_url = output.url()
            print(f"Video URL: {video_url}")
            
            return video_path
            
        except Exception as e:
            print(f"Error generating video with WAN 2.5: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_animated_sequence(self, prompt, duration=10, **kwargs):
        """Main method for creating animated sequences"""
        print(f"=== Starting WAN 2.5 Animation Sequence ===")
        
        # Generate video using WAN 2.5
        video_path = self.generate_video(prompt, duration)
        
        if video_path:
            print(f"=== ✓ Animation sequence completed successfully! ===")
            return video_path
        else:
            print(f"=== ✗ Animation sequence failed ===")
            return None
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print("✓ Temporary files cleaned up")
        except Exception as e:
            print(f"Error cleaning up: {e}")
