from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import os
import threading
import time
from services.animation_service import AnimationService

# Load environment variables
load_dotenv()

# Create Flask application instance
app = Flask(__name__)
CORS(app)

# Configure the application
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'express-animate-secret-key-2024')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/generated', exist_ok=True)

# Initialize animation service
animation_service = AnimationService(os.getenv('REPLICATE_API_TOKEN'))

# Store for tracking generation status
generation_status = {}

@app.route('/')
def home():
    """Landing page route"""
    return render_template('landing.html')

@app.route('/create')
def create():
    """Create animation page route"""
    return render_template('create.html')

@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@app.route('/api/generate', methods=['POST'])
def generate_animation():
    """Generate animation from text prompt"""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt']
        duration = data.get('duration', 10)  # Default 10 seconds
        size = data.get('quality', '1280*720')  # Get size from quality parameter
        negative_prompt = data.get('negative_prompt', '')  # Optional negative prompt
        enable_prompt_expansion = data.get('enable_prompt_expansion', True)  # Prompt expansion
        
        # Debug logging
        print(f"=== AI Video Generation Request ===")
        print(f"Prompt: '{prompt}'")
        print(f"Duration: {duration}")
        print(f"Size: {size}")
        print(f"Negative prompt: '{negative_prompt}'")
        print(f"Enable prompt expansion: {enable_prompt_expansion}")
        print(f"Raw data: {data}")
        print(f"====================================")
        
        if len(prompt.strip()) == 0:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        # Generate unique job ID
        import uuid
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        generation_status[job_id] = {
            'status': 'started',
            'progress': 0,
            'message': 'Starting animation generation...',
            'video_path': None,
            'error': None
        }
        
        # Start generation in background thread
        thread = threading.Thread(
            target=generate_animation_async,
            args=(job_id, prompt, duration, size, negative_prompt, enable_prompt_expansion)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Animation generation started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>')
def get_generation_status(job_id):
    """Get generation status"""
    if job_id not in generation_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(generation_status[job_id])

@app.route('/api/video/<job_id>')
def stream_video(job_id):
    """Stream generated video for preview"""
    if job_id not in generation_status:
        return jsonify({'error': 'Job not found'}), 404
    
    job = generation_status[job_id]
    if job['status'] != 'completed' or not job['video_path']:
        return jsonify({'error': 'Animation not ready'}), 400
    
    try:
        return send_file(
            job['video_path'],
            mimetype='video/mp4'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<job_id>')
def download_video(job_id):
    """Download generated video"""
    if job_id not in generation_status:
        return jsonify({'error': 'Job not found'}), 404
    
    job = generation_status[job_id]
    if job['status'] != 'completed' or not job['video_path']:
        return jsonify({'error': 'Animation not ready'}), 400
    
    try:
        return send_file(
            job['video_path'],
            as_attachment=True,
            download_name=f'express_animate_{job_id}.mp4',
            mimetype='video/mp4'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_animation_async(job_id, prompt, duration, size='1280*720', negative_prompt='', enable_prompt_expansion=True):
    """Generate animation asynchronously using AI video service"""
    try:
        print(f"=== AI Video Async Generation ===")
        print(f"Job ID: {job_id}")
        print(f"Prompt: '{prompt}'")
        print(f"Duration: {duration}")
        print(f"Size: {size}")
        print(f"Negative prompt: '{negative_prompt}'")
        print(f"Enable prompt expansion: {enable_prompt_expansion}")
        print(f"=================================")
        print(f"Starting AI video generation for job {job_id}")
        
        # Update status
        generation_status[job_id].update({
            'status': 'processing',
            'progress': 10,
            'message': 'Initializing AI models...'
        })
        
        # Update progress for AI video generation
        generation_status[job_id].update({
            'progress': 20,
            'message': 'Generating video with AI...'
        })
        
        # Generate the animation using AI video service
        print("Generating animation with AI video service...")
        video_path = animation_service.create_animated_sequence(
            prompt=prompt, 
            duration=duration,
            resolution=size  # Pass size as resolution parameter
        )
        
        if video_path and os.path.exists(video_path):
            # Move video to static folder
            import shutil
            final_path = os.path.join('static/generated', f'{job_id}.mp4')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            
            try:
                shutil.copy2(video_path, final_path)
                print(f"Video copied to: {final_path}")
                
                generation_status[job_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Animation generated successfully!',
                    'video_path': final_path
                })
            except Exception as copy_error:
                print(f"Error copying video file: {copy_error}")
                generation_status[job_id].update({
                    'status': 'failed',
                    'progress': 0,
                    'message': f'Failed to save video: {str(copy_error)}',
                    'error': str(copy_error)
                })
        else:
            generation_status[job_id].update({
                'status': 'failed',
                'progress': 0,
                'message': 'Failed to generate animation',
                'error': 'Generation process failed'
            })
            
    except Exception as e:
        error_msg = f'Generation error: {str(e)}'
        print(f"Exception in async generation for job {job_id}: {error_msg}")
        import traceback
        traceback.print_exc()
        
        generation_status[job_id].update({
            'status': 'failed',
            'progress': 0,
            'message': error_msg,
            'error': str(e)
        })

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests"""
    return '', 204

@app.route('/api/hello')
def api_hello():
    """Simple API endpoint"""
    return jsonify({
        'message': 'Express Animate API is running!',
        'status': 'success',
        'version': '1.0.0'
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Run the application in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000)
