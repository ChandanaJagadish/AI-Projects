# 🎬 Express Animate - AI-Powered Animation Generator

Express Animate is an AI-driven animation generator that converts text prompts into animated video sequences within seconds. Transform your imagination into professional-quality animations using cutting-edge AI technology.

## 🚀 Features

- **⚡ Text-to-Animation**: Convert descriptive prompts into animated videos
- **🎵 Voice Synthesis**: Automatic text-to-speech with natural voices
- **🎨 Multiple Styles**: Cartoon, anime, cinematic, and sketch styles
- **📱 Modern UI**: Responsive web interface with real-time progress tracking
- **🔄 Async Processing**: Background video generation with status updates
- **📥 Easy Download**: Direct video download when generation completes

## 🛠 Technology Stack

### Backend
- **Flask 2.3.3** - Web framework
- **Replicate API** - AI model hosting and inference
- **MoviePy** - Video processing and editing
- **gTTS** - Google Text-to-Speech
- **Pillow** - Image processing

### AI Models
- **Stable Diffusion XL** - Character and scene generation
- **Stable Video Diffusion** - Image-to-video conversion
- **Google TTS** - Voice synthesis

### Frontend
- **Bootstrap 5** - Responsive UI framework
- **JavaScript** - Interactive features and API integration

## 📁 Project Structure

```
expressanimate/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables (API keys)
├── README.md                  # This file
├── services/
│   └── animation_service.py   # AI animation generation service
├── templates/                 # HTML templates
│   ├── base.html             # Base template
│   ├── index.html            # Animation generator interface
│   ├── about.html            # About page
│   ├── 404.html              # 404 error page
│   └── 500.html              # 500 error page
└── static/
    ├── uploads/              # User uploads (auto-created)
    └── generated/            # Generated videos (auto-created)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file with your Replicate API token:

```env
REPLICATE_API_TOKEN=your_replicate_api_token_here
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
```

### 3. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 4. Generate Your First Animation

1. Open `http://localhost:5000` in your browser
2. Enter a descriptive prompt (e.g., "A funny cartoon dog chasing a butterfly in a sunny garden")
3. Select duration and style
4. Click "Generate Animation"
5. Wait for processing and download your video!

## 🎯 API Endpoints

### POST /api/generate
Generate animation from text prompt.

**Request:**
```json
{
    "prompt": "A funny cartoon dog chasing a butterfly",
    "duration": 10
}
```

**Response:**
```json
{
    "job_id": "uuid-string",
    "status": "started",
    "message": "Animation generation started"
}
```

### GET /api/status/{job_id}
Check generation status.

**Response:**
```json
{
    "status": "processing",
    "progress": 45,
    "message": "Generating character images...",
    "video_path": null
}
```

### GET /api/download/{job_id}
Download completed animation video.

## 💡 Example Prompts

- **Animal Adventure**: "A brave little mouse riding on the back of a friendly elephant through a magical forest"
- **Space Story**: "An astronaut floating in space discovers a colorful alien planet with dancing creatures"
- **Fantasy Tale**: "A young wizard casting colorful spells to help flowers bloom in an enchanted garden"

## 🔧 Development

### Running in Development Mode

The application runs in debug mode by default with:
- Auto-reload on file changes
- Detailed error messages
- CORS enabled for frontend development

### Adding New Features

1. **New AI Models**: Add model integrations in `services/animation_service.py`
2. **UI Enhancements**: Modify templates in `templates/`
3. **API Extensions**: Add new routes in `app.py`

### Environment Variables

- `REPLICATE_API_TOKEN`: Your Replicate API token (required)
- `FLASK_ENV`: Set to `production` for production deployment
- `SECRET_KEY`: Flask secret key for sessions

## 🚀 Production Deployment

### Prerequisites
- Python 3.8+
- Replicate API account and token
- Sufficient storage for temporary files

### Deployment Steps

1. **Set Environment Variables**:
   ```bash
   export FLASK_ENV=production
   export REPLICATE_API_TOKEN=your_token
   ```

2. **Install Production Dependencies**:
   ```bash
   pip install gunicorn
   ```

3. **Run with Gunicorn**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### Production Considerations

- Use a reverse proxy (Nginx) for static files
- Implement proper logging and monitoring
- Set up file cleanup for temporary videos
- Configure proper error handling and rate limiting
- Use a production database for job tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Include error logs and reproduction steps

---

**🌟 Turn Your Imagination Into Animation with Express Animate!**
