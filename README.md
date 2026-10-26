# AI Projects

A collection of my AI and machine learning projects.

## Projects

### ExpressAnimate: Text to Animated Video
Takes a text input and turns it into an animated video in stages:

Text → Scene Generation → Image Generation → Animation → Narration → Lip Synchronization → Final Video

The project is split into modules, one for each stage:

- `scene_understanding_model.py`: breaks the text into scenes
- `style_transfer_model.py`: sets the visual style
- `motion_prediction_model.py`: animation and movement
- `audio_synthesis_model.py`: narration
- `video_generation_model.py`: assembles the final video
- `quality_enhancement_model.py`: improves output quality
- `monitoring_system.py`: tracks the pipeline
- `app.py`: runs the application

This work was published in the International Journal of Research and Analytical Reviews (IJRAR).

**Running it:** the project uses the Replicate API. Create a `.env` file in `expressanimate/expressanimate/` and add your own key:

```
REPLICATE_API_TOKEN=your_token_here
```

The `.env` file is ignored by Git, so your key stays on your computer.

### Face Authentication
A FastAPI service that takes two face images and tells you whether they show the same person. It returns a similarity score and the bounding box of each face, using the pre-trained FaceNet model through the DeepFace library.

### Stable Diffusion
`StableDiffusion.ipynb` and `stablediffusion.py` are image-generation experiments with Stable Diffusion.

## Tech
Python, FastAPI, DeepFace (FaceNet), OpenCV, Replicate API, Stable Diffusion

## Contact
Chandana Jagadish | chandanajagadish13@gmail.com | [LinkedIn](https://www.linkedin.com/in/chandana-jagadish-21a6b5304)
