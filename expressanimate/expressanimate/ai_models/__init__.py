"""
ExpressAnimate AI Models Package
Advanced AI model implementations for video generation and processing
"""

from .video_generation_model import VideoGenerationModel
from .audio_synthesis_model import AudioSynthesisModel
from .style_transfer_model import StyleTransferModel
from .motion_prediction_model import MotionPredictionModel
from .scene_understanding_model import SceneUnderstandingModel
from .quality_enhancement_model import QualityEnhancementModel

__version__ = "2.1.0"
__author__ = "ExpressAnimate AI Research Team"

# Model registry for dynamic loading
MODEL_REGISTRY = {
    'video_generation': VideoGenerationModel,
    'audio_synthesis': AudioSynthesisModel,
    'style_transfer': StyleTransferModel,
    'motion_prediction': MotionPredictionModel,
    'scene_understanding': SceneUnderstandingModel,
    'quality_enhancement': QualityEnhancementModel,
}

def get_model(model_name: str, **kwargs):
    """Factory function to get model instances"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")
    
    return MODEL_REGISTRY[model_name](**kwargs)
