"""
ExpressAnimate Machine Learning Pipeline
Enterprise-grade ML pipeline with distributed training and model management
"""

from .model_trainer import ModelTrainer
from .data_pipeline import DataPipeline
from .feature_extractor import FeatureExtractor
from .model_optimizer import ModelOptimizer
from .inference_engine import InferenceEngine
from .experiment_tracker import ExperimentTracker

__version__ = "2.4.1"
__author__ = "ExpressAnimate ML Research Team"

# Pipeline component registry
PIPELINE_REGISTRY = {
    'trainer': ModelTrainer,
    'data_pipeline': DataPipeline,
    'feature_extractor': FeatureExtractor,
    'optimizer': ModelOptimizer,
    'inference_engine': InferenceEngine,
    'experiment_tracker': ExperimentTracker,
}

def create_pipeline_component(component_type: str, **kwargs):
    """Factory function to create ML pipeline components"""
    if component_type not in PIPELINE_REGISTRY:
        raise ValueError(f"Component {component_type} not found in registry")
    
    return PIPELINE_REGISTRY[component_type](**kwargs)
