"""
ExpressAnimate Advanced Video Processing Pipeline
Enterprise-grade video processing with AI-powered optimization
"""

from .video_encoder import VideoEncoder
from .video_decoder import VideoDecoder
from .frame_processor import FrameProcessor
from .codec_optimizer import CodecOptimizer
from .streaming_processor import StreamingProcessor
from .batch_processor import BatchProcessor

__version__ = "1.8.2"
__author__ = "ExpressAnimate Video Processing Team"

# Processing pipeline registry
PROCESSOR_REGISTRY = {
    'encoder': VideoEncoder,
    'decoder': VideoDecoder,
    'frame_processor': FrameProcessor,
    'codec_optimizer': CodecOptimizer,
    'streaming_processor': StreamingProcessor,
    'batch_processor': BatchProcessor,
}

def create_processor(processor_type: str, **kwargs):
    """Factory function to create video processors"""
    if processor_type not in PROCESSOR_REGISTRY:
        raise ValueError(f"Processor {processor_type} not found in registry")
    
    return PROCESSOR_REGISTRY[processor_type](**kwargs)
