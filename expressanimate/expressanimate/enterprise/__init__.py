"""
ExpressAnimate Enterprise Solutions
Enterprise-grade infrastructure and monitoring systems
"""

from .monitoring_system import MonitoringSystem
from .analytics_engine import AnalyticsEngine
from .security_manager import SecurityManager
from .load_balancer import LoadBalancer
from .cache_manager import CacheManager
from .configuration_manager import ConfigurationManager

__version__ = "1.2.0"
__author__ = "ExpressAnimate Enterprise Team"

# Enterprise component registry
ENTERPRISE_REGISTRY = {
    'monitoring': MonitoringSystem,
    'analytics': AnalyticsEngine,
    'security': SecurityManager,
    'load_balancer': LoadBalancer,
    'cache_manager': CacheManager,
    'config_manager': ConfigurationManager,
}

def create_enterprise_component(component_type: str, **kwargs):
    """Factory function to create enterprise components"""
    if component_type not in ENTERPRISE_REGISTRY:
        raise ValueError(f"Component {component_type} not found in registry")
    
    return ENTERPRISE_REGISTRY[component_type](**kwargs)
