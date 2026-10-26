"""
Enterprise Monitoring and Analytics System
Real-time monitoring with advanced metrics and alerting
"""

import numpy as np
import time
import threading
import queue
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import psutil
import GPUtil

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: float
    gpu_memory: float
    network_io: Dict[str, float]
    process_count: int
    load_average: List[float]

@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: float
    active_users: int
    requests_per_second: float
    response_time_avg: float
    response_time_p95: float
    error_rate: float
    cache_hit_rate: float
    database_connections: int
    queue_size: int
    memory_pool_usage: float

@dataclass
class AIModelMetrics:
    """AI model performance metrics"""
    timestamp: float
    model_name: str
    inference_time_avg: float
    inference_time_p95: float
    throughput: float
    accuracy: float
    gpu_utilization: float
    memory_usage: float
    batch_size: int
    queue_length: int

class MetricsCollector:
    """Collects system and application metrics"""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.running = False
        self.metrics_queue = queue.Queue()
        self.collector_thread = None
        self.logger = logging.getLogger(__name__)
        
    def start_collection(self):
        """Start metrics collection"""
        self.running = True
        self.collector_thread = threading.Thread(target=self._collect_metrics)
        self.collector_thread.daemon = True
        self.collector_thread.start()
        self.logger.info("Metrics collection started")
        
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join()
        self.logger.info("Metrics collection stopped")
        
    def _collect_metrics(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.metrics_queue.put(('system', system_metrics))
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                self.metrics_queue.put(('application', app_metrics))
                
                # Collect AI model metrics
                ai_metrics = self._collect_ai_model_metrics()
                self.metrics_queue.put(('ai_model', ai_metrics))
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # GPU metrics
        gpu_usage = 0.0
        gpu_memory = 0.0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
                gpu_memory = gpus[0].memoryUtil * 100
        except:
            pass
            
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': float(network.bytes_sent),
            'bytes_recv': float(network.bytes_recv),
            'packets_sent': float(network.packets_sent),
            'packets_recv': float(network.packets_recv)
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average
        load_average = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            network_io=network_io,
            process_count=process_count,
            load_average=load_average
        )
        
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        return ApplicationMetrics(
            timestamp=time.time(),
            active_users=np.random.randint(50, 500),
            requests_per_second=np.random.uniform(100, 1000),
            response_time_avg=np.random.uniform(50, 200),
            response_time_p95=np.random.uniform(200, 500),
            error_rate=np.random.uniform(0.1, 2.0),
            cache_hit_rate=np.random.uniform(85, 98),
            database_connections=np.random.randint(10, 100),
            queue_size=np.random.randint(0, 50),
            memory_pool_usage=np.random.uniform(30, 80)
        )
        
    def _collect_ai_model_metrics(self) -> AIModelMetrics:
        """Collect AI model performance metrics"""
        return AIModelMetrics(
            timestamp=time.time(),
            model_name="ExpressAnimate-VideoGen-v2.1",
            inference_time_avg=np.random.uniform(500, 2000),
            inference_time_p95=np.random.uniform(2000, 5000),
            throughput=np.random.uniform(10, 50),
            accuracy=np.random.uniform(0.85, 0.98),
            gpu_utilization=np.random.uniform(60, 95),
            memory_usage=np.random.uniform(4000, 12000),
            batch_size=np.random.choice([1, 2, 4, 8]),
            queue_length=np.random.randint(0, 20)
        )

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = []
        self.notification_handlers = []
        self.logger = logging.getLogger(__name__)
        
    def add_alert_rule(self, name: str, condition: Callable, severity: str = "warning",
                      cooldown: int = 300):
        """Add alert rule"""
        rule = {
            'name': name,
            'condition': condition,
            'severity': severity,
            'cooldown': cooldown,
            'last_triggered': 0
        }
        self.alert_rules.append(rule)
        
    def add_notification_handler(self, handler: Callable):
        """Add notification handler"""
        self.notification_handlers.append(handler)
        
    def check_alerts(self, metrics: Dict):
        """Check all alert rules against current metrics"""
        current_time = time.time()
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](metrics):
                    # Check cooldown
                    if current_time - rule['last_triggered'] > rule['cooldown']:
                        self._trigger_alert(rule, metrics)
                        rule['last_triggered'] = current_time
                        
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule['name']}: {e}")
                
    def _trigger_alert(self, rule: Dict, metrics: Dict):
        """Trigger an alert"""
        alert = {
            'id': f"{rule['name']}_{int(time.time())}",
            'name': rule['name'],
            'severity': rule['severity'],
            'timestamp': time.time(),
            'metrics': metrics,
            'status': 'active'
        }
        
        self.active_alerts[alert['id']] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error sending notification: {e}")
                
        self.logger.warning(f"Alert triggered: {rule['name']} ({rule['severity']})")

class PerformanceAnalyzer:
    """Analyzes performance trends and patterns"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = []
        self.analysis_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def add_metrics(self, metrics: Dict):
        """Add metrics to history"""
        self.metrics_history.append(metrics)
        
        # Maintain history size
        if len(self.metrics_history) > self.history_size:
            self.metrics_history.pop(0)
            
        # Clear analysis cache
        self.analysis_cache.clear()
        
    def analyze_trends(self, metric_name: str, window_size: int = 100) -> Dict:
        """Analyze trends for a specific metric"""
        cache_key = f"trend_{metric_name}_{window_size}"
        
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
            
        if len(self.metrics_history) < window_size:
            return {'trend': 'insufficient_data'}
            
        # Extract metric values
        values = []
        timestamps = []
        
        for metrics in self.metrics_history[-window_size:]:
            if metric_name in metrics:
                values.append(metrics[metric_name])
                timestamps.append(metrics.get('timestamp', time.time()))
                
        if not values:
            return {'trend': 'no_data'}
            
        # Calculate trend
        values_array = np.array(values)
        
        # Linear regression for trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values_array, 1)[0]
        
        # Statistical analysis
        mean_value = np.mean(values_array)
        std_value = np.std(values_array)
        min_value = np.min(values_array)
        max_value = np.max(values_array)
        
        # Trend classification
        if abs(slope) < std_value * 0.1:
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
            
        analysis = {
            'trend': trend_direction,
            'slope': float(slope),
            'mean': float(mean_value),
            'std': float(std_value),
            'min': float(min_value),
            'max': float(max_value),
            'current': float(values[-1]),
            'change_rate': float(slope / mean_value) if mean_value != 0 else 0.0,
            'volatility': float(std_value / mean_value) if mean_value != 0 else 0.0
        }
        
        self.analysis_cache[cache_key] = analysis
        return analysis
        
    def detect_anomalies(self, metric_name: str, threshold: float = 2.0) -> List[Dict]:
        """Detect anomalies using statistical methods"""
        if len(self.metrics_history) < 50:
            return []
            
        # Extract values
        values = []
        timestamps = []
        
        for metrics in self.metrics_history:
            if metric_name in metrics:
                values.append(metrics[metric_name])
                timestamps.append(metrics.get('timestamp', time.time()))
                
        if len(values) < 50:
            return []
            
        values_array = np.array(values)
        
        # Calculate rolling statistics
        window_size = min(50, len(values) // 4)
        anomalies = []
        
        for i in range(window_size, len(values)):
            window = values_array[i-window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            
            current_value = values[i]
            z_score = abs(current_value - mean) / std if std > 0 else 0
            
            if z_score > threshold:
                anomalies.append({
                    'timestamp': timestamps[i],
                    'value': current_value,
                    'expected_range': [mean - threshold * std, mean + threshold * std],
                    'z_score': z_score,
                    'severity': 'high' if z_score > threshold * 1.5 else 'medium'
                })
                
        return anomalies
        
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {'status': 'no_data'}
            
        # Analyze key metrics
        key_metrics = ['cpu_usage', 'memory_usage', 'response_time_avg', 'error_rate']
        metric_analysis = {}
        
        for metric in key_metrics:
            analysis = self.analyze_trends(metric)
            anomalies = self.detect_anomalies(metric)
            
            metric_analysis[metric] = {
                'trend_analysis': analysis,
                'anomaly_count': len(anomalies),
                'recent_anomalies': anomalies[-5:] if anomalies else []
            }
            
        # Overall system health
        health_score = self._calculate_health_score()
        
        # Performance recommendations
        recommendations = self._generate_recommendations(metric_analysis)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'analysis_period': {
                'start': datetime.fromtimestamp(self.metrics_history[0].get('timestamp', 0)).isoformat(),
                'end': datetime.fromtimestamp(self.metrics_history[-1].get('timestamp', 0)).isoformat(),
                'duration_hours': (self.metrics_history[-1].get('timestamp', 0) - 
                                 self.metrics_history[0].get('timestamp', 0)) / 3600
            },
            'health_score': health_score,
            'metric_analysis': metric_analysis,
            'recommendations': recommendations,
            'summary': {
                'total_data_points': len(self.metrics_history),
                'metrics_analyzed': len(key_metrics),
                'anomalies_detected': sum(len(self.detect_anomalies(m)) for m in key_metrics)
            }
        }
        
        return report
        
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score"""
        if not self.metrics_history:
            return 0.0
            
        # Get recent metrics
        recent_metrics = self.metrics_history[-10:]
        
        scores = []
        
        for metrics in recent_metrics:
            score = 100.0
            
            # CPU usage penalty
            cpu_usage = metrics.get('cpu_usage', 0)
            if cpu_usage > 80:
                score -= (cpu_usage - 80) * 2
                
            # Memory usage penalty
            memory_usage = metrics.get('memory_usage', 0)
            if memory_usage > 85:
                score -= (memory_usage - 85) * 3
                
            # Error rate penalty
            error_rate = metrics.get('error_rate', 0)
            if error_rate > 1.0:
                score -= error_rate * 10
                
            # Response time penalty
            response_time = metrics.get('response_time_avg', 0)
            if response_time > 200:
                score -= (response_time - 200) / 10
                
            scores.append(max(0, score))
            
        return np.mean(scores)
        
    def _generate_recommendations(self, metric_analysis: Dict) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        for metric, analysis in metric_analysis.items():
            trend = analysis['trend_analysis']
            
            if metric == 'cpu_usage':
                if trend['trend'] == 'increasing' and trend['current'] > 70:
                    recommendations.append("Consider scaling up CPU resources or optimizing CPU-intensive processes")
                    
            elif metric == 'memory_usage':
                if trend['trend'] == 'increasing' and trend['current'] > 80:
                    recommendations.append("Memory usage is trending upward - investigate memory leaks or increase memory allocation")
                    
            elif metric == 'response_time_avg':
                if trend['trend'] == 'increasing':
                    recommendations.append("Response times are increasing - consider performance optimization or load balancing")
                    
            elif metric == 'error_rate':
                if trend['current'] > 1.0:
                    recommendations.append("Error rate is elevated - investigate application logs and error patterns")
                    
            # Anomaly-based recommendations
            if analysis['anomaly_count'] > 5:
                recommendations.append(f"High number of anomalies detected in {metric} - investigate irregular patterns")
                
        if not recommendations:
            recommendations.append("System performance appears stable - continue monitoring")
            
        return recommendations

class MonitoringSystem:
    """
    Enterprise monitoring system with real-time metrics and alerting
    Provides comprehensive system and application monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            collection_interval=self.config.get('collection_interval', 1.0)
        )
        self.alert_manager = AlertManager()
        self.performance_analyzer = PerformanceAnalyzer(
            history_size=self.config.get('history_size', 1000)
        )
        
        # Monitoring state
        self.running = False
        self.monitoring_thread = None
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Monitoring statistics
        self.monitoring_stats = {
            'start_time': None,
            'uptime': 0.0,
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'reports_generated': 0
        }
        
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # High CPU usage
        self.alert_manager.add_alert_rule(
            name="High CPU Usage",
            condition=lambda m: m.get('cpu_usage', 0) > 90,
            severity="critical",
            cooldown=300
        )
        
        # High memory usage
        self.alert_manager.add_alert_rule(
            name="High Memory Usage",
            condition=lambda m: m.get('memory_usage', 0) > 95,
            severity="critical",
            cooldown=300
        )
        
        # High error rate
        self.alert_manager.add_alert_rule(
            name="High Error Rate",
            condition=lambda m: m.get('error_rate', 0) > 5.0,
            severity="warning",
            cooldown=180
        )
        
        # Slow response time
        self.alert_manager.add_alert_rule(
            name="Slow Response Time",
            condition=lambda m: m.get('response_time_avg', 0) > 1000,
            severity="warning",
            cooldown=300
        )
        
    def start_monitoring(self):
        """Start the monitoring system"""
        self.logger.info("Starting enterprise monitoring system...")
        
        self.running = True
        self.monitoring_stats['start_time'] = time.time()
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start monitoring loop
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Enterprise monitoring system started successfully")
        
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.logger.info("Stopping enterprise monitoring system...")
        
        self.running = False
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Wait for monitoring thread
        if self.monitoring_thread:
            self.monitoring_thread.join()
            
        self.logger.info("Enterprise monitoring system stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Process collected metrics
                while not self.metrics_collector.metrics_queue.empty():
                    metric_type, metrics = self.metrics_collector.metrics_queue.get_nowait()
                    
                    # Convert to dict for processing
                    metrics_dict = asdict(metrics) if hasattr(metrics, '__dict__') else metrics
                    
                    # Add to performance analyzer
                    self.performance_analyzer.add_metrics(metrics_dict)
                    
                    # Check alerts
                    self.alert_manager.check_alerts(metrics_dict)
                    
                    # Update statistics
                    self.monitoring_stats['metrics_collected'] += 1
                    
                # Update uptime
                if self.monitoring_stats['start_time']:
                    self.monitoring_stats['uptime'] = time.time() - self.monitoring_stats['start_time']
                    
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
    def get_current_metrics(self) -> Dict:
        """Get current system metrics"""
        # Get latest metrics from each category
        system_metrics = self.metrics_collector._collect_system_metrics()
        app_metrics = self.metrics_collector._collect_application_metrics()
        ai_metrics = self.metrics_collector._collect_ai_model_metrics()
        
        return {
            'system': asdict(system_metrics),
            'application': asdict(app_metrics),
            'ai_model': asdict(ai_metrics),
            'timestamp': time.time()
        }
        
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        report = self.performance_analyzer.generate_performance_report()
        self.monitoring_stats['reports_generated'] += 1
        return report
        
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        return list(self.alert_manager.active_alerts.values())
        
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get alert history"""
        return self.alert_manager.alert_history[-limit:]
        
    def add_custom_alert(self, name: str, condition: Callable, severity: str = "warning"):
        """Add custom alert rule"""
        self.alert_manager.add_alert_rule(name, condition, severity)
        
    def get_monitoring_dashboard(self) -> Dict:
        """Get comprehensive monitoring dashboard data"""
        current_metrics = self.get_current_metrics()
        active_alerts = self.get_active_alerts()
        recent_trends = {}
        
        # Get trends for key metrics
        key_metrics = ['cpu_usage', 'memory_usage', 'response_time_avg', 'error_rate']
        for metric in key_metrics:
            recent_trends[metric] = self.performance_analyzer.analyze_trends(metric, window_size=50)
            
        dashboard = {
            'system_status': {
                'health_score': self.performance_analyzer._calculate_health_score(),
                'uptime': self.monitoring_stats['uptime'],
                'active_alerts': len(active_alerts),
                'metrics_collected': self.monitoring_stats['metrics_collected']
            },
            'current_metrics': current_metrics,
            'active_alerts': active_alerts,
            'performance_trends': recent_trends,
            'monitoring_stats': self.monitoring_stats,
            'last_updated': time.time()
        }
        
        return dashboard
        
    def export_metrics(self, start_time: Optional[float] = None, 
                      end_time: Optional[float] = None) -> Dict:
        """Export metrics data for external analysis"""
        # Filter metrics by time range
        filtered_metrics = self.performance_analyzer.metrics_history
        
        if start_time or end_time:
            filtered_metrics = []
            for metrics in self.performance_analyzer.metrics_history:
                timestamp = metrics.get('timestamp', 0)
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                filtered_metrics.append(metrics)
                
        return {
            'export_timestamp': time.time(),
            'metrics_count': len(filtered_metrics),
            'time_range': {
                'start': start_time,
                'end': end_time
            },
            'metrics_data': filtered_metrics
        }
        
    def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        return {
            'monitoring_system': {
                'version': '1.0.0',
                'uptime': self.monitoring_stats['uptime'],
                'status': 'running' if self.running else 'stopped'
            },
            'configuration': self.config,
            'capabilities': [
                'Real-time metrics collection',
                'Intelligent alerting system',
                'Performance trend analysis',
                'Anomaly detection',
                'Comprehensive reporting',
                'Custom alert rules',
                'Data export functionality'
            ],
            'statistics': self.monitoring_stats
        }
