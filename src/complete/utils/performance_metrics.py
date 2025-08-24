#!/usr/bin/env python3
"""
Performance Metrics Tracker
Logs and maintains performance metrics during model training
"""

from typing import Dict, List, Any
import json
import os

class ModelPerformanceTracker:
    """
    Tracks and logs performance metrics for models
    """
    
    def __init__(self, log_dir='logs', log_file='performance_metrics.json'):
        """Initialize performance tracker"""
        self.metrics = []
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        os.makedirs(log_dir, exist_ok=True)
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics for current era/iteration"""
        self.metrics.append(metrics)
        self._log_metrics()
    
    def _log_metrics(self):
        """Log metrics to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get logged metrics"""
        return self.metrics

if __name__ == "__main__":
    # Example usage
    tracker = ModelPerformanceTracker()
    
    # Simulate metrics
    example_metrics = {
        'epoch': 1,
        'train_loss': 0.5,
        'val_loss': 0.4,
        'train_acc': 80.0,
        'val_acc': 85.0,
        'val_precision': 0.9,
        'val_recall': 0.85,
        'val_f1': 0.87,
        'lr': 0.001
    }
    tracker.update_metrics(example_metrics)
