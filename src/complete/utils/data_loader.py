#!/usr/bin/env python3
"""
Data Loader for DDoS Detection
Simple data loading utilities
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging


class DDoSDataLoader:
    """
    Data loader for DDoS detection datasets
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize data loader"""
        self.config = config
        self.logger = logger
    
    def load_dataset(self, data_path: str, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from file
        
        Args:
            data_path: Path to data directory
            dataset_name: Name of dataset to load
            
        Returns:
            Tuple of (features, labels)
        """
        self.logger.info(f"Loading dataset: {dataset_name}")
        
        # For now, return synthetic data
        # In a real implementation, this would load actual datasets
        return self._generate_sample_data()
    
    def _generate_sample_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample data for testing"""
        np.random.seed(42)
        
        input_size = self.config['model']['input_size']
        sequence_length = self.config['model']['sequence_length']
        
        # Generate random sequences
        X = np.random.randn(num_samples, sequence_length, input_size)
        y = np.random.randint(0, 2, num_samples)
        
        return X, y
