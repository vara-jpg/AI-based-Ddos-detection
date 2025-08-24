#!/usr/bin/env python3
"""
Logger Configuration Utility
Centralized logging setup for the DDoS detection system
"""

import logging
import logging.handlers
import os
from typing import Dict, Any


def setup_logger(name: str, config: Dict[str, Any] = None) -> logging.Logger:
    """
    Setup logger with configuration
    
    Args:
        name: Logger name
        config: Logger configuration dictionary
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = {
            'level': 'INFO',
            'log_file': 'logs/system.log',
            'max_file_size': '10MB',
            'backup_count': 5
        }
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.get('level', 'INFO').upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = config.get('log_file', 'logs/system.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    max_bytes = _parse_size(config.get('max_file_size', '10MB'))
    backup_count = config.get('backup_count', 5)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(getattr(logging, config.get('level', 'INFO').upper()))
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger


def _parse_size(size_str: str) -> int:
    """
    Parse size string (e.g., '10MB', '1GB') to bytes
    
    Args:
        size_str: Size string
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


if __name__ == "__main__":
    # Test logger setup
    test_config = {
        'level': 'DEBUG',
        'log_file': 'logs/test.log',
        'max_file_size': '5MB',
        'backup_count': 3
    }
    
    logger = setup_logger('test_logger', test_config)
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
