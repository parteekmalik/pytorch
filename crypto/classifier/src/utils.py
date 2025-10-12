"""
Utilities for GPU detection, logging, and shared configurations.
"""
import logging
import os
from pathlib import Path
from typing import Tuple
import matplotlib
import warnings

warnings.filterwarnings('ignore')

# Matplotlib configuration
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['figure.autolayout'] = True
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['savefig.bbox'] = 'tight'


def check_gpu_availability() -> Tuple[bool, str]:
    """
    Check if GPU is available via CuPy.
    
    Returns:
        Tuple of (is_available, backend_name)
    """
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True, "CuPy (CUDA)"
    except Exception:
        return False, "NumPy (CPU)"


def get_array_module():
    """
    Get the appropriate array module (CuPy if available, else NumPy).
    
    Returns:
        Module for array operations (cp or np)
    """
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return cp
    except Exception:
        import numpy as np
        return np


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


