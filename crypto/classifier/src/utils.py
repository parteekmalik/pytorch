import logging
import os
from pathlib import Path
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')


def check_gpu_availability() -> Tuple[bool, str]:
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True, "CuPy (CUDA)"
    except Exception:
        return False, "NumPy (CPU)"


def get_array_module():
    import multiprocessing
    import numpy as np
    
    if multiprocessing.current_process().name != 'MainProcess':
        return np
    
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return cp
    except Exception:
        return np


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
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


def ensure_dir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)