"""
OpenCV CUDA-based GPU-accelerated image rendering for cryptocurrency price data.
Uses OpenCV CUDA for true GPU-accelerated line chart rendering.
"""
import numpy as np
import pandas as pd
import cupy as cp
from typing import Dict, Optional
import cv2
from .utils import setup_logger, check_gpu_availability

logger = setup_logger(__name__)

GPU_AVAILABLE, GPU_BACKEND = check_gpu_availability()
logger.info(f"OpenCV CUDA renderer initialized with: {GPU_BACKEND}")


class Renderer:
    """
    GPU-accelerated image renderer using OpenCV CUDA.
    True GPU rendering for line chart generation with better VRAM utilization.
    """
    
    def __init__(self):
        """Initialize OpenCV CUDA renderer."""
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU (CuPy) is required but not available")
        self.gpu_available = True
        logger.info("Renderer initialized")
    
    def render_line_image(
        self,
        sequence: np.ndarray,
        resolution: Dict[str, int],
        line_width: int = 3
    ) -> np.ndarray:
        """
        Render a line plot image from a sequence using OpenCV CUDA.
        
        Args:
            sequence: 1D array of price values
            resolution: Dict with 'width', 'height'
            line_width: Width of the line
            
        Returns:
            2D grayscale image array (normalized to [0, 1])
        """
        return self.render_batch_gpu(sequence.reshape(1, -1), resolution, line_width)[0]
    
    def render_batch_gpu(
        self,
        sequences: np.ndarray,
        resolution: Dict[str, int],
        line_width: int = 3
    ) -> np.ndarray:
        """
        True GPU-accelerated rendering using OpenCV CUDA.
        
        Args:
            sequences: 2D array of sequences (batch_size, seq_len)
            resolution: Dict with 'width', 'height'
            line_width: Width of the line
            
        Returns:
            3D array of images (batch_size, height, width)
        """
        if not self.gpu_available:
            raise RuntimeError("GPU batch rendering requires GPU")
        
        batch_size, seq_len = sequences.shape
        width, height = resolution['width'], resolution['height']
        
        logger.info(f"Rendering {batch_size} images using OpenCV CUDA")
        
        # Move sequences to GPU and normalize
        if not hasattr(sequences, 'get'):
            import cupy as cp
            sequences_gpu = cp.asarray(sequences)
        else:
            sequences_gpu = sequences
        
        # Normalize ALL sequences on GPU in parallel
        seq_min = sequences_gpu.min(axis=1, keepdims=True)
        seq_max = sequences_gpu.max(axis=1, keepdims=True)
        mask = (seq_max > seq_min)
        seqs_norm = cp.where(mask, (sequences_gpu - seq_min) / (seq_max - seq_min), 0.5)
        seqs_cpu = seqs_norm.get()
        
        images = []
        
        for batch_idx in range(batch_size):
            # Create white background on GPU
            gpu_img = cv2.cuda_GpuMat(height, width, cv2.CV_8UC1)
            gpu_img.setTo(255)  # White background
            
            # Prepare line coordinates
            seq = seqs_cpu[batch_idx]
            x_coords = np.linspace(0, width - 1, seq_len)
            y_coords = (1 - seq) * (height - 1)
            
            # Draw line segments on GPU
            points = np.column_stack([x_coords, y_coords]).astype(np.int32)
            for i in range(len(points) - 1):
                pt1 = tuple(points[i])
                pt2 = tuple(points[i + 1])
                cv2.cuda.line(gpu_img, pt1, pt2, 0, line_width)
            
            # Download from GPU to CPU
            img_cpu = gpu_img.download()
            img_array = img_cpu.astype(np.float32) / 255.0
            images.append(img_array)
            
            if batch_idx % 100 == 0:
                logger.info(f"Processed {batch_idx + 1}/{batch_size} images")
        
        result = np.array(images)
        logger.info(f"Completed rendering {batch_size} images")
        return result
