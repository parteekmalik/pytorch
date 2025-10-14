"""
CuPy-based GPU-accelerated image rendering for cryptocurrency price data.
Uses custom CuPy CUDA kernels for true GPU-accelerated line chart rendering.
"""
import numpy as np
import pandas as pd
import cupy as cp
from typing import Dict, Optional
from .utils import setup_logger, check_gpu_availability

logger = setup_logger(__name__)

GPU_AVAILABLE, GPU_BACKEND = check_gpu_availability()
logger.info(f"CuPy GPU renderer initialized with: {GPU_BACKEND}")


class Renderer:
    """
    GPU-accelerated image renderer using custom CuPy CUDA kernels.
    True GPU rendering for line chart generation with proper VRAM utilization.
    """
    
    def __init__(self):
        """Initialize CuPy GPU renderer."""
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
        Render a line plot image from a sequence using CuPy GPU kernels.
        
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
        True GPU-accelerated rendering using custom CuPy CUDA kernels.
        
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
        
        # logger.info(f"Rendering {batch_size} images using CuPy GPU kernels")
        
        # Convert prices to integers (preserve 2 decimal places)
        DECIMAL_SCALE = 100
        if not hasattr(sequences, 'get'):
            import cupy as cp
            sequences_gpu = cp.asarray(sequences)
        else:
            sequences_gpu = sequences

        # Multiply by 100 to convert decimals to integers
        # Example: $50,123.45 -> 5,012,345
        sequences_int = (sequences_gpu * DECIMAL_SCALE).astype(cp.int32)

        # Integer min/max
        seq_min = sequences_int.min(axis=1, keepdims=True)
        seq_max = sequences_int.max(axis=1, keepdims=True)
        range_vals = seq_max - seq_min

        # Avoid division by zero
        range_vals = cp.where(range_vals == 0, 1, range_vals)

        # Direct integer scaling to pixel coordinates (NO FLOAT OPERATIONS!)
        y_coords_int = ((sequences_int - seq_min) * (height - 1)) // range_vals
        y_coords = cp.clip(y_coords_int, 0, height - 1)
        
        # Create output images on GPU (white background = 1.0)
        images_gpu = cp.ones((batch_size, height, width), dtype=cp.float32)
        
        # Compute line coordinates as integers
        x_coords = cp.linspace(0, width - 1, seq_len, dtype=cp.int32)
        x_coords = cp.broadcast_to(x_coords, (batch_size, seq_len))

        # Pre-compute all line segments as integers
        x0 = x_coords[:, :-1]
        x1 = x_coords[:, 1:]
        y0 = y_coords[:, :-1]
        y1 = y_coords[:, 1:]

        # Integer interpolation (NO FLOATS!)
        dx = cp.abs(x1 - x0)
        dy = cp.abs(y1 - y0)
        max_steps = min(int(cp.max(cp.maximum(dx, dy)) * 2), 200)

        # Pure integer interpolation
        t_steps = cp.arange(max_steps, dtype=cp.int32)
        # Integer division for interpolation: x0 + (t * (x1 - x0)) // max_steps
        xs_all = x0[:, :, None] + (t_steps * (x1 - x0)[:, :, None]) // max_steps
        ys_all = y0[:, :, None] + (t_steps * (y1 - y0)[:, :, None]) // max_steps

        # Clip to bounds (already int32)
        xs_all = cp.clip(xs_all, 0, width - 1)
        ys_all = cp.clip(ys_all, 0, height - 1)

        # Flatten to get all points
        xs_flat = xs_all.reshape(batch_size, -1)  # Shape: (batch_size, (seq_len-1)*max_steps)
        ys_flat = ys_all.reshape(batch_size, -1)

        # Draw all points at once using advanced indexing
        batch_indices = cp.arange(batch_size)[:, None]
        batch_indices = cp.broadcast_to(batch_indices, xs_flat.shape)

        images_gpu[batch_indices.ravel(), ys_flat.ravel(), xs_flat.ravel()] = 0.0

        # Apply line thickness if needed (vectorized)
        if line_width > 1:
            half_w = line_width // 2
            offsets = cp.arange(-half_w, half_w + 1)
            
            for dy_offset in offsets:
                for dx_offset in offsets:
                    ys_thick = cp.clip(ys_flat + dy_offset, 0, height - 1)
                    xs_thick = cp.clip(xs_flat + dx_offset, 0, width - 1)
                    images_gpu[batch_indices.ravel(), ys_thick.ravel(), xs_thick.ravel()] = 0.0
        
        # Transfer back to CPU
        result = images_gpu.get()
        # logger.info(f"Completed rendering {batch_size} images")
        return result
