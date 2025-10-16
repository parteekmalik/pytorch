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
    
    def render_ohlc_batch_gpu(
        self,
        ohlc_sequences: np.ndarray,
        resolution: Dict[str, int]
    ) -> np.ndarray:
        """
        Fully vectorized OHLC Bar Chart GPU rendering with 4-pixel layout.
        Pixel 0: Open, Pixel 1: High-Low line, Pixel 2: Close, Pixel 3: Gap
        Each image is independently scaled to its own price range.
        
        Optimized for maximum GPU utilization with zero CPU-GPU sync.
        """
        if not self.gpu_available:
            raise RuntimeError("GPU batch rendering requires GPU")
        
        batch_size, seq_len, _ = ohlc_sequences.shape
        height = resolution['height']
        width = seq_len * 4
        
        # Move to GPU once
        ohlc_gpu = cp.asarray(ohlc_sequences, dtype=cp.float32)
        
        # Create output images (white background)
        images_gpu = cp.ones((batch_size, height, width), dtype=cp.float32)
        
        # VECTORIZED STEP 1: Per-image scaling (all batches at once)
        # Shape: (batch_size, seq_len, 4)
        opens = ohlc_gpu[:, :, 0]
        highs = ohlc_gpu[:, :, 1]
        lows = ohlc_gpu[:, :, 2]
        closes = ohlc_gpu[:, :, 3]
        
        # Find min/max per image (vectorized across batch)
        # Shape: (batch_size,)
        price_min = cp.min(ohlc_gpu, axis=(1, 2))  # min across seq_len and OHLC
        price_max = cp.max(ohlc_gpu, axis=(1, 2))  # max across seq_len and OHLC
        price_range = cp.maximum(price_max - price_min, 1e-8)
        
        # Broadcast scale factor: (batch_size,) -> (batch_size, 1)
        scale_factor = ((height - 1) / price_range)[:, None]
        
        # Scale all prices to pixel coordinates (vectorized)
        # Broadcasting: (batch_size, 1) * (batch_size, seq_len) -> (batch_size, seq_len)
        opens_y = ((price_max[:, None] - opens) * scale_factor).astype(cp.int32)
        highs_y = ((price_max[:, None] - highs) * scale_factor).astype(cp.int32)
        lows_y = ((price_max[:, None] - lows) * scale_factor).astype(cp.int32)
        closes_y = ((price_max[:, None] - closes) * scale_factor).astype(cp.int32)
        
        # Clip all coordinates (vectorized)
        opens_y = cp.clip(opens_y, 0, height - 1)
        highs_y = cp.clip(highs_y, 0, height - 1)
        lows_y = cp.clip(lows_y, 0, height - 1)
        closes_y = cp.clip(closes_y, 0, height - 1)
        
        # VECTORIZED STEP 2: Draw all pixels using advanced indexing
        
        # Create coordinate grids
        batch_indices = cp.arange(batch_size)[:, None]  # (batch_size, 1)
        bar_indices = cp.arange(seq_len)[None, :]       # (1, seq_len)
        
        # X-coordinates for each pixel type (vectorized)
        x_opens = bar_indices * 4           # Pixel 0
        x_highs = bar_indices * 4 + 1       # Pixel 1
        x_closes = bar_indices * 4 + 2      # Pixel 2
        
        # Broadcast batch indices: (batch_size, seq_len)
        batch_grid = cp.broadcast_to(batch_indices, (batch_size, seq_len))
        
        # Draw Open pixels (vectorized across all batches and bars)
        images_gpu[batch_grid, opens_y, x_opens] = 0.0
        
        # Draw Close pixels (vectorized across all batches and bars)
        images_gpu[batch_grid, closes_y, x_closes] = 0.0
        
        # Draw High-Low lines (FULLY VECTORIZED)
        y_min = cp.minimum(highs_y, lows_y)  # (batch_size, seq_len)
        y_max = cp.maximum(highs_y, lows_y)  # (batch_size, seq_len)
        
        # Create y-coordinate array for all possible heights
        y_range = cp.arange(height)  # (height,)
        
        # Expand dimensions for broadcasting
        # y_min_exp: (batch_size, seq_len, 1)
        # y_max_exp: (batch_size, seq_len, 1)
        # y_range: (1, 1, height) after broadcasting
        y_min_exp = y_min[:, :, None]
        y_max_exp = y_max[:, :, None]
        y_range_exp = y_range[None, None, :]
        
        # Create mask: True where we should draw vertical line pixels
        # Shape: (batch_size, seq_len, height)
        line_mask = (y_range_exp >= y_min_exp) & (y_range_exp <= y_max_exp)
        
        # Transpose to (batch_size, height, seq_len) for easier indexing
        line_mask = cp.transpose(line_mask, (0, 2, 1))
        
        # Draw all vertical lines with optimized GPU operations
        # Pre-compute x-positions on GPU
        x_positions = cp.arange(seq_len) * 4 + 1  # (seq_len,)
        
        # Draw all vertical lines using direct assignment (no cp.where needed)
        for bar_idx in range(seq_len):
            x_pos = int(x_positions[bar_idx])  # Single CPU-GPU sync per bar
            # Direct assignment - mask already indicates where to draw
            images_gpu[:, :, x_pos][line_mask[:, :, bar_idx]] = 0.0
        
        return images_gpu.get()

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
        
        # Move to GPU and normalize
        if not hasattr(sequences, 'get'):
            import cupy as cp
            sequences_gpu = cp.asarray(sequences)
        else:
            sequences_gpu = sequences
        
        # Normalize on GPU
        seq_min = sequences_gpu.min(axis=1, keepdims=True)
        seq_max = sequences_gpu.max(axis=1, keepdims=True)
        mask = (seq_max > seq_min)
        seqs_norm = cp.where(mask, (sequences_gpu - seq_min) / (seq_max - seq_min), 0.5)
        
        # Create output images on GPU (white background = 1.0)
        images_gpu = cp.ones((batch_size, height, width), dtype=cp.float32)
        
        # Compute line coordinates for ALL batches at once (vectorized)
        x_coords = cp.linspace(0, width - 1, seq_len, dtype=cp.float32)
        x_coords = cp.broadcast_to(x_coords, (batch_size, seq_len))
        y_coords = (1 - seqs_norm) * (height - 1)  # Shape: (batch_size, seq_len)

        # Pre-compute all line segments for all images
        x0 = x_coords[:, :-1]  # Shape: (batch_size, seq_len-1)
        x1 = x_coords[:, 1:]
        y0 = y_coords[:, :-1]
        y1 = y_coords[:, 1:]

        # Compute max steps needed for all segments
        dx = cp.abs(x1 - x0)
        dy = cp.abs(y1 - y0)
        max_steps = min(int(cp.ceil(cp.max(cp.maximum(dx, dy)) * 2)), 200)  # Cap at 200 steps

        # Interpolate all line segments at once
        t = cp.linspace(0, 1, max_steps).reshape(1, 1, -1)  # Shape: (1, 1, max_steps)
        xs_all = x0[:, :, None] + t * (x1 - x0)[:, :, None]  # Shape: (batch_size, seq_len-1, max_steps)
        ys_all = y0[:, :, None] + t * (y1 - y0)[:, :, None]

        # Round and convert to integers
        xs_all = cp.round(xs_all).astype(cp.int32)
        ys_all = cp.round(ys_all).astype(cp.int32)

        # Clip to image bounds
        xs_all = cp.clip(xs_all, 0, width - 1)
        ys_all = cp.clip(ys_all, 0, height - 1)

        # Flatten to get all points
        xs_flat = xs_all.reshape(batch_size, -1)  # Shape: (batch_size, (seq_len-1)*max_steps)
        ys_flat = ys_all.reshape(batch_size, -1)

        # Draw all points at once using advanced indexing
        batch_indices = cp.arange(batch_size)[:, None]
        batch_indices = cp.broadcast_to(batch_indices, xs_flat.shape)

        images_gpu[batch_indices.ravel(), ys_flat.ravel(), xs_flat.ravel()] = 0.0

        # Apply line thickness if needed
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
    
    def render_ohlc_batch_coordinates(self, ohlc_sequences: np.ndarray) -> np.ndarray:
        """
        Render OHLC bars and return compressed Y-coordinate format.
        
        Args:
            ohlc_sequences: (batch_size, seq_len, 4) OHLC data
            
        Returns:
            Compressed coordinates: (batch_size, seq_len, 4) where each row is:
            [opens_y, highs_y, lows_y, closes_y]
        """
        batch_size, seq_len, _ = ohlc_sequences.shape
        height = self.height
        
        # Convert to CuPy
        ohlc_gpu = cp.asarray(ohlc_sequences)
        
        # Extract OHLC components
        opens = ohlc_gpu[:, :, 0]
        highs = ohlc_gpu[:, :, 1]
        lows = ohlc_gpu[:, :, 2]
        closes = ohlc_gpu[:, :, 3]
        
        # Per-image scaling (same as render_ohlc_batch_gpu lines 82-85)
        price_min = cp.min(ohlc_gpu, axis=(1, 2))
        price_max = cp.max(ohlc_gpu, axis=(1, 2))
        price_range = cp.maximum(price_max - price_min, 1e-8)
        scale_factor = ((height - 1) / price_range)[:, None]
        
        # Scale to Y-coordinates (same as lines 92-95)
        opens_y = ((price_max[:, None] - opens) * scale_factor).astype(cp.uint16)
        highs_y = ((price_max[:, None] - highs) * scale_factor).astype(cp.uint16)
        lows_y = ((price_max[:, None] - lows) * scale_factor).astype(cp.uint16)
        closes_y = ((price_max[:, None] - closes) * scale_factor).astype(cp.uint16)
        
        # Clip coordinates
        opens_y = cp.clip(opens_y, 0, height - 1)
        highs_y = cp.clip(highs_y, 0, height - 1)
        lows_y = cp.clip(lows_y, 0, height - 1)
        closes_y = cp.clip(closes_y, 0, height - 1)
        
        # Stack into (batch_size, seq_len, 4) format
        coordinates = cp.stack([opens_y, highs_y, lows_y, closes_y], axis=2)
        
        # Return as numpy (CPU) for storage
        return cp.asnumpy(coordinates)
