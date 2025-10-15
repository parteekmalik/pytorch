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
    
    def render_candlestick_batch_gpu(
        self,
        ohlc_sequences: np.ndarray,
        resolution: Dict[str, int],
        candle_width_ratio: float = 0.8,
        bullish_color: float = 0.2,
        bearish_color: float = 0.0
    ) -> np.ndarray:
        """
        Fully vectorized GPU candlestick rendering - NO PYTHON LOOPS.
        """
        if not self.gpu_available:
            raise RuntimeError("GPU batch rendering requires GPU")
        
        batch_size, seq_len, _ = ohlc_sequences.shape
        width, height = resolution['width'], resolution['height']
        
        # Move to GPU as float32
        ohlc_gpu = cp.asarray(ohlc_sequences, dtype=cp.float32)
        
        # Extract OHLC
        opens = ohlc_gpu[:, :, 0]
        highs = ohlc_gpu[:, :, 1]
        lows = ohlc_gpu[:, :, 2]
        closes = ohlc_gpu[:, :, 3]
        
        # Vectorized scaling to pixel coordinates
        all_prices = ohlc_gpu.reshape(batch_size, -1)
        price_min = all_prices.min(axis=1, keepdims=True)
        price_max = all_prices.max(axis=1, keepdims=True)
        price_range = cp.maximum(price_max - price_min, 1e-8)
        
        # Scale to pixels (float32 operations are FAST on GPU)
        opens_y = ((price_max - opens) * (height - 1) / price_range).astype(cp.int32)
        highs_y = ((price_max - highs) * (height - 1) / price_range).astype(cp.int32)
        lows_y = ((price_max - lows) * (height - 1) / price_range).astype(cp.int32)
        closes_y = ((price_max - closes) * (height - 1) / price_range).astype(cp.int32)
        
        # Calculate candle positions
        candle_spacing = width / seq_len
        candle_width = int(candle_spacing * candle_width_ratio)
        candle_centers = (cp.arange(seq_len) * candle_spacing + candle_spacing / 2).astype(cp.int32)
        
        # Vectorized coordinate calculations
        x_left_all = cp.maximum(0, candle_centers - candle_width // 2)
        x_right_all = cp.minimum(width - 1, candle_centers + candle_width // 2)
        
        # Create output images
        images_gpu = cp.ones((batch_size, height, width), dtype=cp.float32)
        
        # FULLY VECTORIZED RENDERING - NO LOOPS
        # Create 4D coordinate grids: (batch, height, width, candle)
        batch_idx = cp.arange(batch_size)[:, None, None, None]
        y_coords = cp.arange(height)[None, :, None, None]
        x_coords = cp.arange(width)[None, None, :, None]
        candle_idx = cp.arange(seq_len)[None, None, None, :]
        
        # Broadcast to full shape
        batch_grid = cp.broadcast_to(batch_idx, (batch_size, height, width, seq_len))
        y_grid = cp.broadcast_to(y_coords, (batch_size, height, width, seq_len))
        x_grid = cp.broadcast_to(x_coords, (batch_size, height, width, seq_len))
        
        # Expand candle coordinates to match grid
        wick_top = cp.minimum(highs_y, lows_y)[:, None, None, :]
        wick_bottom = cp.maximum(highs_y, lows_y)[:, None, None, :]
        candle_x = candle_centers[None, None, None, :]
        
        body_top = cp.minimum(opens_y, closes_y)[:, None, None, :]
        body_bottom = cp.maximum(opens_y, closes_y)[:, None, None, :]
        x_left = x_left_all[None, None, None, :]
        x_right = x_right_all[None, None, None, :]
        
        # Create masks for all pixels at once
        wick_mask = (y_grid >= wick_top) & (y_grid <= wick_bottom) & (x_grid == candle_x)
        body_mask = (y_grid >= body_top) & (y_grid <= body_bottom) & \
                    (x_grid >= x_left) & (x_grid <= x_right)
        
        # Determine colors
        is_bullish = (closes >= opens)[:, None, None, :]
        body_colors = cp.where(is_bullish, bullish_color, bearish_color)
        
        # Apply wicks (black) - any candle with wick at this pixel
        images_gpu[wick_mask.any(axis=3)] = 0.0
        
        # Apply bodies with colors - weighted by number of overlapping candles
        body_color_sum = (body_mask * body_colors).sum(axis=3)
        body_count = body_mask.sum(axis=3)
        body_pixels = body_count > 0
        images_gpu[body_pixels] = body_color_sum[body_pixels] / body_count[body_pixels]
        
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
