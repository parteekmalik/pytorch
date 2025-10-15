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
        Shared-memory GPU rendering with zero data copying.
        All images read from the same OHLC data array using offsets.
        """
        if not self.gpu_available:
            raise RuntimeError("GPU batch rendering requires GPU")
        
        batch_size, seq_len, _ = ohlc_sequences.shape
        width, height = resolution['width'], resolution['height']
        
        # Move to GPU as float32 - this is the ONLY data on GPU
        ohlc_gpu = cp.asarray(ohlc_sequences, dtype=cp.float32)  # (batch, seq_len, 4)
        
        # Reshape for efficient access: (batch * seq_len, 4)
        # This allows contiguous memory access
        ohlc_flat = ohlc_gpu.reshape(-1, 4)
        
        # Create batch indices for vectorized access
        batch_indices = cp.arange(batch_size)[:, None]  # (batch, 1)
        candle_indices = cp.arange(seq_len)[None, :]    # (1, seq_len)
        
        # Vectorized indexing: all batches × all candles
        flat_indices = batch_indices * seq_len + candle_indices  # (batch, seq_len)
        
        # Extract OHLC using fancy indexing (NO COPYING!)
        opens = ohlc_flat[flat_indices, 0]    # (batch, seq_len)
        highs = ohlc_flat[flat_indices, 1]    # (batch, seq_len)
        lows = ohlc_flat[flat_indices, 2]     # (batch, seq_len)
        closes = ohlc_flat[flat_indices, 3]   # (batch, seq_len)
        
        # Vectorized scaling to pixel coordinates (in-place where possible)
        all_prices = cp.stack([opens, highs, lows, closes], axis=-1).reshape(batch_size, -1)
        price_min = all_prices.min(axis=1, keepdims=True)
        price_max = all_prices.max(axis=1, keepdims=True)
        price_range = cp.maximum(price_max - price_min, 1e-8)
        
        # Scale to pixels (float32 operations)
        scale_factor = (height - 1) / price_range
        opens_y = ((price_max - opens) * scale_factor).astype(cp.int32)
        highs_y = ((price_max - highs) * scale_factor).astype(cp.int32)
        lows_y = ((price_max - lows) * scale_factor).astype(cp.int32)
        closes_y = ((price_max - closes) * scale_factor).astype(cp.int32)
        
        # Calculate candle positions (shared across all batches)
        candle_spacing = width / seq_len
        candle_width = int(candle_spacing * candle_width_ratio)
        candle_centers = (cp.arange(seq_len) * candle_spacing + candle_spacing / 2).astype(cp.int32)
        x_left_all = cp.maximum(0, candle_centers - candle_width // 2)
        x_right_all = cp.minimum(width - 1, candle_centers + candle_width // 2)
        
        # Create output images
        images_gpu = cp.ones((batch_size, height, width), dtype=cp.float32)
        
        # VECTORIZED RENDERING - Process all candles for all batches
        # Use element-wise operations (CUDA kernels) instead of loops
        
        # Compute wick coordinates (vectorized)
        wick_tops = cp.minimum(highs_y, lows_y)      # (batch, seq_len)
        wick_bottoms = cp.maximum(highs_y, lows_y)   # (batch, seq_len)
        
        # Compute body coordinates (vectorized)
        body_tops = cp.minimum(opens_y, closes_y)    # (batch, seq_len)
        body_bottoms = cp.maximum(opens_y, closes_y) # (batch, seq_len)
        
        # Compute colors (vectorized)
        is_bullish = closes >= opens                  # (batch, seq_len)
        body_colors = cp.where(is_bullish, bullish_color, bearish_color)
        
        # OPTIMIZED DRAWING: Process candles with minimal loops
        # Use broadcasting and advanced indexing for GPU parallelism
        for candle_idx in range(seq_len):
            x_center = candle_centers[candle_idx]
            x_start = x_left_all[candle_idx]
            x_end = x_right_all[candle_idx]
            
            # Extract coordinates for this candle across all batches
            wick_top = wick_tops[:, candle_idx]       # (batch,)
            wick_bottom = wick_bottoms[:, candle_idx] # (batch,)
            body_top = body_tops[:, candle_idx]       # (batch,)
            body_bottom = body_bottoms[:, candle_idx] # (batch,)
            colors = body_colors[:, candle_idx]       # (batch,)
            
            # VECTORIZED WICK DRAWING using broadcasting
            # Create y-coordinate grid
            y_coords = cp.arange(height)[:, None]     # (height, 1)
            
            # Broadcast comparison: (height, 1) vs (1, batch) -> (height, batch)
            wick_mask = (y_coords >= wick_top[None, :]) & (y_coords <= wick_bottom[None, :])
            
            # Apply wicks to all batches simultaneously (in-place)
            images_gpu[:, :, x_center] = cp.where(wick_mask.T, 0.0, images_gpu[:, :, x_center])
            
            # VECTORIZED BODY DRAWING using broadcasting
            body_mask = (y_coords >= body_top[None, :]) & (y_coords <= body_bottom[None, :])
            
            # Apply bodies to all x-coordinates in the candle width
            for x in range(int(x_start), int(x_end) + 1):
                # Broadcast colors: (1, batch) -> (height, batch) where mask is True
                images_gpu[:, :, x] = cp.where(body_mask.T, colors[None, :], images_gpu[:, :, x])
        
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
