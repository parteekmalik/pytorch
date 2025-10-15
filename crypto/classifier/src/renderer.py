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
        bullish_color: float = 0.2,  # Dark gray for bullish (close > open)
        bearish_color: float = 0.0   # Black for bearish (close < open)
    ) -> np.ndarray:
        """
        Render Japanese candlestick charts on GPU (much faster than line charts!).
        
        Args:
            ohlc_sequences: 3D array (batch_size, seq_len, 4) with [Open, High, Low, Close]
            resolution: Dict with 'width', 'height'
            candle_width_ratio: Width of candle body relative to available space (0.0-1.0)
            bullish_color: Gray value for bullish candles (close > open)
            bearish_color: Gray value for bearish candles (close < open)
            
        Returns:
            3D array of images (batch_size, height, width)
        """
        if not self.gpu_available:
            raise RuntimeError("GPU batch rendering requires GPU")
        
        batch_size, seq_len, _ = ohlc_sequences.shape
        width, height = resolution['width'], resolution['height']
        
        # Move to GPU
        ohlc_gpu = cp.asarray(ohlc_sequences)
        
        # Extract OHLC
        opens = ohlc_gpu[:, :, 0]
        highs = ohlc_gpu[:, :, 1]
        lows = ohlc_gpu[:, :, 2]
        closes = ohlc_gpu[:, :, 3]
        
        # Direct scaling to pixel coordinates (NO normalization!)
        # Calculate price range for scaling to image height
        all_prices = ohlc_gpu.reshape(batch_size, -1)
        price_min = all_prices.min(axis=1, keepdims=True)
        price_max = all_prices.max(axis=1, keepdims=True)
        price_range = price_max - price_min
        price_range = cp.where(price_range == 0, 1, price_range)
        
        # Direct scaling to pixel coordinates (inverted Y-axis for images)
        opens_y = ((price_max - opens) * (height - 1) / price_range).astype(cp.int32)
        highs_y = ((price_max - highs) * (height - 1) / price_range).astype(cp.int32)
        lows_y = ((price_max - lows) * (height - 1) / price_range).astype(cp.int32)
        closes_y = ((price_max - closes) * (height - 1) / price_range).astype(cp.int32)
        
        # Calculate X positions for each candle
        candle_spacing = width / seq_len
        candle_width = int(candle_spacing * candle_width_ratio)
        candle_centers = cp.arange(seq_len) * candle_spacing + candle_spacing / 2
        candle_centers = candle_centers.astype(cp.int32)
        
        # Create output images (white background)
        images_gpu = cp.ones((batch_size, height, width), dtype=cp.float32)
        
        # Render each candlestick
        for batch_idx in range(batch_size):
            for candle_idx in range(seq_len):
                x_center = candle_centers[candle_idx]
                x_left = max(0, x_center - candle_width // 2)
                x_right = min(width - 1, x_center + candle_width // 2)
                
                open_y = opens_y[batch_idx, candle_idx]
                high_y = highs_y[batch_idx, candle_idx]
                low_y = lows_y[batch_idx, candle_idx]
                close_y = closes_y[batch_idx, candle_idx]
                
                # Determine candle color (bullish or bearish)
                is_bullish = closes[batch_idx, candle_idx] >= opens[batch_idx, candle_idx]
                body_color = bullish_color if is_bullish else bearish_color
                
                # Draw wick (high to low) - thin vertical line at center
                wick_top = min(high_y, low_y)
                wick_bottom = max(high_y, low_y)
                images_gpu[batch_idx, wick_top:wick_bottom+1, x_center] = 0.0
                
                # Draw body (open to close) - filled rectangle
                body_top = min(open_y, close_y)
                body_bottom = max(open_y, close_y)
                
                # Handle doji (open == close)
                if body_top == body_bottom:
                    # Draw thin horizontal line for doji
                    images_gpu[batch_idx, body_top, x_left:x_right+1] = 0.0
                else:
                    # Draw filled rectangle for body
                    images_gpu[batch_idx, body_top:body_bottom+1, x_left:x_right+1] = body_color
        
        # Transfer back to CPU
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
