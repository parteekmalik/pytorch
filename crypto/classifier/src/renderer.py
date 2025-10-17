import numpy as np
import pandas as pd
import cupy as cp
from typing import Dict, Optional
from .utils import setup_logger, check_gpu_availability

logger = setup_logger(__name__)

GPU_AVAILABLE, GPU_BACKEND = check_gpu_availability()
logger.info(f"CuPy GPU renderer initialized with: {GPU_BACKEND}")


class Renderer:
    
    def __init__(self):
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
        return self.render_batch_gpu(sequence.reshape(1, -1), resolution, line_width)[0]
    
    def render_ohlc_batch_gpu(
        self,
        ohlc_sequences: np.ndarray,
        resolution: Dict[str, int]
    ) -> np.ndarray:
        if not self.gpu_available:
            raise RuntimeError("GPU batch rendering requires GPU")
        
        batch_size, seq_len, _ = ohlc_sequences.shape
        height = resolution['height']
        width = seq_len * 4
        
        ohlc_gpu = cp.asarray(ohlc_sequences, dtype=cp.float32)
        
        images_gpu = cp.ones((batch_size, height, width), dtype=cp.float32)
        
        opens = ohlc_gpu[:, :, 0]
        highs = ohlc_gpu[:, :, 1]
        lows = ohlc_gpu[:, :, 2]
        closes = ohlc_gpu[:, :, 3]
        
        price_min = cp.min(ohlc_gpu, axis=(1, 2))
        price_max = cp.max(ohlc_gpu, axis=(1, 2))
        price_range = cp.maximum(price_max - price_min, 1e-8)
        
        scale_factor = ((height - 1) / price_range)[:, None]
        
        opens_y = ((price_max[:, None] - opens) * scale_factor).astype(cp.int32)
        highs_y = ((price_max[:, None] - highs) * scale_factor).astype(cp.int32)
        lows_y = ((price_max[:, None] - lows) * scale_factor).astype(cp.int32)
        closes_y = ((price_max[:, None] - closes) * scale_factor).astype(cp.int32)
        
        opens_y = cp.clip(opens_y, 0, height - 1)
        highs_y = cp.clip(highs_y, 0, height - 1)
        lows_y = cp.clip(lows_y, 0, height - 1)
        closes_y = cp.clip(closes_y, 0, height - 1)
        
        batch_indices = cp.arange(batch_size)[:, None]
        bar_indices = cp.arange(seq_len)[None, :]
        
        x_opens = bar_indices * 4
        x_highs = bar_indices * 4 + 1
        x_closes = bar_indices * 4 + 2
        
        batch_grid = cp.broadcast_to(batch_indices, (batch_size, seq_len))
        
        images_gpu[batch_grid, opens_y, x_opens] = 0.0
        
        images_gpu[batch_grid, closes_y, x_closes] = 0.0
        
        y_min = cp.minimum(highs_y, lows_y)
        y_max = cp.maximum(highs_y, lows_y)
        
        y_range = cp.arange(height)
        
        y_min_exp = y_min[:, :, None]
        y_max_exp = y_max[:, :, None]
        y_range_exp = y_range[None, None, :]
        
        line_mask = (y_range_exp >= y_min_exp) & (y_range_exp <= y_max_exp)
        
        line_mask = cp.transpose(line_mask, (0, 2, 1))
        
        x_positions = cp.arange(seq_len) * 4 + 1
        
        for bar_idx in range(seq_len):
            x_pos = int(x_positions[bar_idx])
            images_gpu[:, :, x_pos][line_mask[:, :, bar_idx]] = 0.0
        
        return images_gpu.get()

    def render_batch_gpu(
        self,
        sequences: np.ndarray,
        resolution: Dict[str, int],
        line_width: int = 3
    ) -> np.ndarray:
        if not self.gpu_available:
            raise RuntimeError("GPU batch rendering requires GPU")
        
        batch_size, seq_len = sequences.shape
        width, height = resolution['width'], resolution['height']
        
        if not hasattr(sequences, 'get'):
            import cupy as cp
            sequences_gpu = cp.asarray(sequences)
        else:
            sequences_gpu = sequences
        
        seq_min = sequences_gpu.min(axis=1, keepdims=True)
        seq_max = sequences_gpu.max(axis=1, keepdims=True)
        mask = (seq_max > seq_min)
        seqs_norm = cp.where(mask, (sequences_gpu - seq_min) / (seq_max - seq_min), 0.5)
        
        images_gpu = cp.ones((batch_size, height, width), dtype=cp.float32)
        
        x_coords = cp.linspace(0, width - 1, seq_len, dtype=cp.float32)
        x_coords = cp.broadcast_to(x_coords, (batch_size, seq_len))
        y_coords = (1 - seqs_norm) * (height - 1)

        x0 = x_coords[:, :-1]
        x1 = x_coords[:, 1:]
        y0 = y_coords[:, :-1]
        y1 = y_coords[:, 1:]

        dx = cp.abs(x1 - x0)
        dy = cp.abs(y1 - y0)
        max_steps = min(int(cp.ceil(cp.max(cp.maximum(dx, dy)) * 2)), 200)

        t = cp.linspace(0, 1, max_steps).reshape(1, 1, -1)
        xs_all = x0[:, :, None] + t * (x1 - x0)[:, :, None]
        ys_all = y0[:, :, None] + t * (y1 - y0)[:, :, None]

        xs_all = cp.round(xs_all).astype(cp.int32)
        ys_all = cp.round(ys_all).astype(cp.int32)

        xs_all = cp.clip(xs_all, 0, width - 1)
        ys_all = cp.clip(ys_all, 0, height - 1)

        xs_flat = xs_all.reshape(batch_size, -1)
        ys_flat = ys_all.reshape(batch_size, -1)

        batch_indices = cp.arange(batch_size)[:, None]
        batch_indices = cp.broadcast_to(batch_indices, xs_flat.shape)

        images_gpu[batch_indices.ravel(), ys_flat.ravel(), xs_flat.ravel()] = 0.0

        if line_width > 1:
            half_w = line_width // 2
            offsets = cp.arange(-half_w, half_w + 1)
            
            for dy_offset in offsets:
                for dx_offset in offsets:
                    ys_thick = cp.clip(ys_flat + dy_offset, 0, height - 1)
                    xs_thick = cp.clip(xs_flat + dx_offset, 0, width - 1)
                    images_gpu[batch_indices.ravel(), ys_thick.ravel(), xs_thick.ravel()] = 0.0
        
        result = images_gpu.get()
        return result
    
    def render_ohlc_batch_coordinates(self, ohlc_sequences: np.ndarray, height: int) -> np.ndarray:
        batch_size, seq_len, _ = ohlc_sequences.shape
        
        ohlc_gpu = cp.asarray(ohlc_sequences)
        
        opens = ohlc_gpu[:, :, 0]
        highs = ohlc_gpu[:, :, 1]
        lows = ohlc_gpu[:, :, 2]
        closes = ohlc_gpu[:, :, 3]
        
        price_min = cp.min(ohlc_gpu, axis=(1, 2))
        price_max = cp.max(ohlc_gpu, axis=(1, 2))
        price_range = cp.maximum(price_max - price_min, 1e-8)
        scale_factor = ((height - 1) / price_range)[:, None]
        
        opens_y = ((price_max[:, None] - opens) * scale_factor).astype(cp.uint16)
        highs_y = ((price_max[:, None] - highs) * scale_factor).astype(cp.uint16)
        lows_y = ((price_max[:, None] - lows) * scale_factor).astype(cp.uint16)
        closes_y = ((price_max[:, None] - closes) * scale_factor).astype(cp.uint16)
        
        opens_y = cp.clip(opens_y, 0, height - 1)
        highs_y = cp.clip(highs_y, 0, height - 1)
        lows_y = cp.clip(lows_y, 0, height - 1)
        closes_y = cp.clip(closes_y, 0, height - 1)
        
        coordinates = cp.stack([opens_y, highs_y, lows_y, closes_y], axis=2)
        
        return cp.asnumpy(coordinates)