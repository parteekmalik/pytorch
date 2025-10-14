"""
Datashader-based GPU-accelerated image rendering for cryptocurrency price data.
Uses Datashader for efficient GPU-accelerated line chart rendering.
"""
import numpy as np
import pandas as pd
import cupy as cp
from typing import Dict, Optional
import datashader as ds
from datashader.transfer_functions import shade, set_background
from .utils import setup_logger, check_gpu_availability

logger = setup_logger(__name__)

GPU_AVAILABLE, GPU_BACKEND = check_gpu_availability()
logger.info(f"Datashader renderer initialized with: {GPU_BACKEND}")


class Renderer:
    """
    GPU-accelerated image renderer using Datashader.
    Much faster than custom GPU implementation for line chart rendering.
    """
    
    def __init__(self):
        """Initialize Datashader renderer."""
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
        Render a line plot image from a sequence using Datashader.
        
        Args:
            sequence: 1D array of price values
            resolution: Dict with 'width', 'height'
            line_width: Width of the line (not used in Datashader, kept for compatibility)
            
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
        Ultra-fast GPU-accelerated rendering using Datashader.
        
        Args:
            sequences: 2D array of sequences (batch_size, seq_len)
            resolution: Dict with 'width', 'height'
            line_width: Width of the line (not used in Datashader, kept for compatibility)
            
        Returns:
            3D array of images (batch_size, height, width)
        """
        if not self.gpu_available:
            raise RuntimeError("GPU batch rendering requires GPU")
        
        batch_size, seq_len = sequences.shape
        width, height = resolution['width'], resolution['height']
        
        logger.info(f"Rendering {batch_size} images using Datashader")
        
        # Move ALL sequences to GPU and normalize at once
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

        # Convert back to CPU for Datashader (all at once)
        seqs_cpu = seqs_norm.get()

        # Now render with Datashader
        images = []
        for batch_idx in range(batch_size):
            df = pd.DataFrame({
                'x': np.arange(seq_len),
                'y': seqs_cpu[batch_idx]  # Already normalized
            })
            
            canvas = ds.Canvas(plot_width=width, plot_height=height)
            agg = canvas.line(df, 'x', 'y', agg=ds.count())
            img = shade(agg, cmap=['white', 'black'])
            img = set_background(img, 'white')
            img_array = img.to_numpy() / 255.0
            img_array = 1.0 - img_array
            images.append(img_array)
            
            if batch_idx % 100 == 0:
                logger.info(f"Processed {batch_idx + 1}/{batch_size} images")
        
        result = np.array(images)
        logger.info(f"Completed rendering {batch_size} images")
        return result
