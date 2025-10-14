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
        
        images = []
        
        for batch_idx in range(batch_size):
            seq = sequences[batch_idx]
            
            # Convert CuPy array to CPU if needed
            if hasattr(seq, 'get'):
                seq_cpu = seq.get()
            else:
                seq_cpu = seq
            
            # Create DataFrame for Datashader
            df = pd.DataFrame({
                'x': np.arange(seq_len),
                'y': seq_cpu
            })
            
            # Normalize y values to [0, 1] range
            y_min, y_max = df['y'].min(), df['y'].max()
            if y_max > y_min:
                df['y'] = (df['y'] - y_min) / (y_max - y_min)
            else:
                df['y'] = 0.5  # Flat line if no variation
            
            # Create Datashader canvas
            canvas = ds.Canvas(plot_width=width, plot_height=height)
            
            # Render line (GPU-accelerated aggregation)
            agg = canvas.line(df, 'x', 'y', agg=ds.count())
            
            # Convert to grayscale image
            img = shade(agg, cmap=['white', 'black'])
            img = set_background(img, 'white')
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = img.to_numpy()[:, :, 0] / 255.0  # Take only grayscale channel
            
            # Invert colors: Datashader uses black lines on white background
            # We want white background (1.0) with black lines (0.0)
            img_array = 1.0 - img_array  # Invert so lines are black (0.0)
            
            images.append(img_array)
            
            if batch_idx % 100 == 0:
                logger.info(f"Processed {batch_idx + 1}/{batch_size} images")
        
        result = np.array(images)
        logger.info(f"Completed rendering {batch_size} images")
        return result
