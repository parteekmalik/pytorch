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
        
        logger.info(f"Rendering {batch_size} images using Datashader")
        
        images = []
        
        for batch_idx in range(batch_size):
            seq = sequences[batch_idx]
            
            if hasattr(seq, 'get'):
                seq_cpu = seq.get()
            else:
                seq_cpu = seq
            
            df = pd.DataFrame({
                'x': np.arange(seq_len),
                'y': seq_cpu
            })
            
            y_min, y_max = df['y'].min(), df['y'].max()
            if y_max > y_min:
                df['y'] = (df['y'] - y_min) / (y_max - y_min)
            else:
                df['y'] = 0.5
            
            canvas = ds.Canvas(plot_width=width, plot_height=height)
            
            agg = canvas.line(df, 'x', 'y', agg=ds.count())
            
            img = shade(agg, cmap=['white', 'black'])
            img = set_background(img, 'white')
            
            img_array = img.to_numpy()[:, :, 0] / 255.0
            
            img_array = 1.0 - img_array
            
            images.append(img_array)
            
            if batch_idx % 100 == 0:
                logger.info(f"Processed {batch_idx + 1}/{batch_size} images")
        
        result = np.array(images)
        logger.info(f"Completed rendering {batch_size} images")
        return result