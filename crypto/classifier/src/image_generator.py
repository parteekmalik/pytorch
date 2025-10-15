"""
GPU-accelerated image generation (GPU required).
"""
import numpy as np
import os
from typing import Tuple, Dict, Optional
import pandas as pd
from tqdm import tqdm
from .utils import setup_logger, check_gpu_availability, get_array_module
from .image_storage import ImageStorageWriter
from .renderer import Renderer

logger = setup_logger(__name__)

GPU_AVAILABLE, GPU_BACKEND = check_gpu_availability()
logger.info(f"Image generator initialized with: {GPU_BACKEND}")


def create_images_from_data(
    data: pd.DataFrame,
    output_path: str,
    seq_len: int = 100,
    resolution: Optional[Dict[str, int]] = None,
    storage_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    rendering_config: Optional[Dict] = None
) -> str:
    if resolution is None:
        resolution = {'width': 800, 'height': 500}
    
    if storage_config is None:
        storage_config = {'format': 'hdf5', 'mode': 'single', 'images_per_file': 50000}
    
    if rendering_config is None:
        rendering_config = {'mode': 'gpu', 'gpu_batch_size': 1000}
    
    from .data_loader import create_price_sequences, create_ohlc_sequences
    
    # Check if we have OHLC data to work with
    if isinstance(data, pd.DataFrame):
        if data.empty or len(data.columns) == 0:
            raise ValueError("No OHLC data available")
        ohlc_data = data.values  # Shape: (n_samples, 4) with [Open, High, Low, Close]
    else:  # Series case (shouldn't happen with OHLC)
        raise ValueError("Expected DataFrame with OHLC columns, got Series")
    
    sequences = create_ohlc_sequences(ohlc_data, seq_len)
    
    renderer = Renderer()
    
    logger.info(f"Generating {len(sequences)} images")
    logger.info("Rendering mode: GPU")
    logger.info(f"Storage format: hdf5 ({storage_config['mode']} mode)")
    logger.info(f"Resolution: {resolution['width']}x{resolution['height']}")
    
    if metadata is None:
        metadata = {}
    metadata['seq_len'] = seq_len
    metadata['num_sequences'] = len(sequences)
    metadata['rendering_mode'] = 'gpu'
    metadata['chart_type'] = 'candlestick'
    
    return _create_images_storage(
        sequences, output_path, resolution,
        storage_config, metadata, renderer, rendering_config
    )


def _create_images_storage(
    sequences: np.ndarray,
    output_path: str,
    resolution: Dict[str, int],
    storage_config: Dict,
    metadata: Dict,
    renderer: Renderer,
    rendering_config: Dict
) -> str:
    """Create images in HDF5/NPZ/Zarr format using specified renderer."""
    writer = ImageStorageWriter(
        output_path=output_path,
        storage_format=storage_config['format'],
        mode=storage_config['mode'],
        images_per_file=storage_config.get('images_per_file', 50000),
        resolution=resolution,
        metadata=metadata
    )
    
    logger.info("Using GPU batch rendering for storage")
    gpu_batch_size = rendering_config.get('gpu_batch_size', 1000)
    
    # Create global progress bar for all images
    with tqdm(total=len(sequences), desc="Generating images", unit="img") as pbar:
        for start_idx in range(0, len(sequences), gpu_batch_size):
            end_idx = min(start_idx + gpu_batch_size, len(sequences))
            batch_sequences = sequences[start_idx:end_idx]
            
            # Render entire batch on GPU in parallel (vectorized)
            images = renderer.render_candlestick_batch_gpu(batch_sequences, resolution, candle_width_ratio=0.8)
            
            writer.write_batch(images, batch_sequences)
            
            # Update global progress bar
            pbar.update(len(images))
    
    writer.close()
    logger.info(f"Created {len(sequences)} images in {storage_config['format']} format")
    return output_path


