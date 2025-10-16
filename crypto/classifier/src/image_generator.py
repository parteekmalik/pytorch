"""
GPU-accelerated image generation (GPU required).
"""
import numpy as np
import os
from typing import Tuple, Dict, Optional, List
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
    """Generate images from price data using simplified index-based storage."""
    
    if resolution is None:
        resolution = {'height': 500}  # Width auto-calculated
    
    if storage_config is None:
        storage_config = {'format': 'hdf5', 'mode': 'single', 'images_per_file': 50000}
    
    if rendering_config is None:
        rendering_config = {'mode': 'gpu', 'gpu_batch_size': 1000}
    
    from .data_loader import create_ohlc_sequences
    
    # Extract original OHLC data
    original_data = data[['Open', 'High', 'Low', 'Close']].values
    
    # Create sequences (these are just views/slices for rendering)
    sequences = create_ohlc_sequences(data, seq_len)
    
    renderer = Renderer()
    
    if metadata is None:
        metadata = {}
    
    # Pass both sequences and original_data
    return _create_images_storage(
        sequences=sequences,
        original_data=original_data,  # NEW: Pass original data
        output_path=output_path,
        resolution=resolution,
        storage_config=storage_config,
        metadata=metadata,
        renderer=renderer,
        rendering_config=rendering_config
    )


def _create_images_storage(
    sequences: List[np.ndarray],
    original_data: np.ndarray,  # NEW: Pass original data
    output_path: str,
    resolution: Dict[str, int],
    storage_config: Dict,
    metadata: Dict,
    renderer: Renderer,
    rendering_config: Dict
) -> str:
    """Generate images using simplified index-based storage (no indices)."""
    
    seq_len = len(sequences[0])
    num_images = len(sequences)
    
    logger.info(f"Generating {num_images} images from {len(original_data)} data points")
    logger.info("Rendering mode: GPU")
    logger.info(f"Storage format: hdf5 (index-based implicit, no compression)")
    logger.info(f"Resolution: {seq_len * 4}x{resolution['height']} (width auto-calculated)")
    
    # Add metadata
    metadata['seq_len'] = seq_len
    metadata['total_data_points'] = len(original_data)
    
    # Initialize storage writer
    writer = ImageStorageWriter(
        output_path=output_path,
        storage_format=storage_config['format'],
        mode=storage_config['mode'],
        images_per_file=storage_config.get('images_per_file', 50000),
        resolution=resolution,
        metadata=metadata
    )
    
    # Generate coordinates (indices are implicit)
    total_render_time = 0
    total_write_time = 0
    coordinates_list = []
    
    gpu_batch_size = rendering_config.get('gpu_batch_size', 2000)
    
    with tqdm(total=num_images, desc="Generating images", unit="img") as pbar:
        for start_idx in range(0, num_images, gpu_batch_size):
            end_idx = min(start_idx + gpu_batch_size, num_images)
            batch_sequences = sequences[start_idx:end_idx]
            
            # Render coordinates
            import time
            render_start = time.time()
            batch_coordinates = renderer.render_ohlc_batch_coordinates(
                np.array(batch_sequences)
            )
            render_time = time.time() - render_start
            total_render_time += render_time
            
            # Collect coordinates (no indices needed)
            for i in range(len(batch_coordinates)):
                coordinates_list.append(batch_coordinates[i])
            
            pbar.update(len(batch_coordinates))
    
    # Write original data + coordinates (no indices)
    import time
    write_start = time.time()
    writer.write_with_original_data(original_data, coordinates_list)
    writer.close()
    total_write_time = time.time() - write_start
    
    # Log timing
    total_time = total_render_time + total_write_time
    logger.info(f"Timing breakdown for {num_images} images:")
    logger.info(f"  Render time: {total_render_time:.3f}s ({total_render_time/total_time*100:.1f}%)")
    logger.info(f"  Write time: {total_write_time:.3f}s ({total_write_time/total_time*100:.1f}%)")
    logger.info(f"  Total time: {total_time:.3f}s")
    logger.info(f"  Average speed: {num_images/total_time:.1f} img/sec")
    
    logger.info(f"Created {num_images} images in index-based format")
    logger.info(f"Original data points: {len(original_data)}")
    logger.info(f"Storage: coordinates only (implicit indexing, no indices stored)")
    
    return output_path


