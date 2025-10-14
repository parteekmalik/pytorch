"""
GPU-accelerated image generation (GPU required).
"""
import numpy as np
import os
from typing import Tuple, Dict, Optional
import pandas as pd
from .utils import setup_logger, check_gpu_availability, get_array_module
from .image_storage import ImageStorageWriter
from .gpu_renderer import GPURenderer

logger = setup_logger(__name__)

GPU_AVAILABLE, GPU_BACKEND = check_gpu_availability()
logger.info(f"Image generator initialized with: {GPU_BACKEND}")


def create_images_from_data(
    data: pd.DataFrame,
    output_path: str,
    seq_len: int = 100,
    line_width: int = 3,
    batch_size: int = 10000,
    resolution: Optional[Dict[str, int]] = None,
    storage_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    rendering_config: Optional[Dict] = None
) -> str:
    if resolution is None:
        resolution = {'width': 800, 'height': 500, 'dpi': 100}
    
    if storage_config is None:
        storage_config = {'format': 'jpeg', 'mode': 'single', 'images_per_file': 50000}
    
    if rendering_config is None:
        rendering_config = {'mode': 'gpu', 'gpu_batch_size': 1000}
    
    storage_format = storage_config['format'].lower()
    
    from .data_loader import create_price_sequences
    
    # Filter to only keep 'Close' column if DataFrame
    if isinstance(data, pd.DataFrame):
        if 'Close' in data.columns:
            data = data[['Close']]
        else:
            data = pd.DataFrame()  # Empty if Close not available
    
    # Check if we have data to work with
    if isinstance(data, pd.DataFrame):
        if data.empty or len(data.columns) == 0:
            raise ValueError("No 'Close' price data available")
        closing_prices = data.iloc[:, 0].values
    else:  # Series case
        closing_prices = data.values
    
    sequences = create_price_sequences(closing_prices, seq_len)
    
    renderer = GPURenderer()
    
    logger.info(f"Generating {len(sequences)} images")
    logger.info("Rendering mode: GPU")
    logger.info(f"Storage format: {storage_format} ({storage_config['mode']} mode)")
    logger.info(f"Resolution: {resolution['width']}x{resolution['height']} @ {resolution['dpi']} DPI")
    
    if metadata is None:
        metadata = {}
    metadata['seq_len'] = seq_len
    metadata['line_width'] = line_width
    metadata['num_sequences'] = len(sequences)
    metadata['rendering_mode'] = 'gpu'
    
    if storage_format == 'jpeg':
        return _create_images_jpeg(
            sequences, output_path, line_width, resolution, batch_size, renderer
        )
    else:
        return _create_images_storage(
            sequences, output_path, line_width, resolution, batch_size,
            storage_config, metadata, renderer, rendering_config
        )


def _create_images_jpeg(
    sequences: np.ndarray,
    images_folder: str,
    line_width: int,
    resolution: Dict[str, int],
    batch_size: int,
    renderer: GPURenderer
) -> str:
    """Create images as individual JPEG files using specified renderer."""
    if os.path.exists(images_folder) and os.listdir(images_folder):
        logger.info(f"Images already exist in {images_folder}, skipping generation")
        return images_folder
    
    os.makedirs(images_folder, exist_ok=True)
    
    logger.info("Using GPU batch rendering for JPEG generation")
    gpu_batch_size = 1000  # Default batch size
    
    for start_idx in range(0, len(sequences), gpu_batch_size):
        end_idx = min(start_idx + gpu_batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        # Render entire batch on GPU in parallel
        batch_images = renderer.render_batch_gpu(batch_sequences, resolution, line_width)
        
        # Save each image as JPEG using PIL instead of matplotlib
        from PIL import Image
        for i, img in enumerate(batch_images):
            img_idx = start_idx + i
            img_filename = f'price_pattern_{img_idx:06d}.jpg'
            img_path = os.path.join(images_folder, img_filename)
            
            # Convert normalized float image to uint8
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode='L')
            pil_img.save(img_path, 'JPEG', quality=95)
        
        logger.info(f'Processed {end_idx}/{len(sequences)} images (GPU)')
    
    logger.info(f"Created {len(sequences)} JPEG images in {images_folder}")
    return images_folder


def _create_images_storage(
    sequences: np.ndarray,
    output_path: str,
    line_width: int,
    resolution: Dict[str, int],
    batch_size: int,
    storage_config: Dict,
    metadata: Dict,
    renderer: GPURenderer,
    rendering_config: Dict
) -> str:
    """Create images in HDF5/NPZ/Zarr format using specified renderer."""
    writer = ImageStorageWriter(
        output_path=output_path,
        storage_format=storage_config['format'],
        mode=storage_config['mode'],
        images_per_file=storage_config['images_per_file'],
        resolution=resolution,
        metadata=metadata
    )
    
    logger.info("Using GPU batch rendering for storage")
    gpu_batch_size = rendering_config.get('gpu_batch_size', 1000)
    
    for start_idx in range(0, len(sequences), gpu_batch_size):
        end_idx = min(start_idx + gpu_batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        # Render entire batch on GPU in parallel (vectorized)
        images = renderer.render_batch_gpu(batch_sequences, resolution, line_width)
        
        writer.write_batch(images, batch_sequences)
        logger.info(f'Processed {end_idx}/{len(sequences)} images (GPU)')
    
    writer.close()
    logger.info(f"Created {len(sequences)} images in {storage_config['format']} format")
    return output_path


