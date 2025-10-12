"""
GPU-accelerated image generation from price sequences.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from typing import Tuple, Dict, Optional
import pandas as pd
from .utils import setup_logger, check_gpu_availability, get_array_module
from .image_storage import ImageStorageWriter

logger = setup_logger(__name__)

GPU_AVAILABLE, GPU_BACKEND = check_gpu_availability()
logger.info(f"Image generator initialized with: {GPU_BACKEND}")


def sequence_to_image(
    sequence: np.ndarray,
    line_width: int = 3,
    resolution: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Convert a price sequence to a grayscale image with GPU acceleration if available.
    
    Args:
        sequence: 1D array of price values
        line_width: Width of the plotted line
        resolution: Dict with 'width', 'height', 'dpi' (optional)
        
    Returns:
        2D array representing grayscale image (normalized to [0, 1])
    """
    if resolution is None:
        resolution = {'width': 800, 'height': 500, 'dpi': 100}
    
    xp = get_array_module()
    
    seq_gpu = xp.asarray(sequence)
    
    seq_min = xp.min(seq_gpu)
    seq_max = xp.max(seq_gpu)
    
    if seq_max > seq_min:
        normalized_seq = (seq_gpu - seq_min) / (seq_max - seq_min)
    else:
        normalized_seq = xp.zeros_like(seq_gpu)
    
    if hasattr(xp, 'asnumpy'):
        normalized_seq = xp.asnumpy(normalized_seq)
    
    figsize = (resolution['width'] / resolution['dpi'], resolution['height'] / resolution['dpi'])
    fig, ax = plt.subplots(figsize=figsize, dpi=resolution['dpi'])
    ax.set_xlim(0, len(normalized_seq))
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.plot(normalized_seq, linewidth=line_width, color='black', antialiased=True)
    
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    img_gray = np.mean(img_array, axis=2)
    img_normalized = img_gray.astype(np.float32) / 255.0
    
    plt.close(fig)
    return img_normalized


def _process_single_image_jpeg(args: Tuple) -> int:
    """
    Worker function to process and save a single image as JPEG.
    
    Args:
        args: Tuple of (index, sequence, line_width, resolution, output_folder)
        
    Returns:
        Index of processed image
    """
    i, seq, line_width, resolution, images_folder = args
    
    img = sequence_to_image(seq, line_width, resolution)
    
    img_filename = f'price_pattern_{i:06d}.jpg'
    img_path = os.path.join(images_folder, img_filename)
    
    figsize = (resolution['width'] / resolution['dpi'], resolution['height'] / resolution['dpi'])
    fig, ax = plt.subplots(figsize=figsize, dpi=resolution['dpi'])
    ax.imshow(img, cmap='gray', aspect='auto')
    ax.axis('off')
    
    plt.savefig(
        img_path, 
        bbox_inches='tight', 
        pad_inches=0, 
        dpi=resolution['dpi'], 
        facecolor='white', 
        edgecolor='none', 
        format='jpeg'
    )
    plt.close(fig)
    
    return i


def _process_single_image_array(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Worker function to process a single sequence and return array.
    
    Args:
        args: Tuple of (index, sequence, line_width, resolution)
        
    Returns:
        Tuple of (index, image_array)
    """
    i, seq, line_width, resolution = args
    img = sequence_to_image(seq, line_width, resolution)
    return i, img


def create_images_from_data(
    data: pd.DataFrame,
    output_path: str,
    seq_len: int = 100,
    line_width: int = 3,
    batch_size: int = 10000,
    resolution: Optional[Dict[str, int]] = None,
    storage_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Create images from price data with configurable storage format.
    
    Args:
        data: DataFrame with 'Close' column
        output_path: Output path (folder for JPEG, file for HDF5/NPZ/Zarr)
        seq_len: Length of each price sequence
        line_width: Width of line in plots
        batch_size: Number of images to process per batch
        resolution: Dict with 'width', 'height', 'dpi'
        storage_config: Dict with 'format', 'mode', 'images_per_file'
        metadata: Additional metadata to store
        
    Returns:
        Path to output (folder or file)
    """
    if resolution is None:
        resolution = {'width': 800, 'height': 500, 'dpi': 100}
    
    if storage_config is None:
        storage_config = {'format': 'jpeg', 'mode': 'single', 'images_per_file': 50000}
    
    storage_format = storage_config['format'].lower()
    
    from .data_loader import create_price_sequences
    
    closing_prices = data['Close'].values
    sequences = create_price_sequences(closing_prices, seq_len)
    
    logger.info(f"Generating {len(sequences)} images using {GPU_BACKEND}")
    logger.info(f"Storage format: {storage_format} ({storage_config['mode']} mode)")
    logger.info(f"Resolution: {resolution['width']}x{resolution['height']} @ {resolution['dpi']} DPI")
    
    if metadata is None:
        metadata = {}
    metadata['seq_len'] = seq_len
    metadata['line_width'] = line_width
    metadata['num_sequences'] = len(sequences)
    
    if storage_format == 'jpeg':
        return _create_images_jpeg(
            sequences, output_path, line_width, resolution, batch_size
        )
    else:
        return _create_images_storage(
            sequences, output_path, line_width, resolution, batch_size,
            storage_config, metadata
        )


def _create_images_jpeg(
    sequences: np.ndarray,
    images_folder: str,
    line_width: int,
    resolution: Dict[str, int],
    batch_size: int
) -> str:
    """Create images as individual JPEG files."""
    if os.path.exists(images_folder) and os.listdir(images_folder):
        logger.info(f"Images already exist in {images_folder}, skipping generation")
        return images_folder
    
    os.makedirs(images_folder, exist_ok=True)
    
    max_workers = cpu_count() - 1
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        args_list = [
            (i, seq, line_width, resolution, images_folder) 
            for i, seq in enumerate(sequences)
        ]
        
        results = []
        for i in range(0, len(sequences), batch_size):
            batch = args_list[i:i + batch_size]
            batch_results = list(executor.map(_process_single_image_jpeg, batch))
            results.extend(batch_results)
            logger.info(f'Processed {len(results)}/{len(sequences)} images')
    
    logger.info(f"Created {len(results)} JPEG images in {images_folder}")
    return images_folder


def _create_images_storage(
    sequences: np.ndarray,
    output_path: str,
    line_width: int,
    resolution: Dict[str, int],
    batch_size: int,
    storage_config: Dict,
    metadata: Dict
) -> str:
    """Create images in HDF5/NPZ/Zarr format."""
    writer = ImageStorageWriter(
        output_path=output_path,
        storage_format=storage_config['format'],
        mode=storage_config['mode'],
        images_per_file=storage_config['images_per_file'],
        resolution=resolution,
        metadata=metadata
    )
    
    max_workers = cpu_count() - 1
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for start_idx in range(0, len(sequences), batch_size):
            end_idx = min(start_idx + batch_size, len(sequences))
            batch_sequences = sequences[start_idx:end_idx]
            
            args_list = [
                (i, seq, line_width, resolution) 
                for i, seq in enumerate(batch_sequences, start=start_idx)
            ]
            
            results = list(executor.map(_process_single_image_array, args_list))
            results.sort(key=lambda x: x[0])
            
            images = [img for _, img in results]
            seqs = [batch_sequences[i] for i in range(len(batch_sequences))]
            
            writer.write_batch(images, seqs)
            logger.info(f'Processed {end_idx}/{len(sequences)} images')
    
    writer.close()
    logger.info(f"Created {len(sequences)} images in {storage_config['format']} format")
    return output_path


