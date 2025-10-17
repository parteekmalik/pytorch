import numpy as np
from typing import Dict, Optional, List
import pandas as pd
from tqdm import tqdm
from .utils import setup_logger, check_gpu_availability
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
        resolution = {'height': 500}
    
    if storage_config is None:
        storage_config = {'format': 'hdf5', 'mode': 'single', 'images_per_file': 50000}
    
    if rendering_config is None:
        rendering_config = {'mode': 'gpu', 'gpu_batch_size': 1000}
    
    from .data_loader import create_ohlc_sequences
    
    original_data = data[['Open', 'High', 'Low', 'Close']].values
    
    sequences = create_ohlc_sequences(data, seq_len)
    
    renderer = Renderer()
    
    if metadata is None:
        metadata = {}
    
    return _create_images_storage(
        sequences=sequences,
        original_data=original_data,
        output_path=output_path,
        resolution=resolution,
        storage_config=storage_config,
        metadata=metadata,
        renderer=renderer,
        rendering_config=rendering_config
    )


def _create_images_storage(
    sequences: List[np.ndarray],
    original_data: np.ndarray,
    output_path: str,
    resolution: Dict[str, int],
    storage_config: Dict,
    metadata: Dict,
    renderer: Renderer,
    rendering_config: Dict
) -> str:
    
    seq_len = len(sequences[0])
    num_images = len(sequences)
    
    logger.info(f"Generating {num_images} images from {len(original_data)} data points")
    logger.info("Rendering mode: GPU")
    logger.info(f"Storage format: hdf5 (index-based implicit, no compression)")
    logger.info(f"Resolution: {seq_len * 4}x{resolution['height']} (width auto-calculated)")
    
    metadata['seq_len'] = seq_len
    metadata['total_data_points'] = len(original_data)
    
    writer = ImageStorageWriter(
        output_path=output_path,
        storage_format=storage_config['format'],
        mode=storage_config['mode'],
        images_per_file=storage_config.get('images_per_file', 50000),
        resolution=resolution,
        metadata=metadata
    )
    
    total_render_time = 0
    total_write_time = 0
    coordinates_list = []
    
    gpu_batch_size = rendering_config.get('gpu_batch_size', 2000)
    
    with tqdm(total=num_images, desc="Generating images", unit="img") as pbar:
        for start_idx in range(0, num_images, gpu_batch_size):
            end_idx = min(start_idx + gpu_batch_size, num_images)
            batch_sequences = sequences[start_idx:end_idx]
            
            import time
            render_start = time.time()
            batch_coordinates = renderer.render_ohlc_batch_coordinates(
                np.array(batch_sequences),
                height=resolution['height']
            )
            total_render_time += time.time() - render_start
            
            for i in range(len(batch_coordinates)):
                coordinates_list.append(batch_coordinates[i])
            
            pbar.update(len(batch_coordinates))
    
    import time
    write_start = time.time()
    writer.write_with_original_data(original_data, coordinates_list)
    writer.close()
    total_write_time = time.time() - write_start
    
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


