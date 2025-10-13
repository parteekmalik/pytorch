"""
Main pipeline orchestration script for cryptocurrency data processing.
"""
import os
import yaml
from datetime import datetime
from pathlib import Path

from src.utils import setup_logger, ensure_dir, check_gpu_availability
from src.data_loader import download_crypto_data
from src.image_generator import create_images_from_data


def load_config(config_path: str = 'config/config.yaml') -> dict:
    config_full_path = Path(__file__).parent / config_path
    with open(config_full_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_pipeline(config_path: str = 'config/config.yaml'):
    config = load_config(config_path)
    
    # Get absolute path to classifier directory (where pipeline.py lives)
    base_dir = Path(__file__).parent
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Resolve log path relative to base_dir
    log_path = base_dir / config['paths']['logs'] / f'pipeline_{timestamp}.log'
    logger = setup_logger('pipeline', log_file=str(log_path))
    
    logger.info("="*60)
    logger.info("Starting Crypto Classifier Pipeline")
    logger.info("="*60)
    
    gpu_available, backend = check_gpu_availability()
    logger.info(f"Compute Backend: {backend}")
    
    # Ensure all directories exist, resolved relative to base_dir
    for path_name, path_value in config['paths'].items():
        abs_path = base_dir / path_value
        ensure_dir(str(abs_path))
        logger.info(f"Ensured directory: {abs_path}")
    
    data_config = config['data']
    logger.info(f"Data Config: {data_config}")
    
    try:
        logger.info("Step 1: Downloading cryptocurrency data...")
        
        # Resolve cache_dir for data download
        cache_dir = base_dir / config['paths']['raw_data']
        data = download_crypto_data(
            symbol=data_config['symbol'],
            interval=data_config['interval'],
            start_date_str=data_config['start_date'],
            end_date_str=data_config['end_date'],
            cache_dir=str(cache_dir),
            columns=data_config.get('columns', 'Close')
        )
        logger.info(f"Downloaded {len(data)} data points")
        
        logger.info("Step 2: Generating images from price sequences...")
        image_config = config['image']
        
        base_name = (
            f"crypto_{data_config['symbol']}_{data_config['interval']}_"
            f"{data_config['start_date']}_{data_config['end_date']}"
        )
        
        # Always use HDF5 format
        processed_dir = base_dir / config['paths']['processed_data']
        output_path = processed_dir / (base_name + '.h5')
        
        metadata = {
            'symbol': data_config['symbol'],
            'interval': data_config['interval'],
            'start_date': data_config['start_date'],
            'end_date': data_config['end_date']
        }
        
        images_path = create_images_from_data(
            data=data,
            output_path=str(output_path),
            seq_len=image_config['seq_len'],
            line_width=image_config['line_width'],
            batch_size=image_config['batch_size'],
            resolution=image_config['resolution'],
            storage_config={'format': 'hdf5', 'mode': 'single'},
            metadata=metadata,
            rendering_config=image_config.get('rendering', {'mode': 'auto', 'gpu_batch_size': 1000, 'fallback_on_error': True})
        )
        
        # Get HDF5 file info
        from src.image_storage import get_storage_info
        info = get_storage_info(images_path, 'hdf5')
        image_count = info['num_images']
        file_size_mb = info.get('file_size_mb', 0)
        logger.info(f"Created {image_count} images in HDF5 format")
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)
        logger.info(f"Results:")
        logger.info(f"  - Data points: {len(data)}")
        logger.info(f"  - Images: {image_count}")
        logger.info(f"  - Output: {images_path}")
        logger.info(f"  - Log: {log_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_pipeline()


