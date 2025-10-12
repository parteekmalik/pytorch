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
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_full_path = Path(__file__).parent / config_path
    with open(config_full_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_pipeline(config_path: str = 'config/config.yaml'):
    """
    Execute the full data download and image generation pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    config = load_config(config_path)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['paths']['logs'], f'pipeline_{timestamp}.log')
    logger = setup_logger('pipeline', log_file=log_file)
    
    logger.info("="*60)
    logger.info("Starting Crypto Classifier Pipeline")
    logger.info("="*60)
    
    gpu_available, backend = check_gpu_availability()
    logger.info(f"Compute Backend: {backend}")
    
    for path_name, path_value in config['paths'].items():
        ensure_dir(path_value)
        logger.info(f"Ensured directory: {path_value}")
    
    data_config = config['data']
    logger.info(f"Data Config: {data_config}")
    
    try:
        logger.info("Step 1: Downloading cryptocurrency data...")
        data = download_crypto_data(
            symbol=data_config['symbol'],
            interval=data_config['interval'],
            start_date_str=data_config['start_date'],
            end_date_str=data_config['end_date'],
            cache_dir=config['paths']['raw_data']
        )
        logger.info(f"Downloaded {len(data)} data points")
        
        logger.info("Step 2: Generating images from price sequences...")
        image_config = config['image']
        
        base_name = (
            f"crypto_{data_config['symbol']}_{data_config['interval']}_"
            f"{data_config['start_date']}_{data_config['end_date']}"
        )
        
        storage_format = image_config['storage']['format'].lower()
        
        if storage_format == 'jpeg':
            output_path = os.path.join(config['paths']['processed_data'], base_name)
        else:
            ext_map = {'hdf5': '.h5', 'zarr': '.zarr', 'npz': '.npz'}
            ext = ext_map.get(storage_format, '.h5')
            output_path = os.path.join(config['paths']['processed_data'], base_name + ext)
        
        metadata = {
            'symbol': data_config['symbol'],
            'interval': data_config['interval'],
            'start_date': data_config['start_date'],
            'end_date': data_config['end_date']
        }
        
        images_path = create_images_from_data(
            data=data,
            output_path=output_path,
            seq_len=image_config['seq_len'],
            line_width=image_config['line_width'],
            batch_size=image_config['batch_size'],
            resolution=image_config['resolution'],
            storage_config=image_config['storage'],
            metadata=metadata,
            rendering_config=image_config.get('rendering', {'mode': 'auto', 'gpu_batch_size': 1000, 'fallback_on_error': True})
        )
        
        if storage_format == 'jpeg':
            image_count = len([f for f in os.listdir(images_path) if f.endswith('.jpg')])
            logger.info(f"Created {image_count} JPEG images in {images_path}")
        else:
            from src.image_storage import get_storage_info
            if image_config['storage']['mode'] == 'single':
                info = get_storage_info(images_path, storage_format)
                image_count = info['num_images']
                file_size_mb = info.get('file_size_mb', 0)
                logger.info(f"Created {image_count} images in {storage_format.upper()} format")
                logger.info(f"File size: {file_size_mb:.2f} MB")
            else:
                logger.info(f"Created images in batch {storage_format.upper()} format")
                image_count = len(data) - image_config['seq_len'] + 1
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)
        logger.info(f"Results:")
        logger.info(f"  - Data points: {len(data)}")
        logger.info(f"  - Images: {image_count}")
        logger.info(f"  - Output: {images_path}")
        logger.info(f"  - Log: {log_file}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_pipeline()


