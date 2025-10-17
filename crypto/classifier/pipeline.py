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
    
    base_dir = Path(__file__).parent
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    log_path = base_dir / config['paths']['logs'] / f'pipeline_{timestamp}.log'
    logger = setup_logger('pipeline', log_file=str(log_path))
    
    logger.info("="*60)
    logger.info("Starting Crypto Classifier Pipeline")
    logger.info("="*60)
    
    _, backend = check_gpu_availability()
    logger.info(f"Compute Backend: {backend}")
    
    for _, path_value in config['paths'].items():
        abs_path = base_dir / path_value
        ensure_dir(str(abs_path))
        logger.info(f"Ensured directory: {abs_path}")
    
    data_config = config['data']
    logger.info(f"Data Config: {data_config}")
    
    try:
        logger.info("Step 1: Downloading cryptocurrency data...")
        
        raw_data_path = base_dir / config['paths']['raw_data']
        ensure_dir(str(raw_data_path))
        
        data = download_crypto_data(
            symbol=data_config['symbol'],
            interval=data_config['interval'],
            start_date_str=data_config['start_date'],
            end_date_str=data_config['end_date'],
            cache_dir=str(raw_data_path)
        )
        
        logger.info(f"Downloaded {len(data)} data points")
        
        if data_config.get('max_sequences'):
            max_seq = data_config['max_sequences']
            logger.info(f"Limiting to {max_seq} sequences for testing")
            data = data.iloc[:max_seq + config['image']['seq_len'] - 1]
            logger.info(f"Truncated to {len(data)} data points")
        
        logger.info("Step 2: Generating images...")
        
        processed_data_path = base_dir / config['paths']['processed_data']
        ensure_dir(str(processed_data_path))
        
        image_config = config['image']
        output_path = processed_data_path / f"{data_config['symbol']}_{data_config['interval']}_{timestamp}.h5"
        
        metadata = {
            'symbol': data_config['symbol'],
            'interval': data_config['interval'],
            'start_date': data_config['start_date'],
            'end_date': data_config['end_date'],
            'total_data_points': len(data),
            'pipeline_timestamp': timestamp
        }
        
        result_path = create_images_from_data(
            data=data,
            output_path=str(output_path),
            seq_len=image_config['seq_len'],
            resolution=image_config['resolution'],
            metadata=metadata,
            rendering_config=image_config['rendering']
        )
        
        logger.info(f"✓ Images generated successfully: {result_path}")
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()