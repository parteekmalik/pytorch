"""
Example usage of the modular crypto classifier pipeline.
This demonstrates how to use individual modules.
"""

from src.data_loader import download_crypto_data, create_price_sequences
from src.image_generator import create_images_from_data
from src.model_storage import save_model, load_model, list_models
from src.utils import check_gpu_availability, setup_logger
import numpy as np

# Setup logger
logger = setup_logger('example')

# Check GPU availability
gpu_available, backend = check_gpu_availability()
logger.info(f"Using backend: {backend}")

# Example 1: Download data
logger.info("\n=== Example 1: Download Data ===")
data = download_crypto_data(
    symbol='BTCUSDT',
    interval='1m',
    start_date_str='2024-01',
    end_date_str='2024-01',
    cache_dir='crypto/classifier/data/raw'
)
logger.info(f"Downloaded {len(data)} rows")
logger.info(f"Columns: {list(data.columns)}")
logger.info(f"\nFirst few rows:\n{data.head()}")

# Example 2: Create sequences
logger.info("\n=== Example 2: Create Price Sequences ===")
closing_prices = data['Close'].values
sequences = create_price_sequences(closing_prices, seq_len=50)
logger.info(f"Created {len(sequences)} sequences of length 50")
logger.info(f"Shape: {sequences.shape}")

# Example 3: Generate images as JPEG
logger.info("\n=== Example 3a: Generate Images as JPEG ===")
images_folder = create_images_from_data(
    data=data,
    output_path='crypto/classifier/data/processed/example_images_jpeg',
    seq_len=50,
    line_width=3,
    batch_size=100,
    resolution={'width': 800, 'height': 500, 'dpi': 100},
    storage_config={'format': 'jpeg', 'mode': 'single', 'images_per_file': 50000}
)
logger.info(f"JPEG images saved to: {images_folder}")

# Example 3b: Generate images as HDF5
logger.info("\n=== Example 3b: Generate Images as HDF5 ===")
from src.image_storage import load_images_from_storage, get_storage_info

hdf5_path = create_images_from_data(
    data=data,
    output_path='crypto/classifier/data/processed/example_images.h5',
    seq_len=50,
    line_width=3,
    batch_size=100,
    resolution={'width': 800, 'height': 500, 'dpi': 100},
    storage_config={'format': 'hdf5', 'mode': 'single', 'images_per_file': 50000},
    metadata={'symbol': 'BTCUSDT', 'interval': '1m'}
)
logger.info(f"HDF5 file saved to: {hdf5_path}")

# Get info about HDF5 file
info = get_storage_info(hdf5_path, 'hdf5')
logger.info(f"HDF5 Info: {info['num_images']} images, "
           f"shape {info['image_shape']}, "
           f"{info['file_size_mb']:.2f} MB")

# Load some images from HDF5
images, sequences, metadata = load_images_from_storage(hdf5_path, 'hdf5', indices=slice(0, 5))
logger.info(f"Loaded {len(images)} images from HDF5")
logger.info(f"Metadata: {metadata}")

# Example 4: Model storage (demonstration)
logger.info("\n=== Example 4: Model Storage ===")

# Create a dummy model (in practice, this would be your trained model)
dummy_model = {
    'weights': np.random.randn(100, 10),
    'architecture': 'CNN',
    'version': '1.0'
}

# Save model with metadata
model_path = save_model(
    model=dummy_model,
    model_dir='crypto/classifier/models',
    metadata={
        'accuracy': 0.95,
        'loss': 0.05,
        'epochs': 50,
        'dataset': 'BTCUSDT_1m_2024-01',
        'notes': 'Example model for demonstration'
    },
    model_name='example_model'
)
logger.info(f"Model saved to: {model_path}")

# List all models
logger.info("\n=== List All Models ===")
models = list_models('crypto/classifier/models')
for model_info in models:
    logger.info(f"Model: {model_info['name']}")
    logger.info(f"  Size: {model_info['size_mb']:.2f} MB")
    logger.info(f"  Modified: {model_info['modified']}")
    if 'metadata' in model_info:
        logger.info(f"  Metadata: {model_info['metadata']}")

# Load the model back
logger.info("\n=== Load Model ===")
loaded_model, metadata = load_model(model_path)
logger.info(f"Loaded model with metadata: {metadata}")

logger.info("\n=== All Examples Completed! ===")

