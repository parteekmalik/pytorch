# Crypto Classifier Pipeline

Professional machine learning pipeline for cryptocurrency data processing and image generation with GPU acceleration.

## Overview

This pipeline downloads historical cryptocurrency data from Binance, converts price sequences into images, and provides utilities for model storage and versioning.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **GPU Acceleration**: Automatic GPU detection and CuPy acceleration for array operations
- **Caching**: Smart caching of downloaded data to avoid redundant downloads
- **Configuration-Driven**: YAML-based configuration for easy customization
- **Professional Logging**: Structured logging with file and console outputs
- **Model Versioning**: Automated model storage with metadata and timestamps
- **Parallel Processing**: Multi-core image generation with ProcessPoolExecutor

## Project Structure

```
crypto/classifier/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── data_loader.py        # Data downloading and caching
│   ├── image_generator.py    # GPU-accelerated image creation
│   ├── model_storage.py      # Model save/load utilities
│   └── utils.py              # Shared utilities and GPU detection
├── config/
│   └── config.yaml           # Pipeline configuration
├── models/                   # Saved model checkpoints
├── logs/                     # Pipeline execution logs
├── data/
│   ├── raw/                  # Cached ZIP files from Binance
│   └── processed/            # Generated images
├── notebooks/                # Jupyter notebooks
├── pipeline.py               # Main orchestration script
└── README.md                 # This file
```

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. For GPU support (optional but recommended):

```bash
# Install CuPy for your CUDA version
# For CUDA 12.x:
pip install cupy-cuda12x

# For CUDA 11.x:
pip install cupy-cuda11x

# The pipeline will automatically fall back to CPU if GPU is not available
```

## Usage

### Running the Full Pipeline

```bash
cd crypto/classifier
python pipeline.py
```

This will:

1. Download cryptocurrency data from Binance (with caching)
2. Generate images from price sequences
3. Save execution logs

### Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  symbol: BTCUSDT # Trading pair
  interval: 1m # Time interval (1m, 5m, 1h, etc.)
  start_date: "2020-01" # Start date (YYYY-MM)
  end_date: "2024-01" # End date (YYYY-MM)

image:
  seq_len: 100 # Sequence length for images
  resolution:
    height: 500 # Image height in pixels (width auto-calculated: seq_len * 4)
  storage:
    format: hdf5 # Options: hdf5, npz, zarr, jpeg
    mode: single # 'single' = one file, 'batch' = multiple files
    images_per_file: 50000 # Images per file (batch mode only)

paths:
  raw_data: crypto/classifier/data/raw
  processed_data: crypto/classifier/data/processed
  models: crypto/classifier/models
  logs: crypto/classifier/logs
```

### Using Individual Modules

#### Data Loading

```python
from src.data_loader import download_crypto_data

data = download_crypto_data(
    symbol='BTCUSDT',
    interval='1m',
    start_date_str='2020-01',
    end_date_str='2024-01',
    cache_dir='crypto/classifier/data/raw'
)
```

#### Image Generation

```python
from src.image_generator import create_images_from_data

# HDF5 format (single file, recommended for millions of images)
hdf5_file = create_images_from_data(
    data=data,
    output_path='crypto/classifier/data/processed/my_images.h5',
    seq_len=100,
    resolution={'height': 500},  # Width auto-calculated: seq_len * 4
    storage_config={'format': 'hdf5', 'mode': 'single', 'images_per_file': 50000},
    metadata={'symbol': 'BTCUSDT', 'interval': '1m'}
)

# Load images from HDF5
from src.image_storage import load_images_from_storage
images, sequences, metadata = load_images_from_storage(
    hdf5_file, 'hdf5', indices=slice(0, 100)
)
```

#### Model Storage

```python
from src.model_storage import save_model, load_model, list_models

# Save a model
save_model(
    model=my_model,
    model_dir='crypto/classifier/models',
    metadata={'accuracy': 0.95, 'epochs': 100}
)

# Load a model
model, metadata = load_model('crypto/classifier/models/model_20240101_120000.pkl')

# List all models
models = list_models('crypto/classifier/models')
```

## OHLC Bar Chart Format

The pipeline generates OHLC (Open, High, Low, Close) bar charts with a fixed pixel layout:

### Bar Structure
Each OHLC bar uses exactly **4 pixels** horizontally:
- **Pixel 0**: High-Low vertical line
- **Pixel 1**: Open price (single pixel)
- **Pixel 2**: Close price (single pixel)
- **Pixel 3**: Gap (empty space for separation)

### Image Dimensions
- **Height**: Configurable (default: 500 pixels)
- **Width**: Auto-calculated as `seq_len * 4` pixels
- **Example**: 100 bars = 400 pixels wide

### Visual Features
- **White background** (pixel value: 1.0)
- **Black bars** (pixel value: 0.0) 
- **Fixed spacing** between bars for consistent visualization
- **GPU-accelerated rendering** using CuPy for high performance

### Configuration
```yaml
image:
  seq_len: 100  # Number of bars per image
  resolution:
    height: 500  # Height in pixels (width auto-calculated)
```

## GPU Acceleration

The pipeline automatically detects NVIDIA GPUs and uses CuPy for accelerated array operations:

- **Array normalization**: GPU-accelerated min-max scaling
- **Array operations**: Fast reshaping and transformations
- **Automatic fallback**: Seamlessly falls back to NumPy if GPU unavailable

Check GPU status in logs:

```
INFO - Image generator initialized with: CuPy (CUDA)
```

## Storage Formats

The pipeline supports multiple storage formats to efficiently handle millions of images:

### HDF5 (Recommended for Large Datasets)

- **Single file**: Store millions of images in one `.h5` file
- **Fast access**: Efficient random access by index
- **Compression**: Built-in gzip compression (30-50% space savings)
- **Metadata**: Store configuration and metadata alongside images
- **Python native**: Excellent support via `h5py`

```yaml
storage:
  format: hdf5
  mode: single # One file for all images
```

### NPZ (NumPy Compressed)

- **NumPy native**: Uses NumPy's compressed archive format
- **Simple**: Easy to use with NumPy arrays
- **Compressed**: Automatic compression
- **Batch mode**: Can split into multiple files

```yaml
storage:
  format: npz
  mode: batch # Multiple files
  images_per_file: 50000
```

### Zarr (Cloud-Optimized)

- **Scalable**: Designed for large datasets
- **Cloud-ready**: Works well with cloud storage
- **Fast compression**: Uses Blosc compressor
- **Chunked**: Efficient chunked storage

```yaml
storage:
  format: zarr
  mode: single
```

### JPEG (Individual Files)

- **Traditional**: One file per image
- **Compatible**: Works with any image viewer
- **Not recommended**: For millions of images (file system overhead)

```yaml
storage:
  format: jpeg
  mode: single
```

### Storage Mode Comparison

| Format | Best For           | File Count  | Compression | Random Access |
| ------ | ------------------ | ----------- | ----------- | ------------- |
| HDF5   | Millions of images | 1 file      | ✓           | Fast          |
| NPZ    | Medium datasets    | 1 or batch  | ✓           | Fast          |
| Zarr   | Cloud storage      | 1 directory | ✓           | Fast          |
| JPEG   | Small datasets     | Many files  | Moderate    | Slow          |

## Logging

Logs are saved in `logs/pipeline_YYYYMMDD_HHMMSS.log` with timestamps and structured information:

- Pipeline execution steps
- GPU detection status
- Download progress
- Image generation progress
- Error messages with stack traces

## Model Versioning

Models are saved with:

- Timestamp-based naming: `model_YYYYMMDD_HHMMSS.pkl`
- Metadata file: `model_YYYYMMDD_HHMMSS_metadata.json`
- Compression: joblib with level 3 compression

Metadata includes:

- Training metrics
- Configuration parameters
- Timestamp
- Custom fields

## Performance

- **Caching**: Downloads are cached to avoid redundant API calls
- **Parallel Processing**: Uses all CPU cores minus one for image generation
- **GPU Acceleration**: Up to 10x faster array operations with CUDA
- **Batch Processing**: Configurable batch sizes for memory efficiency

## Troubleshooting

### GPU Not Detected

If you have NVIDIA GPU but it's not detected:

1. Check CUDA installation: `nvidia-smi`
2. Install correct CuPy version for your CUDA
3. The pipeline will automatically fall back to CPU

### Out of Memory

If image generation runs out of memory:

1. Reduce `batch_size` in `config/config.yaml`
2. Reduce `seq_len` to generate fewer images
3. Process data in smaller date ranges

### Download Errors

If data downloads fail:

1. Check internet connection
2. Verify the date range is valid
3. Check Binance Vision availability
4. Cached files in `data/raw/` will be reused

## License

MIT License
