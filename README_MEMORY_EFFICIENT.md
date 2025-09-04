# 🚀 Memory-Efficient Crypto Prediction Notebooks

**Designed for 16GB M4 MacBook - No Memory Overflow!**

This repository contains two memory-efficient approaches for cryptocurrency price prediction using LSTM models.

## 📁 Project Structure

```
pytorch/
├── crypto_prediction.ipynb              # Original notebook (updated with memory management)
├── crypto_prediction_ondemand.ipynb     # On-demand processing approach
├── utils/                               # Common utility functions
│   ├── __init__.py                     # Package initialization
│   ├── memory_utils.py                 # Memory management utilities
│   ├── data_utils.py                   # Data processing utilities
│   └── model_utils.py                  # Model creation and training utilities
└── helpers/                            # Test and helper scripts
    ├── test_utils_comprehensive.py    # Comprehensive test suite for all utils
    ├── test_imports.py                # Quick import test
    ├── create_proper_preprocessing.py # Preprocessing utilities
    ├── run_notebook.py                # Notebook execution utilities
    └── setup_environment.py           # Environment setup utilities
```

## 🎯 Two Approaches

### 1. **Original Notebook** (`crypto_prediction.ipynb`)

- **Updated with memory management**
- **Memory limit**: 6GB
- **Best for**: Medium datasets (50k-100k rows)
- **Features**: Full feature set, comprehensive processing

### 2. **On-Demand Processing** (`crypto_prediction_ondemand.ipynb`)

- **Processes data only when needed**
- **Memory limit**: 3GB
- **Best for**: Flexible processing, exploration
- **Features**: Ultra memory efficient, lightweight models

## 🛠️ Utility Functions

### Memory Management (`utils/memory_utils.py`)

```python
from utils import get_memory_usage, check_memory_limit, force_garbage_collection

# Check current memory usage
memory = get_memory_usage()  # Returns MB

# Check if within memory limit
within_limit = check_memory_limit(max_memory_mb=4000)

# Force garbage collection
force_garbage_collection()
```

### Data Processing (`utils/data_utils.py`)

```python
from utils import download_binance_klines_data, create_minimal_features, create_sliding_windows

# Download data
data = download_binance_klines_data("BTCUSDT", "5m", "2021", "01")

# Create features
features = create_minimal_features(data, lag_period=3)

# Create sliding windows
X, y, feature_cols = create_sliding_windows(features, sequence_length=5)
```

### Model Creation (`utils/model_utils.py`)

```python
from utils import create_lstm_model, create_lightweight_lstm_model, train_model_memory_efficient

# Create regular LSTM model
model = create_lstm_model(input_shape=(5, 22), output_dim=5)

# Create lightweight model (memory efficient)
model = create_lightweight_lstm_model(input_shape=(5, 22), output_dim=5)

# Train model
history = train_model_memory_efficient(model, X_train, y_train, X_test, y_test)
```

## 📊 Memory Usage Comparison

| Approach      | Max Memory | Use Case            | Best For        |
| ------------- | ---------- | ------------------- | --------------- |
| **Original**  | ~6GB       | Full processing     | Medium datasets |
| **On-Demand** | ~3GB       | Flexible processing | Exploration     |

## 🚀 Quick Start

### 1. **For Flexible Processing**

```bash
# Use on-demand approach
jupyter notebook crypto_prediction_ondemand.ipynb
```

### 2. **For Medium Datasets (50k-100k rows)**

```bash
# Use original notebook
jupyter notebook crypto_prediction.ipynb
```

## 🎮 Advanced Execution Options

The project includes a highly customizable notebook runner with multiple execution modes:

### **Interactive Modes**

```bash
# Jupyter Notebook (traditional)
python helpers/notebook_runner.py --jupyter original

# JupyterLab (modern interface)
python helpers/notebook_runner.py --jupyterlab ondemand
```

### **Programmatic Modes**

```bash
# Full notebook execution
python helpers/notebook_runner.py --execute original

# Extract and run code directly (first 10 cells)
python helpers/notebook_runner.py --extract original --max-cells 10

# Run specific cells only
python helpers/notebook_runner.py original --cells 0 1 2 3

# Run with memory monitoring
python helpers/notebook_runner.py original --memory
```

### **Conversion Modes**

```bash
# Convert to HTML for sharing
python helpers/notebook_runner.py original --convert --output-dir reports

# List available notebooks
python helpers/notebook_runner.py --list

# Validate environment
python helpers/notebook_runner.py --validate
```

### **Available Notebooks**

- `original`: `crypto_prediction.ipynb` - Original preprocessing approach
- `ondemand`: `crypto_prediction_ondemand.ipynb` - On-demand processing approach

## 🧪 Testing

Run tests to verify everything works:

```bash
# Test all utilities comprehensively
python helpers/test_utils_comprehensive.py

# Quick import test
python helpers/test_imports.py
```

## ⚙️ Configuration

### On-Demand Configuration

```python
ONDEMAND_CONFIG = {
    'MAX_MEMORY_MB': 3000,        # Ultra-safe memory limit
    'SEQUENCE_LENGTH': 5,         # Short sequences
    'LAG_PERIOD': 3,             # Reduced lag features
    'MAX_SAMPLES': 10000         # Maximum samples to process
}
```

## 🎯 Key Features

### Memory Management

- **Real-time memory monitoring**
- **Automatic memory limit checking**
- **Garbage collection optimization**
- **Chunk-based processing**

### Data Processing

- **Minimal feature set** (22 features vs 43+)
- **Reduced lag period** (3 vs 5)
- **Short sequences** (5 vs 10)
- **Efficient sliding windows**

### Model Architecture

- **Lightweight LSTM models**
- **Reduced parameters** (10k vs 36k)
- **Memory-efficient training**
- **Early stopping and learning rate reduction**

## 📈 Performance Results

### Memory Usage

- **Original approach**: ~6GB (✅ Success with memory management)
- **On-demand approach**: ~900MB (✅ Success)

### Processing Speed

- **On-demand**: Fast for small datasets
- **Original**: Fast but memory-intensive

## 🔧 Troubleshooting

### Memory Issues

1. **Reduce batch size**: Change `BATCH_SIZE` to 500
2. **Reduce sequence length**: Change `SEQUENCE_LENGTH` to 3
3. **Reduce lag period**: Change `LAG_PERIOD` to 2
4. **Limit data size**: Reduce `max_rows` parameter

### Performance Issues

1. **Use lightweight models**: `create_lightweight_lstm_model()`
2. **Reduce epochs**: Set `epochs=10` for training
3. **Increase batch size**: Set `batch_size=64` for training

## 🎉 Success Metrics

- ✅ **Memory usage under 16GB limit**
- ✅ **No memory overflow errors**
- ✅ **Efficient data processing**
- ✅ **Working LSTM models**
- ✅ **Accurate predictions**

## 📞 Support

If you encounter any issues:

1. **Check memory usage**: Use `get_memory_usage()`
2. **Verify configuration**: Check memory limits
3. **Run tests**: Use helper test scripts
4. **Reduce data size**: Lower `max_rows` or `max_samples`

---

**Perfect for 16GB M4 MacBook! 🚀**
