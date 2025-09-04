# Cryptocurrency Price Prediction with LSTM

A refined implementation for predicting Bitcoin price movements using LSTM neural networks.

## Quick Setup

### Option 1: Automatic Setup (Recommended)
```bash
# Create virtual environment and install packages
python3 -m venv crypto_env
source crypto_env/bin/activate
pip install -r requirements.txt

# Test installation
python test_imports.py

# Run the notebook
python run_notebook.py
```

### Option 2: Manual Installation
```bash
# Create virtual environment
python3 -m venv crypto_env
source crypto_env/bin/activate

# Install packages
pip install -r requirements.txt

# Test installation
python test_imports.py

# Start Jupyter
jupyter notebook
```

### Option 3: Test Installation Only
```bash
source crypto_env/bin/activate
python test_imports.py
```

## Required Packages

- pandas >= 1.3.0
- numpy >= 1.21.0
- tensorflow >= 2.8.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- requests >= 2.25.0
- jupyter >= 1.0.0
- ipython >= 7.0.0

## Usage

1. Install the required packages using one of the methods above
2. Open `refined_crypto_prediction.ipynb` in Jupyter Notebook
3. Run all cells sequentially

## Features

- **Data Loading**: Downloads Bitcoin data from Binance
- **Feature Engineering**: Creates lag features, rolling statistics, and time-based features
- **LSTM Model**: Multi-output prediction for Open, High, Low, Close, and Volume
- **Visualization**: Comprehensive plots for analysis and evaluation
- **Prediction**: Functions to make predictions on new data

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Run the setup script or install packages manually
2. **SSL Warning**: This is a warning about urllib3 and OpenSSL - it doesn't affect functionality
3. **Data Download Issues**: Check your internet connection and Binance API availability

### Getting Help

If you encounter issues:
1. Run `python test_imports.py` to check package installation
2. Check that all cells are run in order
3. Ensure you have a stable internet connection for data download
