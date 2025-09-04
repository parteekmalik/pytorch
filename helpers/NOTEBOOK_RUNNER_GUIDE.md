# 🚀 Notebook Runner - Customizable Execution System

A highly customizable system for running Jupyter notebooks with multiple execution modes and options.

## 📋 Available Execution Modes

### 1. **Interactive Modes**

- **Jupyter Notebook**: Traditional notebook interface
- **JupyterLab**: Modern notebook interface

### 2. **Programmatic Modes**

- **Execute**: Full notebook execution using nbconvert
- **Extract**: Extract code and run directly as Python
- **Specific Cells**: Run only selected cells
- **Memory Monitoring**: Run with memory usage tracking

### 3. **Conversion Modes**

- **HTML Export**: Convert notebook to HTML format

## 🎯 Quick Start

```bash
# Activate environment
cd /Users/parteekmalik/github/pytorch
source crypto_env/bin/activate

# List available notebooks
python helpers/notebook_runner.py --list

# Validate environment
python helpers/notebook_runner.py --validate

# Run in Jupyter (interactive)
python helpers/notebook_runner.py --jupyter original

# Run in JupyterLab (interactive)
python helpers/notebook_runner.py --jupyterlab ondemand

# Execute programmatically (first 10 cells)
python helpers/notebook_runner.py --execute original --max-cells 10

# Extract and run code directly
python helpers/notebook_runner.py --extract original --max-cells 5

# Run specific cells (0, 2, 5)
python helpers/notebook_runner.py --cells original 0 2 5

# Convert to HTML
python helpers/notebook_runner.py --convert original

# Run with memory monitoring
python helpers/notebook_runner.py --memory original
```

## 📊 Available Notebooks

- **original**: `crypto_prediction.ipynb` - Original preprocessing approach
- **ondemand**: `crypto_prediction_ondemand.ipynb` - On-demand processing approach

## ⚙️ Advanced Options

### Execution Control

- `--max-cells N`: Limit execution to first N cells
- `--timeout N`: Set execution timeout (seconds)
- `--output-dir DIR`: Set output directory for conversions

### Memory Management

- `--memory`: Monitor memory usage during execution
- Automatic garbage collection
- Memory usage reporting

### Cell Selection

- `--cells 0 2 5`: Run specific cell indices
- Skip empty cells automatically
- Error handling per cell

## 🔧 Environment Requirements

The runner automatically:

- ✅ Validates virtual environment
- ✅ Checks Python executable
- ✅ Verifies required packages
- ✅ Sets up proper paths

## 📈 Usage Examples

### Development Workflow

```bash
# Quick test (first 5 cells)
python helpers/notebook_runner.py --extract original --max-cells 5

# Full execution test
python helpers/notebook_runner.py --execute original

# Interactive development
python helpers/notebook_runner.py --jupyterlab original
```

### Production Workflow

```bash
# Convert to HTML for sharing
python helpers/notebook_runner.py --convert original --output-dir reports

# Run with memory monitoring
python helpers/notebook_runner.py --memory original
```

### Debugging

```bash
# Run specific problematic cells
python helpers/notebook_runner.py --cells original 10 11 12

# Test data processing only
python helpers/notebook_runner.py --extract original --max-cells 8
```

## 🛠️ Troubleshooting

### Common Issues

1. **Environment not activated**: Run `source crypto_env/bin/activate`
2. **Missing packages**: Run `pip install -r requirements.txt`
3. **Notebook not found**: Check file paths in `notebooks` dictionary
4. **Memory issues**: Use `--memory` flag to monitor usage

### Error Handling

- ✅ Graceful error handling per cell
- ✅ Detailed error reporting
- ✅ Continue execution on non-critical errors
- ✅ Memory cleanup on errors

## 📝 Output Examples

### Successful Execution

```
🚀 Extracting and running code from: crypto_prediction.ipynb
============================================================
📊 Executing first 5 code cells
🔄 Executing cell 1/5...
✅ Cell 1 completed
🔄 Executing cell 2/5...
✅ Global configuration loaded successfully!
...
✅ All code cells executed successfully!
```

### Memory Monitoring

```
🚀 Running with memory monitoring: original
============================================================
📊 Initial memory usage: 45.23 MB
...
📊 Final memory usage: 67.89 MB
📊 Memory used: 22.66 MB
```

## 🎯 Best Practices

1. **Start Small**: Use `--max-cells 5` for initial testing
2. **Monitor Memory**: Use `--memory` for large datasets
3. **Interactive Development**: Use `--jupyterlab` for development
4. **Production Testing**: Use `--execute` for full validation
5. **Debugging**: Use `--cells` for specific cell testing

## 🔄 Integration

The runner integrates seamlessly with:

- ✅ Virtual environments
- ✅ Jupyter ecosystem
- ✅ nbconvert
- ✅ Memory monitoring
- ✅ Error handling
- ✅ Custom configurations
