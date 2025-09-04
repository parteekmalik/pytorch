# Simple Notebook Runner Guide

This guide explains how to use the simplified notebook runner to execute Jupyter notebooks using Jupyter's built-in conversion tools.

## Overview

The simplified approach uses Jupyter's `nbconvert` to convert notebooks to Python files and execute them directly. This method is more reliable, faster, and doesn't require complex runner infrastructure.

## Quick Start

### 1. Basic Usage

```bash
# Run a notebook (converts to Python and executes)
python helpers/run_notebook.py crypto_prediction.ipynb

# Run with custom output directory
python helpers/run_notebook.py crypto_prediction.ipynb --output-dir results/

# Keep the generated Python file
python helpers/run_notebook.py crypto_prediction.ipynb --keep-python

# Quiet mode (minimal output)
python helpers/run_notebook.py crypto_prediction.ipynb --quiet
```

### 2. Direct Jupyter Commands

You can also use Jupyter commands directly:

```bash
# Convert and execute in one step
jupyter nbconvert --to python --execute crypto_prediction.ipynb

# Convert to Python and run separately
jupyter nbconvert --to python crypto_prediction.ipynb
python crypto_prediction.py

# Execute and save with outputs
jupyter nbconvert --to notebook --execute crypto_prediction.ipynb --output crypto_prediction_executed.ipynb
```

## Features

- **Simple & Reliable**: Uses Jupyter's built-in tools
- **No Dependencies**: Only requires Jupyter (already installed)
- **Fast Execution**: Direct Python execution without notebook overhead
- **Chart Prevention**: Automatically prevents matplotlib GUI popups
- **Clean Output**: Organized file management in `build/` directory
- **Error Handling**: Clear error messages and proper exit codes

## Command Line Options

| Option          | Description                                     | Default  |
| --------------- | ----------------------------------------------- | -------- |
| `notebook`      | Path to notebook file                           | Required |
| `--output-dir`  | Output directory for generated files            | `build/` |
| `--keep-python` | Keep the generated Python file                  | `False`  |
| `--quiet`       | Suppress verbose output                         | `False`  |
| `--no-cleanup`  | Do not clean up build directory after execution | `False`  |

## Examples

### Example 1: Basic Execution

```bash
# Simple execution
python helpers/run_notebook.py crypto_prediction.ipynb
```

Output:

```
🔄 Converting notebook: crypto_prediction.ipynb
📝 Python file: build/crypto_prediction.py
✅ Notebook converted successfully
🚀 Executing Python script: build/crypto_prediction.py
✅ Execution completed successfully
🧹 Cleaned up temporary Python file
🧹 Cleaned up build directory
```

### Example 2: Keep Python File

```bash
# Keep the generated Python file for inspection
python helpers/run_notebook.py crypto_prediction.ipynb --keep-python
```

Output:

```
🔄 Converting notebook: crypto_prediction.ipynb
📝 Python file: build/crypto_prediction.py
✅ Notebook converted successfully
🚀 Executing Python script: build/crypto_prediction.py
✅ Execution completed successfully
📁 Python file saved to: build/crypto_prediction.py
```

### Example 3: Custom Output Directory

```bash
# Save outputs to specific directory
python helpers/run_notebook.py crypto_prediction.ipynb --output-dir results/
```

### Example 4: Quiet Mode

```bash
# Minimal output for scripting
python helpers/run_notebook.py crypto_prediction.ipynb --quiet
```

## Direct Jupyter Usage

### Convert and Execute

```bash
# One-step conversion and execution
jupyter nbconvert --to python --execute crypto_prediction.ipynb
```

### Execute with Outputs

```bash
# Execute and save with all outputs
jupyter nbconvert --to notebook --execute crypto_prediction.ipynb --output crypto_prediction_executed.ipynb
```

### Convert to Different Formats

```bash
# Convert to HTML
jupyter nbconvert --to html crypto_prediction.ipynb

# Convert to PDF
jupyter nbconvert --to pdf crypto_prediction.ipynb

# Convert to Markdown
jupyter nbconvert --to markdown crypto_prediction.ipynb
```

## Environment Setup

### Virtual Environment

```bash
# Activate virtual environment
source crypto_env/bin/activate

# Run notebook
python helpers/run_notebook.py crypto_prediction.ipynb
```

### Environment Variables

```bash
# Prevent matplotlib GUI (automatically set by the script)
export MPLBACKEND=Agg

# Set Jupyter data directory
export JUPYTER_DATA_DIR=/path/to/jupyter/data
```

## Directory Structure

The runner uses a simple directory structure:

```
project/
├── build/                    # Temporary files (auto-cleaned)
│   └── crypto_prediction.py  # Generated Python file (if --keep-python)
├── helpers/
│   └── run_notebook.py       # The runner script
├── crypto_prediction.ipynb   # Your notebook
└── utils/                    # Your modules
```

## Troubleshooting

### Common Issues

1. **Jupyter Not Found**

   ```bash
   # Solution: Activate virtual environment
   source crypto_env/bin/activate
   which jupyter
   ```

2. **Permission Errors**

   ```bash
   # Solution: Make script executable
   chmod +x helpers/run_notebook.py
   ```

3. **Chart Popups**

   ```bash
   # Solution: Set matplotlib backend (automatically handled)
   export MPLBACKEND=Agg
   ```

4. **Memory Issues**

   ```bash
   # Solution: Use direct Python execution
   jupyter nbconvert --to python crypto_prediction.ipynb
   python crypto_prediction.py
   ```

5. **Module Import Errors**
   ```bash
   # Solution: Ensure you're in the project root directory
   cd /path/to/project
   python helpers/run_notebook.py notebook.ipynb
   ```

## Best Practices

1. **Always use virtual environments** for dependency isolation
2. **Use `--keep-python`** for debugging and inspection
3. **Use `--quiet`** for automated scripts
4. **Check exit codes** in automation scripts
5. **Clean up temporary files** regularly (automatic by default)
6. **Use direct Jupyter commands** for one-off executions

## Automation Examples

### Bash Script

```bash
#!/bin/bash
# run_notebooks.sh

source crypto_env/bin/activate

for notebook in *.ipynb; do
    echo "Running $notebook..."
    python helpers/run_notebook.py "$notebook" --quiet
    if [ $? -eq 0 ]; then
        echo "✅ $notebook completed successfully"
    else
        echo "❌ $notebook failed"
    fi
done
```

### Python Script

```python
#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run_notebook(notebook_path):
    """Run a notebook using the helper script"""
    cmd = [sys.executable, "helpers/run_notebook.py", str(notebook_path), "--quiet"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

# Run all notebooks
for notebook in Path(".").glob("*.ipynb"):
    success = run_notebook(notebook)
    print(f"{'✅' if success else '❌'} {notebook.name}")
```

## Migration from Old System

The old complex runner system has been replaced with this simple approach:

- **Old**: `python helpers/notebook_runner.py notebook.ipynb`
- **New**: `python helpers/run_notebook.py notebook.ipynb`

Benefits of the new approach:

- ✅ Simpler and more reliable
- ✅ Uses standard Jupyter tools
- ✅ Faster execution
- ✅ Better error handling
- ✅ No complex dependencies
- ✅ Easier to maintain

## Support

For issues or questions:

1. Check that Jupyter is installed: `jupyter --version`
2. Ensure virtual environment is activated
3. Check notebook file exists and is valid
4. Review error messages for specific issues
5. Use `--no-cleanup` to inspect generated files for debugging
