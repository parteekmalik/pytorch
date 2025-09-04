#!/usr/bin/env python3
"""
Script to run the cryptocurrency prediction notebook with proper environment setup.
"""

import subprocess
import sys
import os

def run_notebook():
    """Run the notebook using the virtual environment."""
    print("Starting Jupyter Notebook with the virtual environment...")
    print("=" * 60)
    print("INSTRUCTIONS:")
    print("1. The notebook will open in your browser")
    print("2. Open 'refined_crypto_prediction.ipynb'")
    print("3. Run all cells sequentially")
    print("4. The virtual environment is already activated")
    print("=" * 60)
    
    # Get the path to the virtual environment
    venv_path = os.path.join(os.getcwd(), 'crypto_env')
    python_path = os.path.join(venv_path, 'bin', 'python')
    
    try:
        # Start Jupyter notebook
        subprocess.run([python_path, '-m', 'jupyter', 'notebook'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Jupyter: {e}")
        print("\nAlternative: You can manually activate the environment and run:")
        print("source crypto_env/bin/activate")
        print("jupyter notebook")
    except KeyboardInterrupt:
        print("\nNotebook stopped by user.")

if __name__ == "__main__":
    run_notebook()
