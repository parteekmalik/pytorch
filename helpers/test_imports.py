#!/usr/bin/env python3
"""
Test script to verify all required packages are installed correctly.
Run this before executing the main notebook.
"""

def test_imports():
    """Test all required imports for the cryptocurrency prediction notebook."""
    print("Testing imports for cryptocurrency prediction notebook...")
    
    try:
        import requests
        print("✓ requests imported successfully")
    except ImportError as e:
        print(f"✗ requests import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✓ tensorflow imported successfully")
    except ImportError as e:
        print(f"✗ tensorflow import failed: {e}")
        return False
    
    try:
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        from IPython.display import display
        print("✓ IPython.display imported successfully")
    except ImportError as e:
        print(f"✗ IPython.display import failed: {e}")
        return False
    
    print("\n🎉 All imports successful! You can now run the notebook.")
    return True

if __name__ == "__main__":
    test_imports()
