#!/usr/bin/env python3
"""
Quick import test for basic dependencies.
"""

def test_basic_imports():
    """Test basic required imports."""
    print("🧪 Testing basic imports...")
    
    imports = [
        ("pandas", "pd"),
        ("numpy", "np"), 
        ("tensorflow", "tf"),
        ("sklearn.preprocessing", "MinMaxScaler"),
        ("sklearn.metrics", "mean_squared_error"),
        ("matplotlib.pyplot", "plt"),
        ("IPython.display", "display")
    ]
    
    for module, alias in imports:
        try:
            exec(f"import {module} as {alias}")
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            print(f"✗ {module} import failed: {e}")
            return False
    
    print("✅ All basic imports successful!")
    return True

if __name__ == "__main__":
    test_basic_imports()