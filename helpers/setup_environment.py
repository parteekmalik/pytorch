#!/usr/bin/env python3
"""
Setup script to install required packages for the cryptocurrency prediction notebook.
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages using pip."""
    print("Installing required packages for cryptocurrency prediction notebook...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("Error: requirements.txt not found!")
        return False
    
    try:
        # Install packages from requirements.txt
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def test_installation():
    """Test if all packages are properly installed."""
    print("\nTesting installation...")
    try:
        import test_imports
        return test_imports.test_imports()
    except ImportError:
        print("Error: test_imports.py not found!")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("CRYPTOCURRENCY PREDICTION NOTEBOOK SETUP")
    print("=" * 60)
    
    # Install packages
    if install_packages():
        print("\n" + "=" * 60)
        print("PACKAGE INSTALLATION COMPLETED")
        print("=" * 60)
        
        # Test installation
        if test_installation():
            print("\n🎉 Setup completed successfully!")
            print("You can now run the refined_crypto_prediction.ipynb notebook.")
        else:
            print("\n⚠️  Setup completed but some packages may not be working correctly.")
            print("Please check the error messages above.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        print("You may need to install packages manually:")
        print("pip install pandas numpy tensorflow scikit-learn matplotlib requests jupyter ipython")

if __name__ == "__main__":
    main()
