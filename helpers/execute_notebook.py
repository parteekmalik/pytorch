#!/usr/bin/env python3
"""
Execute notebook programmatically to test functionality.
"""

import sys
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import traceback

def execute_notebook(notebook_path, max_cells=None):
    """Execute notebook cells programmatically."""
    print(f"🚀 Executing notebook: {notebook_path}")
    print("=" * 50)
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Create executor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # Execute cells
    total_cells = len(notebook.cells)
    cells_to_execute = min(max_cells, total_cells) if max_cells else total_cells
    
    print(f"📊 Total cells: {total_cells}")
    print(f"📊 Executing: {cells_to_execute}")
    print("-" * 30)
    
    for i, cell in enumerate(notebook.cells[:cells_to_execute]):
        if cell.cell_type == 'code':
            print(f"🔄 Executing cell {i+1}...")
            try:
                ep.preprocess_cell(cell, {}, i)
                print(f"✅ Cell {i+1} completed")
            except Exception as e:
                print(f"❌ Cell {i+1} failed: {e}")
                print(f"   Error type: {type(e).__name__}")
                if "NameError" in str(type(e)):
                    print(f"   Missing variable: {e}")
                elif "ImportError" in str(type(e)):
                    print(f"   Missing import: {e}")
                elif "ValueError" in str(type(e)):
                    print(f"   Value error: {e}")
                return False
        else:
            print(f"⏭️  Skipping cell {i+1} (not code)")
    
    print(f"\n🎉 Successfully executed {cells_to_execute} cells!")
    return True

def main():
    """Main execution function."""
    notebook_path = "/Users/parteekmalik/github/pytorch/crypto_prediction.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    # Execute first 15 cells (data processing and scaling)
    print("📊 Testing data processing pipeline...")
    success = execute_notebook(notebook_path, max_cells=15)
    
    if success:
        print("\n✅ Data processing pipeline works!")
        
        # Ask if user wants to continue with model training
        print("\n🤔 Would you like to continue with model training? (This will take longer)")
        print("   Run the full notebook with: python helpers/execute_notebook.py --full")
        return True
    else:
        print("\n❌ Data processing pipeline failed!")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Execute notebook programmatically')
    parser.add_argument('--full', action='store_true', help='Execute all cells')
    args = parser.parse_args()
    
    if args.full:
        success = execute_notebook("/Users/parteekmalik/github/pytorch/crypto_prediction.ipynb")
    else:
        success = main()
    
    sys.exit(0 if success else 1)
