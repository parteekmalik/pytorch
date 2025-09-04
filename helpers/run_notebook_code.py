#!/usr/bin/env python3
"""
Run notebook code directly as Python script.
This extracts and executes the notebook code without Jupyter dependencies.
"""

import sys
import os
import json
import traceback

# Add utils to path
sys.path.append('/Users/parteekmalik/github/pytorch')

def extract_notebook_code(notebook_path):
    """Extract code cells from notebook."""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    code_cells = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if source.strip():  # Skip empty cells
                code_cells.append(source)
    
    return code_cells

def execute_notebook_code(notebook_path, max_cells=None):
    """Execute notebook code directly."""
    print(f"🚀 Running notebook code: {notebook_path}")
    print("=" * 50)
    
    # Extract code cells
    code_cells = extract_notebook_code(notebook_path)
    total_cells = len(code_cells)
    cells_to_execute = min(max_cells, total_cells) if max_cells else total_cells
    
    print(f"📊 Total code cells: {total_cells}")
    print(f"📊 Executing: {cells_to_execute}")
    print("-" * 30)
    
    # Create a global namespace for execution
    global_namespace = {
        '__name__': '__main__',
        '__file__': notebook_path
    }
    
    for i, code in enumerate(code_cells[:cells_to_execute]):
        print(f"🔄 Executing cell {i+1}...")
        try:
            # Execute the code
            exec(code, global_namespace)
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
            print(f"   Code snippet: {code[:100]}...")
            return False
    
    print(f"\n🎉 Successfully executed {cells_to_execute} cells!")
    return True

def main():
    """Main execution function."""
    notebook_path = "/Users/parteekmalik/github/pytorch/crypto_prediction.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    # Execute first 10 cells (data processing)
    print("📊 Testing data processing pipeline...")
    success = execute_notebook_code(notebook_path, max_cells=10)
    
    if success:
        print("\n✅ Data processing pipeline works!")
        return True
    else:
        print("\n❌ Data processing pipeline failed!")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run notebook code directly')
    parser.add_argument('--full', action='store_true', help='Execute all cells')
    args = parser.parse_args()
    
    if args.full:
        success = execute_notebook_code("/Users/parteekmalik/github/pytorch/crypto_prediction.ipynb")
    else:
        success = main()
    
    sys.exit(0 if success else 1)
