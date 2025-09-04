#!/usr/bin/env python3
"""
Highly customizable notebook execution system with multiple options.
"""

import argparse
import json
import os
import sys
import subprocess
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter, PDFExporter
import jupyter_client

class NotebookRunner:
    """Highly customizable notebook execution system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or "/Users/parteekmalik/github/pytorch"
        self.venv_path = os.path.join(self.project_root, "crypto_env")
        self.python_path = os.path.join(self.venv_path, "bin", "python")
        self.notebooks = {
            "original": "crypto_prediction.ipynb",
            "ondemand": "crypto_prediction_ondemand.ipynb"
        }
    
    def list_available_notebooks(self) -> List[str]:
        """List all available notebooks."""
        notebooks = []
        for name, filename in self.notebooks.items():
            full_path = os.path.join(self.project_root, filename)
            if os.path.exists(full_path):
                notebooks.append(f"{name}: {filename}")
            else:
                notebooks.append(f"{name}: {filename} (NOT FOUND)")
        return notebooks
    
    def validate_environment(self) -> bool:
        """Validate that the environment is properly set up."""
        print("🔍 Validating environment...")
        
        # Check if virtual environment exists
        if not os.path.exists(self.venv_path):
            print(f"❌ Virtual environment not found: {self.venv_path}")
            return False
        
        # Check if Python executable exists
        if not os.path.exists(self.python_path):
            print(f"❌ Python executable not found: {self.python_path}")
            return False
        
        # Check if required packages are installed
        try:
            result = subprocess.run([
                self.python_path, "-c", 
                "import pandas, numpy, sklearn, torch, jupyter; print('✅ All packages available')"
            ], capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Missing packages: {e.stderr}")
            return False
        
        print("✅ Environment validation passed!")
        return True
    
    def run_jupyter_interactive(self, notebook_name: str = "original") -> bool:
        """Run notebook in interactive Jupyter mode."""
        notebook_file = self.notebooks.get(notebook_name)
        if not notebook_file:
            print(f"❌ Unknown notebook: {notebook_name}")
            return False
        
        notebook_path = os.path.join(self.project_root, notebook_file)
        if not os.path.exists(notebook_path):
            print(f"❌ Notebook not found: {notebook_path}")
            return False
        
        print(f"🚀 Starting Jupyter Notebook for: {notebook_file}")
        print("=" * 60)
        print("INSTRUCTIONS:")
        print(f"1. Open '{notebook_file}' in the browser")
        print("2. Run cells sequentially (Shift+Enter)")
        print("3. The virtual environment is already activated")
        print("4. Press Ctrl+C to stop")
        print("=" * 60)
        
        try:
            subprocess.run([self.python_path, "-m", "jupyter", "notebook"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error starting Jupyter: {e}")
            return False
        except KeyboardInterrupt:
            print("\n⏹️  Jupyter stopped by user.")
            return True
    
    def run_jupyterlab_interactive(self, notebook_name: str = "original") -> bool:
        """Run notebook in interactive JupyterLab mode."""
        notebook_file = self.notebooks.get(notebook_name)
        if not notebook_file:
            print(f"❌ Unknown notebook: {notebook_name}")
            return False
        
        print(f"🚀 Starting JupyterLab for: {notebook_file}")
        print("=" * 60)
        print("INSTRUCTIONS:")
        print(f"1. Open '{notebook_file}' in JupyterLab")
        print("2. Run cells sequentially (Shift+Enter)")
        print("3. The virtual environment is already activated")
        print("4. Press Ctrl+C to stop")
        print("=" * 60)
        
        try:
            subprocess.run([self.python_path, "-m", "jupyter", "lab"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error starting JupyterLab: {e}")
            return False
        except KeyboardInterrupt:
            print("\n⏹️  JupyterLab stopped by user.")
            return True
    
    def execute_notebook_programmatically(self, notebook_name: str = "original", 
                                        max_cells: Optional[int] = None,
                                        timeout: int = 600,
                                        stop_on_error: bool = True) -> bool:
        """Execute notebook programmatically using nbconvert."""
        notebook_file = self.notebooks.get(notebook_name)
        if not notebook_file:
            print(f"❌ Unknown notebook: {notebook_name}")
            return False
        
        notebook_path = os.path.join(self.project_root, notebook_file)
        if not os.path.exists(notebook_path):
            print(f"❌ Notebook not found: {notebook_path}")
            return False
        
        print(f"🚀 Executing notebook programmatically: {notebook_file}")
        print("=" * 60)
        
        try:
            # Read notebook
            with open(notebook_path, 'r') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Filter cells if max_cells specified
            if max_cells:
                notebook.cells = notebook.cells[:max_cells]
                print(f"📊 Executing first {max_cells} cells")
            
            # Create executor
            ep = ExecutePreprocessor(
                timeout=timeout,
                kernel_name='python3',
                interrupt_on_timeout=True
            )
            
            # Execute notebook
            ep.preprocess(notebook, {'metadata': {'path': self.project_root}})
            
            print("✅ Notebook executed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error executing notebook: {e}")
            if stop_on_error:
                traceback.print_exc()
            return False
    
    def extract_and_run_code(self, notebook_name: str = "original", 
                           max_cells: Optional[int] = None) -> bool:
        """Extract code from notebook and run it directly."""
        notebook_file = self.notebooks.get(notebook_name)
        if not notebook_file:
            print(f"❌ Unknown notebook: {notebook_name}")
            return False
        
        notebook_path = os.path.join(self.project_root, notebook_file)
        if not os.path.exists(notebook_path):
            print(f"❌ Notebook not found: {notebook_path}")
            return False
        
        print(f"🚀 Extracting and running code from: {notebook_file}")
        print("=" * 60)
        
        try:
            # Read notebook
            with open(notebook_path, 'r') as f:
                notebook_data = json.load(f)
            
            # Extract code cells
            code_cells = []
            for cell in notebook_data['cells']:
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source'])
                    if source.strip():
                        code_cells.append(source)
            
            if max_cells:
                code_cells = code_cells[:max_cells]
                print(f"📊 Executing first {max_cells} code cells")
            
            # Create execution environment
            global_namespace = {
                '__name__': '__main__',
                '__file__': notebook_path
            }
            
            # Add project root to Python path
            sys.path.insert(0, self.project_root)
            
            # Execute each code cell
            for i, code in enumerate(code_cells):
                print(f"🔄 Executing cell {i+1}/{len(code_cells)}...")
                try:
                    exec(code, global_namespace)
                    print(f"✅ Cell {i+1} completed")
                except Exception as e:
                    print(f"❌ Cell {i+1} failed: {e}")
                    print(f"   Error type: {type(e).__name__}")
                    return False
            
            print("✅ All code cells executed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error extracting/running code: {e}")
            traceback.print_exc()
            return False
    
    def convert_to_html(self, notebook_name: str = "original", 
                       output_dir: str = "output") -> bool:
        """Convert notebook to HTML."""
        notebook_file = self.notebooks.get(notebook_name)
        if not notebook_file:
            print(f"❌ Unknown notebook: {notebook_name}")
            return False
        
        notebook_path = os.path.join(self.project_root, notebook_file)
        if not os.path.exists(notebook_path):
            print(f"❌ Notebook not found: {notebook_path}")
            return False
        
        print(f"🔄 Converting to HTML: {notebook_file}")
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Read notebook
            with open(notebook_path, 'r') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Convert to HTML
            html_exporter = HTMLExporter()
            (body, resources) = html_exporter.from_notebook_node(notebook)
            
            # Save HTML
            output_file = os.path.join(output_dir, f"{notebook_name}.html")
            with open(output_file, 'w') as f:
                f.write(body)
            
            print(f"✅ HTML saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error converting to HTML: {e}")
            return False
    
    def run_specific_cells(self, notebook_name: str = "original", 
                          cell_indices: List[int] = None) -> bool:
        """Run specific cells from the notebook."""
        notebook_file = self.notebooks.get(notebook_name)
        if not notebook_file:
            print(f"❌ Unknown notebook: {notebook_name}")
            return False
        
        notebook_path = os.path.join(self.project_root, notebook_file)
        if not os.path.exists(notebook_path):
            print(f"❌ Notebook not found: {notebook_path}")
            return False
        
        print(f"🚀 Running specific cells from: {notebook_file}")
        print(f"📊 Cell indices: {cell_indices}")
        
        try:
            # Read notebook
            with open(notebook_path, 'r') as f:
                notebook_data = json.load(f)
            
            # Get code cells
            code_cells = []
            for i, cell in enumerate(notebook_data['cells']):
                if cell['cell_type'] == 'code':
                    code_cells.append((i, cell))
            
            # Execute specified cells
            global_namespace = {
                '__name__': '__main__',
                '__file__': notebook_path
            }
            sys.path.insert(0, self.project_root)
            
            for cell_idx in cell_indices:
                if cell_idx < len(code_cells):
                    original_idx, cell = code_cells[cell_idx]
                    source = ''.join(cell['source'])
                    if source.strip():
                        print(f"🔄 Executing cell {original_idx} (code cell {cell_idx})...")
                        try:
                            exec(source, global_namespace)
                            print(f"✅ Cell {original_idx} completed")
                        except Exception as e:
                            print(f"❌ Cell {original_idx} failed: {e}")
                            return False
                    else:
                        print(f"⏭️  Skipping empty cell {original_idx}")
                else:
                    print(f"⚠️  Cell index {cell_idx} out of range")
            
            print("✅ Specified cells executed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error running specific cells: {e}")
            return False
    
    def run_with_memory_monitoring(self, notebook_name: str = "original") -> bool:
        """Run notebook with memory monitoring."""
        try:
            import psutil
            import gc
        except ImportError:
            print("❌ psutil not available for memory monitoring")
            return False
        
        print(f"🚀 Running with memory monitoring: {notebook_name}")
        print("=" * 60)
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 Initial memory usage: {initial_memory:.2f} MB")
        
        # Run notebook
        success = self.extract_and_run_code(notebook_name, max_cells=10)
        
        # Get final memory
        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        print(f"📊 Final memory usage: {final_memory:.2f} MB")
        print(f"📊 Memory used: {memory_used:.2f} MB")
        
        return success

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Highly customizable notebook execution system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # List available notebooks
  python notebook_runner.py --list

  # Run original notebook in Jupyter
  python notebook_runner.py --jupyter original

  # Run on-demand notebook in JupyterLab
  python notebook_runner.py --jupyterlab ondemand

  # Execute programmatically (first 10 cells)
  python notebook_runner.py --execute original --max-cells 10

  # Extract and run code directly
  python notebook_runner.py --extract original --max-cells 5

  # Run specific cells (0, 2, 5)
  python notebook_runner.py --cells original 0 2 5

  # Convert to HTML
  python notebook_runner.py --convert original

  # Run with memory monitoring
  python notebook_runner.py --memory original

  # Validate environment
  python notebook_runner.py --validate
        """
    )
    
    # Execution modes
    parser.add_argument('--jupyter', action='store_true',
                       help='Run in interactive Jupyter mode')
    parser.add_argument('--jupyterlab', action='store_true',
                       help='Run in interactive JupyterLab mode')
    parser.add_argument('--execute', action='store_true',
                       help='Execute programmatically using nbconvert')
    parser.add_argument('--extract', action='store_true',
                       help='Extract code and run directly')
    parser.add_argument('--cells', nargs='+', type=int,
                       help='Run specific cell indices')
    parser.add_argument('--convert', action='store_true',
                       help='Convert notebook to HTML')
    parser.add_argument('--memory', action='store_true',
                       help='Run with memory monitoring')
    
    # Notebook selection
    parser.add_argument('notebook', nargs='?', default='original',
                       choices=['original', 'ondemand'],
                       help='Notebook to run (default: original)')
    
    # Options
    parser.add_argument('--max-cells', type=int,
                       help='Maximum number of cells to execute')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout for execution (seconds)')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for conversions')
    parser.add_argument('--list', action='store_true',
                       help='List available notebooks')
    parser.add_argument('--validate', action='store_true',
                       help='Validate environment setup')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = NotebookRunner()
    
    # Handle special commands
    if args.list:
        print("📚 Available notebooks:")
        for notebook in runner.list_available_notebooks():
            print(f"  - {notebook}")
        return
    
    if args.validate:
        success = runner.validate_environment()
        sys.exit(0 if success else 1)
    
    # Determine execution mode
    if args.jupyter:
        success = runner.run_jupyter_interactive(args.notebook)
    elif args.jupyterlab:
        success = runner.run_jupyterlab_interactive(args.notebook)
    elif args.execute:
        success = runner.execute_notebook_programmatically(
            args.notebook, args.max_cells, args.timeout)
    elif args.extract:
        success = runner.extract_and_run_code(args.notebook, args.max_cells)
    elif args.cells is not None:
        success = runner.run_specific_cells(args.notebook, args.cells)
    elif args.convert:
        success = runner.convert_to_html(args.notebook, args.output_dir)
    elif args.memory:
        success = runner.run_with_memory_monitoring(args.notebook)
    else:
        # Default: interactive mode
        print("🚀 Starting interactive mode...")
        print("Available options:")
        print("  --jupyter     : Run in Jupyter")
        print("  --jupyterlab  : Run in JupyterLab")
        print("  --execute     : Execute programmatically")
        print("  --extract     : Extract and run code")
        print("  --cells       : Run specific cells")
        print("  --convert     : Convert to HTML")
        print("  --memory      : Run with memory monitoring")
        print("  --list        : List available notebooks")
        print("  --validate    : Validate environment")
        print("\nUse --help for detailed usage information")
        success = True
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
