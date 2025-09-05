#!/usr/bin/env python3
"""
Simple Notebook Runner using Jupyter nbconvert
Converts notebooks to Python and executes them directly
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def cleanup_build_dir():
    """Clean up the build directory"""
    build_dir = Path("build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

def run_notebook(notebook_path, output_dir="build", keep_python=False, verbose=True, cleanup=True):
    """
    Convert notebook to Python and execute it
    
    Sets matplotlib backend to non-interactive mode to prevent chart popups.
    
    Args:
        notebook_path (str): Path to the notebook file
        output_dir (str): Directory to save outputs (default: build folder)
        keep_python (bool): Keep the generated Python file
        verbose (bool): Print execution details
        cleanup (bool): Clean up build directory after execution
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"❌ Error: Notebook file '{notebook_path}' not found")
        return False
    
    if not notebook_path.suffix == '.ipynb':
        print(f"❌ Error: File '{notebook_path}' is not a Jupyter notebook")
        return False
    
    # Create output directory for all files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Python file path in output directory
    python_file = output_dir / f"{notebook_path.stem}.py"
    
    if verbose:
        print(f"🔄 Converting notebook: {notebook_path}")
        print(f"📝 Python file: {python_file}")
    
    try:
        # Convert notebook to Python
        cmd = [
            'jupyter', 'nbconvert', 
            '--to', 'python',
            '--output-dir', str(output_dir),
            str(notebook_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if verbose:
            print("✅ Notebook converted successfully")
        
        # Execute the Python file
        if verbose:
            print(f"🚀 Executing Python script: {python_file}")
        
        # Set environment to prevent chart popups
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'  # Prevent matplotlib GUI
        
        # Add necessary imports and path fixes to the Python file
        try:
            with open(python_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add sys.path fix for src import
            if 'from src import' in content and 'sys.path.append' not in content:
                # Find the first import statement and add sys.path fix before it
                lines = content.split('\n')
                new_lines = []
                added_sys_path = False
                
                for line in lines:
                    if 'from src import' in line and not added_sys_path:
                        new_lines.append('import sys')
                        new_lines.append('import os')
                        new_lines.append('sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))')
                        new_lines.append('')
                        added_sys_path = True
                    new_lines.append(line)
                
                content = '\n'.join(new_lines)
            
            # Ensure test_config is imported if using src imports
            if 'from src import' in content and 'test_config' not in content:
                # Add test_config to the import statement
                content = content.replace(
                    'from src import (',
                    'from src import (\n    test_config,'
                )
            
            # Add matplotlib backend setting at the top if matplotlib is imported
            if ('import matplotlib' in content or 'import matplotlib.pyplot' in content) and 'matplotlib.use(' not in content:
                # Find the first matplotlib import and add backend setting after it
                lines = content.split('\n')
                new_lines = []
                added_matplotlib_backend = False
                
                for line in lines:
                    if ('import matplotlib' in line or 'import matplotlib.pyplot' in line) and not added_matplotlib_backend:
                        # Add matplotlib import first if only pyplot is imported
                        if 'import matplotlib.pyplot' in line and 'import matplotlib' not in ''.join(new_lines):
                            new_lines.append('import matplotlib')
                        new_lines.append(line)
                        # Add the backend setting after the import
                        new_lines.append('matplotlib.use("Agg")  # Prevent GUI popups')
                        added_matplotlib_backend = True
                    else:
                        new_lines.append(line)
                
                content = '\n'.join(new_lines)
            
            # Replace production config with test config for faster execution
            if 'from src import' in content and 'production_config' in content:
                content = content.replace('production_config', 'test_config')
                # Also update CONFIG_MODE to match the actual configuration being used
                content = content.replace("CONFIG_MODE = 'production'", "CONFIG_MODE = 'test'")
            elif 'from src import' in content and 'test_config' not in content:
                # Only replace if test_config is not already being used
                content = content.replace('config.', 'test_config.')
                content = content.replace('config =', 'test_config =')
                # Also update CONFIG_MODE to match the actual configuration being used
                content = content.replace("CONFIG_MODE = 'production'", "CONFIG_MODE = 'test'")
                
            with open(python_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            pass  # Silently continue if file modification fails
        
        # Add parent directory to Python path for module imports
        python_path = str(notebook_path.parent)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{python_path}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = python_path
        
        # Run from the notebook's parent directory to maintain module paths
        result = subprocess.run(
            [sys.executable, str(python_file)],
            cwd=str(notebook_path.parent),
            env=env,
            check=True
        )
        
        if verbose:
            print("✅ Execution completed successfully")
        
        # Clean up Python file if not keeping it
        if not keep_python:
            python_file.unlink()
        else:
            pass  # Keep the file
        
        # Clean up output directory if requested and no files to keep
        if cleanup and not keep_python:
            cleanup_build_dir()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during execution:")
        print(f"   Command: {' '.join(e.cmd)}")
        print(f"   Return code: {e.returncode}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        # Clean up on error
        if cleanup:
            cleanup_build_dir()
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Jupyter notebooks as Python scripts')
    parser.add_argument('notebook', help='Path to the notebook file')
    parser.add_argument('-o', '--output-dir', help='Output directory for generated files (default: build/)')
    parser.add_argument('-k', '--keep-python', action='store_true', 
                       help='Keep the generated Python file (default: False - files are cleaned up)')
    parser.add_argument('-q', '--quiet', action='store_true', 
                       help='Suppress verbose output')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Do not clean up build directory after execution')
    
    args = parser.parse_args()
    
    success = run_notebook(
        notebook_path=args.notebook,
        output_dir=args.output_dir or "build",
        keep_python=args.keep_python,
        verbose=not args.quiet,
        cleanup=not args.no_cleanup
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
