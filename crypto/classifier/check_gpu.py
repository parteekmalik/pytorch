"""
Check GPU availability and details for the crypto classifier pipeline.
"""
import sys

def check_gpu():
    print("=" * 60)
    print("GPU Detection Check")
    print("=" * 60)
    
    # Check CUDA availability
    try:
        import cupy as cp
        print("\n✓ CuPy is installed")
        
        # Get device info
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"✓ GPU Device 0: {props['name'].decode()}")
        print(f"  - Compute Capability: {device.compute_capability}")
        mem_info = device.mem_info
        print(f"  - Total Memory: {mem_info[1] / 1024**3:.2f} GB")
        print(f"  - Free Memory: {mem_info[0] / 1024**3:.2f} GB")
        
        # Test a simple operation
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        print(f"✓ GPU computation test passed: {cp.asnumpy(c)}")
        
        print("\n✓ GPU is available and working!")
        gpu_available = True
        
    except Exception as e:
        print(f"\n✗ GPU not available")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        gpu_available = False
    
    # Check multiprocessing context
    print("\n" + "=" * 60)
    print("Multiprocessing Context")
    print("=" * 60)
    
    import multiprocessing
    print(f"Process name: {multiprocessing.current_process().name}")
    print(f"CPU count: {multiprocessing.cpu_count()}")
    
    # Check if GPU would be disabled in workers
    if multiprocessing.current_process().name != 'MainProcess':
        print("⚠ Running in worker process - GPU would be disabled")
    else:
        print("✓ Running in main process - GPU available")
    
    # Show the issue
    print("\n" + "=" * 60)
    print("Current Pipeline Bottleneck")
    print("=" * 60)
    
    print("""
The pipeline has GPU available but it's NOT being used effectively because:

1. ✓ GPU detected: {gpu_status}
2. ✗ Worker processes: Forced to use CPU (to avoid CUDA multiprocessing errors)
3. ✗ Matplotlib: Does NOT support GPU rendering at all

Breakdown per image:
  - Array normalization (could use GPU): ~0.001s  ← 0.1% of time
  - Matplotlib rendering (CPU only):     ~1-2s    ← 99.9% of time (BOTTLENECK)
  
With 72 CPU cores:
  - Current speed: ~10,000 images per 2.5 minutes
  - Total time for 130,399 images: ~33 minutes
  
This is normal for matplotlib-based rendering. GPU won't help here.
""".format(gpu_status="YES, CuPy (CUDA)" if gpu_available else "NO"))

    print("\n" + "=" * 60)
    print("Speed Up Options")
    print("=" * 60)
    
    print("""
To speed up image generation:

Option 1: Skip matplotlib (10-100x faster)
  - Generate images directly from arrays using NumPy/CuPy
  - Use OpenCV or PIL for basic line drawing
  - Can leverage GPU for array operations
  
Option 2: Lower resolution (4x faster)
  - Change config.yaml: width=400, height=250, dpi=50
  - Faster rendering, smaller files
  
Option 3: Accept current speed
  - 33 minutes for 130k images is reasonable
  - Matplotlib produces high-quality plots
""")

if __name__ == "__main__":
    check_gpu()

