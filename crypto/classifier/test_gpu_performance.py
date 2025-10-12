"""
Test GPU batch rendering performance.
Run this on Colab T4 to verify the fix is working.
"""
import numpy as np
import time
from src.gpu_renderer import GPURenderer

print("=" * 70)
print("GPU BATCH RENDERING PERFORMANCE TEST")
print("=" * 70)

# Initialize renderer
renderer = GPURenderer(mode='gpu')
print(f"\n✓ Renderer: mode={renderer.mode}, gpu_available={renderer.gpu_available}")

if not renderer.gpu_available:
    print("\n❌ ERROR: GPU not available!")
    print("   Make sure you're running on a GPU instance")
    exit(1)

resolution = {'width': 800, 'height': 500, 'dpi': 100}
line_width = 3

# Test different batch sizes
test_configs = [
    {'name': 'Single Image', 'batch_size': 1, 'iterations': 5},
    {'name': 'Small Batch (10)', 'batch_size': 10, 'iterations': 5},
    {'name': 'Small Batch (100)', 'batch_size': 100, 'iterations': 5},
    {'name': 'Medium Batch (1000)', 'batch_size': 1000, 'iterations': 3},
    {'name': 'Large Batch (5000)', 'batch_size': 5000, 'iterations': 2},
]

print("\n" + "=" * 70)
print("PERFORMANCE BENCHMARKS")
print("=" * 70)

results = []

for config in test_configs:
    batch_size = config['batch_size']
    iterations = config['iterations']
    name = config['name']
    
    print(f"\n🔄 Testing: {name} (batch_size={batch_size}, iterations={iterations})")
    
    times = []
    for i in range(iterations):
        # Generate random sequences
        sequences = np.random.randn(batch_size, 100)
        
        # Time the rendering
        start = time.time()
        
        if batch_size == 1:
            # Test single image rendering
            img = renderer.render_line_image(sequences[0], resolution, line_width)
        else:
            # Test batch rendering
            images = renderer.render_batch_gpu(sequences, resolution, line_width)
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"   Iteration {i+1}/{iterations}: {elapsed:.4f}s ({batch_size/elapsed:.1f} img/s)")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    img_per_sec = batch_size / avg_time
    
    results.append({
        'name': name,
        'batch_size': batch_size,
        'avg_time': avg_time,
        'std_time': std_time,
        'img_per_sec': img_per_sec
    })
    
    print(f"   ✓ Average: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"   ✓ Throughput: {img_per_sec:.1f} images/second")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Test':<20} {'Batch Size':<12} {'Time (s)':<12} {'Images/sec':<15}")
print("-" * 70)

for r in results:
    print(f"{r['name']:<20} {r['batch_size']:<12} {r['avg_time']:<12.4f} {r['img_per_sec']:<15.1f}")

# Speedup calculation
if len(results) > 1:
    baseline_ips = results[0]['img_per_sec']
    best_ips = max(r['img_per_sec'] for r in results)
    speedup = best_ips / baseline_ips
    
    print("\n" + "=" * 70)
    print(f"✅ SPEEDUP: {speedup:.1f}x faster (batch vs single)")
    print("=" * 70)

# Estimate time for full dataset
print("\n" + "=" * 70)
print("ESTIMATED PROCESSING TIMES")
print("=" * 70)

best_throughput = max(r['img_per_sec'] for r in results)

dataset_sizes = [
    ('85k images (2 months)', 85887),
    ('500k images (1 year)', 500000),
    ('10M images (20 years)', 10000000),
    ('40M images (80 years)', 40000000),
]

print(f"\nUsing best throughput: {best_throughput:.1f} images/second\n")

for desc, size in dataset_sizes:
    time_seconds = size / best_throughput
    
    if time_seconds < 60:
        time_str = f"{time_seconds:.1f} seconds"
    elif time_seconds < 3600:
        time_str = f"{time_seconds/60:.1f} minutes"
    else:
        time_str = f"{time_seconds/3600:.1f} hours"
    
    print(f"{desc:<30} → {time_str}")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
print("\n💡 If batch rendering is NOT faster than single image:")
print("   1. Check nvidia-smi shows high GPU memory usage")
print("   2. Verify CuPy version: python -c 'import cupy; print(cupy.__version__)'")
print("   3. Check logs for 'falling back to matplotlib' warnings")
print("\n✅ If you see 1000+ images/second, the fix is working! 🚀\n")

