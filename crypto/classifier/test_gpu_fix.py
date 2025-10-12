"""
Quick test script to verify GPU rendering works without scipy.
Run this on Colab T4 to confirm the fix.
"""
import numpy as np
from src.gpu_renderer import GPURenderer

print("=" * 60)
print("Testing GPU Renderer (No scipy, No smoothing)")
print("=" * 60)

# Test GPU detection
renderer = GPURenderer(mode='auto')
print(f"\n✓ Renderer initialized")
print(f"  Mode: {renderer.mode}")
print(f"  GPU Available: {renderer.gpu_available}")

# Test rendering a single image
print(f"\n✓ Testing single image rendering...")
sequence = np.random.randn(100)
resolution = {'width': 800, 'height': 500, 'dpi': 100}

try:
    img = renderer.render_line_image(sequence, resolution, line_width=3)
    print(f"  ✓ Success! Image shape: {img.shape}")
    print(f"  ✓ Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  ✓ No scipy errors!")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test batch rendering if GPU available
if renderer.gpu_available:
    print(f"\n✓ Testing batch rendering (10 images)...")
    try:
        sequences = np.random.randn(10, 100)
        images = []
        for seq in sequences:
            img = renderer.render_line_image(seq, resolution, line_width=3)
            images.append(img)
        print(f"  ✓ Success! Generated {len(images)} images")
        print(f"  ✓ Batch shape: {np.array(images).shape}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n" + "=" * 60)
print("GPU Renderer Test Complete!")
print("=" * 60)
print("\nIf you see '✓ No scipy errors!' above, the fix works!")
print("You can now run: python pipeline.py")

