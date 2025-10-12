"""
Visual verification of GPU-rendered lines.
Generates sample images to verify line quality after vectorization.
"""
import numpy as np
import matplotlib.pyplot as plt
from src.gpu_renderer import GPURenderer

print("=" * 70)
print("GPU Line Rendering Visual Verification")
print("=" * 70)

# Initialize renderer
renderer = GPURenderer(mode='auto')
print(f"\nRenderer mode: {renderer.mode}, GPU available: {renderer.gpu_available}")

if not renderer.gpu_available:
    print("\n⚠️  GPU not available - this test requires GPU")
    print("   Run this on Colab T4 to verify GPU rendering")
    exit(0)

# Create test sequences with different patterns
test_sequences = {
    'Linear Up': np.linspace(0, 1, 100),
    'Linear Down': np.linspace(1, 0, 100),
    'Sine Wave': np.sin(np.linspace(0, 4*np.pi, 100)),
    'Random Walk': np.cumsum(np.random.randn(100)),
    'Step Function': np.repeat([0, 1, 0.5, 0.2, 0.8], 20),
    'Spiky': np.random.randn(100),
}

resolution = {'width': 800, 'height': 500, 'dpi': 100}
line_width = 3

print(f"\nGenerating {len(test_sequences)} test images...")
print(f"Resolution: {resolution['width']}x{resolution['height']}")
print(f"Line width: {line_width}")

# Render each pattern
images = {}
for name, seq in test_sequences.items():
    print(f"  Rendering: {name}...")
    img = renderer.render_line_image(seq, resolution, line_width)
    images[name] = img

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('GPU-Rendered Line Images (Vectorized)', fontsize=16)

for idx, (name, img) in enumerate(images.items()):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    ax.imshow(img, cmap='gray', aspect='auto')
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
output_path = 'crypto/classifier/gpu_rendering_verification.png'
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"\n✓ Saved visualization to: {output_path}")

# Test batch rendering
print(f"\n" + "=" * 70)
print("Testing Batch Rendering")
print("=" * 70)

batch_sequences = np.array([seq for seq in test_sequences.values()])
print(f"\nBatch size: {len(batch_sequences)}")
print(f"Rendering batch...")

import time
start = time.time()
batch_images = renderer.render_batch_gpu(batch_sequences, resolution, line_width)
elapsed = time.time() - start

print(f"✓ Batch rendered in {elapsed:.3f}s")
print(f"✓ Throughput: {len(batch_sequences)/elapsed:.1f} images/second")
print(f"✓ Output shape: {batch_images.shape}")

# Verify batch matches individual renders
print(f"\nVerifying batch rendering matches individual rendering...")
for idx, (name, img_individual) in enumerate(images.items()):
    img_batch = batch_images[idx]
    
    # Check if images match (allow small floating point differences)
    max_diff = np.max(np.abs(img_individual - img_batch))
    
    if max_diff < 0.01:  # Less than 1% difference
        print(f"  ✓ {name}: MATCH (max diff: {max_diff:.6f})")
    else:
        print(f"  ✗ {name}: MISMATCH (max diff: {max_diff:.6f})")

print(f"\n" + "=" * 70)
print("Verification Complete!")
print("=" * 70)
print(f"\nCheck '{output_path}' to visually inspect line quality.")
print("Lines should be smooth and continuous.")

