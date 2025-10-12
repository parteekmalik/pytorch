#!/bin/bash
# CUDA Diagnostics Script

echo "============================================================"
echo "CUDA Diagnostics"
echo "============================================================"

echo -e "\n1. Checking NVIDIA GPU with nvidia-smi:"
echo "-----------------------------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "✗ nvidia-smi not found"
fi

echo -e "\n2. CUDA Environment Variables:"
echo "-----------------------------------------------------------"
env | grep -i cuda || echo "No CUDA environment variables set"

echo -e "\n3. CUDA Libraries:"
echo "-----------------------------------------------------------"
ldconfig -p | grep -i cuda || echo "No CUDA libraries found in ldconfig"

echo -e "\n4. Python CuPy Check:"
echo "-----------------------------------------------------------"
python3 -c "
try:
    import cupy as cp
    print('✓ CuPy installed:', cp.__version__)
    try:
        print('✓ CUDA available:', cp.cuda.is_available())
        print('✓ Device count:', cp.cuda.runtime.getDeviceCount())
        print('✓ Device 0:', cp.cuda.Device(0).name)
    except Exception as e:
        print('✗ CUDA error:', e)
except ImportError:
    print('✗ CuPy not installed')
"

echo -e "\n5. PyTorch CUDA Check:"
echo "-----------------------------------------------------------"
python3 -c "
try:
    import torch
    print('✓ PyTorch installed:', torch.__version__)
    print('✓ CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('✓ CUDA version:', torch.version.cuda)
        print('✓ Device count:', torch.cuda.device_count())
        print('✓ Device name:', torch.cuda.get_device_name(0))
except ImportError:
    print('✗ PyTorch not installed')
"

echo -e "\n============================================================"

