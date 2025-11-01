#!/bin/bash
# Setup script for NVIDIA GPU support

echo "=========================================="
echo "NVIDIA GPU Setup for Chandra OCR"
echo "=========================================="
echo ""
echo "Detected GPU: NVIDIA Quadro P620"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "❌ Don't run this script as root/sudo"
    echo "   Run commands individually as needed"
    exit 1
fi

echo "Step 1: Check current driver status"
echo "-----------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi || echo "⚠️  nvidia-smi exists but driver not working"
else
    echo "❌ nvidia-smi not found"
fi

echo ""
echo "Step 2: Check available drivers"
echo "-----------------------------------"
if command -v ubuntu-drivers &> /dev/null; then
    echo "Checking available NVIDIA drivers..."
    ubuntu-drivers devices
    echo ""
    echo "Recommended: Install recommended driver with:"
    echo "  sudo ubuntu-drivers autoinstall"
else
    echo "Manual installation:"
    echo "  sudo apt update"
    echo "  sudo apt install nvidia-driver-535  # or latest version"
fi

echo ""
echo "Step 3: After driver installation"
echo "-----------------------------------"
echo "1. Reboot your system:"
echo "   sudo reboot"
echo ""
echo "2. After reboot, verify GPU is working:"
echo "   nvidia-smi"
echo ""
echo "3. Test PyTorch GPU access:"
echo "   python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
echo ""
echo "4. Once GPU is working, run extraction script:"
echo "   python examples/extract_ec_data.py"
echo ""
echo "=========================================="
echo "Note: GPU will make processing 10-20x faster!"
echo "=========================================="



