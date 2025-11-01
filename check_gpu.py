#!/usr/bin/env python3
"""
Check GPU availability and configuration for Chandra OCR.
"""

import subprocess
import sys


def check_gpu():
    """Check GPU availability and provide setup instructions."""
    print("=" * 70)
    print("GPU Detection and Configuration Check")
    print("=" * 70)
    print()
    
    # Check nvidia-smi
    print("1. Checking NVIDIA Driver:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ NVIDIA driver is installed and working")
            print()
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU' in line and ('Name' in line or 'Tesla' in line or 'GeForce' in line or 'Quadro' in line):
                    print(f"   {line.strip()}")
        else:
            print("   ❌ NVIDIA driver not working properly")
    except FileNotFoundError:
        print("   ❌ nvidia-smi not found")
    except subprocess.TimeoutExpired:
        print("   ⚠️  nvidia-smi timed out")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Check PyTorch CUDA
    print("2. Checking PyTorch CUDA Support:")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print()
            print("   ✅ GPU is ready to use!")
            return True
        else:
            print("   ❌ CUDA not available in PyTorch")
            print("   Possible reasons:")
            print("     - NVIDIA driver not installed/configured")
            print("     - PyTorch was installed without CUDA support")
            print("     - GPU not properly connected")
    except ImportError:
        print("   ❌ PyTorch not installed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Check hardware
    print("3. Checking Hardware:")
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        gpu_lines = [line for line in result.stdout.split('\n') if 'VGA' in line or '3D' in line or 'Display' in line]
        if gpu_lines:
            print("   Graphics devices found:")
            for line in gpu_lines:
                print(f"     • {line.strip()}")
                
            nvidia_gpus = [line for line in gpu_lines if 'NVIDIA' in line or 'nvidia' in line.lower()]
            if nvidia_gpus:
                print()
                print("   ✅ NVIDIA GPU hardware detected!")
            else:
                print()
                print("   ⚠️  No NVIDIA GPU found - only integrated graphics detected")
                print("   Note: Integrated graphics cannot run CUDA models")
        else:
            print("   ⚠️  Could not detect graphics devices")
    except Exception as e:
        print(f"   ⚠️  Could not check hardware: {e}")
    
    print()
    print("=" * 70)
    print()
    
    return False


def provide_recommendations():
    """Provide GPU setup recommendations."""
    print("Recommendations:")
    print()
    print("If you have an NVIDIA GPU but it's not detected:")
    print()
    print("1. Install NVIDIA Driver:")
    print("   Ubuntu/Debian:")
    print("     sudo apt update")
    print("     sudo apt install nvidia-driver-535  # or latest version")
    print("     sudo reboot")
    print()
    print("2. Verify Driver Installation:")
    print("     nvidia-smi")
    print()
    print("3. If using a cloud/remote GPU:")
    print("   - Ensure GPU is properly allocated")
    print("   - Check if CUDA_VISIBLE_DEVICES is set correctly")
    print("   - Verify GPU access permissions")
    print()
    print("4. If PyTorch doesn't detect GPU after driver install:")
    print("   - Reinstall PyTorch with CUDA support:")
    print("     pip uninstall torch")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print()


if __name__ == "__main__":
    has_gpu = check_gpu()
    if not has_gpu:
        provide_recommendations()
    
    sys.exit(0 if has_gpu else 1)



