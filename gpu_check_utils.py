#!/usr/bin/env python3
"""
GPU check utilities for Streamlit UI.
"""

import subprocess
from typing import Dict, Optional, Tuple


def check_nvidia_driver() -> Tuple[bool, Dict]:
    """
    Check NVIDIA driver availability.
    
    Returns:
        Tuple of (is_available: bool, info: dict)
    """
    info = {
        'available': False,
        'driver_version': None,
        'gpu_name': None,
        'gpu_memory': None,
        'gpu_count': 0,
        'cuda_version': None,
        'error': None
    }
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info['available'] = True
            lines = result.stdout.split('\n')
            
            # Extract driver version
            for line in lines:
                if 'Driver Version:' in line:
                    parts = line.split('Driver Version:')
                    if len(parts) > 1:
                        info['driver_version'] = parts[1].strip().split()[0]
            
            # Extract CUDA version
            for line in lines:
                if 'CUDA Version:' in line:
                    parts = line.split('CUDA Version:')
                    if len(parts) > 1:
                        info['cuda_version'] = parts[1].strip().split()[0]
            
            # Extract GPU info
            gpu_found = False
            for line in lines:
                if ('GPU' in line or 'Tesla' in line or 'GeForce' in line or 'Quadro' in line or 'RTX' in line or 'GTX' in line) and not gpu_found:
                    # Look for GPU name
                    if any(keyword in line for keyword in ['Tesla', 'GeForce', 'Quadro', 'RTX', 'GTX', 'A100', 'V100']):
                        info['gpu_name'] = line.strip()
                        gpu_found = True
                    # Look for memory
                    if 'MiB' in line:
                        memory_parts = line.split('MiB')
                        if memory_parts:
                            try:
                                memory_str = memory_parts[0].split()[-1]
                                info['gpu_memory'] = f"{int(memory_str) / 1024:.1f} GB"
                            except:
                                pass
            
            # Count GPUs
            info['gpu_count'] = result.stdout.count('|  ')  # Simple count
            if info['gpu_count'] == 0:
                info['gpu_count'] = 1 if info['gpu_name'] else 0
                
    except FileNotFoundError:
        info['error'] = 'nvidia-smi not found'
    except subprocess.TimeoutExpired:
        info['error'] = 'nvidia-smi timed out'
    except Exception as e:
        info['error'] = str(e)
    
    return info['available'], info


def check_pytorch_cuda() -> Tuple[bool, Dict]:
    """
    Check PyTorch CUDA support.
    
    Returns:
        Tuple of (is_available: bool, info: dict)
    """
    info = {
        'available': False,
        'pytorch_version': None,
        'cuda_available': False,
        'cuda_version': None,
        'cudnn_version': None,
        'gpu_count': 0,
        'gpus': [],
        'error': None
    }
    
    try:
        import torch
        info['pytorch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            info['available'] = True
            try:
                info['cuda_version'] = torch.version.cuda
            except:
                pass
            
            try:
                info['cudnn_version'] = str(torch.backends.cudnn.version())
            except:
                pass
            
            info['gpu_count'] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_gb': f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f}"
                }
                info['gpus'].append(gpu_info)
    except ImportError:
        info['error'] = 'PyTorch not installed'
    except Exception as e:
        info['error'] = str(e)
    
    return info['cuda_available'], info


def check_gpu_comprehensive() -> Dict:
    """
    Perform comprehensive GPU check.
    
    Returns:
        Dictionary with all GPU information
    """
    result = {
        'nvidia_driver': {},
        'pytorch_cuda': {},
        'overall_status': 'unknown',
        'gpu_ready': False,
        'recommendations': []
    }
    
    # Check NVIDIA driver
    nvidia_available, nvidia_info = check_nvidia_driver()
    result['nvidia_driver'] = {
        'available': nvidia_available,
        **nvidia_info
    }
    
    # Check PyTorch CUDA
    pytorch_available, pytorch_info = check_pytorch_cuda()
    result['pytorch_cuda'] = {
        'available': pytorch_available,
        **pytorch_info
    }
    
    # Determine overall status
    if nvidia_available and pytorch_available:
        result['overall_status'] = 'ready'
        result['gpu_ready'] = True
    elif nvidia_available and not pytorch_available:
        result['overall_status'] = 'driver_only'
        result['recommendations'].append('PyTorch CUDA support not available. Consider reinstalling PyTorch with CUDA.')
    elif not nvidia_available:
        result['overall_status'] = 'not_available'
        result['recommendations'].append('NVIDIA driver not detected. GPU acceleration will not be available.')
    
    return result

