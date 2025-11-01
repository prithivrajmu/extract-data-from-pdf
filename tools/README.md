# Utility Tools

This directory contains diagnostic and utility scripts for the PDF extraction project.

## Available Tools

### `diagnose_chandra.py`
Diagnostic script to check Chandra OCR model loading status and provide recommendations.

**Usage:**
```bash
python tools/diagnose_chandra.py
```

**What it checks:**
- Model file download status
- Model shard completeness
- Active downloads
- Running processes
- GPU availability
- System memory

**Output:**
- Shows which model shards are complete, incomplete, or missing
- Provides recommendations for optimization
- Exits with status code indicating if model is ready

### `setup_gpu.sh`
Setup script for NVIDIA GPU support with Chandra OCR.

**Usage:**
```bash
bash tools/setup_gpu.sh
```

**What it does:**
- Checks current NVIDIA driver status
- Shows available drivers
- Provides installation instructions
- Guides through GPU setup process

**Note:** This script provides guidance and checks - it doesn't install drivers automatically for safety reasons.

## When to Use

- **Before first extraction**: Run `diagnose_chandra.py` to check if models are downloaded
- **GPU setup issues**: Use `setup_gpu.sh` to troubleshoot GPU configuration
- **Performance problems**: Use `diagnose_chandra.py` to identify bottlenecks
- **Troubleshooting**: Both tools provide diagnostic information

## Integration

These tools are standalone utilities and are not imported by the main application code. They can be run independently to diagnose issues or check system status.

