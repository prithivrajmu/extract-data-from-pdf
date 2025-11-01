# Troubleshooting Guide

## Error: "No CUDA GPUs are available"

### Problem
Chandra OCR is trying to use CUDA/GPU but your GPU driver isn't loaded, causing the process to fail.

### Solution (Already Applied)
The script now automatically:
1. Detects if GPU is available
2. Sets `CUDA_VISIBLE_DEVICES=''` when GPU is not available (forces CPU mode)
3. Prevents CUDA initialization errors

### Manual Fix (if needed)
If you still see CUDA errors, explicitly set environment variable before running:

```bash
export CUDA_VISIBLE_DEVICES=''
python extract_ec_data_pretty.py
```

## Error: "Failed to extract text from Chandra OCR output"

### Possible Causes:
1. **CUDA error during processing** (see above)
2. **Model not fully downloaded** - check with `python diagnose_chandra.py`
3. **Insufficient memory** - need ~18GB RAM for CPU inference
4. **Output files not created** - check error messages above

### Solutions:

#### Check model download status:
```bash
python diagnose_chandra.py
```

#### Monitor download progress:
```bash
python monitor_download.py
```

#### Check available memory:
```bash
free -h
```

You need at least 18GB available for CPU inference.

## Slow Processing

### CPU Mode (Current)
- Model loading: 5-10 minutes
- Per page: 1-2 minutes  
- **Total for 17 pages: 20-40 minutes**

### GPU Mode (After reboot with GPU driver)
- Model loading: 30 seconds - 2 minutes
- Per page: 5-10 seconds
- **Total for 17 pages: 2-5 minutes**

### To Enable GPU:
1. Reboot system (loads NVIDIA driver)
2. Verify: `nvidia-smi`
3. Run script again - will auto-detect GPU

## Progress Indicators

The script now shows:
- ⏳ Real-time progress every 5 seconds
- CPU usage percentage
- RAM usage
- Time elapsed

If you see no progress updates for 10+ minutes, it may be stuck.

## Common Issues

### Issue: Process hangs at "Loading model weights"
**Solution**: This is normal! Takes 5-10 minutes on CPU. Progress indicators show it's working.

### Issue: "Model still downloading"
**Solution**: Wait for download to complete. Monitor with `python monitor_download.py`

### Issue: Multiple processes running
**Solution**: Kill duplicates: `pkill -f "chandra.*hf"` (keep only one)

### Issue: Out of memory
**Solution**: Close other applications, or use GPU which requires less RAM

## Getting Help

Run diagnostics:
```bash
python diagnose_chandra.py
python check_gpu.py
```

Check logs:
- Look for error messages in terminal output
- Check `*_ocr_text.txt` files if created



