#!/usr/bin/env python3
"""
Diagnostic script to check Chandra OCR model loading status and provide recommendations.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_model_files():
    """Check if model files are complete."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--datalab-to--chandra"
    
    print("=" * 70)
    print("Chandra OCR Model Status Check")
    print("=" * 70)
    print()
    
    if not cache_dir.exists():
        print("❌ Model cache directory not found")
        print("   The model has not been downloaded yet.")
        return False
    
    # Find snapshot directory
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        print("❌ No snapshots found in cache")
        return False
    
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        print("❌ No snapshots found")
        return False
    
    snapshot = snapshots[0]
    print(f"📁 Snapshot: {snapshot.name}")
    print()
    
    # Check for model shards
    model_files = sorted(list(snapshot.glob("model-*-of-*.safetensors")))
    index_file = snapshot / "model.safetensors.index.json"
    
    if not index_file.exists():
        print("❌ Model index file not found")
        return False
    
    # Read index to see how many shards expected
    import json
    try:
        with open(index_file) as f:
            index_data = json.load(f)
            # Find unique shard filenames from weight_map
            weight_map = index_data.get("weight_map", {})
            shard_files = set()
            for weight_name, shard_file in weight_map.items():
                shard_files.add(shard_file)
            total_shards = len(shard_files) if shard_files else 4
    except Exception as e:
        # Fallback: check existing files
        total_shards = 4  # Default for this model
    
    print(f"📊 Expected model shards: {total_shards}")
    print(f"📦 Found model shards: {len(model_files)}")
    print()
    
    # Check each shard
    missing = []
    incomplete = []
    complete = []
    
    for i in range(1, total_shards + 1):
        shard_file = snapshot / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        if shard_file.exists():
            size = shard_file.stat().st_size
            if size < 1000:  # Symlink to blob
                # Check actual blob
                if shard_file.is_symlink():
                    blob_path = shard_file.readlink()
                    if not blob_path.is_absolute():
                        blob_path = cache_dir / "blobs" / blob_path.name
                    if blob_path.exists():
                        blob_size = blob_path.stat().st_size
                        if blob_size > 1000000:  # > 1MB
                            complete.append(i)
                        else:
                            incomplete.append(i)
                    else:
                        incomplete.append(i)
                else:
                    incomplete.append(i)
            else:
                complete.append(i)
        else:
            missing.append(i)
    
    print("Model shard status:")
    if complete:
        print(f"  ✅ Complete: {len(complete)} shards {complete}")
    if incomplete:
        print(f"  ⏳ Incomplete: {len(incomplete)} shards {incomplete}")
    if missing:
        print(f"  ❌ Missing: {len(missing)} shards {missing}")
    print()
    
    # Check for incomplete downloads
    blobs_dir = cache_dir / "blobs"
    if blobs_dir.exists():
        incomplete_files = list(blobs_dir.glob("*.incomplete"))
        if incomplete_files:
            print(f"⚠️  Active downloads detected: {len(incomplete_files)}")
            for inc_file in incomplete_files:
                size = inc_file.stat().st_size / (1024**3)  # GB
                print(f"   • {inc_file.name[:20]}... ({size:.2f} GB)")
    
    # Check running processes
    print()
    print("Running processes:")
    result = subprocess.run(
        ["pgrep", "-f", "chandra.*hf"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        pids = result.stdout.strip().split()
        print(f"  ⚠️  Found {len(pids)} running chandra processes: {', '.join(pids)}")
        print("  💡 Tip: Having multiple processes can slow down download/loading")
    else:
        print("  ✓ No running chandra processes")
    
    # Check GPU
    print()
    print("GPU Status:")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✅ GPU detected and available")
        print("  💡 GPU will significantly speed up model loading and inference")
    else:
        print("  ⚠️  No GPU detected - using CPU only")
        print("  ⚠️  CPU inference will be MUCH slower (10-20x slower)")
        print("  💡 Consider using GPU for faster processing")
    
    print()
    print("=" * 70)
    
    if missing or incomplete:
        print("📥 Model is still downloading or incomplete")
        print("   Estimated time remaining: 5-15 minutes depending on connection")
        print("   Recommended: Wait for download to complete before processing")
        return False
    else:
        print("✅ Model files are complete!")
        return True


def check_memory():
    """Check available system memory."""
    print()
    print("System Memory:")
    try:
        result = subprocess.run(["free", "-h"], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            mem_line = lines[1].split()
            total = mem_line[1]
            used = mem_line[2]
            available = mem_line[6] if len(mem_line) > 6 else mem_line[3]
            print(f"  Total: {total}")
            print(f"  Used: {used}")
            print(f"  Available: {available}")
            print()
            print("  💡 Model requires ~18GB RAM for CPU inference")
            print("  💡 With GPU: requires ~10GB GPU VRAM")
    except (OSError, subprocess.SubprocessError) as error:
        print("  ⚠️  Could not check memory")
        print(f"      Details: {error}")


def recommendations():
    """Provide optimization recommendations."""
    print()
    print("=" * 70)
    print("Recommendations:")
    print("=" * 70)
    print()
    print("1. 🛑 If model is still downloading:")
    print("   • Wait for download to complete (check progress above)")
    print("   • Don't run multiple chandra processes simultaneously")
    print()
    print("2. ⚡ To speed up model loading:")
    print("   • Use GPU if available (10-20x faster)")
    print("   • Ensure sufficient RAM (18GB+ for CPU, 10GB+ VRAM for GPU)")
    print("   • Close other memory-intensive applications")
    print()
    print("3. 🚀 After model is loaded:")
    print("   • First inference is slower (JIT compilation)")
    print("   • Subsequent runs will be faster")
    print("   • Consider using batch processing for multiple PDFs")
    print()
    print("4. 💾 Cache location:")
    print(f"   {Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--datalab-to--chandra'}")
    print("   • Model files are ~18GB total")
    print("   • Once downloaded, cached locally for future use")
    print()


if __name__ == "__main__":
    complete = check_model_files()
    check_memory()
    recommendations()
    
    if not complete:
        print()
        print("⚠️  Model download/loading is in progress.")
        print("   Please wait for it to complete before processing PDFs.")
        sys.exit(1)
    else:
        print()
        print("✅ Model is ready! You can proceed with extraction.")
        sys.exit(0)

