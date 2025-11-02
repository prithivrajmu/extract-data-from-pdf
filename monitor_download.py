#!/usr/bin/env python3
"""
Monitor Chandra OCR model download progress in real-time.
"""

import time
from pathlib import Path


def monitor_download():
    """Monitor the model download progress."""
    cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--datalab-to--chandra"
        / "blobs"
    )

    print("=" * 70)
    print("Chandra OCR Model Download Monitor")
    print("=" * 70)
    print()
    print("Monitoring download progress...")
    print("Press Ctrl+C to stop")
    print()

    if not cache_dir.exists():
        print("❌ Cache directory not found")
        return

    last_size = {}

    try:
        while True:
            incomplete_files = list(cache_dir.glob("*.incomplete"))

            if not incomplete_files:
                print("\n✅ No active downloads - checking if model is complete...")
                time.sleep(2)
                continue

            for inc_file in incomplete_files:
                current_size = inc_file.stat().st_size
                size_gb = current_size / (1024**3)

                # Get previous size
                if inc_file.name in last_size:
                    prev_size = last_size[inc_file.name]
                    diff = current_size - prev_size
                    diff_gb = diff / (1024**3)
                    speed_mbps = (diff_gb * 1024) / 2  # 2 second interval

                    if diff > 0:
                        print(
                            f"\r⬇️  Downloading: {size_gb:.2f} GB | Speed: {speed_mbps:.1f} MB/s",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r⬇️  Downloading: {size_gb:.2f} GB | Waiting...",
                            end="",
                            flush=True,
                        )
                else:
                    print(f"\r⬇️  Downloading: {size_gb:.2f} GB", end="", flush=True)

                last_size[inc_file.name] = current_size

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")
        print()
        # Final status
        incomplete_files = list(cache_dir.glob("*.incomplete"))
        if incomplete_files:
            print("⚠️  Download still in progress")
            for inc_file in incomplete_files:
                size_gb = inc_file.stat().st_size / (1024**3)
                print(f"   {inc_file.name[:30]}... ({size_gb:.2f} GB)")
        else:
            print("✅ No active downloads")


if __name__ == "__main__":
    monitor_download()
