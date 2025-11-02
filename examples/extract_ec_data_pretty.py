#!/usr/bin/env python3
"""
User-friendly EC Data Extraction Script with formatted Chandra OCR output.
Filters out technical details and shows only relevant, human-readable progress.
"""

import os
import re
import json
import subprocess
import tempfile
import threading
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional

# Note: This script has its own format_chandra_line() function for formatting output


def format_chandra_line(line: str) -> Optional[str]:
    """
    Format a single line from Chandra OCR output to be human-readable.
    Returns None if line should be filtered out, otherwise returns formatted line.
    """
    line = line.rstrip()

    # Skip empty lines
    if not line.strip():
        return None

    # Important status messages
    if "Chandra CLI - Starting OCR processing" in line:
        return "🚀 Starting OCR processing..."

    if "Loading model with method" in line:
        return "📥 Loading AI model (this may take a few minutes)..."

    # Skip model configuration blocks (too technical)
    if "Model config" in line or "Qwen3VLConfig" in line:
        return None
    if line.strip().startswith(
        (
            "_attn_",
            "architectures",
            "base_model",
            "dtype",
            "text_config",
            "vision_config",
            "hidden_size",
            "num_attention",
            "intermediate_size",
            "vocab_size",
            "max_position",
            "rope_",
            "tie_word",
            "use_cache",
            "transformers_version",
            "video_token",
            "image_token",
            "deepstack",
            "depth",
            "patch_size",
            "spatial_merge",
            "temporal_patch",
            "vision_end",
            "vision_start",
        )
    ):
        return None
    if line.strip() in ["}", "{"] or line.strip().startswith("  "):
        return None

    # Show file loading with progress
    if "loading" in line.lower() and "from cache" in line.lower():
        if "configuration file" in line.lower():
            return "  → Loading configuration files..."
        elif "weights file" in line.lower() or "model.safetensors" in line.lower():
            # Extract progress if available
            match = re.search(r"(\d+)/(\d+)", line)
            if match:
                current, total = match.groups()
                percent = int((int(current) / int(total)) * 100)
                return f"  → Loading model weights ({percent}%)"
            return "  → Loading model weights..."

    # Show page processing
    page_match = re.search(r"page\s+(\d+)", line, re.IGNORECASE)
    if page_match:
        page_num = page_match.group(1)
        return f"📄 Processing page {page_num}..."

    # Show percentage progress
    if "%" in line:
        percent_match = re.search(r"(\d+)%", line)
        if percent_match:
            percent = percent_match.group(1)
            return f"⏳ Progress: {percent}%"

    # Show download progress
    if "downloading" in line.lower():
        size_match = re.search(r"([\d.]+)\s*(MB|GB|KB)", line, re.IGNORECASE)
        if size_match:
            size, unit = size_match.groups()
            return f"⬇️  Downloading: {size} {unit.upper()}"
        return "⬇️  Downloading model files..."

    # Show batch info
    if "batch" in line.lower() and "size" in line.lower():
        return "ℹ️  Using default batch size of 1"

    # Show completion/status
    if "completed" in line.lower() or "finished" in line.lower():
        return f"✅ {line.strip()}"

    # Show errors/warnings
    if "error" in line.lower():
        return f"❌ Error: {line.strip()}"
    if "warning" in line.lower():
        return f"⚠️  Warning: {line.strip()}"

    # Skip very long technical lines
    if len(line) > 150:
        return None

    # Skip JSON/config lines
    if re.match(r'^[\s\[\]{}",:]+$', line):
        return None

    # Show short meaningful messages
    if len(line) < 100:
        if any(
            keyword in line.lower()
            for keyword in [
                "processing",
                "extracting",
                "saving",
                "output",
                "input",
                "method",
            ]
        ):
            # Clean up the message
            clean = line.strip()
            if not any(
                tech in clean for tech in ["Qwen3VL", "config", "safetensors.index"]
            ):
                return f"ℹ️  {clean}"

    return None


def extract_text_from_pdf(pdf_path: str, use_structure: bool = True):
    """
    Extract text from PDF using Chandra OCR with user-friendly output.

    Args:
        pdf_path: Path to the PDF file
        use_structure: If True, also returns structured OCR data from JSON

    Returns:
        Extracted text from all pages, and optionally structured data
    """
    print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
    print("🔧 Using Chandra OCR (datalab-to/chandra from Hugging Face)")
    print()

    try:
        # Create a temporary directory for Chandra output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "chandra_output")
            os.makedirs(output_dir, exist_ok=True)

            print("=" * 70)
            print("⚠️  First run may take 10-20 minutes!")
            print("   • Downloading AI model (~2GB) - happens once")
            print("   • Processing PDF pages")
            print("   • Subsequent runs will be much faster")
            print("=" * 70)
            print()

            # Enable Hugging Face progress bars
            # Configure GPU usage - automatically use GPU if available
            env = os.environ.copy()
            env["TRANSFORMERS_VERBOSITY"] = "info"
            env["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

            # Check if GPU is available and configure accordingly
            try:
                import torch

                if torch.cuda.is_available():
                    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
                    print("✅ GPU detected - using GPU acceleration")
                    print(f"   GPU: {torch.cuda.get_device_name(0)}")
                else:
                    # Force CPU mode - prevent CUDA initialization errors
                    # Set multiple environment variables to ensure CPU usage
                    env["CUDA_VISIBLE_DEVICES"] = ""  # Empty = no GPU
                    env["TORCH_DEVICE"] = "cpu"
                    env["HF_DEVICE_MAP"] = "cpu"  # HuggingFace device map
                    # Prevent PyTorch from trying CUDA
                    env["PYTORCH_CUDA_ALLOC_CONF"] = ""
                    print("⚠️  GPU not available - forcing CPU mode")
                    print("   (This is normal and will work, just slower)")
            except Exception as e:
                # Force CPU mode if torch check fails
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["TORCH_DEVICE"] = "cpu"
                env["HF_DEVICE_MAP"] = "cpu"
                env["PYTORCH_CUDA_ALLOC_CONF"] = ""
                print(f"⚠️  GPU check failed - forcing CPU mode: {e}")

            # Always set these to ensure CPU mode is enforced
            # Remove any existing CUDA settings
            if (
                "CUDA_VISIBLE_DEVICES" not in env
                or env.get("CUDA_VISIBLE_DEVICES") == ""
            ):
                # Double-check and enforce CPU mode
                import torch

                if not torch.cuda.is_available():
                    # Create a monkey-patch wrapper to force CPU
                    # This will be used when Chandra initializes
                    print("🔧 Configuring for CPU-only execution...")

            # Global flag to track if we've seen output recently
            last_output_time = [time.time()]
            output_buffer = []
            output_lock = threading.Lock()

            def print_formatted_output(pipe, pipe_name=""):
                """Print formatted output from a pipe in real-time with progress indicators"""
                try:
                    line_count = 0
                    for line in pipe:
                        line_count += 1
                        formatted = format_chandra_line(line)
                        if formatted:
                            with output_lock:
                                output_buffer.append(formatted)
                                last_output_time[0] = time.time()
                                sys.stdout.write(formatted + "\n")
                                sys.stdout.flush()
                        # Also show raw important lines even if not formatted
                        elif (
                            "loading weights" in line.lower() or "model" in line.lower()
                        ):
                            with output_lock:
                                last_output_time[0] = time.time()
                except Exception as e:
                    pass  # Pipe closed

            # Progress monitor function - shows activity even when no output
            def monitor_progress(process):
                """Monitor process and show progress indicators"""
                import psutil

                start_time = time.time()
                last_status = time.time()
                dots = 0

                try:
                    proc = psutil.Process(process.pid)
                except (psutil.Error, ProcessLookupError):
                    proc = None

                while process.poll() is None:
                    current_time = time.time()
                    elapsed = current_time - start_time

                    # Show progress every 5 seconds
                    if current_time - last_status >= 5:
                        elapsed_min = int(elapsed // 60)
                        elapsed_sec = int(elapsed % 60)

                        # Check if process is still alive and using resources
                        if proc:
                            try:
                                # Initialize CPU percent (first call returns 0)
                                cpu_percent = proc.cpu_percent(interval=None)
                                mem_info = proc.memory_info()
                                mem_mb = mem_info.rss / 1024 / 1024

                                # Show status if no recent output
                                if current_time - last_output_time[0] > 10:
                                    dots = (dots + 1) % 4
                                    status_dots = "." * dots
                                    sys.stdout.write(
                                        f"\r⏳ Loading model weights{status_dots} ({elapsed_min}m {elapsed_sec}s, CPU: {cpu_percent:.1f}%, RAM: {mem_mb:.0f}MB)     "
                                    )
                                    sys.stdout.flush()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                            except Exception:
                                pass

                        last_status = current_time

                    time.sleep(1)

                    # Warn if taking too long
                    if (
                        elapsed > 600 and current_time - last_output_time[0] > 30
                    ):  # 10 min with no output
                        print(
                            f"\n⚠️  Model loading is taking longer than expected ({elapsed_min} minutes)"
                        )
                        print("   This is normal for CPU inference with large models")
                        print("   Consider using GPU for 10-20x faster loading")
                        last_output_time[0] = current_time  # Reset warning timer

                sys.stdout.write("\r" + " " * 80 + "\r")  # Clear progress line
                sys.stdout.flush()

            # Run Chandra OCR via CLI
            process = subprocess.Popen(
                ["chandra", pdf_path, output_dir, "--method", "hf"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            print("🚀 Starting OCR process...\n")
            print("💡 Loading model weights can take 5-10 minutes on CPU")
            print("   Progress indicators will show below...\n")

            # Create threads to print both streams with formatting
            stdout_thread = threading.Thread(
                target=print_formatted_output,
                args=(process.stdout, "stdout"),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=print_formatted_output,
                args=(process.stderr, "stderr"),
                daemon=True,
            )

            # Progress monitor thread
            try:
                import psutil

                progress_thread = threading.Thread(
                    target=monitor_progress, args=(process,), daemon=True
                )
                progress_thread.start()
            except ImportError:
                print("⚠️  psutil not available - progress monitoring disabled")
                print("   Install with: pip install psutil")

            stdout_thread.start()
            stderr_thread.start()

            # Wait for both threads to complete
            stdout_thread.join()
            stderr_thread.join()

            process.wait()
            returncode = process.returncode

            print()
            print("=" * 70)
            if returncode == 0:
                print("✅ OCR processing completed successfully!")
            else:
                print(f"⚠️  OCR processing finished with return code: {returncode}")
                print("Checking for output files anyway...")
            print("=" * 70)
            print()

            # Look for JSON output file
            json_file = os.path.join(output_dir, "output.json")
            if not os.path.exists(json_file):
                for file in os.listdir(output_dir):
                    if file.endswith(".json"):
                        json_file = os.path.join(output_dir, file)
                        break

            structured_data = None
            full_text = ""

            if os.path.exists(json_file):
                print(
                    f"📂 Loading extracted data from {os.path.basename(json_file)}..."
                )
                with open(json_file, "r", encoding="utf-8") as f:
                    structured_data = json.load(f)

                # Extract text from structured data
                text_parts = []
                if "pages" in structured_data:
                    for page in structured_data["pages"]:
                        page_text = []
                        if "blocks" in page:
                            for block in page["blocks"]:
                                if "text" in block:
                                    page_text.append(block["text"])
                        text_parts.append("\n".join(page_text))

                full_text = "\n\n".join(text_parts)
            else:
                # Fallback: try to extract text from markdown or HTML output
                for file in os.listdir(output_dir):
                    if file.endswith(".md"):
                        md_file = os.path.join(output_dir, file)
                        with open(md_file, "r", encoding="utf-8") as f:
                            full_text = f.read()
                        break
                    elif file.endswith(".html"):
                        html_file = os.path.join(output_dir, file)
                        with open(html_file, "r", encoding="utf-8") as f:
                            html_content = f.read()
                        full_text = re.sub(r"<[^>]+>", "", html_content)
                        break

            if not full_text:
                # Check if there are any output files at all
                output_files = list(os.listdir(output_dir))
                if output_files:
                    error_msg = f"Failed to extract text from Chandra OCR output.\n"
                    error_msg += f"Output directory contains: {output_files}\n"
                    error_msg += f"Return code: {returncode}\n"
                    if returncode != 0:
                        error_msg += (
                            "Chandra OCR reported an error (see messages above)."
                        )
                    raise Exception(error_msg)
                else:
                    raise Exception(
                        "No output files generated by Chandra OCR. Check error messages above."
                    )

            print("✓ Text extraction completed\n")

            if use_structure:
                return full_text, structured_data
            return full_text

    except FileNotFoundError:
        raise Exception(
            "Chandra OCR not found. Please install it using: "
            "pip install chandra-ocr or uv pip install chandra-ocr"
        )
    except Exception as e:
        print(f"\n❌ Error processing PDF with Chandra OCR: {e}")
        raise


def parse_table_rows(
    text: str, structured_data: Optional[List] = None
) -> List[Dict[str, str]]:
    """
    Parse OCR text to extract table rows with the required fields.
    Only includes rows that have Plot No./ information.
    """
    lines = text.split("\n")
    rows = []

    # Pattern to identify serial numbers
    serial_pattern = re.compile(r"^\s*(\d+)\s+")

    # Patterns for extracting different fields
    plot_no_pattern = re.compile(
        r"Plot\s+No[./:\s]*([A-Z0-9/\s\-]+?)(?:\s|$|[,\n])", re.IGNORECASE
    )
    survey_pattern = re.compile(
        r"Survey\s+No[./:\s]*([A-Z0-9/\s\-]+?)(?:\s|$|[,\n])", re.IGNORECASE
    )
    doc_pattern = re.compile(r"([A-Z]{1,4}[0-9/]{2,}\s*\d{4})", re.IGNORECASE)

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]
        serial_match = serial_pattern.match(line)

        if serial_match:
            serial_no = serial_match.group(1)
            current_row = {
                "Sr.No": serial_no,
                "Document No.& Year": "",
                "Name of Executant(s)": "",
                "Name of Claimant(s)": "",
                "Survey No./": "",
                "Plot No./": "",
            }

            # Combine current line with following lines to form a complete row
            row_text_parts = [line]
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j]
                if serial_pattern.match(next_line) and j > i + 1:
                    break
                row_text_parts.append(next_line)

            row_text = " ".join(row_text_parts)

            # Extract Plot No. (must have this field)
            plot_match = plot_no_pattern.search(row_text)
            if plot_match:
                current_row["Plot No./"] = plot_match.group(1).strip()
            else:
                plot_alt_pattern = re.compile(r"\b([0-9]+/[0-9]+|[A-Z]\d+[A-Z]?)\b")
                alt_match = plot_alt_pattern.search(line)
                if alt_match:
                    current_row["Plot No./"] = alt_match.group(1).strip()

            # Only process rows that have Plot No.
            if current_row["Plot No./"]:
                # Extract Survey No.
                survey_match = survey_pattern.search(row_text)
                if survey_match:
                    current_row["Survey No./"] = survey_match.group(1).strip()

                # Extract Document No. & Year
                doc_match = doc_pattern.search(row_text)
                if doc_match:
                    current_row["Document No.& Year"] = doc_match.group(1).strip()

                # Extract names
                name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b")
                name_matches = name_pattern.findall(row_text)

                valid_names = []
                for name in name_matches:
                    name_lower = name.lower()
                    if not any(
                        keyword in name_lower
                        for keyword in ["plot", "survey", "document", "no", "year"]
                    ):
                        if len(name.split()) >= 2:
                            valid_names.append(name)

                if len(valid_names) >= 1:
                    current_row["Name of Executant(s)"] = valid_names[0]
                if len(valid_names) >= 2:
                    current_row["Name of Claimant(s)"] = valid_names[1]

                rows.append(current_row)

            i += 1
        else:
            i += 1

    return rows


def extract_data_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """Main function to extract data from a PDF file."""
    filename = os.path.basename(pdf_path)

    # Extract text using OCR
    result = extract_text_from_pdf(pdf_path, use_structure=False)
    if isinstance(result, tuple):
        text, structured_data = result
    else:
        text = result
        structured_data = None

    # Save OCR text for debugging
    debug_file = pdf_path.replace(".pdf", "_ocr_text.txt")
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"💾 Saved OCR text to: {debug_file}\n")

    # Parse table rows
    print("🔍 Parsing table data...")
    rows = parse_table_rows(text, structured_data)

    # Add filename to each row
    for row in rows:
        row["filename"] = filename

    # Filter rows that have Plot No./ information
    filtered_rows = [row for row in rows if row.get("Plot No./", "").strip()]

    print(f"✓ Found {len(filtered_rows)} rows with Plot No./ information\n")

    return filtered_rows


def main():
    """Main function to run the extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract data from EC PDF files using Chandra OCR (Pretty Output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default test file
  %(prog)s --file path/to/file.pdf           # Use custom PDF file
  %(prog)s --input path/to/file.pdf          # Alternative syntax
        """,
    )
    parser.add_argument(
        "--file",
        "--input",
        dest="pdf_file",
        default="test_file/RG EC 103 3.pdf",
        help="Path to the PDF file to process (default: test_file/RG EC 103 3.pdf)",
    )

    args = parser.parse_args()
    pdf_file = args.pdf_file

    # Convert to absolute path if relative
    if not os.path.isabs(pdf_file):
        if not os.path.exists(pdf_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            test_path = os.path.join(parent_dir, pdf_file)
            if os.path.exists(test_path):
                pdf_file = test_path
            else:
                project_root = os.path.dirname(script_dir)
                pdf_file = os.path.join(project_root, pdf_file)

    if not os.path.exists(pdf_file):
        print(f"❌ Error: File {pdf_file} not found!")
        print(f"Please check the path and try again.")
        return

    print("=" * 70)
    print(" " * 20 + "EC Data Extraction Script")
    print("=" * 70)
    print()

    try:
        # Extract data
        rows = extract_data_from_pdf(pdf_file)

        if not rows:
            print("⚠️  No rows with Plot No./ information found.")
            return

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Reorder columns
        columns_order = [
            "filename",
            "Sr.No",
            "Document No.& Year",
            "Name of Executant(s)",
            "Name of Claimant(s)",
            "Survey No./",
            "Plot No./",
        ]
        df = df[columns_order]

        # Display results
        print("=" * 70)
        print(" " * 25 + "Extracted Data")
        print("=" * 70)
        print()
        print(df.to_string(index=False))
        print()

        # Save to CSV
        output_file = pdf_file.replace(".pdf", "_extracted.csv")
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"💾 Data saved to: {output_file}")

        # Save to Excel
        output_excel = pdf_file.replace(".pdf", "_extracted.xlsx")
        df.to_excel(output_excel, index=False, engine="openpyxl")
        print(f"💾 Data saved to: {output_excel}")
        print()
        print("=" * 70)
        print("✅ Extraction complete!")
        print("=" * 70)

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ Error occurred during extraction")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback

        print()
        print("Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
