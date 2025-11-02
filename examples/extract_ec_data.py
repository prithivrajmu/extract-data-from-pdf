#!/usr/bin/env python3
"""
Script to extract data from EC (Encumbrance Certificate) PDF files using OCR.
Extracts: filename, Sr.No, Document No.& Year, Name of Executant(s),
Name of Claimant(s), Survey No./, Plot No./
Only processes rows that have Plot No./ information.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from typing import Optional

import pandas as pd


def extract_text_from_pdf(pdf_path: str, use_structure: bool = True):
    """
    Extract text from PDF using Chandra OCR (datalab-to/chandra from Hugging Face).

    Uses the Chandra OCR model which is highly accurate at extracting text from images
    and PDFs while preserving layout information. Supports tables, forms, and complex layouts.

    Model: https://huggingface.co/datalab-to/chandra

    Args:
        pdf_path: Path to the PDF file
        use_structure: If True, also returns structured OCR data from JSON

    Returns:
        Extracted text from all pages, and optionally structured data
    """
    print(f"Processing PDF with Chandra OCR: {pdf_path}")
    print("Using datalab-to/chandra model from Hugging Face")

    try:
        # Create a temporary directory for Chandra output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "chandra_output")
            os.makedirs(output_dir, exist_ok=True)

            print("Running Chandra OCR...")
            print("=" * 60)
            print("⚠️  IMPORTANT: First run may take 10-20 minutes!")
            print("   - Downloading models (~2GB) - happens once")
            print("   - Processing PDF pages")
            print("   - Subsequent runs will be much faster")
            print("=" * 60)
            print("\nProcessing your PDF now...")

            # Enable Hugging Face progress bars for download visibility
            # Configure GPU usage - automatically use GPU if available
            env = os.environ.copy()
            env["TRANSFORMERS_VERBOSITY"] = "info"
            env["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # Enable progress bars

            # Check if GPU is available and configure accordingly
            try:
                import torch

                if torch.cuda.is_available():
                    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
                    print("✅ GPU detected - will use GPU acceleration")
                    print(f"   GPU: {torch.cuda.get_device_name(0)}")
                    print(
                        f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                    )
                else:
                    print("⚠️  GPU not available - using CPU (will be slower)")
            except ImportError:
                print("⚠️  PyTorch not available - GPU check skipped")
            except Exception as e:
                print(f"⚠️  Could not check GPU: {e}")

            # Function to print output from a pipe in real-time
            def print_output(pipe, prefix=""):
                """Print output from a pipe in real-time"""
                try:
                    for line in pipe:
                        if line:  # Only print non-empty lines
                            sys.stdout.write(prefix + line)
                            sys.stdout.flush()
                except Exception:
                    pass  # Pipe closed

            # Run Chandra OCR via CLI with real-time output
            # --method hf = HuggingFace method (uses datalab-to/chandra model)
            process = subprocess.Popen(
                ["chandra", pdf_path, output_dir, "--method", "hf"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr separately for download progress
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,  # Pass environment variables
            )

            # Print both stdout and stderr in real-time using threads
            print("\nChandra OCR progress:")
            print("(Download progress will appear below)\n")

            # Create threads to print both streams simultaneously
            stdout_thread = threading.Thread(
                target=print_output, args=(process.stdout,), daemon=True
            )
            stderr_thread = threading.Thread(
                target=print_output,
                args=(process.stderr, ""),  # Download progress usually on stderr
                daemon=True,
            )

            stdout_thread.start()
            stderr_thread.start()

            # Wait for both threads to complete
            stdout_thread.join()
            stderr_thread.join()

            process.wait()
            returncode = process.returncode

            print("\n" + "=" * 60)
            print(f"Chandra OCR finished (return code: {returncode})")

            if returncode != 0:
                print(f"Chandra OCR completed with return code: {returncode}")
                print("Checking for output files anyway...")
                # Continue anyway, output might still be generated

            # Look for JSON output file
            json_file = os.path.join(output_dir, "output.json")
            if not os.path.exists(json_file):
                # Try alternative output file names
                for file in os.listdir(output_dir):
                    if file.endswith(".json"):
                        json_file = os.path.join(output_dir, file)
                        break

            structured_data = None
            full_text = ""

            if os.path.exists(json_file):
                print(f"Loading structured data from {json_file}")
                with open(json_file, encoding="utf-8") as f:
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
                        with open(md_file, encoding="utf-8") as f:
                            full_text = f.read()
                        break
                    elif file.endswith(".html"):
                        html_file = os.path.join(output_dir, file)
                        # Simple HTML text extraction
                        with open(html_file, encoding="utf-8") as f:
                            html_content = f.read()
                        # Remove HTML tags (simple approach)
                        full_text = re.sub(r"<[^>]+>", "", html_content)
                        break

            if not full_text:
                raise Exception("Failed to extract text from Chandra OCR output")

            if use_structure:
                return full_text, structured_data
            return full_text

    except FileNotFoundError:
        raise Exception(
            "Chandra OCR not found. Please install it using: "
            "pip install chandra-ocr or uv pip install chandra-ocr"
        )
    except Exception as e:
        print(f"Error processing PDF with Chandra OCR: {e}")
        raise


def parse_table_rows(
    text: str, structured_data: Optional[list] = None  # noqa: UP006
) -> list[dict[str, str]]:
    """
    Parse OCR text to extract table rows with the required fields.
    Only includes rows that have Plot No./ information.

    Args:
        text: OCR extracted text
        structured_data: Optional structured OCR data with bounding boxes

    Returns:
        List of dictionaries containing extracted row data
    """
    lines = text.split("\n")
    rows = []

    # Pattern to identify serial numbers (usually numeric at the start)
    serial_pattern = re.compile(r"^\s*(\d+)\s+")

    # Patterns for extracting different fields
    plot_no_pattern = re.compile(
        r"Plot\s+No[./:\s]*([A-Z0-9/\s\-]+?)(?:\s|$|[,\n])", re.IGNORECASE
    )
    survey_pattern = re.compile(
        r"Survey\s+No[./:\s]*([A-Z0-9/\s\-]+?)(?:\s|$|[,\n])", re.IGNORECASE
    )
    doc_pattern = re.compile(r"([A-Z]{1,4}[0-9/]{2,}\s*\d{4})", re.IGNORECASE)

    # Split text into potential rows - look for lines starting with numbers
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if line starts with a serial number
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
            # Look ahead up to 10 lines or until next serial number
            row_text_parts = [line]

            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j]
                # Stop if we hit another serial number
                if serial_pattern.match(next_line) and j > i + 1:
                    break
                row_text_parts.append(next_line)

            # Combine all parts for this row
            row_text = " ".join(row_text_parts)

            # Extract Plot No. (must have this field)
            plot_match = plot_no_pattern.search(row_text)
            if plot_match:
                current_row["Plot No./"] = plot_match.group(1).strip()
            else:
                # Also try without explicit "Plot No" label (just look for common plot patterns)
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

                # Extract names - look for text segments that look like names
                # Names are typically 2-6 words, may contain commas, and don't match other patterns
                name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b")
                name_matches = name_pattern.findall(row_text)

                # Filter out matches that are part of patterns like "Plot No", "Survey No", etc.
                valid_names = []
                for name in name_matches:
                    name_lower = name.lower()
                    if not any(
                        keyword in name_lower
                        for keyword in ["plot", "survey", "document", "no", "year"]
                    ):
                        if len(name.split()) >= 2:  # At least 2 words
                            valid_names.append(name)

                if len(valid_names) >= 1:
                    current_row["Name of Executant(s)"] = valid_names[0]
                if len(valid_names) >= 2:
                    current_row["Name of Claimant(s)"] = valid_names[1]

                rows.append(current_row)

            # Move to next potential row
            i += 1
        else:
            i += 1

    return rows


def extract_data_from_pdf(pdf_path: str) -> list[dict[str, str]]:
    """
    Main function to extract data from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries with extracted row data
    """
    filename = os.path.basename(pdf_path)

    # Extract text using OCR
    result = extract_text_from_pdf(pdf_path, use_structure=False)
    if isinstance(result, tuple):
        text, structured_data = result
    else:
        text = result
        structured_data = None

    # Save OCR text for debugging (optional)
    debug_file = pdf_path.replace(".pdf", "_ocr_text.txt")
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved OCR text to: {debug_file}")

    # Parse table rows
    rows = parse_table_rows(text, structured_data)

    # Add filename to each row
    for row in rows:
        row["filename"] = filename

    # Filter rows that have Plot No./ information
    filtered_rows = [row for row in rows if row.get("Plot No./", "").strip()]

    print(f"Extracted {len(filtered_rows)} rows with Plot No./ information")

    return filtered_rows


def main():
    """Main function to run the extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract data from EC (Encumbrance Certificate) PDF files using OCR",
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
        # Try relative to current directory first
        if not os.path.exists(pdf_file):
            # Try relative to script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(
                script_dir
            )  # Go up from examples/ to project root
            test_path = os.path.join(parent_dir, pdf_file)
            if os.path.exists(test_path):
                pdf_file = test_path
            else:
                # Try as absolute from project root
                project_root = os.path.dirname(script_dir)
                pdf_file = os.path.join(project_root, pdf_file)

    if not os.path.exists(pdf_file):
        print(f"Error: File not found: {pdf_file}")
        print("Please check the path and try again.")
        return

    print("=" * 60)
    print("EC Data Extraction Script")
    print("=" * 60)

    try:
        # Extract data
        rows = extract_data_from_pdf(pdf_file)

        if not rows:
            print("\nNo rows with Plot No./ information found.")
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
        print("\n" + "=" * 60)
        print("Extracted Data:")
        print("=" * 60)
        print(df.to_string(index=False))

        # Save to CSV
        output_file = pdf_file.replace(".pdf", "_extracted.csv")
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nData saved to: {output_file}")

        # Save to Excel
        output_excel = pdf_file.replace(".pdf", "_extracted.xlsx")
        df.to_excel(output_excel, index=False, engine="openpyxl")
        print(f"Data saved to: {output_excel}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
