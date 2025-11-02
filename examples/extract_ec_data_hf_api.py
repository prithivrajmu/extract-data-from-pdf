#!/usr/bin/env python3
"""
Script to extract data from EC PDF files using Chandra OCR via HuggingFace Inference Providers API.
Uses the new router endpoint (not the deprecated api-inference.huggingface.co).

Note: The Chandra model may not be available on HuggingFace Inference API.
If you get errors, consider using extract_ec_data_api.py with Datalab API instead.

Documentation: https://huggingface.co/docs/inference-providers
"""

import argparse
import base64
import os
import re
import time
from io import BytesIO
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

# Load environment variables from .env file
load_dotenv()


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def chandra_ocr_hf_api(image: Image.Image, api_key: str, retries: int = 3) -> str:
    """
    Send image to Chandra OCR via HuggingFace Inference Providers API.

    Args:
        image: PIL Image object
        api_key: HuggingFace API token (required)
        retries: Number of retry attempts if API is loading

    Returns:
        Extracted text (markdown format)

    Note: Uses the new router endpoint instead of deprecated api-inference.huggingface.co
    """
    # New HuggingFace Inference Providers API endpoint
    api_url = "https://router.huggingface.co/hf-inference/models/datalab-to/chandra"

    # Convert image to base64
    img_base64 = image_to_base64(image)

    # Prepare headers
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Prepare payload - try different formats
    payload_formats = [
        # Format 1: Direct base64 string
        img_base64,
        # Format 2: With inputs wrapper
        {"inputs": img_base64},
        # Format 3: With image data URL
        {"inputs": f"data:image/png;base64,{img_base64}"},
    ]

    # Try with retries
    for attempt in range(retries):
        for payload_format_idx, payload in enumerate(payload_formats):
            try:
                print(
                    f"   🌐 Sending to HF API (attempt {attempt + 1}/{retries}, format {payload_format_idx + 1})..."
                )

                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload if isinstance(payload, dict) else {"inputs": payload},
                    timeout=120,
                )

                # Check if model is loading
                if response.status_code == 503:
                    try:
                        error_data = response.json()
                        wait_time = error_data.get("estimated_time", 30)
                    except ValueError:
                        wait_time = 30
                    print(f"   ⏳ Model is loading, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                # Handle errors
                if response.status_code != 200:
                    error_msg = response.text[:500]
                    if response.status_code == 404:
                        print("   ⚠️  Model not found on Inference Providers API (404)")
                        print(
                            "   💡 The Chandra model may not be available on Inference Providers"
                        )
                        print(
                            "   💡 Alternative: Use extract_ec_data_api.py with Datalab API"
                        )
                        if attempt < retries - 1:
                            continue
                        raise Exception(
                            "Model not available on Inference Providers API (404)"
                        )
                    elif response.status_code == 401:
                        print("   ⚠️  API key invalid or missing")
                        print(
                            "   💡 Get free API key at: https://huggingface.co/settings/tokens"
                        )
                        raise Exception(f"Authentication failed: {error_msg}")
                    else:
                        if attempt < retries - 1:
                            print(
                                f"   ⚠️  Error {response.status_code}, trying next format..."
                            )
                            time.sleep(2)
                            continue
                        raise Exception(
                            f"API error {response.status_code}: {error_msg}"
                        )

                # Parse response
                try:
                    result = response.json()
                except ValueError:
                    return response.text

                # Extract text from response
                if isinstance(result, dict):
                    # Try common response fields
                    for key in [
                        "generated_text",
                        "text",
                        "markdown",
                        "output",
                        "result",
                    ]:
                        if key in result:
                            return str(result[key])
                    # If list in result, try first item
                    if "generated_texts" in result and isinstance(
                        result["generated_texts"], list
                    ):
                        return str(result["generated_texts"][0])
                    # Single key-value pair
                    if len(result) == 1:
                        return str(list(result.values())[0])
                    return str(result)
                elif isinstance(result, list) and len(result) > 0:
                    first_item = result[0]
                    if isinstance(first_item, dict):
                        return first_item.get("generated_text", str(first_item))
                    return str(first_item)
                elif isinstance(result, str):
                    return result
                else:
                    return str(result)

            except requests.exceptions.Timeout:
                if payload_format_idx < len(payload_formats) - 1:
                    continue  # Try next format
                if attempt < retries - 1:
                    print("   ⚠️  Timeout, retrying...")
                    time.sleep(5)
                    break
                raise
            except Exception as e:
                if payload_format_idx < len(payload_formats) - 1:
                    continue  # Try next format
                if attempt < retries - 1:
                    print(f"   ⚠️  Error: {e}, retrying...")
                    time.sleep(5)
                    break
                raise

    raise Exception("Failed to get response from HuggingFace API after all attempts")


def extract_text_from_pdf_hf_api(pdf_path: str, api_key: str) -> tuple:
    """
    Extract text from PDF using Chandra OCR via HuggingFace Inference Providers API.

    Args:
        pdf_path: Path to the PDF file
        api_key: HuggingFace API token (required)

    Returns:
        Extracted text and structured data
    """
    print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
    print("🔧 Using Chandra OCR via HuggingFace Inference Providers API")
    print("   📖 New endpoint: router.huggingface.co/hf-inference/")
    print()

    try:
        # Convert PDF to images
        print("📄 Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=300)
        print(f"   Converted {len(images)} pages")
        print()

        # Process each page
        all_text_parts = []
        structured_data: dict[str, Any] = {"pages": []}

        for i, image in enumerate(images):
            print(f"📄 Processing page {i+1}/{len(images)}...")

            try:
                # Get OCR result from API
                page_text = chandra_ocr_hf_api(image, api_key)

                all_text_parts.append(page_text)
                structured_data["pages"].append({"page_num": i + 1, "text": page_text})

                print(f"   ✓ Page {i+1} completed")

                # Small delay to avoid rate limiting
                if i < len(images) - 1:
                    time.sleep(1)

            except Exception as e:
                print(f"   ❌ Error processing page {i+1}: {e}")
                all_text_parts.append(f"\n[Error processing page {i+1}]\n")

        full_text = "\n\n".join(all_text_parts)
        print("\n✅ Text extraction completed")

        return full_text, structured_data

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise


def parse_table_rows(text: str) -> list[dict[str, str]]:
    """Parse OCR text to extract table rows with the required fields."""
    lines = text.split("\n")
    rows = []

    serial_pattern = re.compile(r"^\s*(\d+)\s+")
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

            row_text_parts = [line]
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j]
                if serial_pattern.match(next_line) and j > i + 1:
                    break
                row_text_parts.append(next_line)

            row_text = " ".join(row_text_parts)

            plot_match = plot_no_pattern.search(row_text)
            if plot_match:
                current_row["Plot No./"] = plot_match.group(1).strip()
            else:
                plot_alt_pattern = re.compile(r"\b([0-9]+/[0-9]+|[A-Z]\d+[A-Z]?)\b")
                alt_match = plot_alt_pattern.search(line)
                if alt_match:
                    current_row["Plot No./"] = alt_match.group(1).strip()

            if current_row["Plot No./"]:
                survey_match = survey_pattern.search(row_text)
                if survey_match:
                    current_row["Survey No./"] = survey_match.group(1).strip()

                doc_match = doc_pattern.search(row_text)
                if doc_match:
                    current_row["Document No.& Year"] = doc_match.group(1).strip()

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


def extract_data_from_pdf(
    pdf_path: str, api_key: str | None = None
) -> list[dict[str, str]]:
    """Main function to extract data from a PDF file."""
    if not api_key:
        raise ValueError(
            "HF_API_KEY is required. Get one from https://huggingface.co/settings/tokens"
        )

    filename = os.path.basename(pdf_path)

    text, structured_data = extract_text_from_pdf_hf_api(pdf_path, api_key)

    # Save OCR text for debugging
    debug_file = pdf_path.replace(".pdf", "_ocr_text.txt")
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n💾 Saved OCR text to: {debug_file}\n")

    # Parse table rows
    print("🔍 Parsing table data...")
    rows = parse_table_rows(text)

    for row in rows:
        row["filename"] = filename

    filtered_rows = [row for row in rows if row.get("Plot No./", "").strip()]
    print(f"✓ Found {len(filtered_rows)} rows with Plot No./ information\n")

    return filtered_rows


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract data from EC PDF files using HuggingFace Inference Providers API",
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
        print("Please check the path and try again.")
        return

    # Get API key from .env file (loaded by load_dotenv()) or environment variables
    api_key = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_TOKEN")

    print("=" * 70)
    print(" " * 20 + "EC Data Extraction (HF API Mode)")
    print("=" * 70)
    print()

    if api_key:
        print("✅ Using HuggingFace API key")
        print("   📖 Using new Inference Providers API endpoint")
    else:
        print("❌ No API key provided - HuggingFace API requires an API key")
        print("   📖 Get API key: https://huggingface.co/settings/tokens")
        print("   📖 Documentation: https://huggingface.co/docs/inference-providers")
        print("   💡 Add to .env: HF_API_KEY=your_token_here")
        print()
        print(
            "   ⚠️  Note: If Chandra model is not available, use extract_ec_data_api.py"
        )
        print("      with Datalab API instead")
        return
    print()

    try:
        rows = extract_data_from_pdf(pdf_file, api_key)

        if not rows:
            print("⚠️  No rows with Plot No./ information found.")
            return

        df = pd.DataFrame(rows)
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

        print("=" * 70)
        print(" " * 25 + "Extracted Data")
        print("=" * 70)
        print()
        print(df.to_string(index=False))
        print()

        output_file = pdf_file.replace(".pdf", "_extracted.csv")
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"💾 Data saved to: {output_file}")

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

        traceback.print_exc()


if __name__ == "__main__":
    main()
