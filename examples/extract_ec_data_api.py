#!/usr/bin/env python3
"""
Script to extract data from EC PDF files using Datalab Marker API.
No local model download required - uses Datalab's managed API endpoint.

Documentation: https://documentation.datalab.to/docs/recipes/marker/conversion-api-overview
"""

import argparse
import os
import re
import time
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def submit_pdf_to_datalab(
    pdf_path: str, api_key: str, output_format: str = "markdown"
) -> dict:
    """
    Submit PDF to Datalab Marker API for conversion.

    Args:
        pdf_path: Path to PDF file
        api_key: Datalab API key
        output_format: Output format ('markdown', 'json', or 'html')

    Returns:
        Initial response with request_id and check_url
    """
    api_url = "https://www.datalab.to/api/v1/marker"

    headers = {"X-Api-Key": api_key}

    with open(pdf_path, "rb") as f:
        form_data = {
            "file": (os.path.basename(pdf_path), f, "application/pdf"),
            "force_ocr": (None, True),  # Force OCR for better accuracy
            "paginate": (None, False),
            "output_format": (None, output_format),
            "use_llm": (None, False),  # Set to True for better accuracy but slower
            "strip_existing_ocr": (None, False),
            "disable_image_extraction": (None, False),
        }

        response = requests.post(api_url, files=form_data, headers=headers)
        response.raise_for_status()
        return response.json()


def poll_datalab_status(
    check_url: str, api_key: str, max_polls: int = 300, poll_interval: int = 2
) -> dict:
    """
    Poll Datalab API to check if PDF conversion is complete.

    Args:
        check_url: URL to poll for status
        api_key: Datalab API key
        max_polls: Maximum number of polling attempts
        poll_interval: Seconds to wait between polls

    Returns:
        Final response with converted content
    """
    headers = {"X-Api-Key": api_key}

    for i in range(max_polls):
        response = requests.get(check_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "complete":
            return data
        elif data.get("status") == "failed":
            error_msg = data.get("error", "Unknown error")
            raise Exception(f"Conversion failed: {error_msg}")
        else:
            if i % 10 == 0:  # Print progress every 10 polls
                print(f"   ⏳ Still processing... (poll {i+1}/{max_polls})")
            time.sleep(poll_interval)

    raise Exception(f"Conversion timed out after {max_polls} polling attempts")


def extract_text_from_pdf_datalab_api(pdf_path: str, api_key: str) -> tuple:
    """
    Extract text from PDF using Datalab Marker API.

    Args:
        pdf_path: Path to the PDF file
        api_key: Datalab API key (required)

    Returns:
        Extracted text and structured data
    """
    print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
    print("🔧 Using Datalab Marker API")
    print()

    try:
        # Step 1: Submit PDF for processing
        print("📤 Submitting PDF to Datalab API...")
        initial_response = submit_pdf_to_datalab(
            pdf_path, api_key, output_format="markdown"
        )

        if not initial_response.get("success"):
            error = initial_response.get("error", "Unknown error")
            raise Exception(f"Failed to submit PDF: {error}")

        request_id = initial_response.get("request_id")
        check_url = initial_response.get("request_check_url")

        print(f"   ✓ PDF submitted successfully (Request ID: {request_id})")
        print("   🔄 Polling for completion...")

        # Step 2: Poll for completion
        result = poll_datalab_status(check_url, api_key, max_polls=300, poll_interval=2)

        if not result.get("success"):
            error = result.get("error", "Unknown error")
            raise Exception(f"Conversion failed: {error}")

        # Step 3: Extract results
        output_format = result.get("output_format", "markdown")
        full_text = result.get(output_format, "")
        page_count = result.get("page_count", 0)

        print(f"   ✓ Conversion complete! ({page_count} pages processed)")
        print()

        # Build structured data
        structured_data: dict[str, Any] = {
            "pages": [],
            "page_count": page_count,
            "metadata": result.get("metadata", {}),
        }

        # If paginated, split by page delimiters
        if page_count > 1:
            # Try to split by page markers if present
            pages = full_text.split("\n\n---\n\n")  # Common page delimiter
            for i, page_text in enumerate(pages):
                structured_data["pages"].append(
                    {"page_num": i + 1, "text": page_text.strip()}
                )
        else:
            structured_data["pages"].append({"page_num": 1, "text": full_text})

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
        raise ValueError("DATALAB_API_KEY is required. Get one from https://datalab.to")

    filename = os.path.basename(pdf_path)

    text, structured_data = extract_text_from_pdf_datalab_api(pdf_path, api_key)

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
        description="Extract data from EC PDF files using Datalab Marker API",
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
        print(f"❌ Error: File not found: {pdf_file}")
        print("Please check the path and try again.")
        return

    # Get API key from .env file (loaded by load_dotenv()) or environment variables
    api_key = os.environ.get("DATALAB_API_KEY") or os.environ.get("DATALAB_API_TOKEN")

    print("=" * 70)
    print(" " * 20 + "EC Data Extraction (API Mode)")
    print("=" * 70)
    print()

    if api_key:
        print("✅ Using Datalab API key")
    else:
        print("❌ No API key provided - Datalab API requires an API key")
        print("   📖 Get API key: https://datalab.to (sign up for free)")
        print(
            "   📖 Documentation: https://documentation.datalab.to/docs/recipes/marker/conversion-api-overview"
        )
        print("   💡 Add to .env: DATALAB_API_KEY=your_key_here")
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
