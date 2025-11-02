#!/usr/bin/env python3
"""
Fast CPU-only EC Data Extraction using EasyOCR.
Much faster than Chandra on CPU - processes pages in seconds instead of minutes.
"""

import argparse
import os
import re
import time

import easyocr
import numpy as np
import pandas as pd
from pdf2image import convert_from_path

# Try to import fuzzy matching libraries (optional but helpful)
try:
    from fuzzywuzzy import fuzz

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print(
        "💡 Tip: Install fuzzywuzzy for better name matching: pip install fuzzywuzzy python-Levenshtein"
    )


def extract_text_from_pdf_easyocr(pdf_path: str) -> str:
    """
    Extract text from PDF using EasyOCR (fast CPU mode).

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text from all pages
    """
    print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
    print("🔧 Using EasyOCR (Fast CPU mode)")
    print()

    try:
        print("=" * 70)
        print("⚠️  First run will download EasyOCR models (~100MB)")
        print("   Subsequent runs will be much faster!")
        print("=" * 70)
        print()

        # Initialize EasyOCR reader (English only for speed)
        print("📥 Initializing EasyOCR...")
        print("   (First run downloads models - one time only)")
        start_init = time.time()
        reader = easyocr.Reader(["en"], gpu=False)  # Force CPU mode
        init_time = time.time() - start_init
        print(f"   ✓ EasyOCR ready ({init_time:.1f}s)")
        print()

        # Convert PDF to images
        print("📄 Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=300)
        print(f"   Converted {len(images)} pages")
        print()

        # Process each page
        all_text_parts = []

        for i, image in enumerate(images):
            print(f"📄 Processing page {i+1}/{len(images)}...", end="", flush=True)
            start_time = time.time()

            # Convert PIL Image to numpy array for EasyOCR
            img_array = np.array(image)

            # Run OCR on the image
            results = reader.readtext(img_array)

            # Combine all text detections
            page_text_lines = []
            for bbox, text, confidence in results:
                if confidence > 0.5:  # Filter low confidence results
                    page_text_lines.append(text)

            page_text = "\n".join(page_text_lines)
            all_text_parts.append(page_text)

            elapsed = time.time() - start_time
            print(f" ✓ ({elapsed:.1f}s)")

        full_text = "\n\n".join(all_text_parts)
        print("\n✅ Text extraction completed")

        return full_text

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise


def normalize_ocr_text(text: str) -> str:
    """
    Normalize OCR text to handle common OCR errors.
    """
    # Common OCR character substitutions
    substitutions = {
        "O": "0",  # Sometimes O is confused with 0 in numbers
        "I": "1",  # I can be confused with 1
        "S": "5",  # S can be confused with 5
        "l": "1",  # lowercase l can be 1
        "|": "1",  # pipe can be 1
    }

    # Only substitute in number contexts
    normalized = text
    # Replace common OCR errors in numeric patterns
    for old, new in substitutions.items():
        # Replace in number-like contexts (surrounded by digits or /)
        normalized = re.sub(r"(?<=\d)" + re.escape(old) + r"(?=\d|/)", new, normalized)

    return normalized


def fuzzy_match_pattern(pattern: str, text: str, threshold: int = 70) -> str | None:
    """
    Use fuzzy matching to find pattern in text, handling OCR errors.
    """
    if not FUZZY_AVAILABLE:
        return None

    # Normalize text
    text_normalized = normalize_ocr_text(text)

    # Try exact match first
    if pattern.lower() in text_normalized.lower():
        return pattern

    # Try fuzzy match
    words = text_normalized.split()
    for i in range(len(words) - len(pattern.split()) + 1):
        candidate = " ".join(words[i : i + len(pattern.split())])
        if fuzz.partial_ratio(pattern.lower(), candidate.lower()) >= threshold:
            return candidate

    return None


def extract_document_number_fuzzy(row_text: str) -> str:
    """
    Extract document number with fuzzy matching and multiple strategies.
    Returns best match or empty string.
    """
    # Strategy 1: Direct pattern match (XXXX/YYYY)
    doc_patterns = [
        re.compile(r"(\d{2,6}/\d{4})"),  # Standard: 1439/2005
        re.compile(r"(\d{2,6}[\/\-]\d{4})"),  # With dash or slash variants
        re.compile(r"([A-Z]?\d{2,6}[\/\-]\d{4})"),  # With optional prefix
    ]

    candidates = []
    for pattern in doc_patterns:
        matches = pattern.findall(row_text)
        for match in matches:
            # Normalize (fix OCR errors)
            normalized = normalize_ocr_text(match)
            parts = re.split(r"[/\-]", normalized)
            if len(parts) == 2:
                year = parts[1]
                if len(year) == 4 and year.isdigit():
                    year_int = int(year)
                    if 1900 <= year_int <= 2100:
                        candidates.append((normalized, 100))  # High confidence

    # Strategy 2: Look for patterns near keywords
    keywords = ["document", "doc", "reg", "registration"]
    for keyword in keywords:
        keyword_match = re.search(
            rf"{keyword}[^0-9]*?(\d{{2,6}}[/\-]\d{{4}})", row_text, re.IGNORECASE
        )
        if keyword_match:
            doc_candidate = normalize_ocr_text(keyword_match.group(1))
            parts = doc_candidate.split("/")
            if len(parts) == 2 and len(parts[1]) == 4:
                candidates.append((doc_candidate, 90))

    # Strategy 3: Fuzzy match known patterns
    if FUZZY_AVAILABLE:
        # Look for number/number patterns and validate
        all_patterns = re.findall(r"\d+[/\-]\d+", row_text)
        for pattern in all_patterns:
            normalized = normalize_ocr_text(pattern)
            parts = re.split(r"[/\-]", normalized)
            if len(parts) == 2 and len(parts[1]) == 4:
                year = parts[1]
                if year.isdigit() and 1900 <= int(year) <= 2100:
                    # Check if it matches a document pattern
                    confidence = fuzz.partial_ratio(pattern, normalized)
                    if confidence >= 70:
                        candidates.append((normalized, confidence))

    if candidates:
        # Sort by confidence and return best
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return ""


def extract_plot_numbers_fuzzy(row_text: str, row_lines: list[str]) -> str:
    """
    Extract plot numbers using multiple strategies and fuzzy matching.
    """
    plot_numbers = []

    # Strategy 1: Look for "Village &" pattern
    village_patterns = [
        re.compile(r"Village[^&]*&[^,]*?((?:\d+[/\-]\d+[,\s]*)+)", re.IGNORECASE),
        re.compile(
            r"Vilage[^&]*&[^,]*?((?:\d+[/\-]\d+[,\s]*)+)", re.IGNORECASE
        ),  # OCR typo
        re.compile(
            r"VIlage[^&]*&[^,]*?((?:\d+[/\-]\d+[,\s]*)+)", re.IGNORECASE
        ),  # OCR typo
    ]

    for pattern in village_patterns:
        match = pattern.search(row_text)
        if match:
            plot_text = match.group(1)
            plot_pattern = re.compile(r"(\d{1,4}[/\-]\d{1,3})")
            matches = plot_pattern.findall(plot_text)
            for plot in matches:
                normalized = normalize_ocr_text(plot)
                parts = re.split(r"[/\-]", normalized)
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    if 1 <= int(parts[0]) <= 9999 and 1 <= int(parts[1]) <= 999:
                        plot_numbers.append(normalized)

    # Strategy 2: Look for comma-separated lists of numbers
    if not plot_numbers:
        # Look for lines with multiple number/number patterns
        for line in row_lines:
            # Pattern: "103/3, 104/1, 85/8"
            if "," in line and "/" in line:
                plot_pattern = re.compile(r"(\d{1,4}[/\-]\d{1,3})")
                matches = plot_pattern.findall(line)
                for plot in matches:
                    normalized = normalize_ocr_text(plot)
                    parts = re.split(r"[/\-]", normalized)
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        # Ensure it's not a document number (4-digit second part = year)
                        if (
                            len(parts[1]) <= 3
                            and 1 <= int(parts[0]) <= 9999
                            and 1 <= int(parts[1]) <= 999
                        ):
                            plot_numbers.append(normalized)

    # Strategy 3: Look for standalone plot patterns (but exclude document numbers)
    if not plot_numbers:
        plot_pattern = re.compile(r"\b(\d{1,4}[/\-]\d{1,3})\b")
        all_matches = plot_pattern.findall(row_text)
        for plot in all_matches:
            normalized = normalize_ocr_text(plot)
            parts = re.split(r"[/\-]", normalized)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                # Document numbers have 4-digit years
                if (
                    len(parts[1]) <= 3
                    and 1 <= int(parts[0]) <= 9999
                    and 1 <= int(parts[1]) <= 999
                ):
                    # Additional validation: check context
                    plot_context = row_text[
                        max(0, row_text.find(plot) - 20) : row_text.find(plot) + 50
                    ]
                    if (
                        "village" in plot_context.lower()
                        or "schedule" in plot_context.lower()
                        or "plot" in plot_context.lower()
                    ):
                        plot_numbers.append(normalized)

    # Remove duplicates and sort
    if plot_numbers:
        unique_plots = []
        seen = set()
        for plot in plot_numbers:
            if plot not in seen:
                seen.add(plot)
                unique_plots.append(plot)

        # Sort by first number, then second
        unique_plots.sort(key=lambda x: (int(x.split("/")[0]), int(x.split("/")[1])))
        return ", ".join(unique_plots)

    return ""


def extract_names_fuzzy(row_text: str, row_lines: list[str]) -> tuple[str, str]:
    """
    Extract names using fuzzy matching and multiple strategies.
    Returns (executant_name, claimant_name).
    """
    executant = ""
    claimant = ""

    # Strategy 1: Look for name-like patterns (2-6 capitalized words)
    name_patterns = [
        re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b"),  # Normal case
        re.compile(r"\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,4})\b"),  # All caps (OCR)
        re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\s+[A-Z][a-z]+)\b"),  # Mixed
    ]

    all_name_candidates = []
    for pattern in name_patterns:
        matches = pattern.findall(row_text)
        all_name_candidates.extend(matches)

    # Strategy 2: Look for text near keywords
    executant_keywords = ["executant", "executant(s)", "exec", "seller", "transferor"]
    claimant_keywords = ["claimant", "claimant(s)", "claim", "buyer", "transferee"]

    for keyword in executant_keywords:
        keyword_match = re.search(
            rf"{keyword}[:\s]+([A-Z][A-Za-z\s]+{{2,20}})", row_text, re.IGNORECASE
        )
        if keyword_match:
            name_candidate = keyword_match.group(1).strip()
            if len(name_candidate.split()) >= 2:
                all_name_candidates.append(name_candidate)

    for keyword in claimant_keywords:
        keyword_match = re.search(
            rf"{keyword}[:\s]+([A-Z][A-Za-z\s]+{{2,20}})", row_text, re.IGNORECASE
        )
        if keyword_match:
            name_candidate = keyword_match.group(1).strip()
            if len(name_candidate.split()) >= 2:
                all_name_candidates.append(name_candidate)

    # Filter and validate names
    valid_names = []
    exclude_keywords = [
        "plot",
        "survey",
        "document",
        "village",
        "schedule",
        "date",
        "registration",
        "nature",
        "page",
        "vol",
        "no",
        "year",
        "pr",
        "consideration",
        "value",
        "august",
        "june",
        "december",
        "january",
    ]

    for name in all_name_candidates:
        name_lower = name.lower().strip()
        # Filter out keywords and short strings
        if not any(keyword in name_lower for keyword in exclude_keywords):
            if len(name.split()) >= 2 and len(name) > 5:
                # Avoid Roman numerals, single letters, dates
                if not re.match(r"^[IVX]+$", name) and not re.match(
                    r"^\d+[/-]\d+", name
                ):
                    # Check if it looks like a real name (has vowels, proper structure)
                    if any(vowel in name_lower for vowel in "aeiou"):
                        valid_names.append(name.strip())

    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in valid_names:
        name_norm = name.lower().strip()
        if name_norm not in seen:
            seen.add(name_norm)
            unique_names.append(name)

    # Use fuzzy matching to merge similar names
    if FUZZY_AVAILABLE and len(unique_names) > 1:
        merged_names = []
        for name in unique_names:
            is_duplicate = False
            for existing in merged_names:
                if fuzz.ratio(name.lower(), existing.lower()) > 85:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged_names.append(name)
        unique_names = merged_names

    # Assign names (first = executant, second = claimant if available)
    if len(unique_names) >= 1:
        executant = unique_names[0]
    if len(unique_names) >= 2:
        claimant = unique_names[1]

    return executant, claimant


def extract_survey_number_fuzzy(row_text: str, header_value: str) -> str:
    """
    Extract survey number with fuzzy matching.
    """
    if header_value:
        return header_value

    # Try multiple patterns
    survey_patterns = [
        re.compile(r"Survey[^0-9]*([0-9]+[/\-][0-9]+)", re.IGNORECASE),
        re.compile(r"Survey[^0-9]*([0-9]+)", re.IGNORECASE),
        re.compile(r"([0-9]+[/\-][0-9]+).*[Ss]urvey", re.IGNORECASE),
    ]

    for pattern in survey_patterns:
        match = pattern.search(row_text)
        if match:
            survey_val = normalize_ocr_text(match.group(1))
            # Validate format
            if "/" in survey_val or "-" in survey_val:
                parts = re.split(r"[/\-]", survey_val)
                if len(parts) == 2 and all(p.isdigit() for p in parts):
                    return survey_val

    return ""


def parse_table_rows(text: str) -> list[dict[str, str]]:
    """
    Parse OCR text to extract table rows with the required fields.
    Only includes rows that have Plot No information.
    Fields: filename, Sr.No, Document No.& Year, Name of Executant(s),
    Name of Claimant(s), Survey No, Plot No
    """
    lines = text.split("\n")
    rows = []

    # Clean lines
    lines = [line.strip() for line in lines if line.strip()]

    # First, try to find the Survey No from the header (appears early in document)
    survey_no_header = ""
    for line in lines[:50]:  # Check first 50 lines
        if "Survey" in line:
            # Look for pattern like "Survey 103/3"
            survey_header_match = re.search(
                r"Survey[^0-9]*([0-9]+/[0-9]+)", line, re.IGNORECASE
            )
            if survey_header_match:
                survey_no_header = survey_header_match.group(1)
                break
        # Also check if just a number like "103/3" appears near "Survey" text
        if re.match(r"^\d+/\d+$", line):
            # Check if nearby lines mention Survey
            idx = lines.index(line)
            nearby = " ".join(lines[max(0, idx - 3) : min(len(lines), idx + 3)])
            if "survey" in nearby.lower():
                survey_no_header = line
                break

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if line is a standalone serial number (just digits)
        # Or starts with a serial number followed by content
        serial_match = None
        if re.match(r"^\s*\d+\s*$", line):
            # Standalone number - likely serial number
            serial_match = re.match(r"^\s*(\d+)\s*$", line)
        elif re.match(r"^\s*\d+\s+", line):
            # Number followed by text
            serial_match = re.match(r"^\s*(\d+)\s+", line)

        if serial_match:
            serial_no = serial_match.group(1)

            # Skip if it's clearly part of a date (e.g., "2005" alone)
            if (
                len(serial_no) == 4
                and int(serial_no) >= 1900
                and int(serial_no) <= 2100
            ):
                # Might be a year, check context
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if "/" in next_line or re.match(r"\d{1,2}-[A-Za-z]{3}", next_line):
                        # Likely part of date, skip
                        i += 1
                        continue

            current_row = {
                "Sr.No": serial_no,
                "Document No.& Year": "",
                "Name of Executant(s)": "",
                "Name of Claimant(s)": "",
                "Survey No": survey_no_header,  # Use header value as default
                "Plot No": "",
            }

            # Look ahead to collect all related information for this row
            # Table rows typically span 10-25 lines
            row_text_parts = [line]
            row_lines = [line]

            for j in range(i + 1, min(i + 30, len(lines))):
                next_line = lines[j]

                # Stop if we hit another clear serial number (standalone or at start)
                if j > i + 2:  # Give some buffer
                    if re.match(r"^\s*\d+\s*$", next_line):
                        # Check if next few lines suggest a new row
                        lookahead = " ".join(lines[j : min(j + 3, len(lines))])
                        if re.search(
                            r"\d{2,6}/\d{4}", lookahead
                        ):  # Has document number
                            break

                    # Also stop if we see a pattern like "X" where X is next serial
                    if re.match(r"^\s*\d+\s+", next_line):
                        next_serial = re.match(r"^\s*(\d+)\s+", next_line).group(1)
                        if (
                            next_serial.isdigit()
                            and int(next_serial) == int(serial_no) + 1
                        ):
                            break

                row_text_parts.append(next_line)
                row_lines.append(next_line)

            row_text = " ".join(row_text_parts)

            # Normalize row text to handle OCR errors
            row_text_normalized = normalize_ocr_text(row_text)

            # Extract Document No. & Year using fuzzy matching
            current_row["Document No.& Year"] = extract_document_number_fuzzy(
                row_text_normalized
            )

            # Extract Plot No using fuzzy matching
            current_row["Plot No"] = extract_plot_numbers_fuzzy(
                row_text_normalized, row_lines
            )

            # Extract Survey No using fuzzy matching
            current_row["Survey No"] = extract_survey_number_fuzzy(
                row_text_normalized, survey_no_header
            )

            # Extract Names using fuzzy matching
            executant, claimant = extract_names_fuzzy(row_text_normalized, row_lines)
            current_row["Name of Executant(s)"] = executant
            current_row["Name of Claimant(s)"] = claimant

            # Only include rows that have Plot No
            if current_row["Plot No"]:
                rows.append(current_row)

            i += 1
        else:
            i += 1

    return rows


def extract_data_from_pdf(pdf_path: str) -> list[dict[str, str]]:
    """Main function to extract data from a PDF file."""
    filename = os.path.basename(pdf_path)

    # Extract text using EasyOCR
    text = extract_text_from_pdf_easyocr(pdf_path)

    # Save OCR text for debugging
    debug_file = pdf_path.replace(".pdf", "_ocr_text.txt")
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n💾 Saved OCR text to: {debug_file}\n")

    # Parse table rows
    print("🔍 Parsing table data...")
    rows = parse_table_rows(text)

    # Add filename to each row
    for row in rows:
        row["filename"] = filename

    # Filter rows that have Plot No information
    filtered_rows = [row for row in rows if row.get("Plot No", "").strip()]
    print(f"✓ Found {len(filtered_rows)} rows with Plot No information\n")

    return filtered_rows


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract data from EC PDF files using EasyOCR (Fast CPU)",
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

    print("=" * 70)
    print(" " * 20 + "EC Data Extraction (EasyOCR - Fast CPU)")
    print("=" * 70)
    print()

    try:
        rows = extract_data_from_pdf(pdf_file)

        if not rows:
            print("⚠️  No rows with Plot No information found.")
            print("\n💡 Tip: Check the OCR text file to see what was extracted:")
            print(f"   {pdf_file.replace('.pdf', '_ocr_text.txt')}")
            return

        # Create DataFrame
        df = pd.DataFrame(rows)
        columns_order = [
            "filename",
            "Sr.No",
            "Document No.& Year",
            "Name of Executant(s)",
            "Name of Claimant(s)",
            "Survey No",
            "Plot No",
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
