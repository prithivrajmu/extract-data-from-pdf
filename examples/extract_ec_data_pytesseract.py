#!/usr/bin/env python3
"""
EC Data Extraction using PyTesseract (Google's Tesseract OCR).
Fast, lightweight OCR engine that works well for text-based PDFs.
Requires Tesseract-OCR to be installed on the system.
"""

import os
import re
import argparse
from pathlib import Path
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from typing import List, Dict, Optional
import time

# Try to import fuzzy matching libraries (optional but helpful)
try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


def extract_text_from_pdf_pytesseract(pdf_path: str) -> str:
    """
    Extract text from PDF using PyTesseract (Google's Tesseract OCR).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from all pages
    """
    print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
    print("🔧 Using PyTesseract (Google's Tesseract OCR)")
    print()
    
    try:
        # Check if Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            raise RuntimeError(
                "Tesseract OCR is not installed or not in PATH.\n"
                "Please install Tesseract:\n"
                "  - Linux: sudo apt-get install tesseract-ocr\n"
                "  - macOS: brew install tesseract\n"
                "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
            )
        
        print("=" * 70)
        print("📝 PyTesseract - Fast OCR Engine")
        print("=" * 70)
        print()
        
        # Convert PDF to images
        print(f"📄 Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=300)
        print(f"   Converted {len(images)} pages")
        print()
        
        # Process each page
        all_text_parts = []
        
        for i, image in enumerate(images):
            print(f"📄 Processing page {i+1}/{len(images)}...", end='', flush=True)
            start_time = time.time()
            
            # Run OCR on the image
            # Use page segmentation mode 6 (Assume a single uniform block of text)
            page_text = pytesseract.image_to_string(image, config='--psm 6')
            
            all_text_parts.append(page_text)
            
            elapsed = time.time() - start_time
            print(f" ✓ ({elapsed:.1f}s)")
        
        full_text = '\n\n'.join(all_text_parts)
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
    # Replace common OCR mistakes
    replacements = {
        '|': 'I',
        '0': 'O',  # Be careful with this one
        '@': 'a',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def parse_table_rows(text: str) -> List[Dict[str, str]]:
    """
    Parse table rows from extracted OCR text.
    Extracts: Sr.No, Document No.& Year, Name of Executant(s), 
    Name of Claimant(s), Survey No, Plot No
    """
    lines = text.split('\n')
    rows = []
    current_row = {}
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        # Look for serial number pattern (1, 2, 3, etc.)
        sr_no_match = re.match(r'^(\d+)[\.\)]?\s*', line)
        if sr_no_match and len(sr_no_match.group(1)) <= 3:
            # If we have a complete row, save it
            if current_row and 'Plot No' in current_row:
                rows.append(current_row)
            
            # Start new row
            current_row = {
                'Sr.No': sr_no_match.group(1),
                'Document No.& Year': '',
                'Name of Executant(s)': '',
                'Name of Claimant(s)': '',
                'Survey No': '',
                'Plot No': ''
            }
            
            # Try to extract Document No.& Year from same line
            rest_of_line = line[len(sr_no_match.group(0)):].strip()
            doc_year_match = re.search(r'([A-Z0-9]+/\d{4})', rest_of_line)
            if doc_year_match:
                current_row['Document No.& Year'] = doc_year_match.group(1)
        
        # Look for Document No.& Year pattern (XXXX/YYYY)
        elif 'Document No.& Year' not in current_row or not current_row.get('Document No.& Year'):
            doc_year_match = re.search(r'([A-Z0-9]+/\d{4})', line)
            if doc_year_match:
                current_row['Document No.& Year'] = doc_year_match.group(1)
        
        # Look for Plot No pattern
        plot_match = re.search(r'(?:Plot\s*No[\.:]?\s*)([A-Z0-9/]+)', line, re.IGNORECASE)
        if plot_match:
            current_row['Plot No'] = plot_match.group(1).strip()
        
        # Look for Survey No pattern
        survey_match = re.search(r'(?:Survey\s*No[\.:]?\s*)([A-Z0-9/]+)', line, re.IGNORECASE)
        if survey_match:
            current_row['Survey No'] = survey_match.group(1).strip()
        
        # Look for Executant name (usually comes before Claimant)
        if 'Executant' in line or 'Executant' in line.lower():
            # Name might be on same line or next line
            name_match = re.search(r'(?:Executant[\(s\)]*[:\-]?\s*)(.+?)(?:\s*Claimant|$)', line, re.IGNORECASE)
            if name_match:
                current_row['Name of Executant(s)'] = name_match.group(1).strip()
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not re.match(r'^(\d+)[\.\)]', next_line):
                    current_row['Name of Executant(s)'] = next_line
                    i += 1
        
        # Look for Claimant name
        if 'Claimant' in line or 'claimant' in line.lower():
            name_match = re.search(r'(?:Claimant[\(s\)]*[:\-]?\s*)(.+)', line, re.IGNORECASE)
            if name_match:
                current_row['Name of Claimant(s)'] = name_match.group(1).strip()
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not re.match(r'^(\d+)[\.\)]', next_line):
                    current_row['Name of Claimant(s)'] = next_line
                    i += 1
        
        i += 1
    
    # Add last row if it exists
    if current_row and 'Plot No' in current_row:
        rows.append(current_row)
    
    return rows


def extract_data_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """Main function to extract data from a PDF file."""
    filename = os.path.basename(pdf_path)
    
    # Extract text using PyTesseract
    text = extract_text_from_pdf_pytesseract(pdf_path)
    
    # Save OCR text for debugging
    debug_file = pdf_path.replace('.pdf', '_ocr_text.txt')
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\n💾 Saved OCR text to: {debug_file}\n")
    
    # Parse table rows
    print("🔍 Parsing table data...")
    rows = parse_table_rows(text)
    
    # Add filename to each row
    for row in rows:
        row['filename'] = filename
    
    # Filter rows that have Plot No information
    filtered_rows = [row for row in rows if row.get('Plot No', '').strip()]
    print(f"✓ Found {len(filtered_rows)} rows with Plot No information\n")
    
    return filtered_rows


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract data from EC PDF files using PyTesseract OCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                                    # Use default test file
  %(prog)s --file path/to/file.pdf           # Use custom PDF file
  %(prog)s --input path/to/file.pdf          # Alternative syntax
        '''
    )
    parser.add_argument(
        '--file', '--input',
        dest='pdf_file',
        default='test_file/RG EC 103 3.pdf',
        help='Path to the PDF file to process (default: test_file/RG EC 103 3.pdf)'
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
    print(" " * 20 + "EC Data Extraction (PyTesseract)")
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
        columns_order = ['filename', 'Sr.No', 'Document No.& Year', 
                        'Name of Executant(s)', 'Name of Claimant(s)', 
                        'Survey No', 'Plot No']
        df = df[columns_order]
        
        # Display results
        print("=" * 70)
        print(" " * 25 + "Extracted Data")
        print("=" * 70)
        print()
        print(df.to_string(index=False))
        print()
        
        # Save to CSV
        output_file = pdf_file.replace('.pdf', '_extracted.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"💾 Data saved to: {output_file}")
        
        # Save to Excel
        output_excel = pdf_file.replace('.pdf', '_extracted.xlsx')
        df.to_excel(output_excel, index=False, engine='openpyxl')
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

