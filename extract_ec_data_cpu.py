#!/usr/bin/env python3
"""
CPU-only version of EC Data Extraction using Chandra OCR Python API.
Forces CPU mode explicitly to avoid CUDA errors.
"""

import os
import re
import json
import tempfile
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional

# Force CPU mode before importing torch or chandra
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_DEVICE_MAP'] = 'cpu'

# Import torch first and monkey-patch to force CPU
import torch
original_cuda_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False  # Force CPU mode

# Patch .cuda() and .to('cuda') methods to prevent CUDA usage
_original_module_to = torch.nn.Module.to
_original_module_cuda = torch.nn.Module.cuda

def _cpu_safe_to(self, device=None, *args, **kwargs):
    """Override .to() to force CPU if CUDA is requested"""
    if device is not None:
        device_str = str(device).lower()
        if 'cuda' in device_str:
            device = 'cpu'
    return _original_module_to(self, device, *args, **kwargs)

def _cpu_safe_cuda(self, device=None):
    """Override .cuda() to use CPU instead"""
    return self.to('cpu')

torch.nn.Module.to = _cpu_safe_to
torch.nn.Module.cuda = _cpu_safe_cuda

# Also patch Tensor methods
if hasattr(torch.Tensor, 'to'):
    _original_tensor_to = torch.Tensor.to
    def _cpu_safe_tensor_to(self, device=None, *args, **kwargs):
        if device is not None:
            device_str = str(device).lower()
            if 'cuda' in device_str:
                device = 'cpu'
        return _original_tensor_to(self, device, *args, **kwargs)
    torch.Tensor.to = _cpu_safe_tensor_to

if hasattr(torch.Tensor, 'cuda'):
    _original_tensor_cuda = torch.Tensor.cuda
    def _cpu_safe_tensor_cuda(self, device=None):
        return self.to('cpu')
    torch.Tensor.cuda = _cpu_safe_tensor_cuda

from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from pdf2image import convert_from_path
from PIL import Image


def extract_text_from_pdf_cpu(pdf_path: str) -> tuple:
    """
    Extract text from PDF using Chandra OCR with explicit CPU mode.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text and structured data
    """
    print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
    print("🔧 Using Chandra OCR Python API (CPU mode)")
    print()
    
    try:
        # Ensure CPU mode
        import torch
        if torch.cuda.is_available():
            print("⚠️  Warning: GPU detected but forcing CPU mode")
            print("   (To use GPU, reboot system and run extract_ec_data_pretty.py)")
        else:
            print("ℹ️  Running in CPU mode (GPU not available)")
        print()
        
        print("=" * 70)
        print("⚠️  First run may take 10-20 minutes!")
        print("   • Loading AI model (~18GB) - cached after first time")
        print("   • Processing PDF pages (1-2 min per page on CPU)")
        print("=" * 70)
        print()
        
        # Initialize InferenceManager with explicit CPU mode
        print("📥 Initializing model (CPU mode)...")
        print("   This may take 5-10 minutes on first run...")
        
        # Force device_map='cpu' in the manager
        # Also patch the model loading to ensure CPU
        try:
            from transformers import AutoModel
            original_from_pretrained = AutoModel.from_pretrained
            
            def cpu_from_pretrained(*args, **kwargs):
                """Wrapper to force CPU device_map"""
                kwargs['device_map'] = 'cpu'
                kwargs.pop('torch_dtype', None)  # Remove dtype that might force CUDA
                return original_from_pretrained(*args, **kwargs)
            
            AutoModel.from_pretrained = cpu_from_pretrained
        except Exception as e:
            print(f"   Warning: Could not patch model loading: {e}")
        
        manager = InferenceManager(method="hf")
        
        # Force model to CPU if it was loaded on CUDA
        try:
            if hasattr(manager, 'model') and manager.model is not None:
                # Get the actual model object
                model = manager.model
                if hasattr(model, 'to'):
                    model = model.to('cpu')
                    print("   ✓ Model forced to CPU")
                # Also check for model attribute
                if hasattr(model, 'model'):
                    model.model = model.model.to('cpu')
        except Exception as e:
            print(f"   Warning: Could not force model to CPU: {e}")
        
        # Convert PDF to images
        print(f"\n📄 Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=300)
        print(f"   Converted {len(images)} pages")
        
        # Process each page
        all_text_parts = []
        structured_data = {
            'pages': []
        }
        
        for i, image in enumerate(images):
            print(f"\n📄 Processing page {i+1}/{len(images)}...")
            
            try:
                # Create batch input
                batch = [
                    BatchInputItem(
                        image=image,
                        prompt_type="ocr_layout"
                    )
                ]
                
                # Generate OCR result
                print(f"   Running OCR (CPU mode - this may take 5-10 minutes per page)")
                print(f"   ⏳ Very slow on CPU - please be patient, it IS working...")
                
                # Ensure model is on CPU before generation
                try:
                    if hasattr(manager, 'model') and manager.model is not None:
                        model = manager.model
                        if hasattr(model, 'to'):
                            model.to('cpu')
                        if hasattr(model, 'device') and str(model.device) != 'cpu':
                            model = model.to('cpu')
                except:
                    pass
                
                try:
                    result = manager.generate(batch)[0]
                except RuntimeError as e:
                    if 'CUDA' in str(e) or 'cuda' in str(e).lower():
                        print(f"   ⚠️  CUDA error caught, forcing model to CPU and retrying...")
                        # Force everything to CPU and retry
                        try:
                            if hasattr(manager, 'model'):
                                manager.model = manager.model.to('cpu')
                        except:
                            pass
                        result = manager.generate(batch)[0]
                    else:
                        raise
                
                # Extract text
                if hasattr(result, 'markdown'):
                    page_text = result.markdown
                elif hasattr(result, 'raw'):
                    from chandra.output import parse_markdown
                    page_text = parse_markdown(result.raw)
                else:
                    page_text = str(result)
                
                all_text_parts.append(page_text)
                
                # Store structured data if available
                if hasattr(result, 'raw'):
                    structured_data['pages'].append({
                        'page_num': i + 1,
                        'text': page_text,
                        'raw': result.raw if hasattr(result, 'raw') else None
                    })
                
                print(f"   ✓ Page {i+1} completed")
                
            except Exception as e:
                print(f"   ❌ Error processing page {i+1}: {e}")
                all_text_parts.append(f"\n[Error processing page {i+1}]\n")
        
        full_text = '\n\n'.join(all_text_parts)
        print("\n✅ Text extraction completed")
        
        return full_text, structured_data
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def parse_table_rows(text: str) -> List[Dict[str, str]]:
    """Parse OCR text to extract table rows with the required fields."""
    lines = text.split('\n')
    rows = []
    
    serial_pattern = re.compile(r'^\s*(\d+)\s+')
    plot_no_pattern = re.compile(r'Plot\s+No[./:\s]*([A-Z0-9/\s\-]+?)(?:\s|$|[,\n])', re.IGNORECASE)
    survey_pattern = re.compile(r'Survey\s+No[./:\s]*([A-Z0-9/\s\-]+?)(?:\s|$|[,\n])', re.IGNORECASE)
    doc_pattern = re.compile(r'([A-Z]{1,4}[0-9/]{2,}\s*\d{4})', re.IGNORECASE)
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        serial_match = serial_pattern.match(line)
        
        if serial_match:
            serial_no = serial_match.group(1)
            current_row = {
                'Sr.No': serial_no,
                'Document No.& Year': '',
                'Name of Executant(s)': '',
                'Name of Claimant(s)': '',
                'Survey No./': '',
                'Plot No./': ''
            }
            
            row_text_parts = [line]
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j]
                if serial_pattern.match(next_line) and j > i + 1:
                    break
                row_text_parts.append(next_line)
            
            row_text = ' '.join(row_text_parts)
            
            plot_match = plot_no_pattern.search(row_text)
            if plot_match:
                current_row['Plot No./'] = plot_match.group(1).strip()
            else:
                plot_alt_pattern = re.compile(r'\b([0-9]+/[0-9]+|[A-Z]\d+[A-Z]?)\b')
                alt_match = plot_alt_pattern.search(line)
                if alt_match:
                    current_row['Plot No./'] = alt_match.group(1).strip()
            
            if current_row['Plot No./']:
                survey_match = survey_pattern.search(row_text)
                if survey_match:
                    current_row['Survey No./'] = survey_match.group(1).strip()
                
                doc_match = doc_pattern.search(row_text)
                if doc_match:
                    current_row['Document No.& Year'] = doc_match.group(1).strip()
                
                name_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b')
                name_matches = name_pattern.findall(row_text)
                
                valid_names = []
                for name in name_matches:
                    name_lower = name.lower()
                    if not any(keyword in name_lower for keyword in ['plot', 'survey', 'document', 'no', 'year']):
                        if len(name.split()) >= 2:
                            valid_names.append(name)
                
                if len(valid_names) >= 1:
                    current_row['Name of Executant(s)'] = valid_names[0]
                if len(valid_names) >= 2:
                    current_row['Name of Claimant(s)'] = valid_names[1]
                
                rows.append(current_row)
            
            i += 1
        else:
            i += 1
    
    return rows


def extract_data_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """Main function to extract data from a PDF file."""
    filename = os.path.basename(pdf_path)
    
    text, structured_data = extract_text_from_pdf_cpu(pdf_path)
    
    # Save OCR text for debugging
    debug_file = pdf_path.replace('.pdf', '_ocr_text.txt')
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\n💾 Saved OCR text to: {debug_file}\n")
    
    # Parse table rows
    print("🔍 Parsing table data...")
    rows = parse_table_rows(text)
    
    for row in rows:
        row['filename'] = filename
    
    filtered_rows = [row for row in rows if row.get('Plot No./', '').strip()]
    print(f"✓ Found {len(filtered_rows)} rows with Plot No./ information\n")
    
    return filtered_rows


def main():
    """Main function."""
    pdf_file = "ec/RG EC 103 3.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ Error: File {pdf_file} not found!")
        return
    
    print("=" * 70)
    print(" " * 20 + "EC Data Extraction (CPU Mode)")
    print("=" * 70)
    print()
    
    try:
        rows = extract_data_from_pdf(pdf_file)
        
        if not rows:
            print("⚠️  No rows with Plot No./ information found.")
            return
        
        df = pd.DataFrame(rows)
        columns_order = ['filename', 'Sr.No', 'Document No.& Year', 
                        'Name of Executant(s)', 'Name of Claimant(s)', 
                        'Survey No./', 'Plot No./']
        df = df[columns_order]
        
        print("=" * 70)
        print(" " * 25 + "Extracted Data")
        print("=" * 70)
        print()
        print(df.to_string(index=False))
        print()
        
        output_file = pdf_file.replace('.pdf', '_extracted.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"💾 Data saved to: {output_file}")
        
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
        traceback.print_exc()


if __name__ == "__main__":
    main()

