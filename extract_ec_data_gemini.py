#!/usr/bin/env python3
"""
Script to extract data from EC (Encumbrance Certificate) PDF files using Google Gemini AI.
Extracts: filename, Sr.No, Document No.& Year, Name of Executant(s), 
Name of Claimant(s), Survey No., Plot No.
Only processes rows that have Plot No. information.
"""

import os
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not installed.")
    print("Please install it using: pip install google-generativeai")
    sys.exit(1)


def setup_gemini_client() -> genai.GenerativeModel:
    """
    Setup Google Gemini API client.
    
    Requires GEMINI_API_KEY environment variable.
    
    Returns:
        Configured Gemini model instance
    """
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("=" * 70)
        print("❌ Error: GEMINI_API_KEY environment variable not set!")
        print("=" * 70)
        print("\nTo get your API key:")
        print("1. Go to https://aistudio.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Set it as an environment variable:")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        print("\nOr create a .env file with:")
        print("   GEMINI_API_KEY=your-api-key-here")
        print()
        sys.exit(1)
    
    try:
        genai.configure(api_key=api_key)
        
        # Try different models: gemini-2.5-flash (tested and working)
        # According to https://ai.google.dev/api/models, use baseModelId without "models/" prefix
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        
        # Remove models/ prefix if present - Python SDK uses baseModelId format
        model_name_clean = model_name.replace('models/', '')
        
        try:
            # Python SDK expects baseModelId format (e.g., "gemini-2.0-flash") not "models/gemini-2.0-flash"
            model = genai.GenerativeModel(model_name_clean)
            
            # Store model name for later reference
            model._model_name = model_name_clean
            print(f"✅ Gemini API configured (using model: {model_name_clean})")
            return model
        except Exception as e:
            print(f"⚠️  Model {model_name_clean} not available: {e}")
            # Try alternative flash models (try 2.5-flash first, then 2.0-flash, then vision)
            fallback_models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-pro-vision']
            for fallback_model in fallback_models:
                if fallback_model == model_name_clean:
                    continue  # Skip if it's the same as what we already tried
                try:
                    print(f"   Trying {fallback_model} as fallback...")
                    # Python SDK expects baseModelId format without "models/" prefix
                    fallback_clean = fallback_model.replace('models/', '')
                    model = genai.GenerativeModel(fallback_clean)
                    # Store model name for later reference
                    model._model_name = fallback_clean
                    print(f"✅ Gemini API configured (using model: {fallback_clean})")
                    return model
                except Exception as e2:
                    print(f"   {fallback_model} also not available: {e2}")
                    continue
            raise Exception("No available Gemini models found. Please check your API key and model availability.")
    except Exception as e:
        print(f"❌ Error setting up Gemini API: {e}")
        sys.exit(1)


def create_lenient_extraction_prompt() -> str:
    """
    Create a more lenient prompt for extracting EC data when initial extraction returns no rows.
    Allows some fields to be missing but still requires Plot No.
    
    Returns:
        Lenient prompt string for Gemini AI
    """
    prompt = """You are an expert at extracting structured data from EC (Encumbrance Certificate) documents.

This is a SECOND PASS with more lenient rules. Extract rows even if one or two fields are missing, BUT Plot Number field MUST be present and filled.

Analyze the PDF document and extract table rows. Use fuzzy matching to identify columns/fields - headers may have variations in spelling, spacing, punctuation, or language.

ONLY extract rows where the Plot Number field has a value (is NOT empty). This is REQUIRED - Plot Number must be present.

Use fuzzy matching to identify these fields in the table:
1. Serial Number / Sr.No / Sr. No. / Sr No / Serial No - Look for serial number column (usually first column, numeric)
2. Document No.& Year / Document Number / Document No / Doc No / Document - Look for document number and year column
3. Name of Executant(s) / Executant / Executants / Name of Executant - Look for executant names column
4. Name of Claimant(s) / Claimant / Claimants / Name of Claimant - Look for claimant names column
5. Survey No. / Survey No / Survey Number / Survey - Look for survey number column
6. Plot No. / Plot No / Plot Number / Plot / Plot No./ - Look for plot number column (MANDATORY - only include rows where this has a value)

LENIENT EXTRACTION RULES (Second Pass):
- Extract rows even if 1-2 fields are missing (e.g., Serial No, Document No., Executant, Claimant, or Survey No. can be missing)
- BUT Plot Number field is MANDATORY - DO NOT include rows without Plot Number
- If a row has Plot Number but is missing other fields, STILL include it and use empty string "" for missing fields
- Use intelligent fuzzy matching to map table columns to these 6 fields
- Be more flexible - if a field name doesn't match exactly, try to find similar columns
- Preserve the exact text as it appears in the document, including Tamil/regional language characters
- If a field contains newlines or multiple items, preserve them as-is (use \n for newlines in JSON strings)
- Extract data exactly as it appears - do not modify or summarize
- Return the data as a JSON array of objects
- Each object MUST have these exact keys: "Sr.No", "Document No.& Year", "Name of Executant(s)", "Name of Claimant(s)", "Survey No.", "Plot No."
- If any field is missing or cannot be found (except Plot No. which is required), use an empty string "" for that field

Return ONLY valid JSON array, no additional text, no explanations, no markdown code blocks. The format should be:
[
  {
    "Sr.No": "1",
    "Document No.& Year": "",
    "Name of Executant(s)": "",
    "Name of Claimant(s)": "1. ரகுராமன்",
    "Survey No.": "",
    "Plot No.": "18"
  },
  ...
]
"""
    return prompt


def parse_json_response(response_text: str) -> List[Dict]:
    """
    Robustly parse JSON response with multiple fallback strategies.
    
    Args:
        response_text: Raw response text from AI
        
    Returns:
        Parsed JSON data as a list
    """
    # Clean up response text
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    
    # Strategy 1: Try direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Try to find JSON array in the text
    import re
    json_match = re.search(r'\[[\s\S]*\]', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Try to find multiple JSON objects and combine them
    json_objects = re.findall(r'\{[^{}]*\}', text, re.DOTALL)
    if json_objects:
        parsed_objects = []
        for obj_str in json_objects:
            try:
                # Try to balance braces if needed
                if obj_str.count('{') != obj_str.count('}'):
                    continue
                parsed_objects.append(json.loads(obj_str))
            except json.JSONDecodeError:
                continue
        if parsed_objects:
            return parsed_objects
    
    # Strategy 4: Try to fix common JSON issues
    # Fix unescaped newlines in strings
    fixed_text = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    # Fix single quotes to double quotes
    fixed_text = fixed_text.replace("'", '"')
    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Try to extract just the JSON array content and reconstruct
    # Remove any text before first [ and after last ]
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_content = text[start_idx:end_idx + 1]
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass
    
    # If all strategies fail, return empty list
    print(f"⚠️  Could not parse JSON. Response preview: {text[:200]}...")
    return []


def create_extraction_prompt() -> str:
    """
    Create the prompt for extracting EC data from PDF.
    Simplified: AI identifies fields, returns JSON, only Plot No. required.
    
    Returns:
        Prompt string for Gemini AI
    """
    prompt = """Extract data from this EC (Encumbrance Certificate) document.

TASK: Identify table columns and extract rows where Plot Number has a value.

REQUIREMENTS:
1. Find all table rows in the document
2. Identify columns automatically (headers may vary in spelling/format/language)
3. Extract these fields: Serial Number, Document Number & Year, Executant Name(s), Claimant Name(s), Survey Number, Plot Number
4. ONLY include rows where Plot Number field has a value (not empty)
5. Other fields can be empty if not found

RETURN FORMAT: Valid JSON array only, no extra text.

JSON Structure:
- Each object must have these keys: "Sr.No", "Document No.& Year", "Name of Executant(s)", "Name of Claimant(s)", "Survey No.", "Plot No."
- Use empty string "" for missing fields
- Preserve exact text including Tamil/regional characters
- Use \\n for line breaks within fields

Example:
[
  {"Sr.No": "1", "Document No.& Year": "1439/2005", "Name of Executant(s)": "Name1", "Name of Claimant(s)": "Name2", "Survey No.": "103/3", "Plot No.": "18"},
  {"Sr.No": "2", "Document No.& Year": "", "Name of Executant(s)": "", "Name of Claimant(s)": "Name3", "Survey No.": "", "Plot No.": "20"}
]

Return ONLY the JSON array, nothing else."""
    return prompt


def should_skip_pdf(pdf_path: str) -> bool:
    """
    Check if PDF already has output files and should be skipped.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        True if output files exist (should skip), False otherwise
    """
    csv_output = pdf_path.replace('.pdf', '_extracted_gemini.csv')
    excel_output = pdf_path.replace('.pdf', '_extracted_gemini.xlsx')
    
    # Skip if either CSV or Excel output exists
    return os.path.exists(csv_output) or os.path.exists(excel_output)


def extract_data_from_pdf_gemini(pdf_path: str, model: genai.GenerativeModel, max_retries: int = 3, model_name: str = None) -> List[Dict[str, str]]:
    """
    Extract data from PDF using Google Gemini AI with retry logic.
    
    Args:
        pdf_path: Path to the PDF file
        model: Configured Gemini model instance
        max_retries: Maximum number of retry attempts for API calls
        model_name: Name of the model being used (for fallback checks)
        
    Returns:
        List of dictionaries with extracted row data
    """
    filename = os.path.basename(pdf_path)
    current_model_name = model_name or getattr(model, '_model_name', 'unknown')
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"📄 Processing PDF: {filename}")
    print("🤖 Using Google Gemini AI for extraction...")
    print()
    
    pdf_file = None
    
    try:
        # Upload PDF file to Gemini API with retry logic
        print("📤 Uploading PDF to Gemini API...")
        for retry in range(max_retries):
            try:
                pdf_file = genai.upload_file(path=pdf_path)
                print(f"✅ PDF uploaded (file URI: {pdf_file.uri})")
                break
            except Exception as e:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                    print(f"   ⚠️  Upload failed (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"   Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to upload PDF after {max_retries} attempts: {e}")
        
        # Wait for file to be processed (if needed)
        max_wait_time = 60  # Maximum 60 seconds wait
        wait_interval = 2
        elapsed_time = 0
        
        while pdf_file.state.name == "PROCESSING" and elapsed_time < max_wait_time:
            print("   Waiting for file to be processed...")
            time.sleep(wait_interval)
            elapsed_time += wait_interval
            pdf_file = genai.get_file(pdf_file.name)
        
        if pdf_file.state.name == "PROCESSING":
            raise Exception(f"File processing timeout after {max_wait_time} seconds")
        
        if pdf_file.state.name == "FAILED":
            raise Exception("PDF file upload failed")
        
        # Create extraction prompt (standard)
        prompt = create_extraction_prompt()
        
        # Submit to Gemini with retry logic
        print("🔍 Extracting data with AI...")
        print("   (This may take 30-60 seconds)")
        
        response = None
        for retry in range(max_retries):
            try:
                # Try generating with explicit model specification for file uploads
                # Some models require explicit model name when using file uploads
                try:
                    response = model.generate_content(
                        [prompt, pdf_file],
                        generation_config={
                            "temperature": 0.1,  # Low temperature for consistent, structured output
                            "response_mime_type": "application/json",  # Request JSON response
                        }
                    )
                except Exception as gen_error:
                    # If error is about model format, try recreating model with explicit name
                    if "model name format" in str(gen_error).lower() or "400" in str(gen_error):
                        print(f"   ⚠️  Model format issue detected, trying alternative approach...")
                        # Recreate model instance
                        model_name_clean = getattr(model, '_model_name', 'gemini-2.0-flash')
                        model = genai.GenerativeModel(model_name_clean)
                        response = model.generate_content(
                            [prompt, pdf_file],
                            generation_config={
                                "temperature": 0.1,
                                "response_mime_type": "application/json",
                            }
                        )
                    else:
                        raise gen_error
                break
            except Exception as e:
                error_msg = str(e)
                
                # Check for rate limiting
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    if retry < max_retries - 1:
                        wait_time = (retry + 1) * 10  # Longer wait for rate limits: 10s, 20s, 30s
                        print(f"   ⚠️  Rate limit hit (attempt {retry + 1}/{max_retries})")
                        print(f"   Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                
                # Check for model not found
                if "not found" in error_msg or "not supported" in error_msg:
                    print(f"⚠️  Error: {error_msg}")
                    print("\nTrying to list available models...")
                    try:
                        models = genai.list_models()
                        print("Available models:")
                        for m in models:
                            if 'generateContent' in m.supported_generation_methods:
                                print(f"  - {m.name}")
                        print("\n💡 Try setting GEMINI_MODEL environment variable to one of the above models.")
                    except:
                        pass
                
                if retry == max_retries - 1:
                    raise Exception(f"Failed to extract data after {max_retries} attempts: {e}")
                
                wait_time = (retry + 1) * 5
                print(f"   ⚠️  Extraction failed (attempt {retry + 1}/{max_retries}): {e}")
                print(f"   Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        if response is None:
            raise Exception("Failed to get response from Gemini API after all retries")
        
        # Extract and parse JSON from response using robust parser
        response_text = response.text.strip()
        extracted_data = parse_json_response(response_text)
        
        # Validate data format
        if not isinstance(extracted_data, list):
            if isinstance(extracted_data, dict):
                print("⚠️  Warning: Expected list, got dict. Wrapping in list.")
                extracted_data = [extracted_data]
            else:
                print("⚠️  Warning: Unexpected data type, using empty list.")
                extracted_data = []
        
        # Ensure all rows have required fields and add filename
        formatted_rows = []
        for row in extracted_data:
            if not isinstance(row, dict):
                continue
            
            formatted_row = {
                'filename': filename,
                'Sr.No': str(row.get('Sr.No', '')).strip(),
                'Document No.& Year': str(row.get('Document No.& Year', '')).strip(),
                'Name of Executant(s)': str(row.get('Name of Executant(s)', '')).strip(),
                'Name of Claimant(s)': str(row.get('Name of Claimant(s)', '')).strip(),
                'Survey No.': str(row.get('Survey No.', row.get('Survey No./', ''))).strip(),
                'Plot No.': str(row.get('Plot No.', row.get('Plot No./', ''))).strip(),
            }
            
            # Only include rows with Plot No. filled (non-empty)
            if formatted_row['Plot No.'].strip():
                formatted_rows.append(formatted_row)
        
        # Double verification: If no rows found, try with lenient prompt
        if len(formatted_rows) == 0:
            print("⚠️  No rows found with standard extraction.")
            print("🔄 Attempting double verification with lenient rules...")
            print("   (Allowing rows with 1-2 missing fields, but Plot No. is still required)")
            print()
            
            # Use lenient prompt for retry
            lenient_prompt = create_lenient_extraction_prompt()
            
            try:
                print("🔍 Re-extracting with lenient rules...")
                lenient_response = model.generate_content(
                    [lenient_prompt, pdf_file],
                    generation_config={
                        "temperature": 0.2,  # Slightly higher temperature for more flexible extraction
                        "response_mime_type": "application/json",
                    }
                )
                
                # Extract and parse JSON from lenient response using robust parser
                lenient_response_text = lenient_response.text.strip()
                lenient_extracted_data = parse_json_response(lenient_response_text)
                
                # Validate lenient data format
                if not isinstance(lenient_extracted_data, list):
                    if isinstance(lenient_extracted_data, dict):
                        lenient_extracted_data = [lenient_extracted_data]
                    else:
                        lenient_extracted_data = []
                
                # Process lenient results
                lenient_rows = []
                for row in lenient_extracted_data:
                    if not isinstance(row, dict):
                        continue
                    
                    formatted_row = {
                        'filename': filename,
                        'Sr.No': str(row.get('Sr.No', '')).strip(),
                        'Document No.& Year': str(row.get('Document No.& Year', '')).strip(),
                        'Name of Executant(s)': str(row.get('Name of Executant(s)', '')).strip(),
                        'Name of Claimant(s)': str(row.get('Name of Claimant(s)', '')).strip(),
                        'Survey No.': str(row.get('Survey No.', row.get('Survey No./', ''))).strip(),
                        'Plot No.': str(row.get('Plot No.', row.get('Plot No./', ''))).strip(),
                    }
                    
                    # Only include rows with Plot No. filled (non-empty)
                    if formatted_row['Plot No.'].strip():
                        lenient_rows.append(formatted_row)
                
                if lenient_rows:
                    print(f"✅ Lenient extraction found {len(lenient_rows)} rows with Plot No. information")
                    formatted_rows = lenient_rows
                else:
                    print("⚠️  Lenient extraction also found no rows with Plot No. information")
                    # Final fallback: Try with Gemini Vision Pro model
                    if current_model_name != 'gemini-pro-vision':
                        print()
                        print("🔄 Attempting final fallback with Gemini Vision Pro model...")
                        try:
                            vision_model = genai.GenerativeModel('gemini-pro-vision')
                            print("   Using gemini-pro-vision for extraction...")
                            
                            vision_response = vision_model.generate_content(
                                [lenient_prompt, pdf_file],
                                generation_config={
                                    "temperature": 0.3,
                                    "response_mime_type": "application/json",
                                }
                            )
                            
                            # Extract and parse JSON from vision model response
                            vision_response_text = vision_response.text.strip()
                            vision_extracted_data = parse_json_response(vision_response_text)
                            
                            # Validate vision data format
                            if not isinstance(vision_extracted_data, list):
                                if isinstance(vision_extracted_data, dict):
                                    vision_extracted_data = [vision_extracted_data]
                                else:
                                    vision_extracted_data = []
                            
                            # Process vision model results
                            vision_rows = []
                            for row in vision_extracted_data:
                                if not isinstance(row, dict):
                                    continue
                                
                                formatted_row = {
                                    'filename': filename,
                                    'Sr.No': str(row.get('Sr.No', '')).strip(),
                                    'Document No.& Year': str(row.get('Document No.& Year', '')).strip(),
                                    'Name of Executant(s)': str(row.get('Name of Executant(s)', '')).strip(),
                                    'Name of Claimant(s)': str(row.get('Name of Claimant(s)', '')).strip(),
                                    'Survey No.': str(row.get('Survey No.', row.get('Survey No./', ''))).strip(),
                                    'Plot No.': str(row.get('Plot No.', row.get('Plot No./', ''))).strip(),
                                }
                                
                                # Only include rows with Plot No. filled (non-empty)
                                if formatted_row['Plot No.'].strip():
                                    vision_rows.append(formatted_row)
                            
                            if vision_rows:
                                print(f"✅ Vision Pro model found {len(vision_rows)} rows with Plot No. information")
                                formatted_rows = vision_rows
                            else:
                                print("⚠️  Vision Pro model also found no rows with Plot No. information")
                        except Exception as vision_error:
                            print(f"⚠️  Vision Pro model extraction failed: {vision_error}")
            except Exception as e:
                print(f"⚠️  Lenient extraction failed: {e}")
                print("   Continuing with empty results from first extraction...")
        
        # Clean up uploaded file after processing
        if pdf_file:
            try:
                genai.delete_file(pdf_file.name)
                print("🧹 Cleaned up uploaded file")
            except Exception as e:
                print(f"   Note: Could not delete uploaded file: {e}")
        
        print(f"✅ Extracted {len(formatted_rows)} rows with Plot No. information")
        print()
        
        return formatted_rows
        
    except Exception as e:
        print(f"❌ Error processing PDF {filename}: {e}")
        # Clean up file if it was uploaded
        if pdf_file:
            try:
                genai.delete_file(pdf_file.name)
            except:
                pass
        # Don't re-raise, let the caller handle it
        raise Exception(f"Failed to process {filename}: {e}")


def process_batch(pdf_files: List[str], model: genai.GenerativeModel, delay_between_files: int = 5, model_name: str = None) -> Dict[str, List[Dict[str, str]]]:
    """
    Process multiple PDF files in batch with delays to avoid rate limiting.
    
    Args:
        pdf_files: List of PDF file paths
        model: Configured Gemini model instance
        delay_between_files: Seconds to wait between processing files (default: 5)
        
    Returns:
        Dictionary mapping filename to list of extracted rows
    """
    all_results = {}
    total_files = len(pdf_files)
    successful = 0
    failed = 0
    
    print(f"📦 Processing {total_files} PDF file(s) in batch...")
    print(f"⏱️  Delay between files: {delay_between_files} seconds (to avoid rate limits)")
    print("=" * 70)
    print()
    
    for i, pdf_file in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_file)
        print(f"[{i}/{total_files}] Processing: {filename}")
        print("-" * 70)
        
        if not os.path.exists(pdf_file):
            print(f"❌ Error: File {pdf_file} not found! Skipping...")
            print()
            failed += 1
            all_results[filename] = []
            continue
        
        # Check if output files already exist
        if should_skip_pdf(pdf_file):
            csv_file = pdf_file.replace('.pdf', '_extracted_gemini.csv')
            excel_file = pdf_file.replace('.pdf', '_extracted_gemini.xlsx')
            existing_files = []
            if os.path.exists(csv_file):
                existing_files.append(os.path.basename(csv_file))
            if os.path.exists(excel_file):
                existing_files.append(os.path.basename(excel_file))
            print(f"⏭️  Skipping {filename} - output files already exist: {', '.join(existing_files)}")
            print()
            # Load existing data instead of processing
            try:
                if os.path.exists(csv_file):
                    df_existing = pd.read_csv(csv_file)
                    existing_rows = df_existing.to_dict('records')
                    all_results[filename] = existing_rows
                    successful += 1
                    print(f"   📂 Loaded {len(existing_rows)} rows from existing output")
                    print()
                else:
                    all_results[filename] = []
            except Exception as e:
                print(f"   ⚠️  Could not load existing output: {e}")
                all_results[filename] = []
            continue
        
        try:
            rows = extract_data_from_pdf_gemini(pdf_file, model, model_name=model_name)
            
            # Ensure filename is correctly set in all rows
            for row in rows:
                if 'filename' not in row or row['filename'] != filename:
                    row['filename'] = filename
            
            all_results[filename] = rows
            successful += 1
            print(f"✅ Completed: {filename} ({len(rows)} rows extracted)")
            print()
            
            # Save individual file results immediately (checkpoint)
            if rows:
                df = pd.DataFrame(rows)
                columns_order = ['filename', 'Sr.No', 'Document No.& Year', 
                                'Name of Executant(s)', 'Name of Claimant(s)', 
                                'Survey No.', 'Plot No.']
                df = df[columns_order]
                
                output_csv = pdf_file.replace('.pdf', '_extracted_gemini.csv')
                output_excel = pdf_file.replace('.pdf', '_extracted_gemini.xlsx')
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
                
                df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                df.to_excel(output_excel, index=False, engine='openpyxl')
                print(f"   💾 Individual files saved: {output_csv}, {output_excel}")
                print()
                
        except Exception as e:
            failed += 1
            print(f"❌ Error processing {filename}: {e}")
            print("   Continuing with next file...")
            print()
            all_results[filename] = []
            # Continue processing other files even if one fails
            continue
        
        # Add delay between files to avoid rate limiting (except after last file)
        if i < total_files:
            print(f"⏳ Waiting {delay_between_files} seconds before next file...")
            print()
            time.sleep(delay_between_files)
    
    print("=" * 70)
    print(f"📊 Batch Processing Summary: {successful} successful, {failed} failed")
    print("=" * 70)
    print()
    
    return all_results


def process_directory(directory: str, model: genai.GenerativeModel, model_name: str, delay_between_files: int) -> Dict[str, List[Dict[str, str]]]:
    """
    Process all PDFs in a directory.
    
    Args:
        directory: Directory name (e.g., 'ec3', 'ec2', 'ec')
        model: Configured Gemini model instance
        model_name: Name of the model being used
        delay_between_files: Delay between processing files
        
    Returns:
        Dictionary mapping filename to list of extracted rows
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"⚠️  Directory {directory}/ not found, skipping...")
        return {}
    
    # Get all PDF files in directory
    pdf_files = sorted([str(f) for f in dir_path.glob("*.pdf")])
    
    if not pdf_files:
        print(f"📁 No PDF files found in {directory}/ directory")
        return {}
    
    print("=" * 70)
    print(f"📁 Processing directory: {directory}/")
    print("=" * 70)
    print()
    
    # Filter out PDFs that already have outputs
    pdfs_to_process = []
    skipped_count = 0
    
    for pdf_file in pdf_files:
        if should_skip_pdf(pdf_file):
            skipped_count += 1
        else:
            pdfs_to_process.append(pdf_file)
    
    if skipped_count > 0:
        print(f"⏭️  Skipping {skipped_count} PDF(s) with existing outputs")
        print()
    
    if not pdfs_to_process:
        print(f"✅ All PDFs in {directory}/ already have output files")
        print()
        # Load existing outputs for combination
        all_existing_rows = {}
        for pdf_file in pdf_files:
            csv_file = pdf_file.replace('.pdf', '_extracted_gemini.csv')
            if os.path.exists(csv_file):
                try:
                    df_existing = pd.read_csv(csv_file)
                    filename = os.path.basename(pdf_file)
                    all_existing_rows[filename] = df_existing.to_dict('records')
                except:
                    pass
        return all_existing_rows
    
    print(f"📋 Found {len(pdfs_to_process)} PDF(s) to process (skipped {skipped_count} with existing outputs)")
    print()
    
    # Process PDFs
    all_results = process_batch(pdfs_to_process, model, delay_between_files=delay_between_files, model_name=model_name)
    
    # Also load existing outputs for skipped files
    for pdf_file in pdf_files:
        if should_skip_pdf(pdf_file):
            csv_file = pdf_file.replace('.pdf', '_extracted_gemini.csv')
            filename = os.path.basename(pdf_file)
            if filename not in all_results:
                try:
                    if os.path.exists(csv_file):
                        df_existing = pd.read_csv(csv_file)
                        all_results[filename] = df_existing.to_dict('records')
                except:
                    all_results[filename] = []
    
    return all_results


def main():
    """Main function to run the extraction script with batch processing."""
    # Process directories in order: ec3, ec2, ec
    directories = ["ec3", "ec2", "ec"]
    
    print("=" * 70)
    print(" " * 10 + "EC Data Extraction - Batch Processing (Google Gemini AI)")
    print("=" * 70)
    print()
    print(f"📂 Processing directories: {', '.join(directories)}")
    print()
    
    try:
        # Setup Gemini client (once for all directories)
        model = setup_gemini_client()
        model_name = getattr(model, '_model_name', None)
        print()
        
        # Delay between files (can be adjusted via environment variable)
        delay_between_files = int(os.getenv('GEMINI_BATCH_DELAY', '5'))
        
        # Process each directory sequentially
        all_directory_results = {}
        
        for directory in directories:
            dir_results = process_directory(directory, model, model_name, delay_between_files)
            all_directory_results[directory] = dir_results
            
            # Create combined output for this directory
            if dir_results:
                all_rows = []
                for filename, rows in dir_results.items():
                    for row in rows:
                        row['filename'] = filename  # Ensure correct filename mapping
                    all_rows.extend(rows)
                
                if all_rows:
                    df_all = pd.DataFrame(all_rows)
                    columns_order = ['filename', 'Sr.No', 'Document No.& Year', 
                                    'Name of Executant(s)', 'Name of Claimant(s)', 
                                    'Survey No.', 'Plot No.']
                    df_all = df_all[columns_order]
                    
                    # Save combined output in the directory
                    combined_excel = f"{directory}/{directory}_combine_output.xlsx"
                    os.makedirs(os.path.dirname(combined_excel) or '.', exist_ok=True)
                    
                    df_all.to_excel(combined_excel, index=False, engine='openpyxl')
                    print(f"💾 Combined output saved: {combined_excel}")
                    print(f"   Total rows: {len(all_rows)}")
                    print()
            
            # Add delay between directories
            if directory != directories[-1]:
                print(f"⏳ Waiting {delay_between_files} seconds before next directory...")
                print()
                time.sleep(delay_between_files)
        
        # Overall summary
        print("=" * 70)
        print(" " * 25 + "Overall Summary")
        print("=" * 70)
        print()
        
        total_processed = 0
        for directory, dir_results in all_directory_results.items():
            dir_total = sum(len(rows) for rows in dir_results.values())
            total_processed += dir_total
            print(f"  {directory}/: {dir_total} rows from {len(dir_results)} files")
        
        print()
        print(f"📊 Total across all directories: {total_processed} rows")
        print()
        print("=" * 70)
        print("✅ Batch extraction complete for all directories!")
        print("=" * 70)
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ Error occurred during batch extraction")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

