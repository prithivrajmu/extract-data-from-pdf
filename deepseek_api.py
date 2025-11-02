#!/usr/bin/env python3
"""
Deepseek API integration for EC (Encumbrance Certificate) PDF extraction.
Similar structure to Gemini API integration.
"""

import os
import json
import base64
import time
from typing import List, Dict, Optional


def setup_deepseek_client(api_key: str, base_url: str = "https://api.deepseek.com"):
    """
    Setup Deepseek API client.

    Args:
        api_key: Deepseek API key
        base_url: API base URL (default: https://api.deepseek.com)

    Returns:
        Tuple of (api_key, base_url) for use in API calls
    """
    if not api_key:
        raise ValueError(
            "Deepseek API key is required. Get one from https://platform.deepseek.com"
        )

    return api_key, base_url


def create_extraction_prompt() -> str:
    """
    Create the prompt for extracting EC data from PDF.
    Similar to Gemini prompt structure.

    Returns:
        Prompt string for Deepseek AI
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


def create_lenient_extraction_prompt() -> str:
    """
    Create a more lenient prompt for extracting EC data.

    Returns:
        Lenient prompt string for Deepseek AI
    """
    prompt = """You are an expert at extracting structured data from EC (Encumbrance Certificate) documents.

This is a SECOND PASS with more lenient rules. Extract rows even if one or two fields are missing, BUT Plot Number field MUST be present and filled.

Analyze the PDF document and extract table rows. Use fuzzy matching to identify columns/fields - headers may have variations in spelling, spacing, punctuation, or language."""  # noqa: E501

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
    Same as Gemini implementation.

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

    json_match = re.search(r"\[[\s\S]*\]", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try to find multiple JSON objects and combine them
    json_objects = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    if json_objects:
        parsed_objects = []
        for obj_str in json_objects:
            try:
                # Try to balance braces if needed
                if obj_str.count("{") != obj_str.count("}"):
                    continue
                parsed_objects.append(json.loads(obj_str))
            except json.JSONDecodeError:
                continue
        if parsed_objects:
            return parsed_objects

    # Strategy 4: Try to fix common JSON issues
    fixed_text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    fixed_text = fixed_text.replace("'", '"')
    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass

    # Strategy 5: Try to extract just the JSON array content
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_content = text[start_idx : end_idx + 1]
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass

    # If all strategies fail, return empty list
    return []


def pdf_to_base64(pdf_path: str) -> str:
    """
    Convert PDF file to base64 string.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Base64 encoded string of PDF
    """
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        return base64.b64encode(pdf_bytes).decode("utf-8")


def extract_data_from_pdf_deepseek(
    pdf_path: str,
    api_key: str,
    custom_fields: Optional[List[str]] = None,
    max_retries: int = 3,
) -> List[Dict[str, str]]:
    """
    Extract data from PDF using Deepseek API with retry logic.

    Args:
        pdf_path: Path to the PDF file
        api_key: Deepseek API key
        custom_fields: Optional list of custom field names to extract
        max_retries: Maximum number of retry attempts for API calls

    Returns:
        List of dictionaries with extracted row data
    """
    import requests

    filename = os.path.basename(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Setup client
    api_key, base_url = setup_deepseek_client(api_key)

    # Convert PDF to base64
    try:
        _ = pdf_to_base64(pdf_path)  # PDF conversion validated
    except Exception as e:
        raise Exception(f"Failed to read PDF file: {e}")

    # Prepare the prompt with custom fields if provided
    if custom_fields:
        from prompt_utils import (
            create_custom_extraction_prompt,
            create_lenient_custom_prompt,
        )

        prompt = create_custom_extraction_prompt(custom_fields)

        def lenient_prompt_func():
            return create_lenient_custom_prompt(custom_fields)
    else:
        prompt = create_extraction_prompt()
        lenient_prompt_func = create_lenient_extraction_prompt

    # Prepare API request
    api_endpoint = f"{base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Deepseek supports vision, so we'll use the chat completions endpoint with images
    # Since PDFs need to be converted, we'll convert PDF pages to images first
    try:
        from pdf2image import convert_from_path
        import io

        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=200)

        # Convert first page (or all pages) to base64
        image_parts = []
        for i, image in enumerate(images[:10]):  # Limit to first 10 pages
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                }
            )

        # Create messages with images
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + image_parts,
            }
        ]

    except ImportError:
        # Fallback: use text-only if pdf2image not available
        raise Exception(
            "pdf2image library required for Deepseek PDF processing. Install with: pip install pdf2image"
        )
    except Exception as e:
        raise Exception(f"Failed to process PDF images: {e}")

    # Make API request with retry logic
    response = None
    for retry in range(max_retries):
        try:
            payload = {
                "model": "deepseek-chat",  # Deepseek chat model
                "messages": messages,
                "temperature": 0.1,
                "response_format": (
                    {"type": "json_object"} if "json" in prompt.lower() else None
                ),
            }

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            response = requests.post(
                api_endpoint, headers=headers, json=payload, timeout=120
            )

            # Check for rate limiting
            if response.status_code == 429:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 10
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception("Rate limit exceeded. Please try again later.")

            response.raise_for_status()
            break

        except requests.exceptions.RequestException as e:
            if retry == max_retries - 1:
                raise Exception(
                    f"Failed to call Deepseek API after {max_retries} attempts: {e}"
                )
            wait_time = (retry + 1) * 5
            time.sleep(wait_time)

    if response is None:
        raise Exception("Failed to get response from Deepseek API")

    # Parse response
    try:
        response_data = response.json()
        response_text = response_data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise Exception(f"Failed to parse Deepseek API response: {e}")

    # Parse JSON from response
    extracted_data = parse_json_response(response_text)

    # Validate data format
    if not isinstance(extracted_data, list):
        if isinstance(extracted_data, dict):
            extracted_data = [extracted_data]
        else:
            extracted_data = []

    # Format rows - dynamically handle fields
    formatted_rows = []
    for row in extracted_data:
        if not isinstance(row, dict):
            continue

        formatted_row = {"filename": filename}

        if custom_fields:
            # Use custom fields
            for field in custom_fields:
                formatted_row[field] = str(row.get(field, "")).strip()
        else:
            # Use default fields
            formatted_row.update(
                {
                    "Sr.No": str(row.get("Sr.No", "")).strip(),
                    "Document No.& Year": str(
                        row.get("Document No.& Year", "")
                    ).strip(),
                    "Name of Executant(s)": str(
                        row.get("Name of Executant(s)", "")
                    ).strip(),
                    "Name of Claimant(s)": str(
                        row.get("Name of Claimant(s)", "")
                    ).strip(),
                    "Survey No.": str(
                        row.get("Survey No.", row.get("Survey No./", ""))
                    ).strip(),
                    "Plot No.": str(
                        row.get("Plot No.", row.get("Plot No./", ""))
                    ).strip(),
                }
            )

        # Only include rows with at least one non-empty field (besides filename)
        non_empty_fields = [
            v for k, v in formatted_row.items() if k != "filename" and v.strip()
        ]
        if non_empty_fields:
            formatted_rows.append(formatted_row)

    # If no rows found, try with lenient prompt
    if len(formatted_rows) == 0:
        if custom_fields:
            lenient_prompt = lenient_prompt_func()
        else:
            lenient_prompt = create_lenient_extraction_prompt()

        messages_lenient = [
            {
                "role": "user",
                "content": [{"type": "text", "text": lenient_prompt}] + image_parts,
            }
        ]

        try:
            payload_lenient = {
                "model": "deepseek-chat",
                "messages": messages_lenient,
                "temperature": 0.2,
            }

            response_lenient = requests.post(
                api_endpoint, headers=headers, json=payload_lenient, timeout=120
            )
            response_lenient.raise_for_status()

            response_data_lenient = response_lenient.json()
            response_text_lenient = response_data_lenient["choices"][0]["message"][
                "content"
            ].strip()

            lenient_extracted_data = parse_json_response(response_text_lenient)

            if not isinstance(lenient_extracted_data, list):
                if isinstance(lenient_extracted_data, dict):
                    lenient_extracted_data = [lenient_extracted_data]
                else:
                    lenient_extracted_data = []

            lenient_rows = []
            for row in lenient_extracted_data:
                if not isinstance(row, dict):
                    continue

                formatted_row = {"filename": filename}

                if custom_fields:
                    for field in custom_fields:
                        formatted_row[field] = str(row.get(field, "")).strip()
                else:
                    formatted_row.update(
                        {
                            "Sr.No": str(row.get("Sr.No", "")).strip(),
                            "Document No.& Year": str(
                                row.get("Document No.& Year", "")
                            ).strip(),
                            "Name of Executant(s)": str(
                                row.get("Name of Executant(s)", "")
                            ).strip(),
                            "Name of Claimant(s)": str(
                                row.get("Name of Claimant(s)", "")
                            ).strip(),
                            "Survey No.": str(
                                row.get("Survey No.", row.get("Survey No./", ""))
                            ).strip(),
                            "Plot No.": str(
                                row.get("Plot No.", row.get("Plot No./", ""))
                            ).strip(),
                        }
                    )

                # Only include rows with at least one non-empty field
                non_empty_fields = [
                    v for k, v in formatted_row.items() if k != "filename" and v.strip()
                ]
                if non_empty_fields:
                    lenient_rows.append(formatted_row)

            if lenient_rows:
                formatted_rows = lenient_rows
        except Exception:
            # If lenient extraction fails, continue with empty results
            pass

    return formatted_rows
