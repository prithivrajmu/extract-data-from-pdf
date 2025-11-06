#!/usr/bin/env python3
"""
Utility functions for generating extraction prompts with custom fields.
"""


def create_custom_extraction_prompt(
    custom_fields: list[str] | None = None,
    required_fields: list[str] | None = None,
    preset_name: str = "encumbrance",
) -> str:
    """
    Create extraction prompt with custom fields or preset fields.

    Args:
        custom_fields: List of custom field names to extract. If None, uses preset.
        required_fields: List of fields that must have values. If None, uses preset defaults.
        preset_name: Preset to use when custom_fields is None (default: "encumbrance")

    Returns:
        Prompt string for AI extraction
    """
    # Normalize field names
    if custom_fields:
        normalized_fields = [f.strip() for f in custom_fields if f.strip()]
    else:
        # Use preset fields
        from field_presets import get_preset_fields

        preset_fields = get_preset_fields(preset_name)
        if preset_fields:
            normalized_fields = preset_fields
        else:
            # Fallback to encumbrance fields
            normalized_fields = [
                "Sr.No",
                "Document No.& Year",
                "Name of Executant(s)",
                "Name of Claimant(s)",
                "Survey No.",
                "Plot No.",
            ]

    # Get required fields from preset if not provided
    if required_fields is None:
        from field_presets import get_preset_required_fields

        required_fields = get_preset_required_fields(preset_name) or []
        # If still empty, use first field or common required fields
        if not required_fields and normalized_fields:
            # Default: no required fields unless specified
            required_fields = []

    # Build field list description
    field_list = "\n".join(
        [f"{i+1}. {field}" for i, field in enumerate(normalized_fields)]
    )

    # Build required fields description
    required_list = ", ".join(required_fields)

    # Build JSON structure example
    json_example = "{\n"
    for field in normalized_fields:
        json_example += f'    "{field}": "{field} value",\n'
    json_example = json_example.rstrip(",\n") + "\n}"

    # Build requirements text for required fields
    if required_fields:
        required_text = f"4. ONLY include rows where required fields have values: {required_list}\n5. Other fields can be empty if not found"
    else:
        required_text = "4. Include all rows found in the document (no required fields specified)\n5. Fields can be empty if not found"

    prompt = f"""Extract data from this PDF document.

TASK: Identify table columns and extract rows based on the specified fields.

REQUIREMENTS:
1. Find all table rows in the document
2. Identify columns automatically (headers may vary in spelling/format/language)
3. Extract these fields:
{field_list}
{required_text}

RETURN FORMAT: Valid JSON array only, no extra text.

JSON Structure:
- Each object must have these keys: {', '.join([f'"{f}"' for f in normalized_fields])}
- Use empty string "" for missing fields
- Preserve exact text including regional characters
- Use \\n for line breaks within fields

Example:
[
  {json_example.replace('value', 'value1')},
  {json_example.replace('value', 'value2')}
]

Return ONLY the JSON array, nothing else."""

    return prompt


def create_lenient_custom_prompt(
    custom_fields: list[str] | None = None,
    required_fields: list[str] | None = None,
    preset_name: str = "encumbrance",
) -> str:
    """
    Create lenient extraction prompt with custom fields or preset fields (for retry attempts).

    Args:
        custom_fields: List of custom field names to extract. If None, uses preset.
        required_fields: List of fields that must have values. If None, uses preset defaults.
        preset_name: Preset to use when custom_fields is None (default: "encumbrance")

    Returns:
        Lenient prompt string for AI extraction
    """
    # Normalize field names
    if custom_fields:
        normalized_fields = [f.strip() for f in custom_fields if f.strip()]
    else:
        # Use preset fields
        from field_presets import get_preset_fields

        preset_fields = get_preset_fields(preset_name)
        if preset_fields:
            normalized_fields = preset_fields
        else:
            # Fallback to encumbrance fields
            normalized_fields = [
                "Sr.No",
                "Document No.& Year",
                "Name of Executant(s)",
                "Name of Claimant(s)",
                "Survey No.",
                "Plot No.",
            ]

    # Get required fields from preset if not provided
    if required_fields is None:
        from field_presets import get_preset_required_fields

        required_fields = get_preset_required_fields(preset_name) or []

    field_list = "\n".join(
        [
            f"{i+1}. {field} - Look for {field.lower()} column"
            for i, field in enumerate(normalized_fields)
        ]
    )
    required_list = ", ".join(required_fields)

    # Build lenient prompt with required fields handling
    if required_fields:
        required_instruction = f"ONLY extract rows where required fields have values (is NOT empty): {required_list}"
        lenient_rules = f"LENIENT EXTRACTION RULES (Second Pass):\n- Extract rows even if 1-2 optional fields are missing\n- BUT required fields ({required_list}) are MANDATORY - DO NOT include rows without these"
    else:
        required_instruction = "Extract all rows found in the document (no required fields specified)"
        lenient_rules = "LENIENT EXTRACTION RULES (Second Pass):\n- Extract rows even if 1-2 fields are missing\n- All fields are optional"

    prompt = f"""You are an expert at extracting structured data from PDF documents.

This is a SECOND PASS with more lenient rules. Extract rows even if some fields are missing, BUT required fields MUST be present and filled (if any specified).

Analyze the PDF document and extract table rows. Use fuzzy matching to identify columns/fields - headers may have variations in spelling, spacing, punctuation, or language.

{required_instruction}

Use fuzzy matching to identify these fields in the table:
{field_list}

{lenient_rules}
- If a row has required fields (or any fields if none required) but is missing other fields, STILL include it and use empty string "" for missing fields
- Use intelligent fuzzy matching to map table columns to the specified fields
- Be flexible - if a field name doesn't match exactly, try to find similar columns
- Preserve the exact text as it appears in the document, including regional language characters
- If a field contains newlines or multiple items, preserve them as-is (use \\n for newlines in JSON strings)
- Extract data exactly as it appears - do not modify or summarize
- Return the data as a JSON array of objects
- Each object MUST have these exact keys: {', '.join([f'"{f}"' for f in normalized_fields])}
- If any optional field is missing or cannot be found, use an empty string "" for that field

Return ONLY valid JSON array, no additional text, no explanations, no markdown code blocks."""

    return prompt
