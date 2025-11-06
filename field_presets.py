#!/usr/bin/env python3
"""
Field presets configuration module for PDF data extraction.

This module provides field preset definitions and management utilities.
Presets allow users to quickly select predefined field configurations for
common document types, while also supporting custom field definitions.
"""

from typing import Any

# Field preset definitions
PRESETS: dict[str, dict[str, Any]] = {
    "encumbrance": {
        "name": "Encumbrance Certificate",
        "fields": [
            "Sr.No",
            "Document No.& Year",
            "Name of Executant(s)",
            "Name of Claimant(s)",
            "Survey No.",
            "Plot No.",
        ],
        "required_fields": ["Plot No."],
        "description": "Standard fields for Tamil Nadu Encumbrance Certificates",
    },
    "invoice": {
        "name": "Invoice Document",
        "fields": [
            "Invoice Number",
            "Date",
            "Vendor",
            "Bill To",
            "Amount",
            "Tax",
            "Total",
            "Due Date",
            "Payment Terms",
        ],
        "required_fields": ["Invoice Number", "Amount"],
        "description": "Standard fields for invoice extraction",
    },
    "receipt": {
        "name": "Receipt Document",
        "fields": [
            "Receipt Number",
            "Date",
            "Merchant",
            "Items",
            "Quantity",
            "Unit Price",
            "Total Amount",
            "Payment Method",
        ],
        "required_fields": ["Receipt Number"],
        "description": "Standard fields for receipt extraction",
    },
}


def get_field_preset(preset_name: str) -> dict[str, Any] | None:
    """
    Get field preset configuration by name.

    Args:
        preset_name: Name of the preset (e.g., "encumbrance")

    Returns:
        Dictionary containing preset configuration, or None if not found
        Structure: {
            "name": str,
            "fields": list[str],
            "required_fields": list[str],
            "description": str
        }
    """
    return PRESETS.get(preset_name.lower())


def get_available_presets() -> dict[str, dict[str, Any]]:
    """
    Get all available field presets.

    Returns:
        Dictionary mapping preset names to preset configurations
    """
    return PRESETS.copy()


def register_field_preset(
    preset_name: str,
    fields: list[str],
    required_fields: list[str] | None = None,
    name: str | None = None,
    description: str = "",
) -> None:
    """
    Register a new field preset or update an existing one.

    Args:
        preset_name: Unique identifier for the preset (case-insensitive)
        fields: List of field names to extract
        required_fields: List of fields that must have values (default: empty list)
        name: Display name for the preset (default: preset_name)
        description: Description of what this preset is for
    """
    preset_name_lower = preset_name.lower()

    if required_fields is None:
        required_fields = []

    PRESETS[preset_name_lower] = {
        "name": name or preset_name,
        "fields": fields,
        "required_fields": required_fields,
        "description": description,
    }


def get_preset_fields(preset_name: str) -> list[str] | None:
    """
    Get the field list for a preset.

    Args:
        preset_name: Name of the preset

    Returns:
        List of field names, or None if preset not found
    """
    preset = get_field_preset(preset_name)
    return preset.get("fields") if preset else None


def get_preset_required_fields(preset_name: str) -> list[str] | None:
    """
    Get the required fields list for a preset.

    Args:
        preset_name: Name of the preset

    Returns:
        List of required field names, or None if preset not found
    """
    preset = get_field_preset(preset_name)
    return preset.get("required_fields") if preset else None


def list_preset_names() -> list[str]:
    """
    Get list of all available preset names.

    Returns:
        List of preset names
    """
    return list(PRESETS.keys())


def preset_exists(preset_name: str) -> bool:
    """
    Check if a preset exists.

    Args:
        preset_name: Name of the preset to check

    Returns:
        True if preset exists, False otherwise
    """
    return preset_name.lower() in PRESETS

