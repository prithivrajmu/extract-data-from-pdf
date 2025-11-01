#!/usr/bin/env python3
"""
Helper utilities for file handling, field filtering, and data formatting.
"""

import os
import pandas as pd
from typing import List, Dict, Set, Tuple
from pathlib import Path


def filter_fields(data: List[Dict], selected_fields: Set[str]) -> List[Dict]:
    """
    Filter dictionary list to only include selected fields.
    Always includes 'filename' field even if not selected.
    
    Args:
        data: List of dictionaries with extracted data
        selected_fields: Set of field names to include
        
    Returns:
        Filtered list of dictionaries
    """
    if not data:
        return []
    
    # Always include filename
    selected_fields = selected_fields.copy()
    selected_fields.add('filename')
    
    # Filter each row
    filtered_data = []
    for row in data:
        filtered_row = {k: v for k, v in row.items() if k in selected_fields}
        filtered_data.append(filtered_row)
    
    return filtered_data


def format_dataframe(df: pd.DataFrame, field_order: List[str] = None) -> pd.DataFrame:
    """
    Format DataFrame with consistent column order.
    
    Args:
        df: Input DataFrame
        field_order: Desired column order (filename always first)
        
    Returns:
        Formatted DataFrame
    """
    if df.empty:
        return df
    
    # Default field order
    if field_order is None:
        field_order = [
            'filename',
            'Sr.No',
            'Document No.& Year',
            'Name of Executant(s)',
            'Name of Claimant(s)',
            'Survey No.',
            'Plot No.'
        ]
    
    # Ensure filename is first
    if 'filename' in df.columns:
        field_order = ['filename'] + [f for f in field_order if f != 'filename']
    
    # Reorder columns (only include columns that exist)
    available_cols = [col for col in field_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in field_order]
    
    df = df[available_cols + remaining_cols]
    
    return df


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in MB.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0


def format_file_size(size_mb: float) -> str:
    """
    Format file size as human-readable string.
    
    Args:
        size_mb: Size in MB
        
    Returns:
        Formatted string (e.g., "1.5 MB" or "0.5 MB")
    """
    if size_mb < 1:
        return f"{size_mb * 1024:.1f} KB"
    else:
        return f"{size_mb:.2f} MB"


def validate_pdf_file(file) -> Tuple[bool, str]:
    """
    Validate uploaded PDF file.
    
    Args:
        file: Uploaded file object (Streamlit UploadedFile)
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if file is None:
        return False, "No file provided"
    
    # Check file extension
    if not file.name.lower().endswith('.pdf'):
        return False, f"File '{file.name}' is not a PDF file"
    
    # Check file size (max 50 MB)
    max_size_mb = 50
    file_size_mb = len(file.getvalue()) / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        return False, f"File '{file.name}' is too large ({file_size_mb:.2f} MB). Maximum size is {max_size_mb} MB."
    
    return True, ""


def get_default_fields() -> List[str]:
    """
    Get list of default extraction fields.
    
    Returns:
        List of field names
    """
    return [
        'filename',
        'Sr.No',
        'Document No.& Year',
        'Name of Executant(s)',
        'Name of Claimant(s)',
        'Survey No.',
        'Plot No.'
    ]


def get_field_descriptions() -> Dict[str, str]:
    """
    Get descriptions for each field.
    
    Returns:
        Dictionary mapping field names to descriptions
    """
    return {
        'filename': 'Source PDF filename',
        'Sr.No': 'Serial number',
        'Document No.& Year': 'Document number and year',
        'Name of Executant(s)': 'Name(s) of executant(s)',
        'Name of Claimant(s)': 'Name(s) of claimant(s)',
        'Survey No.': 'Survey number',
        'Plot No.': 'Plot number (required)'
    }


def save_results_to_file(df: pd.DataFrame, output_path: str, format: str = 'csv') -> bool:
    """
    Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        format: File format ('csv', 'excel', 'json', 'md', 'markdown')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        format_lower = format.lower()
        
        if format_lower == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif format_lower in ['excel', 'xlsx']:
            df.to_excel(output_path, index=False, engine='openpyxl')
        elif format_lower == 'json':
            df.to_json(output_path, orient='records', indent=2, force_ascii=False)
        elif format_lower in ['md', 'markdown']:
            save_dataframe_to_markdown(df, output_path)
        else:
            return False
        
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


def save_dataframe_to_markdown(df: pd.DataFrame, output_path: str) -> bool:
    """
    Save DataFrame as Markdown table.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            headers = '| ' + ' | '.join(df.columns) + ' |'
            f.write(headers + '\n')
            
            # Write separator
            separators = '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'
            f.write(separators + '\n')
            
            # Write data rows
            for _, row in df.iterrows():
                row_values = []
                for col in df.columns:
                    value = str(row[col]) if pd.notna(row[col]) else ''
                    # Escape pipe characters and newlines
                    value = value.replace('|', '\\|').replace('\n', ' ')
                    row_values.append(value)
                row_line = '| ' + ' | '.join(row_values) + ' |'
                f.write(row_line + '\n')
        
        return True
    except Exception as e:
        print(f"Error saving markdown file: {e}")
        return False


def dataframe_to_json_string(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to JSON string.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        JSON string
    """
    return df.to_json(orient='records', indent=2, force_ascii=False)


def dataframe_to_markdown_string(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to Markdown table string.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        Markdown table string
    """
    md_lines = []
    
    # Header
    headers = '| ' + ' | '.join(df.columns) + ' |'
    md_lines.append(headers)
    
    # Separator
    separators = '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'
    md_lines.append(separators)
    
    # Data rows
    for _, row in df.iterrows():
        row_values = []
        for col in df.columns:
            value = str(row[col]) if pd.notna(row[col]) else ''
            # Escape pipe characters and newlines
            value = value.replace('|', '\\|').replace('\n', '<br>')
            row_values.append(value)
        row_line = '| ' + ' | '.join(row_values) + ' |'
        md_lines.append(row_line)
    
    return '\n'.join(md_lines)

