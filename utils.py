#!/usr/bin/env python3
"""
Helper utilities for file handling, field filtering, and data formatting.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
from datetime import datetime


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


def dataframe_to_json_string(df: pd.DataFrame, format: str = 'standard', metadata: Optional[Dict] = None) -> str:
    """
    Convert DataFrame to JSON string with multiple format options.
    
    Args:
        df: DataFrame to convert
        format: JSON format ('standard', 'structured', 'multi_file', 'unified')
        metadata: Optional metadata dictionary (used for structured/unified formats)
        
    Returns:
        JSON string
    """
    format_lower = format.lower()
    
    if format_lower == 'standard':
        # Default format - backward compatible
        return df.to_json(orient='records', indent=2, force_ascii=False)
    
    elif format_lower == 'structured':
        return create_structured_json_output(df, metadata)
    
    elif format_lower == 'multi_file':
        return create_multi_file_json(df, metadata)
    
    elif format_lower == 'unified':
        return create_unified_json(df, metadata)
    
    else:
        raise ValueError(f"Unknown JSON format: {format}. Use 'standard', 'structured', 'multi_file', or 'unified'.")


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


def create_structured_json_output(df: pd.DataFrame, metadata: Optional[Dict] = None) -> str:
    """
    Create structured JSON output with metadata wrapper.
    
    Args:
        df: DataFrame to convert
        metadata: Optional metadata dictionary
        
    Returns:
        JSON string with metadata and data
    """
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    # Add default metadata
    default_metadata = {
        'extraction_timestamp': datetime.now().isoformat(),
        'total_rows': len(df),
        'fields': list(df.columns),
        'files_processed': df['filename'].nunique() if 'filename' in df.columns else 0
    }
    
    # Merge with provided metadata
    final_metadata = {**default_metadata, **metadata}
    
    # Convert DataFrame to records
    data = df.to_dict(orient='records')
    
    # Create structured output
    output = {
        'metadata': final_metadata,
        'data': data
    }
    
    return json.dumps(output, indent=2, ensure_ascii=False)


def create_multi_file_json(df: pd.DataFrame, metadata: Optional[Dict] = None) -> str:
    """
    Create multi-file JSON output organized by filename.
    
    Args:
        df: DataFrame to convert (must have 'filename' column for multi-file support)
        metadata: Optional metadata dictionary
        
    Returns:
        JSON string organized by filename
    """
    if 'filename' not in df.columns:
        # If no filename column, fallback to structured format
        return create_structured_json_output(df, metadata)
    
    # Group by filename
    file_data = {}
    for filename in df['filename'].unique():
        file_df = df[df['filename'] == filename]
        file_data[filename] = file_df.to_dict(orient='records')
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    default_metadata = {
        'extraction_timestamp': datetime.now().isoformat(),
        'total_files': len(file_data),
        'total_rows': len(df),
        'fields': list(df.columns)
    }
    
    final_metadata = {**default_metadata, **metadata}
    
    # Create multi-file output
    output = {
        **file_data,
        'metadata': final_metadata
    }
    
    return json.dumps(output, indent=2, ensure_ascii=False)


def create_unified_json(df: pd.DataFrame, metadata: Optional[Dict] = None) -> str:
    """
    Create unified JSON output with metadata embedded in each record.
    
    Args:
        df: DataFrame to convert
        metadata: Optional metadata dictionary (will be embedded in each row)
        
    Returns:
        JSON string with metadata in each record
    """
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    # Add default metadata
    default_metadata = {
        'extraction_timestamp': datetime.now().isoformat(),
        'total_rows': len(df),
        'fields': list(df.columns)
    }
    
    final_metadata = {**default_metadata, **metadata}
    
    # Convert DataFrame to records and add metadata to each
    records = df.to_dict(orient='records')
    unified_records = []
    
    for record in records:
        unified_record = {
            **record,
            '_metadata': final_metadata
        }
        unified_records.append(unified_record)
    
    return json.dumps(unified_records, indent=2, ensure_ascii=False)

