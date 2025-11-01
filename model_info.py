#!/usr/bin/env python3
"""
Model information and links for local and HuggingFace OCR models.
"""

from typing import List, Dict


def get_local_ocr_models() -> List[Dict[str, str]]:
    """
    Get list of available local OCR models with information.
    
    Returns:
        List of dictionaries with model information
    """
    return [
        {
            'id': 'datalab-to/chandra',
            'name': 'Chandra OCR',
            'description': 'High-accuracy OCR model for documents, tables, and forms. Best for structured data extraction.',
            'size': '~2GB',
            'url': 'https://huggingface.co/datalab-to/chandra',
            'download_time': '10-20 minutes (first time)',
            'supports_gpu': True,
            'supports_cpu': True
        },
        {
            'id': 'PaddleOCR',
            'name': 'PaddleOCR (via EasyOCR)',
            'description': 'Fast OCR model, lightweight and quick. Good for simple text extraction.',
            'size': '~100MB',
            'url': 'https://github.com/JaidedAI/EasyOCR',
            'download_time': '1-2 minutes (first time)',
            'supports_gpu': False,
            'supports_cpu': True
        }
    ]


def get_huggingface_ocr_models() -> List[Dict[str, str]]:
    """
    Get list of available HuggingFace OCR models via API.
    
    Returns:
        List of dictionaries with model information
    """
    return [
        {
            'id': 'datalab-to/chandra',
            'name': 'Chandra OCR',
            'description': 'High-accuracy OCR model for documents, tables, and forms.',
            'url': 'https://huggingface.co/datalab-to/chandra',
            'api_compatible': True
        },
        {
            'id': 'microsoft/table-transformer',
            'name': 'Table Transformer',
            'description': 'Microsoft table extraction model.',
            'url': 'https://huggingface.co/microsoft/table-transformer',
            'api_compatible': False,
            'note': 'May require specific API setup'
        },
        {
            'id': 'microsoft/table-transformer-structure-recognition',
            'name': 'Table Transformer Structure',
            'description': 'Structure recognition for tables.',
            'url': 'https://huggingface.co/microsoft/table-transformer-structure-recognition',
            'api_compatible': False,
            'note': 'May require specific API setup'
        }
    ]


def get_model_search_url() -> str:
    """
    Get URL for searching HuggingFace OCR models.
    
    Returns:
        Search URL
    """
    return "https://huggingface.co/models?pipeline_tag=document-question-answering&sort=trending&search=ocr"


def get_model_info_by_id(model_id: str, model_type: str = 'local') -> Dict[str, str]:
    """
    Get information for a specific model.
    
    Args:
        model_id: Model identifier
        model_type: 'local' or 'huggingface'
        
    Returns:
        Model information dictionary or None
    """
    if model_type == 'local':
        models = get_local_ocr_models()
    else:
        models = get_huggingface_ocr_models()
    
    for model in models:
        if model['id'] == model_id:
            return model
    
    return None

