#!/usr/bin/env python3
"""
Model information and links for local and HuggingFace OCR models.
Now supports dynamic fetching from HuggingFace Hub API.
"""

from typing import List, Dict, Optional
from model_fetcher import (
    get_available_local_models,
    get_popular_ocr_models,
    clear_cache,
    fetch_ocr_models_from_huggingface
)


def get_local_ocr_models(api_key: Optional[str] = None, use_dynamic: bool = True) -> List[Dict[str, str]]:
    """
    Get list of available local OCR models with information.
    Now supports dynamic fetching from HuggingFace Hub API.
    
    Args:
        api_key: Optional HuggingFace API key for authenticated requests
        use_dynamic: Whether to fetch models dynamically from HuggingFace
        
    Returns:
        List of dictionaries with model information
    """
    if use_dynamic:
        # Try to get models dynamically
        try:
            models = get_available_local_models(api_key=api_key)
            if models:
                return models
        except Exception:
            # Fallback to static list
            pass
    
    # Fallback to popular models
    return get_popular_ocr_models()


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

