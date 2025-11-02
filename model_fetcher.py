#!/usr/bin/env python3
"""
Dynamic model fetcher from HuggingFace Hub API.
Fetches available OCR and document processing models.
"""

import requests
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path


# Cache file location
CACHE_DIR = Path.home() / ".cache" / "extract_tn_ec"
CACHE_FILE = CACHE_DIR / "model_cache.json"
CACHE_DURATION = timedelta(hours=24)  # Cache for 24 hours


def ensure_cache_dir():
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cache() -> Optional[Dict]:
    """Load cached model data."""
    if not CACHE_FILE.exists():
        return None

    try:
        with open(CACHE_FILE, "r") as f:
            cache_data = json.load(f)
            cache_time = datetime.fromisoformat(cache_data.get("timestamp", ""))
            if datetime.now() - cache_time < CACHE_DURATION:
                return cache_data.get("models")
    except Exception:
        pass

    return None


def save_cache(models: List[Dict]):
    """Save model data to cache."""
    ensure_cache_dir()
    cache_data = {"timestamp": datetime.now().isoformat(), "models": models}
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=2)


def fetch_ocr_models_from_huggingface(
    api_key: Optional[str] = None,
    limit: int = 50,
    task: str = "document-question-answering",
    for_api_use: bool = False,
) -> List[Dict[str, str]]:
    """
    Fetch OCR models from HuggingFace Hub API.

    Args:
        api_key: Optional HuggingFace API key (for authenticated requests)
        limit: Maximum number of models to fetch
        task: Task filter (document-question-answering, image-to-text, etc.)

    Returns:
        List of model dictionaries
    """
    models = []

    try:
        # Check cache first
        cached_models = load_cache()
        if cached_models:
            return cached_models

        # HuggingFace Hub API endpoint
        base_url = "https://huggingface.co/api/models"

        # Try multiple API endpoints and search strategies
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Try fetching models with different search terms
        search_terms = ["ocr", "document", "table", "chandra", "layout"]
        all_models = []
        seen_ids = set()

        for search_term in search_terms[:2]:  # Limit to avoid too many requests
            try:
                params = {
                    "search": search_term,
                    "sort": "downloads",
                    "direction": "-1",
                    "limit": str(min(limit, 20)),
                }

                response = requests.get(
                    base_url, params=params, headers=headers, timeout=10
                )
                response.raise_for_status()
                data = response.json()

                # Handle both list and dict responses
                if isinstance(data, dict):
                    data = data.get("models", [])

                for item in data:
                    model_id = item.get("id", "")
                    if model_id and model_id not in seen_ids:
                        seen_ids.add(model_id)
                        all_models.append(item)

                if len(all_models) >= limit:
                    break

            except Exception as e:
                print(f"Warning: Search for '{search_term}' failed: {e}")
                continue

        data = all_models[:limit]

        for item in data:
            # Filter for relevant models
            model_id = item.get("id", "")
            if not model_id:
                continue

            # Get model info
            model_info = {
                "id": model_id,
                "name": model_id.split("/")[-1].replace("-", " ").title(),
                "description": item.get("modelId", "") or f"{model_id} OCR model",
                "url": f"https://huggingface.co/{model_id}",
                "downloads": item.get("downloads", 0),
                "likes": item.get("likes", 0),
                "pipeline_tag": item.get("pipeline_tag", ""),
                "verified": item.get("gated", False)
                or model_id.startswith(("microsoft/", "google/", "datalab-to/")),
            }

            # Estimate size based on model name/type
            if "large" in model_id.lower():
                model_info["size"] = "~3-5GB"
            elif "base" in model_id.lower() or "small" in model_id.lower():
                model_info["size"] = "~500MB-1GB"
            else:
                model_info["size"] = "~1-2GB"

            model_info["download_time"] = (
                "5-15 minutes (first time)" if not for_api_use else "N/A (API-based)"
            )
            model_info["supports_gpu"] = True
            model_info["supports_cpu"] = True

            # For API use, check if model is API-compatible
            if for_api_use:
                model_info["api_compatible"] = (
                    True  # Most models on HF can be used via API
                )
                # Known compatible models
                known_api_models = [
                    "datalab-to/chandra",
                    "microsoft/trocr",
                    "microsoft/table-transformer",
                ]
                model_info["api_compatible"] = any(
                    known in model_id.lower() for known in known_api_models
                )

            models.append(model_info)

        # Cache the results
        if models:
            save_cache(models)

    except requests.RequestException as e:
        print(f"Warning: Could not fetch models from HuggingFace API: {e}")
        # Return empty list if API fails
        return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

    return models


def get_popular_ocr_models() -> List[Dict[str, str]]:
    """
    Get a curated list of popular OCR models (fallback if API fails).

    Returns:
        List of popular OCR models
    """
    return [
        {
            "id": "datalab-to/chandra",
            "name": "Chandra OCR",
            "description": "High-accuracy OCR model for documents, tables, and forms. Best for structured data extraction.",
            "size": "~2GB",
            "url": "https://huggingface.co/datalab-to/chandra",
            "download_time": "10-20 minutes (first time)",
            "supports_gpu": True,
            "supports_cpu": True,
            "downloads": 10000,
            "verified": True,
        },
        {
            "id": "microsoft/table-transformer",
            "name": "Table Transformer",
            "description": "Microsoft table extraction model for structured document parsing.",
            "size": "~1.5GB",
            "url": "https://huggingface.co/microsoft/table-transformer",
            "download_time": "5-10 minutes (first time)",
            "supports_gpu": True,
            "supports_cpu": True,
            "downloads": 5000,
            "verified": True,
        },
        {
            "id": "microsoft/trocr-base-printed",
            "name": "TrOCR Base (Printed)",
            "description": "Transformer-based OCR for printed text recognition.",
            "size": "~500MB",
            "url": "https://huggingface.co/microsoft/trocr-base-printed",
            "download_time": "2-5 minutes (first time)",
            "supports_gpu": True,
            "supports_cpu": True,
            "downloads": 8000,
            "verified": True,
        },
        {
            "id": "microsoft/trocr-base-handwritten",
            "name": "TrOCR Base (Handwritten)",
            "description": "Transformer-based OCR for handwritten text recognition.",
            "size": "~500MB",
            "url": "https://huggingface.co/microsoft/trocr-base-handwritten",
            "download_time": "2-5 minutes (first time)",
            "supports_gpu": True,
            "supports_cpu": True,
            "downloads": 6000,
            "verified": True,
        },
        {
            "id": "facebook/detr-resnet-50",
            "name": "DETR ResNet-50",
            "description": "Object detection model that can be used for document layout analysis.",
            "size": "~150MB",
            "url": "https://huggingface.co/facebook/detr-resnet-50",
            "download_time": "1-3 minutes (first time)",
            "supports_gpu": True,
            "supports_cpu": True,
            "downloads": 15000,
            "verified": True,
        },
        {
            "id": "layoutlm-base-uncased",
            "name": "LayoutLM Base",
            "description": "Multimodal pre-trained model for document understanding and OCR.",
            "size": "~400MB",
            "url": "https://huggingface.co/microsoft/layoutlm-base-uncased",
            "download_time": "2-4 minutes (first time)",
            "supports_gpu": True,
            "supports_cpu": True,
            "downloads": 7000,
            "verified": True,
        },
        {
            "id": "PaddleOCR",
            "name": "PaddleOCR (via EasyOCR)",
            "description": "Fast OCR model, lightweight and quick. Good for simple text extraction.",
            "size": "~100MB",
            "url": "https://github.com/JaidedAI/EasyOCR",
            "download_time": "1-2 minutes (first time)",
            "supports_gpu": False,
            "supports_cpu": True,
            "downloads": 0,
            "verified": False,
        },
    ]


def get_available_local_models(
    api_key: Optional[str] = None, use_cache: bool = True
) -> List[Dict[str, str]]:
    """
    Get available local models, trying to fetch from HuggingFace API first.

    Args:
        api_key: Optional HuggingFace API key
        use_cache: Whether to use cached results

    Returns:
        List of available models
    """
    # Try to fetch from HuggingFace API
    if not use_cache or not load_cache():
        try:
            hf_models = fetch_ocr_models_from_huggingface(api_key=api_key)
            if hf_models:
                # Merge with popular models, prioritizing fetched ones
                popular = get_popular_ocr_models()
                popular_ids = {m["id"] for m in popular}
                # Add popular models that weren't fetched
                for model in popular:
                    if model["id"] not in {m["id"] for m in hf_models}:
                        hf_models.append(model)
                return hf_models
        except Exception as e:
            print(f"Failed to fetch models from API: {e}")

    # Use cached models if available
    cached = load_cache()
    if cached:
        return cached

    # Fallback to popular models
    return get_popular_ocr_models()


def clear_cache():
    """Clear the model cache."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()


def get_model_info(model_id: str, models: List[Dict]) -> Optional[Dict]:
    """Get information for a specific model from the list."""
    for model in models:
        if model["id"] == model_id:
            return model
    return None
