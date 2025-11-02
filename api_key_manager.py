#!/usr/bin/env python3
"""
API Key Manager for storing and loading API keys locally.
Keys are stored in .env file (preferred) or api_keys.txt (fallback).
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv, set_key

from logging_config import get_logger

logger = get_logger(__name__)


def get_env_file_path() -> str:
    """
    Get the path to .env file.

    Returns:
        Path to .env file
    """
    env_path = find_dotenv()
    if env_path:
        return env_path
    # If no .env exists, create one in current directory
    return os.path.join(os.getcwd(), ".env")


def save_api_key(provider: str, api_key: str) -> bool:
    """
    Save API key to .env file.

    Args:
        provider: Provider name (datalab, huggingface, gemini, deepseek)
        api_key: API key value

    Returns:
        True if successful, False otherwise
    """
    try:
        env_file = get_env_file_path()

        # Create .env file if it doesn't exist
        if not os.path.exists(env_file):
            Path(env_file).touch()

        # Map provider names to environment variable names
        env_var_map = {
            "datalab": "DATALAB_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "hf": "HUGGINGFACE_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }

        env_var = env_var_map.get(provider.lower(), f"{provider.upper()}_API_KEY")

        # Set key in .env file
        set_key(env_file, env_var, api_key)

        # Reload environment
        load_dotenv(env_file, override=True)

        return True
    except Exception as e:
        logger.error(f"Error saving API key for {provider}: {e}")
        return False


def load_api_key(provider: str) -> str | None:
    """
    Load API key from .env file.

    Args:
        provider: Provider name (datalab, huggingface, gemini, deepseek)

    Returns:
        API key value or None if not found
    """
    try:
        load_dotenv()  # Load .env file

        # Map provider names to environment variable names
        env_var_map = {
            "datalab": ["DATALAB_API_KEY", "DATALAB_API_TOKEN"],
            "huggingface": ["HUGGINGFACE_API_KEY", "HF_API_KEY", "HF_TOKEN"],
            "hf": ["HUGGINGFACE_API_KEY", "HF_API_KEY", "HF_TOKEN"],
            "gemini": ["GEMINI_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
        }

        env_vars = env_var_map.get(provider.lower(), [f"{provider.upper()}_API_KEY"])

        # Try each possible environment variable name
        for env_var in env_vars:
            api_key = os.getenv(env_var)
            if api_key:
                return api_key

        return None
    except Exception as e:
        logger.error(f"Error loading API key for {provider}: {e}")
        return None


def save_all_api_keys(keys: dict) -> bool:
    """
    Save multiple API keys at once.

    Args:
        keys: Dictionary with provider names as keys and API keys as values

    Returns:
        True if all successful, False otherwise
    """
    success = True
    for provider, api_key in keys.items():
        if api_key:  # Only save non-empty keys
            if not save_api_key(provider, api_key):
                success = False
    return success


def get_storage_info() -> str:
    """
    Get information about where keys are stored.

    Returns:
        String describing storage location
    """
    env_file = get_env_file_path()
    return f"API keys are stored locally in: {env_file}\n\n⚠️  Please make a backup copy of this file to keep your keys safe!"
