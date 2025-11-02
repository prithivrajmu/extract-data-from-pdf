#!/usr/bin/env python3
"""
Test modules for validating API keys.
Provides test functions for Datalab, HuggingFace, Gemini, and Deepseek APIs.
"""

import requests
from typing import Dict, Tuple


def test_datalab_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Test Datalab API key by making a simple request.

    Args:
        api_key: Datalab API key to test

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty"

    try:
        # Test by trying to list available models or check API status
        headers = {"X-Api-Key": api_key}

        # Try a simple endpoint to validate the key
        # Note: This is a placeholder - adjust based on actual Datalab API
        response = requests.get(
            "https://www.datalab.to/api/v1/status", headers=headers, timeout=10
        )

        if response.status_code == 200:
            return True, "✅ Datalab API key is valid"
        elif response.status_code == 401:
            return False, "❌ Invalid API key (unauthorized)"
        elif response.status_code == 403:
            return False, "❌ API key lacks required permissions"
        else:
            return False, f"❌ API returned status code: {response.status_code}"

    except requests.exceptions.Timeout:
        return False, "❌ Connection timeout - check your internet connection"
    except requests.exceptions.RequestException as e:
        return False, f"❌ Connection error: {str(e)}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def test_huggingface_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Test HuggingFace API key by listing available models.

    Args:
        api_key: HuggingFace API key (token) to test

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty"

    try:
        headers = {"Authorization": f"Bearer {api_key}"}

        # Test by trying to access user info or model listing
        response = requests.get(
            "https://huggingface.co/api/whoami", headers=headers, timeout=10
        )

        if response.status_code == 200:
            user_info = response.json()
            username = user_info.get("name", "Unknown")
            return True, f"✅ HuggingFace API key is valid (User: {username})"
        elif response.status_code == 401:
            return False, "❌ Invalid API key (unauthorized)"
        else:
            return False, f"❌ API returned status code: {response.status_code}"

    except requests.exceptions.Timeout:
        return False, "❌ Connection timeout - check your internet connection"
    except requests.exceptions.RequestException as e:
        return False, f"❌ Connection error: {str(e)}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def test_gemini_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Test Gemini API key by listing available models.

    Args:
        api_key: Gemini API key to test

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty"

    try:
        import google.generativeai as genai

        # Configure Gemini with the API key
        genai.configure(api_key=api_key)

        # Try to list models
        try:
            models = genai.list_models()

            # Check if we can access models
            model_count = sum(1 for _ in models)

            if model_count > 0:
                return (
                    True,
                    f"✅ Gemini API key is valid ({model_count} models available)",
                )
            else:
                return True, "✅ Gemini API key is valid"

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "401" in error_msg
            ):
                return False, "❌ Invalid API key (unauthorized)"
            elif "quota" in error_msg or "429" in error_msg:
                return False, "❌ API quota exceeded"
            else:
                return False, f"❌ Error accessing models: {str(e)}"

    except ImportError:
        return False, "❌ google-generativeai package not installed"
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "authentication" in error_msg:
            return False, "❌ Invalid API key"
        else:
            return False, f"❌ Error: {str(e)}"


def test_deepseek_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Test Deepseek API key by making a simple request.

    Args:
        api_key: Deepseek API key to test

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty"

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Test by making a simple chat completion request
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5,
        }

        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )

        if response.status_code == 200:
            return True, "✅ Deepseek API key is valid"
        elif response.status_code == 401:
            return False, "❌ Invalid API key (unauthorized)"
        elif response.status_code == 403:
            return False, "❌ API key lacks required permissions"
        elif response.status_code == 429:
            return False, "❌ Rate limit exceeded"
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get(
                    "message", f"Status {response.status_code}"
                )
            except ValueError:
                error_msg = f"Status {response.status_code}"
            return False, f"❌ API error: {error_msg}"

    except requests.exceptions.Timeout:
        return False, "❌ Connection timeout - check your internet connection"
    except requests.exceptions.RequestException as e:
        return False, f"❌ Connection error: {str(e)}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def test_all_api_keys(keys: Dict[str, str]) -> Dict[str, Tuple[bool, str]]:
    """
    Test all provided API keys.

    Args:
        keys: Dictionary with provider names as keys and API keys as values

    Returns:
        Dictionary with test results for each provider
    """
    results = {}

    test_functions = {
        "datalab": test_datalab_api_key,
        "huggingface": test_huggingface_api_key,
        "hf": test_huggingface_api_key,
        "gemini": test_gemini_api_key,
        "deepseek": test_deepseek_api_key,
    }

    for provider, api_key in keys.items():
        if api_key and api_key.strip():
            test_func = test_functions.get(provider.lower())
            if test_func:
                results[provider] = test_func(api_key)
            else:
                results[provider] = (False, f"❌ Unknown provider: {provider}")
        else:
            results[provider] = (None, "No API key provided")

    return results
