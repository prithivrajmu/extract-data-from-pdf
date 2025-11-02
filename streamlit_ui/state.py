"""Session state helpers for the Streamlit application."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from api_key_manager import load_api_key
from utils import get_default_fields


def initialize_session_state(default_fields: Iterable[str] | None = None) -> None:
    """Ensure required keys exist in ``st.session_state``."""

    if default_fields is None:
        default_fields = get_default_fields()

    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {
            "huggingface": "",
            "datalab": "",
            "gemini": "",
            "deepseek": "",
        }

    st.session_state.setdefault("api_key_tests", {})
    st.session_state.setdefault("extraction_results", None)
    st.session_state.setdefault("selected_fields", set(default_fields))
    st.session_state.setdefault("custom_fields", [])
    st.session_state.setdefault("selected_ocr_method", "EasyOCR")


def load_saved_api_keys() -> None:
    """Load API keys from local storage into session state."""

    for provider in ("huggingface", "datalab", "gemini", "deepseek"):
        saved_key = load_api_key(provider)
        if saved_key and not st.session_state.api_keys.get(provider):
            st.session_state.api_keys[provider] = saved_key
