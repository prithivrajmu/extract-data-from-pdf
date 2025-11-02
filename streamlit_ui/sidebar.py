"""Sidebar UI components for the Streamlit application."""

from __future__ import annotations

import streamlit as st

from api_key_manager import get_storage_info, save_all_api_keys
from gemini_test_modules import (
    test_gemini_basic,
    test_gemini_file_upload,
    test_gemini_json_upload,
)
from gpu_check_utils import check_gpu_comprehensive
from model_fetcher import clear_cache
from model_info import (
    get_huggingface_ocr_models,
    get_local_ocr_models,
    get_model_search_url,
)
from test_api_keys import (
    test_datalab_api_key,
    test_deepseek_api_key,
    test_gemini_api_key,
    test_huggingface_api_key,
)
from utils import get_default_fields, get_field_descriptions


def render_sidebar() -> dict:
    """
    Render the entire sidebar configuration UI.

    Returns:
        Dictionary containing configuration values:
        - ocr_method: Selected extraction method
        - local_model: Selected local model (if applicable)
        - hf_model: Selected HuggingFace model (if applicable)
        - use_cpu_mode: Whether to force CPU mode
        - use_pretty_output: Whether to use formatted output
        - api_keys: Dictionary of API keys
        - auto_detect_fields: Whether to auto-detect fields
        - use_custom_fields: Whether to use only custom fields
        - output_formats: List of selected output formats
        - json_format: Selected JSON format
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        # OCR Method Selection
        st.subheader("🔧 Extraction Method")

        # Define methods with tooltips/descriptions and extraction types
        method_descriptions = {
            "EasyOCR": (
                "Fast CPU-based OCR using EasyOCR library. "
                "Lightweight (~100MB models), quick setup, good for simple text extraction. "
                "No API key needed."
            ),
            "PyTesseract": (
                "Google's Tesseract OCR engine via PyTesseract. "
                "Fast, lightweight, works well for text-based PDFs. "
                "Requires Tesseract-OCR installed on system."
            ),
            "Local Model": (
                "Download and run OCR models locally (Chandra OCR). "
                "High accuracy for documents/tables. First run downloads ~2GB models. "
                "Supports GPU/CPU."
            ),
            "HuggingFace": (
                "Use HuggingFace Inference API for OCR. "
                "Requires HuggingFace API key. "
                "Cloud-based, no local model download needed."
            ),
            "Datalab API": (
                "High-accuracy OCR via Datalab Marker API. "
                "Best for structured documents and tables. "
                "Requires Datalab API key. Recommended for production."
            ),
            "Gemini AI": (
                "Google's Gemini AI for intelligent data extraction. "
                "Understands context and can extract custom fields. "
                "Requires Gemini API key."
            ),
            "Deepseek AI": (
                "Deepseek AI for document extraction with vision capabilities. "
                "Can extract custom fields. Requires Deepseek API key."
            ),
        }

        # Define extraction types for each method (shown in dropdown)
        method_types = {
            "EasyOCR": "CPU OCR",
            "PyTesseract": "OCR",
            "Local Model": "Local OCR",
            "HuggingFace": "Cloud OCR",
            "Datalab API": "Cloud OCR",
            "Gemini AI": "AI Extraction",
            "Deepseek AI": "AI Extraction",
        }

        # Create formatted options with extraction type visible in dropdown
        method_options_raw = list(method_descriptions.keys())
        method_options_formatted = [
            f"{method} ({method_types.get(method, '')})"
            for method in method_options_raw
        ]

        # Create mapping from formatted to raw method names
        method_mapping = dict(zip(method_options_formatted, method_options_raw))

        # Get current selection index
        current_method = st.session_state.get(
            "selected_ocr_method", method_options_raw[0]
        )
        current_index = (
            method_options_raw.index(current_method)
            if current_method in method_options_raw
            else 0
        )

        selected_formatted = st.selectbox(
            "Select extraction method",
            options=method_options_formatted,
            index=current_index,
            help="Choose an extraction method. The type (OCR/AI) is shown in parentheses. See description below for details.",
            key="ocr_method_select",
        )

        # Extract actual method name from formatted selection
        ocr_method = method_mapping.get(selected_formatted, method_options_raw[0])
        st.session_state.selected_ocr_method = ocr_method

        # Show detailed description for selected method with info box
        if ocr_method in method_descriptions:
            extraction_type = method_types.get(ocr_method, "")
            st.info(
                f"**{ocr_method} ({extraction_type}):** {method_descriptions[ocr_method]}"
            )

        # Local Model Selection
        local_model = None
        use_cpu_mode = False
        use_pretty_output = False

        if ocr_method == "Local Model":
            st.markdown("---")
            st.subheader("🖥️ Local Model Configuration")
            st.info(
                "📥 Models will be downloaded automatically on first use (~2GB for Chandra)"
            )

            # Get HuggingFace API key if available (defined later in sidebar)
            hf_api_key_available = st.session_state.api_keys.get("huggingface", "")

            # Model fetching options
            col_refresh, col_info = st.columns([1, 2])
            with col_refresh:
                refresh_models = st.button(
                    "🔄 Refresh Model List",
                    help="Fetch latest models from HuggingFace Hub",
                )

            if refresh_models:
                clear_cache()
                st.success(
                    "✅ Cache cleared! Model list will refresh on next selection."
                )

            # Try to use HuggingFace API key if available
            use_dynamic = bool(hf_api_key_available) or refresh_models

            # Model selection
            try:
                with st.spinner("Loading available models..." if use_dynamic else ""):
                    local_models = get_local_ocr_models(
                        api_key=hf_api_key_available if hf_api_key_available else None,
                        use_dynamic=use_dynamic,
                    )
            except Exception as e:
                st.warning(
                    f"⚠️ Could not fetch models dynamically: {e}. Using default models."
                )
                local_models = get_local_ocr_models(use_dynamic=False)

            model_options = [f"{m['name']} ({m['id']})" for m in local_models]
            model_ids = [m["id"] for m in local_models]

            selected_model_idx = st.selectbox(
                "Select OCR Model",
                range(len(model_options)),
                format_func=lambda x: model_options[x],
                help="Choose which OCR model to download and use locally",
            )
            local_model = model_ids[selected_model_idx]

            # Show model info
            selected_model_info = local_models[selected_model_idx]

            # Check if model is supported
            try:
                from model_loaders import is_model_supported

                is_supported, support_reason = is_model_supported(local_model)
            except Exception:
                # Fallback check
                is_supported = (
                    local_model == "datalab-to/chandra"
                    or local_model.startswith("datalab-to/chandra")
                )
                support_reason = (
                    "Chandra CLI model"
                    if is_supported
                    else "Needs custom implementation"
                )

            with st.expander(f"ℹ️ About {selected_model_info['name']}"):
                st.write(f"**Description:** {selected_model_info['description']}")
                st.write(
                    f"**Model Size:** {selected_model_info.get('size', 'Unknown')}"
                )
                st.write(
                    f"**Download Time:** {selected_model_info.get('download_time', 'Unknown')}"
                )
                st.write(
                    f"**GPU Support:** {'Yes' if selected_model_info.get('supports_gpu') else 'No'}"
                )
                st.write(
                    f"**CPU Support:** {'Yes' if selected_model_info.get('supports_cpu') else 'No'}"
                )
                if selected_model_info.get("downloads"):
                    st.write(f"**Downloads:** {selected_model_info['downloads']:,}")
                if selected_model_info.get("verified"):
                    st.write("**Status:** ✅ Verified")

                # Show support status with detailed reason
                if is_supported:
                    st.success(f"✅ **Supported** - {support_reason}")
                else:
                    st.warning(f"⚠️ **Limited Support** - {support_reason}")
                    st.info(
                        "💡 You can still try this model - it will attempt to load via transformers library. Some models may need additional dependencies."
                    )

                st.markdown(
                    f"**Model Page:** [{selected_model_info['name']}]({selected_model_info['url']})"
                )

            # Info about dynamic fetching
            if not hf_api_key_available:
                st.caption(
                    "💡 **Tip:** Add your HuggingFace API key below to fetch the latest models automatically"
                )
            else:
                st.caption(
                    "✅ Using HuggingFace API key - models are fetched automatically"
                )

            # Options for Chandra model
            if local_model == "datalab-to/chandra":
                use_cpu_mode = st.checkbox(
                    "Force CPU Mode",
                    value=False,
                    help="Force CPU usage even if GPU is available (slower but more stable)",
                )
                use_pretty_output = st.checkbox(
                    "Use Formatted Output",
                    value=False,
                    help="Use human-readable output formatting (filters technical details)",
                )

        # HuggingFace Model Selection
        hf_model = None
        if ocr_method == "HuggingFace":
            st.markdown("---")
            st.subheader("🤗 Model Selection")
            st.info(
                "📡 Models are fetched from HuggingFace Hub. Requires HuggingFace API key for best results."
            )

            # Get HuggingFace API key if available (from session state, will be updated when user enters it)
            hf_api_key_for_models = st.session_state.api_keys.get("huggingface", "")

            # Model fetching options
            col_refresh_hf, col_info_hf = st.columns([1, 2])
            with col_refresh_hf:
                refresh_hf_models = st.button(
                    "🔄 Refresh Model List",
                    key="refresh_hf",
                    help="Fetch latest models from HuggingFace Hub",
                )

            if refresh_hf_models:
                clear_cache()
                st.success("✅ Cache cleared! Model list will refresh.")

            # Determine if we should use dynamic fetching
            use_dynamic_hf = bool(hf_api_key_for_models) or refresh_hf_models

            # Fetch available models
            try:
                with st.spinner(
                    "Loading available HuggingFace models..." if use_dynamic_hf else ""
                ):
                    hf_models = get_huggingface_ocr_models(
                        api_key=(
                            hf_api_key_for_models if hf_api_key_for_models else None
                        ),
                        use_dynamic=use_dynamic_hf,
                    )
            except Exception as e:
                st.warning(
                    f"⚠️ Could not fetch models dynamically: {e}. Using default models."
                )
                hf_models = get_huggingface_ocr_models(use_dynamic=False)

            if not hf_models:
                st.error(
                    "No models available. Please check your HuggingFace API key or internet connection."
                )
                hf_models = get_huggingface_ocr_models(use_dynamic=False)

            hf_model_options = [f"{m['name']} ({m['id']})" for m in hf_models]
            hf_model_ids = [m["id"] for m in hf_models]

            selected_hf_idx = st.selectbox(
                "Select HuggingFace Model",
                range(len(hf_model_options)),
                format_func=lambda x: hf_model_options[x],
                index=0,  # Default to first model
                help="Choose which HuggingFace model to use via API",
            )
            hf_model = hf_model_ids[selected_hf_idx]

            # Show model info and link
            selected_hf_info = hf_models[selected_hf_idx]
            with st.expander(f"ℹ️ About {selected_hf_info['name']}"):
                st.write(f"**Description:** {selected_hf_info['description']}")
                st.write(f"**Model ID:** `{selected_hf_info['id']}`")
                st.write(
                    f"**API Compatible:** {'✅ Yes' if selected_hf_info.get('api_compatible', True) else '⚠️ Limited'}"
                )
                if selected_hf_info.get("downloads"):
                    st.write(f"**Downloads:** {selected_hf_info['downloads']:,}")
                if selected_hf_info.get("verified"):
                    st.write("**Status:** ✅ Verified")
                if selected_hf_info.get("size"):
                    st.write(f"**Model Size:** {selected_hf_info['size']}")
                if selected_hf_info.get("note"):
                    st.info(f"**Note:** {selected_hf_info['note']}")
                st.markdown(
                    f"**Model Page:** [{selected_hf_info['name']}]({selected_hf_info['url']})"
                )

            # Info about dynamic fetching
            if not hf_api_key_for_models:
                st.caption(
                    "💡 **Tip:** Add your HuggingFace API key below to fetch the latest models automatically"
                )
            else:
                st.caption(
                    "✅ Using HuggingFace API key - models are fetched automatically"
                )

            # Link to browse more models
            st.markdown(
                f"[🔍 Browse more OCR models on HuggingFace]({get_model_search_url()})"
            )

            # Fallback: allow manual entry
            st.markdown("**Or enter custom model name:**")
            custom_model = st.text_input(
                "Custom Model Name",
                value="",
                placeholder="username/model-name",
                help="Enter a custom HuggingFace model name (e.g., datalab-to/chandra)",
            )
            if custom_model:
                hf_model = custom_model
                st.info(f"Using custom model: `{hf_model}`")

        # API Key Management Section
        st.markdown("---")
        st.subheader("🔑 API Keys")

        # Initialize API key variables (even when menu is hidden, they're still used in code)
        hf_api_key = st.session_state.api_keys.get("huggingface", "")
        dl_api_key = st.session_state.api_keys.get("datalab", "")
        gem_api_key = st.session_state.api_keys.get("gemini", "")
        ds_api_key = st.session_state.api_keys.get("deepseek", "")

        # Use expander similar to Advanced Testing section
        with st.expander("API Key Configuration", expanded=False):
            # Show storage info
            with st.expander("ℹ️ About API Key Storage"):
                st.info(get_storage_info())

            # HuggingFace API Key
            st.markdown("#### HuggingFace API Key")
            hf_key_col1, hf_key_col2 = st.columns([3, 1])
            with hf_key_col1:
                hf_api_key = st.text_input(
                    "HF API Key",
                    value=hf_api_key,
                    type="password",
                    key="hf_key_input",
                    label_visibility="collapsed",
                )
            with hf_key_col2:
                if st.button("Test", key="test_hf", use_container_width=True):
                    if hf_api_key:
                        with st.spinner("Testing..."):
                            success, message = test_huggingface_api_key(hf_api_key)
                            st.session_state.api_key_tests["huggingface"] = (
                                success,
                                message,
                            )
                            if success:
                                st.session_state.api_keys["huggingface"] = hf_api_key
                    else:
                        st.session_state.api_key_tests["huggingface"] = (
                            False,
                            "Please enter an API key",
                        )

            if "huggingface" in st.session_state.api_key_tests:
                success, msg = st.session_state.api_key_tests["huggingface"]
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

            # Datalab API Key
            st.markdown("#### Datalab API Key")
            dl_key_col1, dl_key_col2 = st.columns([3, 1])
            with dl_key_col1:
                dl_api_key = st.text_input(
                    "Datalab API Key",
                    value=dl_api_key,
                    type="password",
                    key="dl_key_input",
                    label_visibility="collapsed",
                )
            with dl_key_col2:
                if st.button("Test", key="test_dl", use_container_width=True):
                    if dl_api_key:
                        with st.spinner("Testing..."):
                            success, message = test_datalab_api_key(dl_api_key)
                            st.session_state.api_key_tests["datalab"] = (
                                success,
                                message,
                            )
                            if success:
                                st.session_state.api_keys["datalab"] = dl_api_key
                    else:
                        st.session_state.api_key_tests["datalab"] = (
                            False,
                            "Please enter an API key",
                        )

            if "datalab" in st.session_state.api_key_tests:
                success, msg = st.session_state.api_key_tests["datalab"]
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

            # Gemini API Key
            st.markdown("#### Gemini API Key")
            gem_key_col1, gem_key_col2 = st.columns([3, 1])
            with gem_key_col1:
                gem_api_key = st.text_input(
                    "Gemini API Key",
                    value=gem_api_key,
                    type="password",
                    key="gem_key_input",
                    label_visibility="collapsed",
                )
            with gem_key_col2:
                if st.button("Test", key="test_gem", use_container_width=True):
                    if gem_api_key:
                        with st.spinner("Testing..."):
                            success, message = test_gemini_api_key(gem_api_key)
                            st.session_state.api_key_tests["gemini"] = (
                                success,
                                message,
                            )
                            if success:
                                st.session_state.api_keys["gemini"] = gem_api_key
                    else:
                        st.session_state.api_key_tests["gemini"] = (
                            False,
                            "Please enter an API key",
                        )

            if "gemini" in st.session_state.api_key_tests:
                success, msg = st.session_state.api_key_tests["gemini"]
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

            # Deepseek API Key
            st.markdown("#### Deepseek API Key")
            ds_key_col1, ds_key_col2 = st.columns([3, 1])
            with ds_key_col1:
                ds_api_key = st.text_input(
                    "Deepseek API Key",
                    value=ds_api_key,
                    type="password",
                    key="ds_key_input",
                    label_visibility="collapsed",
                )
            with ds_key_col2:
                if st.button("Test", key="test_ds", use_container_width=True):
                    if ds_api_key:
                        with st.spinner("Testing..."):
                            success, message = test_deepseek_api_key(ds_api_key)
                            st.session_state.api_key_tests["deepseek"] = (
                                success,
                                message,
                            )
                            if success:
                                st.session_state.api_keys["deepseek"] = ds_api_key
                    else:
                        st.session_state.api_key_tests["deepseek"] = (
                            False,
                            "Please enter an API key",
                        )

            if "deepseek" in st.session_state.api_key_tests:
                success, msg = st.session_state.api_key_tests["deepseek"]
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

            # Save Keys Button
            st.markdown("---")
            if st.button("💾 Save All Keys Locally", use_container_width=True):
                keys_to_save = {
                    "huggingface": hf_api_key
                    or st.session_state.api_keys.get("huggingface", ""),
                    "datalab": dl_api_key
                    or st.session_state.api_keys.get("datalab", ""),
                    "gemini": gem_api_key
                    or st.session_state.api_keys.get("gemini", ""),
                    "deepseek": ds_api_key
                    or st.session_state.api_keys.get("deepseek", ""),
                }
                if save_all_api_keys(keys_to_save):
                    st.success("✅ All API keys saved to .env file!")
                    st.info("⚠️ Please make a backup copy of the .env file.")
                else:
                    st.error("❌ Failed to save some API keys")

        # Field Selection
        st.markdown("---")
        st.subheader("📋 Field Selection")

        # Auto-detect Fields Option
        auto_detect_fields = st.checkbox(
            "Auto-detect Fields",
            value=False,
            help="Automatically detect field names from PDF before extraction. Works best with AI-based methods (Gemini/Deepseek).",
        )

        # Initialize detected fields in session state
        if "detected_fields" not in st.session_state:
            st.session_state.detected_fields = None
        if "field_detection_mode" not in st.session_state:
            st.session_state.field_detection_mode = "unified"

        # Field Detection Mode (for multiple files)
        if auto_detect_fields:
            field_detection_mode = st.radio(
                "Field Detection Mode (for multiple files):",
                options=["unified", "per_file"],
                format_func=lambda x: (
                    "Unified (detect once from first file)"
                    if x == "unified"
                    else "Per-file (detect from each file)"
                ),
                index=0,
                help="Unified: Detect fields once from the first file and use for all. Per-file: Detect from each file and merge.",
            )
            st.session_state.field_detection_mode = field_detection_mode

        # Custom Fields Input
        with st.expander("➕ Add Custom Fields", expanded=False):
            st.info("Define your own fields to extract. Enter one field name per line.")
            custom_fields_text = st.text_area(
                "Custom Fields",
                placeholder="Enter field names (one per line):\nVillage Name\nRegistration Date\nProperty Type\netc.",
                height=100,
                help="Enter custom field names that exist in your PDF. One field per line.",
            )

            if custom_fields_text:
                custom_fields_list = [
                    f.strip() for f in custom_fields_text.split("\n") if f.strip()
                ]
                st.session_state.custom_fields = custom_fields_list
            else:
                st.session_state.custom_fields = []

        # Use custom fields or default
        use_custom_fields = st.checkbox(
            "Use Custom Fields Only",
            value=False,
            help="If checked, only extract the custom fields you defined above. Otherwise, use default fields.",
        )

        if use_custom_fields and st.session_state.get("custom_fields"):
            # Use only custom fields
            st.session_state.selected_fields = set(
                ["filename"] + st.session_state.custom_fields
            )
            st.info(f"Extracting {len(st.session_state.custom_fields)} custom field(s)")
        else:
            # Use default fields with selection
            default_fields = get_default_fields()
            field_descriptions = get_field_descriptions()

            # Select all / Deselect all buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Select All", use_container_width=True, key="select_all_fields"
                ):
                    # Update selected fields
                    st.session_state.selected_fields = set(default_fields)
                    # Update individual checkbox widget states
                    for field in default_fields:
                        st.session_state[f"field_{field}"] = True
                    st.rerun()
            with col2:
                if st.button(
                    "Deselect All", use_container_width=True, key="deselect_all_fields"
                ):
                    # Update selected fields - always keep filename
                    st.session_state.selected_fields = {"filename"}
                    # Update individual checkbox widget states
                    for field in default_fields:
                        if field == "filename":
                            st.session_state[f"field_{field}"] = True
                        else:
                            st.session_state[f"field_{field}"] = False
                    st.rerun()

            # Field checkboxes - sync with session state
            for field in default_fields:
                # Read from session state (either from selected_fields or widget state)
                checked = field in st.session_state.selected_fields
                disabled = field == "filename"  # Always include filename
                description = field_descriptions.get(field, "")

                # Use widget state if available, otherwise use selected_fields
                widget_key = f"field_{field}"
                if widget_key in st.session_state:
                    checked = st.session_state[widget_key]

                checkbox_value = st.checkbox(
                    field,
                    value=checked,
                    disabled=disabled,
                    help=description,
                    key=widget_key,
                )

                # Sync widget state back to selected_fields
                if checkbox_value:
                    st.session_state.selected_fields.add(field)
                else:
                    if not disabled:
                        st.session_state.selected_fields.discard(field)

            # Add custom fields to selection if any
            if st.session_state.get("custom_fields"):
                st.markdown("**Custom Fields (will also be extracted):**")
                for custom_field in st.session_state.custom_fields:
                    st.session_state.selected_fields.add(custom_field)
                    st.text(f"  ✓ {custom_field}")

        # Output Format Selection
        st.markdown("---")
        st.subheader("💾 Output Format")
        st.info("⚠️ **Required:** Select at least one output format before processing")
        output_formats = st.multiselect(
            "Select output format(s)",
            options=["CSV", "Excel (XLSX)", "JSON", "Markdown (MD)"],
            default=["CSV"],
            help="You can select multiple formats. At least one format must be selected to proceed.",
        )

        if not output_formats:
            st.warning("⚠️ Please select at least one output format")

        # JSON Format Selection (only shown if JSON is selected)
        json_format = "standard"
        if "JSON" in output_formats:
            st.markdown("#### JSON Format Options")
            json_format = st.selectbox(
                "JSON Format",
                options=["standard", "structured", "multi_file", "unified"],
                format_func=lambda x: {
                    "standard": "Standard (array of records)",
                    "structured": "Structured (with metadata wrapper)",
                    "multi_file": "Multi-file (organized by filename)",
                    "unified": "Unified (metadata in each record)",
                }.get(x, x),
                index=0,
                help="Choose how JSON data should be structured",
            )
            st.session_state.json_format = json_format

        # GPU Check Section
        st.markdown("---")
        st.subheader("🖥️ GPU Status")

        with st.expander("Check GPU Availability", expanded=False):
            if st.button("🔍 Check GPU", use_container_width=True):
                with st.spinner("Checking GPU status..."):
                    gpu_info = check_gpu_comprehensive()

                    # Overall Status
                    status_emoji = {
                        "ready": "✅",
                        "driver_only": "⚠️",
                        "not_available": "❌",
                        "unknown": "❓",
                    }
                    status_text = {
                        "ready": "GPU Ready",
                        "driver_only": "Driver Only",
                        "not_available": "Not Available",
                        "unknown": "Unknown",
                    }

                    status = gpu_info["overall_status"]
                    st.markdown(
                        f"### {status_emoji.get(status, '❓')} Status: {status_text.get(status, 'Unknown')}"
                    )

                    # NVIDIA Driver Check
                    st.markdown("#### 📦 NVIDIA Driver")
                    nvidia = gpu_info["nvidia_driver"]
                    if nvidia["available"]:
                        st.success("✅ NVIDIA driver detected")
                        if nvidia["driver_version"]:
                            st.text(f"Version: {nvidia['driver_version']}")
                        if nvidia["cuda_version"]:
                            st.text(f"CUDA: {nvidia['cuda_version']}")
                        if nvidia["gpu_name"]:
                            st.text(f"GPU: {nvidia['gpu_name']}")
                        if nvidia["gpu_memory"]:
                            st.text(f"Memory: {nvidia['gpu_memory']}")
                    else:
                        st.error("❌ NVIDIA driver not available")
                        if nvidia["error"]:
                            st.text(f"Error: {nvidia['error']}")

                    # PyTorch CUDA Check
                    st.markdown("#### 🔥 PyTorch CUDA Support")
                    pytorch = gpu_info["pytorch_cuda"]
                    if pytorch["available"]:
                        st.success("✅ PyTorch CUDA support available")
                        if pytorch["pytorch_version"]:
                            st.text(f"PyTorch: {pytorch['pytorch_version']}")
                        if pytorch["cuda_version"]:
                            st.text(f"CUDA: {pytorch['cuda_version']}")
                        if pytorch["cudnn_version"]:
                            st.text(f"cuDNN: {pytorch['cudnn_version']}")

                        if pytorch["gpu_count"] > 0:
                            st.text(f"GPUs Detected: {pytorch['gpu_count']}")
                            for gpu in pytorch["gpus"]:
                                st.text(
                                    f"  • GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']} GB)"
                                )
                    else:
                        st.warning("⚠️ PyTorch CUDA support not available")
                        if pytorch["error"]:
                            st.text(f"Error: {pytorch['error']}")
                        elif pytorch["pytorch_version"]:
                            st.text(
                                f"PyTorch {pytorch['pytorch_version']} installed, but CUDA not available"
                            )
                            st.info(
                                "💡 Tip: Reinstall PyTorch with CUDA support for GPU acceleration"
                            )

                    # Recommendations
                    if gpu_info["recommendations"]:
                        st.markdown("#### 💡 Recommendations")
                        for rec in gpu_info["recommendations"]:
                            st.info(rec)

                    # Summary
                    if gpu_info["gpu_ready"]:
                        st.success(
                            "🎉 GPU is ready for acceleration! Some extraction methods (like EasyOCR) can use GPU for faster processing."
                        )
                    else:
                        st.info(
                            "ℹ️ GPU acceleration not available. Extraction will use CPU, which may be slower but still functional."
                        )

                    # Store in session state
                    st.session_state.gpu_info = gpu_info

            # Show cached GPU info if available
            if "gpu_info" in st.session_state:
                gpu_info = st.session_state.gpu_info
                status = gpu_info["overall_status"]
                status_emoji = {
                    "ready": "✅",
                    "driver_only": "⚠️",
                    "not_available": "❌",
                    "unknown": "❓",
                }

                if status == "ready":
                    st.success(f"{status_emoji.get(status)} GPU Ready")
                elif status == "driver_only":
                    st.warning(f"{status_emoji.get(status)} GPU Driver Only")
                else:
                    st.info(f"{status_emoji.get(status)} GPU Not Available")

        # Advanced Testing Section
        st.markdown("---")
        st.subheader("🧪 Advanced Testing")

        with st.expander("Gemini API Tests", expanded=False):
            st.info("Run advanced tests to verify your Gemini API setup")
            st.caption("💡 Test PDFs should be placed in the `test_file/` directory")

            # Show available test files
            import os

            test_file_dir = "test_file"
            if os.path.exists(test_file_dir):
                test_files = [
                    f for f in os.listdir(test_file_dir) if f.lower().endswith(".pdf")
                ]
                if test_files:
                    st.success(
                        f"✅ Found {len(test_files)} test file(s): {', '.join(test_files)}"
                    )
                else:
                    st.warning("⚠️ No PDF files found in `test_file/` directory")
            else:
                st.info(
                    "ℹ️ `test_file/` directory not found. Tests will use fallback locations"
                )

            # Check if Gemini API key is available
            gemini_key = gem_api_key or st.session_state.api_keys.get("gemini", "")

            if not gemini_key:
                st.warning("⚠️ Gemini API key required for testing")
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button(
                        "Test 1: Basic",
                        use_container_width=True,
                        help="Test model creation, text generation, and JSON output",
                    ):
                        with st.spinner("Running basic tests..."):
                            success, results = test_gemini_basic(gemini_key)

                            if success:
                                st.success("✅ Basic Tests Passed")
                            else:
                                st.error("❌ Basic Tests Failed")

                            # Show test details
                            for test in results.get("tests", []):
                                if test["status"] == "success":
                                    st.success(f"✓ {test['name']}: {test['message']}")
                                elif test["status"] == "error":
                                    st.error(f"✗ {test['name']}: {test['message']}")
                                else:
                                    st.warning(f"⚠ {test['name']}: {test['message']}")

                            if results.get("errors"):
                                for error in results["errors"]:
                                    st.error(f"Error: {error}")

                            st.session_state.gemini_test_basic = results

                with col2:
                    if st.button(
                        "Test 2: File Upload",
                        use_container_width=True,
                        help="Test PDF file upload and processing (uses test_file/ directory)",
                    ):
                        with st.spinner(
                            "Testing file upload (this may take 30-60 seconds)..."
                        ):
                            success, results = test_gemini_file_upload(gemini_key)

                            if success:
                                st.success("✅ File Upload Test Passed")
                                if results.get("model_used"):
                                    st.info(f"Model used: {results['model_used']}")
                            else:
                                st.error("❌ File Upload Test Failed")

                            # Show test details
                            for test in results.get("tests", []):
                                if test["status"] == "success":
                                    st.success(f"✓ {test['name']}: {test['message']}")
                                elif test["status"] == "error":
                                    st.error(f"✗ {test['name']}: {test['message']}")
                                else:
                                    st.warning(f"⚠ {test['name']}: {test['message']}")

                            if results.get("errors"):
                                for error in results["errors"]:
                                    st.error(f"Error: {error}")

                            st.session_state.gemini_test_file_upload = results

                with col3:
                    if st.button(
                        "Test 3: JSON Upload",
                        use_container_width=True,
                        help="Test PDF upload with JSON output format (uses test_file/ directory)",
                    ):
                        with st.spinner(
                            "Testing JSON output with file upload (this may take 30-60 seconds)..."
                        ):
                            success, results = test_gemini_json_upload(gemini_key)

                            if success:
                                st.success("✅ JSON Upload Test Passed")
                                if results.get("model_used"):
                                    st.info(f"Model used: {results['model_used']}")
                            else:
                                st.error("❌ JSON Upload Test Failed")

                            # Show test details
                            for test in results.get("tests", []):
                                if test["status"] == "success":
                                    st.success(f"✓ {test['name']}: {test['message']}")
                                elif test["status"] == "error":
                                    st.error(f"✗ {test['name']}: {test['message']}")
                                else:
                                    st.warning(f"⚠ {test['name']}: {test['message']}")

                            if results.get("errors"):
                                for error in results["errors"]:
                                    st.error(f"Error: {error}")

                            st.session_state.gemini_test_json_upload = results

                # Run All Tests Button
                if st.button(
                    "🚀 Run All Tests", use_container_width=True, type="primary"
                ):
                    with st.spinner(
                        "Running all Gemini API tests (this may take 2-3 minutes)..."
                    ):
                        test_results = {}

                        # Test 1: Basic
                        st.markdown("#### Test 1: Basic Functionality")
                        success1, results1 = test_gemini_basic(gemini_key)
                        test_results["basic"] = (success1, results1)

                        # Test 2: File Upload
                        st.markdown("#### Test 2: File Upload")
                        success2, results2 = test_gemini_file_upload(gemini_key)
                        test_results["file_upload"] = (success2, results2)

                        # Test 3: JSON Upload
                        st.markdown("#### Test 3: JSON Output with File Upload")
                        success3, results3 = test_gemini_json_upload(gemini_key)
                        test_results["json_upload"] = (success3, results3)

                        # Summary
                        st.markdown("### 📊 Test Summary")
                        all_passed = success1 and success2 and success3

                        if all_passed:
                            st.success(
                                "🎉 All tests passed! Your Gemini API setup is working correctly."
                            )
                        else:
                            st.warning("⚠️ Some tests failed. Check the details above.")

                        # Store all results
                        st.session_state.gemini_all_tests = test_results

    # Return configuration dictionary
    return {
        "ocr_method": ocr_method,
        "local_model": local_model,
        "hf_model": hf_model,
        "use_cpu_mode": use_cpu_mode,
        "use_pretty_output": use_pretty_output,
        "api_keys": {
            "huggingface": hf_api_key
            or st.session_state.api_keys.get("huggingface", ""),
            "datalab": dl_api_key or st.session_state.api_keys.get("datalab", ""),
            "gemini": gem_api_key or st.session_state.api_keys.get("gemini", ""),
            "deepseek": ds_api_key or st.session_state.api_keys.get("deepseek", ""),
        },
        "auto_detect_fields": auto_detect_fields,
        "use_custom_fields": use_custom_fields,
        "output_formats": output_formats,
        "json_format": json_format,
    }
