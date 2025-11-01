#!/usr/bin/env python3
"""
Streamlit webapp for EC (Encumbrance Certificate) PDF extraction.
Modern, clean UI with multiple OCR options and API key management.
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import time
from typing import Dict, List, Set
from pathlib import Path

# Import our modules
from api_key_manager import save_api_key, load_api_key, save_all_api_keys, get_storage_info
from test_api_keys import (
    test_datalab_api_key,
    test_huggingface_api_key,
    test_gemini_api_key,
    test_deepseek_api_key
)
from extraction_service import extract_data, process_multiple_files, save_uploaded_file
from utils import (
    filter_fields,
    format_dataframe,
    get_default_fields,
    get_field_descriptions,
    validate_pdf_file,
    format_file_size,
    get_file_size_mb
)
from gpu_check_utils import check_gpu_comprehensive
from gemini_test_modules import (
    test_gemini_basic,
    test_gemini_file_upload,
    test_gemini_json_upload
)
from model_info import (
    get_local_ocr_models,
    get_huggingface_ocr_models,
    get_model_search_url
)
from model_fetcher import clear_cache, fetch_ocr_models_from_huggingface

# Page configuration
st.set_page_config(
    page_title="EC Data Extraction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            'huggingface': '',
            'datalab': '',
            'gemini': '',
            'deepseek': ''
        }
    
    if 'api_key_tests' not in st.session_state:
        st.session_state.api_key_tests = {}
    
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    
    if 'selected_fields' not in st.session_state:
        st.session_state.selected_fields = set(get_default_fields())
    
    if 'custom_fields' not in st.session_state:
        st.session_state.custom_fields = []


def load_saved_api_keys():
    """Load API keys from .env file."""
    providers = ['huggingface', 'datalab', 'gemini', 'deepseek']
    for provider in providers:
        saved_key = load_api_key(provider)
        if saved_key and not st.session_state.api_keys.get(provider):
            st.session_state.api_keys[provider] = saved_key


def main():
    """Main application function."""
    initialize_session_state()
    load_saved_api_keys()
    
    # Header
    st.markdown('<div class="main-header">📄 PDF Data Extraction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Extract structured data from PDF documents using multiple OCR and AI methods</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OCR Method Selection
        st.subheader("🔧 Extraction Method")
        
        # Define methods with tooltips/descriptions
        method_descriptions = {
            "EasyOCR": "Fast CPU-based OCR using EasyOCR library. Lightweight (~100MB models), quick setup, good for simple text extraction. No API key needed.",
            "PyTesseract": "Google's Tesseract OCR engine via PyTesseract. Fast, lightweight, works well for text-based PDFs. Requires Tesseract-OCR installed on system.",
            "Local Model": "Download and run OCR models locally (Chandra OCR). High accuracy for documents/tables. First run downloads ~2GB models. Supports GPU/CPU.",
            "HuggingFace": "Use HuggingFace Inference API for OCR. Requires HuggingFace API key. Cloud-based, no local model download needed.",
            "Datalab API": "High-accuracy OCR via Datalab Marker API. Best for structured documents and tables. Requires Datalab API key. Recommended for production.",
            "Gemini AI": "Google's Gemini AI for intelligent data extraction. Understands context and can extract custom fields. Requires Gemini API key.",
            "Deepseek AI": "Deepseek AI for document extraction with vision capabilities. Can extract custom fields. Requires Deepseek API key."
        }
        
        # Create selectbox with formatted labels that include descriptions
        method_options = list(method_descriptions.keys())
        
        ocr_method = st.selectbox(
            "Select extraction method",
            options=method_options,
            index=0,
            help="Choose an OCR method. Hover over each option to see details, or see description below.",
            key="ocr_method_select"
        )
        
        # Show detailed description for selected method with info box
        if ocr_method in method_descriptions:
            st.info(f"**{ocr_method}:** {method_descriptions[ocr_method]}")
        
        # Local Model Selection
        local_model = None
        use_cpu_mode = False
        use_pretty_output = False
        
        if ocr_method == "Local Model":
            st.markdown("---")
            st.subheader("🖥️ Local Model Configuration")
            st.info("📥 Models will be downloaded automatically on first use (~2GB for Chandra)")
            
            # Get HuggingFace API key if available (defined later in sidebar)
            hf_api_key_available = st.session_state.api_keys.get('huggingface', '')
            
            # Model fetching options
            col_refresh, col_info = st.columns([1, 2])
            with col_refresh:
                refresh_models = st.button("🔄 Refresh Model List", help="Fetch latest models from HuggingFace Hub")
            
            if refresh_models:
                clear_cache()
                st.success("✅ Cache cleared! Model list will refresh on next selection.")
            
            # Try to use HuggingFace API key if available
            use_dynamic = bool(hf_api_key_available) or refresh_models
            
            # Model selection
            try:
                with st.spinner("Loading available models..." if use_dynamic else ""):
                    local_models = get_local_ocr_models(
                        api_key=hf_api_key_available if hf_api_key_available else None,
                        use_dynamic=use_dynamic
                    )
            except Exception as e:
                st.warning(f"⚠️ Could not fetch models dynamically: {e}. Using default models.")
                local_models = get_local_ocr_models(use_dynamic=False)
            
            model_options = [f"{m['name']} ({m['id']})" for m in local_models]
            model_ids = [m['id'] for m in local_models]
            
            selected_model_idx = st.selectbox(
                "Select OCR Model",
                range(len(model_options)),
                format_func=lambda x: model_options[x],
                help="Choose which OCR model to download and use locally"
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
                is_supported = local_model == 'datalab-to/chandra' or local_model.startswith('datalab-to/chandra')
                support_reason = "Chandra CLI model" if is_supported else "Needs custom implementation"
            
            with st.expander(f"ℹ️ About {selected_model_info['name']}"):
                st.write(f"**Description:** {selected_model_info['description']}")
                st.write(f"**Model Size:** {selected_model_info.get('size', 'Unknown')}")
                st.write(f"**Download Time:** {selected_model_info.get('download_time', 'Unknown')}")
                st.write(f"**GPU Support:** {'Yes' if selected_model_info.get('supports_gpu') else 'No'}")
                st.write(f"**CPU Support:** {'Yes' if selected_model_info.get('supports_cpu') else 'No'}")
                if selected_model_info.get('downloads'):
                    st.write(f"**Downloads:** {selected_model_info['downloads']:,}")
                if selected_model_info.get('verified'):
                    st.write("**Status:** ✅ Verified")
                
                # Show support status with detailed reason
                if is_supported:
                    st.success(f"✅ **Supported** - {support_reason}")
                else:
                    st.warning(f"⚠️ **Limited Support** - {support_reason}")
                    st.info("💡 You can still try this model - it will attempt to load via transformers library. Some models may need additional dependencies.")
                
                st.markdown(f"**Model Page:** [{selected_model_info['name']}]({selected_model_info['url']})")
            
            # Info about dynamic fetching
            if not hf_api_key_available:
                st.caption("💡 **Tip:** Add your HuggingFace API key below to fetch the latest models automatically")
            else:
                st.caption("✅ Using HuggingFace API key - models are fetched automatically")
            
            # Options for Chandra model
            if local_model == 'datalab-to/chandra':
                use_cpu_mode = st.checkbox(
                    "Force CPU Mode",
                    value=False,
                    help="Force CPU usage even if GPU is available (slower but more stable)"
                )
                use_pretty_output = st.checkbox(
                    "Use Formatted Output",
                    value=False,
                    help="Use human-readable output formatting (filters technical details)"
                )
        
        # HuggingFace Model Selection
        hf_model = None
        if ocr_method == "HuggingFace":
            st.markdown("---")
            st.subheader("🤗 Model Selection")
            st.info("📡 Models are fetched from HuggingFace Hub. Requires HuggingFace API key for best results.")
            
            # Get HuggingFace API key if available (from session state, will be updated when user enters it)
            hf_api_key_for_models = st.session_state.api_keys.get('huggingface', '')
            
            # Model fetching options
            col_refresh_hf, col_info_hf = st.columns([1, 2])
            with col_refresh_hf:
                refresh_hf_models = st.button("🔄 Refresh Model List", key="refresh_hf", help="Fetch latest models from HuggingFace Hub")
            
            if refresh_hf_models:
                from model_fetcher import clear_cache
                clear_cache()
                st.success("✅ Cache cleared! Model list will refresh.")
            
            # Determine if we should use dynamic fetching
            use_dynamic_hf = bool(hf_api_key_for_models) or refresh_hf_models
            
            # Fetch available models
            try:
                with st.spinner("Loading available HuggingFace models..." if use_dynamic_hf else ""):
                    hf_models = get_huggingface_ocr_models(
                        api_key=hf_api_key_for_models if hf_api_key_for_models else None,
                        use_dynamic=use_dynamic_hf
                    )
            except Exception as e:
                st.warning(f"⚠️ Could not fetch models dynamically: {e}. Using default models.")
                hf_models = get_huggingface_ocr_models(use_dynamic=False)
            
            if not hf_models:
                st.error("No models available. Please check your HuggingFace API key or internet connection.")
                hf_models = get_huggingface_ocr_models(use_dynamic=False)
            
            hf_model_options = [f"{m['name']} ({m['id']})" for m in hf_models]
            hf_model_ids = [m['id'] for m in hf_models]
            
            selected_hf_idx = st.selectbox(
                "Select HuggingFace Model",
                range(len(hf_model_options)),
                format_func=lambda x: hf_model_options[x],
                index=0,  # Default to first model
                help="Choose which HuggingFace model to use via API"
            )
            hf_model = hf_model_ids[selected_hf_idx]
            
            # Show model info and link
            selected_hf_info = hf_models[selected_hf_idx]
            with st.expander(f"ℹ️ About {selected_hf_info['name']}"):
                st.write(f"**Description:** {selected_hf_info['description']}")
                st.write(f"**Model ID:** `{selected_hf_info['id']}`")
                st.write(f"**API Compatible:** {'✅ Yes' if selected_hf_info.get('api_compatible', True) else '⚠️ Limited'}")
                if selected_hf_info.get('downloads'):
                    st.write(f"**Downloads:** {selected_hf_info['downloads']:,}")
                if selected_hf_info.get('verified'):
                    st.write("**Status:** ✅ Verified")
                if selected_hf_info.get('size'):
                    st.write(f"**Model Size:** {selected_hf_info['size']}")
                if selected_hf_info.get('note'):
                    st.info(f"**Note:** {selected_hf_info['note']}")
                st.markdown(f"**Model Page:** [{selected_hf_info['name']}]({selected_hf_info['url']})")
            
            # Info about dynamic fetching
            if not hf_api_key_for_models:
                st.caption("💡 **Tip:** Add your HuggingFace API key below to fetch the latest models automatically")
            else:
                st.caption("✅ Using HuggingFace API key - models are fetched automatically")
            
            # Link to browse more models
            st.markdown(f"[🔍 Browse more OCR models on HuggingFace]({get_model_search_url()})")
            
            # Fallback: allow manual entry
            st.markdown("**Or enter custom model name:**")
            custom_model = st.text_input(
                "Custom Model Name",
                value="",
                placeholder="username/model-name",
                help="Enter a custom HuggingFace model name (e.g., datalab-to/chandra)"
            )
            if custom_model:
                hf_model = custom_model
                st.info(f"Using custom model: `{hf_model}`")
        
        # API Key Management Section
        st.markdown("---")
        st.subheader("🔑 API Keys")
        
        # Show storage info
        with st.expander("ℹ️ About API Key Storage"):
            st.info(get_storage_info())
        
        # HuggingFace API Key
        st.markdown("#### HuggingFace API Key")
        hf_key_col1, hf_key_col2 = st.columns([3, 1])
        with hf_key_col1:
            hf_api_key = st.text_input(
                "HF API Key",
                value=st.session_state.api_keys.get('huggingface', ''),
                type="password",
                key="hf_key_input",
                label_visibility="collapsed"
            )
        with hf_key_col2:
            if st.button("Test", key="test_hf", use_container_width=True):
                if hf_api_key:
                    with st.spinner("Testing..."):
                        success, message = test_huggingface_api_key(hf_api_key)
                        st.session_state.api_key_tests['huggingface'] = (success, message)
                        if success:
                            st.session_state.api_keys['huggingface'] = hf_api_key
                else:
                    st.session_state.api_key_tests['huggingface'] = (False, "Please enter an API key")
        
        if 'huggingface' in st.session_state.api_key_tests:
            success, msg = st.session_state.api_key_tests['huggingface']
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
                value=st.session_state.api_keys.get('datalab', ''),
                type="password",
                key="dl_key_input",
                label_visibility="collapsed"
            )
        with dl_key_col2:
            if st.button("Test", key="test_dl", use_container_width=True):
                if dl_api_key:
                    with st.spinner("Testing..."):
                        success, message = test_datalab_api_key(dl_api_key)
                        st.session_state.api_key_tests['datalab'] = (success, message)
                        if success:
                            st.session_state.api_keys['datalab'] = dl_api_key
                else:
                    st.session_state.api_key_tests['datalab'] = (False, "Please enter an API key")
        
        if 'datalab' in st.session_state.api_key_tests:
            success, msg = st.session_state.api_key_tests['datalab']
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
                value=st.session_state.api_keys.get('gemini', ''),
                type="password",
                key="gem_key_input",
                label_visibility="collapsed"
            )
        with gem_key_col2:
            if st.button("Test", key="test_gem", use_container_width=True):
                if gem_api_key:
                    with st.spinner("Testing..."):
                        success, message = test_gemini_api_key(gem_api_key)
                        st.session_state.api_key_tests['gemini'] = (success, message)
                        if success:
                            st.session_state.api_keys['gemini'] = gem_api_key
                else:
                    st.session_state.api_key_tests['gemini'] = (False, "Please enter an API key")
        
        if 'gemini' in st.session_state.api_key_tests:
            success, msg = st.session_state.api_key_tests['gemini']
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
                value=st.session_state.api_keys.get('deepseek', ''),
                type="password",
                key="ds_key_input",
                label_visibility="collapsed"
            )
        with ds_key_col2:
            if st.button("Test", key="test_ds", use_container_width=True):
                if ds_api_key:
                    with st.spinner("Testing..."):
                        success, message = test_deepseek_api_key(ds_api_key)
                        st.session_state.api_key_tests['deepseek'] = (success, message)
                        if success:
                            st.session_state.api_keys['deepseek'] = ds_api_key
                else:
                    st.session_state.api_key_tests['deepseek'] = (False, "Please enter an API key")
        
        if 'deepseek' in st.session_state.api_key_tests:
            success, msg = st.session_state.api_key_tests['deepseek']
            if success:
                st.success(msg)
            else:
                st.error(msg)
        
        # Save Keys Button
        st.markdown("---")
        if st.button("💾 Save All Keys Locally", use_container_width=True):
            keys_to_save = {
                'huggingface': hf_api_key or st.session_state.api_keys.get('huggingface', ''),
                'datalab': dl_api_key or st.session_state.api_keys.get('datalab', ''),
                'gemini': gem_api_key or st.session_state.api_keys.get('gemini', ''),
                'deepseek': ds_api_key or st.session_state.api_keys.get('deepseek', '')
            }
            if save_all_api_keys(keys_to_save):
                st.success("✅ All API keys saved to .env file!")
                st.info("⚠️ Please make a backup copy of the .env file.")
            else:
                st.error("❌ Failed to save some API keys")
        
        # Field Selection
        st.markdown("---")
        st.subheader("📋 Field Selection")
        
        # Custom Fields Input
        with st.expander("➕ Add Custom Fields", expanded=False):
            st.info("Define your own fields to extract. Enter one field name per line.")
            custom_fields_text = st.text_area(
                "Custom Fields",
                placeholder="Enter field names (one per line):\nVillage Name\nRegistration Date\nProperty Type\netc.",
                height=100,
                help="Enter custom field names that exist in your PDF. One field per line."
            )
            
            if custom_fields_text:
                custom_fields_list = [f.strip() for f in custom_fields_text.split('\n') if f.strip()]
                st.session_state.custom_fields = custom_fields_list
            else:
                st.session_state.custom_fields = []
        
        # Use custom fields or default
        use_custom_fields = st.checkbox(
            "Use Custom Fields Only",
            value=False,
            help="If checked, only extract the custom fields you defined above. Otherwise, use default fields."
        )
        
        if use_custom_fields and st.session_state.get('custom_fields'):
            # Use only custom fields
            st.session_state.selected_fields = set(['filename'] + st.session_state.custom_fields)
            st.info(f"Extracting {len(st.session_state.custom_fields)} custom field(s)")
        else:
            # Use default fields with selection
            default_fields = get_default_fields()
            field_descriptions = get_field_descriptions()
            
            # Select all / Deselect all buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All", use_container_width=True):
                    st.session_state.selected_fields = set(default_fields)
            with col2:
                if st.button("Deselect All", use_container_width=True):
                    st.session_state.selected_fields = {'filename'}  # Always keep filename
            
            # Field checkboxes
            for field in default_fields:
                checked = field in st.session_state.selected_fields
                disabled = field == 'filename'  # Always include filename
                description = field_descriptions.get(field, '')
                
                if st.checkbox(
                    field,
                    value=checked,
                    disabled=disabled,
                    help=description,
                    key=f"field_{field}"
                ):
                    st.session_state.selected_fields.add(field)
                else:
                    if not disabled:
                        st.session_state.selected_fields.discard(field)
            
            # Add custom fields to selection if any
            if st.session_state.get('custom_fields'):
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
            help="You can select multiple formats. At least one format must be selected to proceed."
        )
        
        if not output_formats:
            st.warning("⚠️ Please select at least one output format")
        
        # GPU Check Section
        st.markdown("---")
        st.subheader("🖥️ GPU Status")
        
        with st.expander("Check GPU Availability", expanded=False):
            if st.button("🔍 Check GPU", use_container_width=True):
                with st.spinner("Checking GPU status..."):
                    gpu_info = check_gpu_comprehensive()
                    
                    # Overall Status
                    status_emoji = {
                        'ready': '✅',
                        'driver_only': '⚠️',
                        'not_available': '❌',
                        'unknown': '❓'
                    }
                    status_text = {
                        'ready': 'GPU Ready',
                        'driver_only': 'Driver Only',
                        'not_available': 'Not Available',
                        'unknown': 'Unknown'
                    }
                    
                    status = gpu_info['overall_status']
                    st.markdown(f"### {status_emoji.get(status, '❓')} Status: {status_text.get(status, 'Unknown')}")
                    
                    # NVIDIA Driver Check
                    st.markdown("#### 📦 NVIDIA Driver")
                    nvidia = gpu_info['nvidia_driver']
                    if nvidia['available']:
                        st.success("✅ NVIDIA driver detected")
                        if nvidia['driver_version']:
                            st.text(f"Version: {nvidia['driver_version']}")
                        if nvidia['cuda_version']:
                            st.text(f"CUDA: {nvidia['cuda_version']}")
                        if nvidia['gpu_name']:
                            st.text(f"GPU: {nvidia['gpu_name']}")
                        if nvidia['gpu_memory']:
                            st.text(f"Memory: {nvidia['gpu_memory']}")
                    else:
                        st.error(f"❌ NVIDIA driver not available")
                        if nvidia['error']:
                            st.text(f"Error: {nvidia['error']}")
                    
                    # PyTorch CUDA Check
                    st.markdown("#### 🔥 PyTorch CUDA Support")
                    pytorch = gpu_info['pytorch_cuda']
                    if pytorch['available']:
                        st.success("✅ PyTorch CUDA support available")
                        if pytorch['pytorch_version']:
                            st.text(f"PyTorch: {pytorch['pytorch_version']}")
                        if pytorch['cuda_version']:
                            st.text(f"CUDA: {pytorch['cuda_version']}")
                        if pytorch['cudnn_version']:
                            st.text(f"cuDNN: {pytorch['cudnn_version']}")
                        
                        if pytorch['gpu_count'] > 0:
                            st.text(f"GPUs Detected: {pytorch['gpu_count']}")
                            for gpu in pytorch['gpus']:
                                st.text(f"  • GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']} GB)")
                    else:
                        st.warning("⚠️ PyTorch CUDA support not available")
                        if pytorch['error']:
                            st.text(f"Error: {pytorch['error']}")
                        elif pytorch['pytorch_version']:
                            st.text(f"PyTorch {pytorch['pytorch_version']} installed, but CUDA not available")
                            st.info("💡 Tip: Reinstall PyTorch with CUDA support for GPU acceleration")
                    
                    # Recommendations
                    if gpu_info['recommendations']:
                        st.markdown("#### 💡 Recommendations")
                        for rec in gpu_info['recommendations']:
                            st.info(rec)
                    
                    # Summary
                    if gpu_info['gpu_ready']:
                        st.success("🎉 GPU is ready for acceleration! Some extraction methods (like EasyOCR) can use GPU for faster processing.")
                    else:
                        st.info("ℹ️ GPU acceleration not available. Extraction will use CPU, which may be slower but still functional.")
                    
                    # Store in session state
                    st.session_state.gpu_info = gpu_info
            
            # Show cached GPU info if available
            if 'gpu_info' in st.session_state:
                gpu_info = st.session_state.gpu_info
                status = gpu_info['overall_status']
                status_emoji = {'ready': '✅', 'driver_only': '⚠️', 'not_available': '❌', 'unknown': '❓'}
                
                if status == 'ready':
                    st.success(f"{status_emoji.get(status)} GPU Ready")
                elif status == 'driver_only':
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
            test_file_dir = "test_file"
            if os.path.exists(test_file_dir):
                test_files = [f for f in os.listdir(test_file_dir) if f.lower().endswith('.pdf')]
                if test_files:
                    st.success(f"✅ Found {len(test_files)} test file(s): {', '.join(test_files)}")
                else:
                    st.warning("⚠️ No PDF files found in `test_file/` directory")
            else:
                st.info("ℹ️ `test_file/` directory not found. Tests will use fallback locations")
            
            # Check if Gemini API key is available
            gemini_key = gem_api_key or st.session_state.api_keys.get('gemini', '')
            
            if not gemini_key:
                st.warning("⚠️ Gemini API key required for testing")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Test 1: Basic", use_container_width=True, help="Test model creation, text generation, and JSON output"):
                        with st.spinner("Running basic tests..."):
                            success, results = test_gemini_basic(gemini_key)
                            
                            if success:
                                st.success("✅ Basic Tests Passed")
                            else:
                                st.error("❌ Basic Tests Failed")
                            
                            # Show test details
                            for test in results.get('tests', []):
                                if test['status'] == 'success':
                                    st.success(f"✓ {test['name']}: {test['message']}")
                                elif test['status'] == 'error':
                                    st.error(f"✗ {test['name']}: {test['message']}")
                                else:
                                    st.warning(f"⚠ {test['name']}: {test['message']}")
                            
                            if results.get('errors'):
                                for error in results['errors']:
                                    st.error(f"Error: {error}")
                            
                            st.session_state.gemini_test_basic = results
                
                with col2:
                    if st.button("Test 2: File Upload", use_container_width=True, help="Test PDF file upload and processing (uses test_file/ directory)"):
                        with st.spinner("Testing file upload (this may take 30-60 seconds)..."):
                            success, results = test_gemini_file_upload(gemini_key)
                            
                            if success:
                                st.success("✅ File Upload Test Passed")
                                if results.get('model_used'):
                                    st.info(f"Model used: {results['model_used']}")
                            else:
                                st.error("❌ File Upload Test Failed")
                            
                            # Show test details
                            for test in results.get('tests', []):
                                if test['status'] == 'success':
                                    st.success(f"✓ {test['name']}: {test['message']}")
                                elif test['status'] == 'error':
                                    st.error(f"✗ {test['name']}: {test['message']}")
                                else:
                                    st.warning(f"⚠ {test['name']}: {test['message']}")
                            
                            if results.get('errors'):
                                for error in results['errors']:
                                    st.error(f"Error: {error}")
                            
                            st.session_state.gemini_test_file_upload = results
                
                with col3:
                    if st.button("Test 3: JSON Upload", use_container_width=True, help="Test PDF upload with JSON output format (uses test_file/ directory)"):
                        with st.spinner("Testing JSON output with file upload (this may take 30-60 seconds)..."):
                            success, results = test_gemini_json_upload(gemini_key)
                            
                            if success:
                                st.success("✅ JSON Upload Test Passed")
                                if results.get('model_used'):
                                    st.info(f"Model used: {results['model_used']}")
                            else:
                                st.error("❌ JSON Upload Test Failed")
                            
                            # Show test details
                            for test in results.get('tests', []):
                                if test['status'] == 'success':
                                    st.success(f"✓ {test['name']}: {test['message']}")
                                elif test['status'] == 'error':
                                    st.error(f"✗ {test['name']}: {test['message']}")
                                else:
                                    st.warning(f"⚠ {test['name']}: {test['message']}")
                            
                            if results.get('errors'):
                                for error in results['errors']:
                                    st.error(f"Error: {error}")
                            
                            st.session_state.gemini_test_json_upload = results
                
                # Run All Tests Button
                if st.button("🚀 Run All Tests", use_container_width=True, type="primary"):
                    with st.spinner("Running all Gemini API tests (this may take 2-3 minutes)..."):
                        test_results = {}
                        
                        # Test 1: Basic
                        st.markdown("#### Test 1: Basic Functionality")
                        success1, results1 = test_gemini_basic(gemini_key)
                        test_results['basic'] = (success1, results1)
                        
                        # Test 2: File Upload
                        st.markdown("#### Test 2: File Upload")
                        success2, results2 = test_gemini_file_upload(gemini_key)
                        test_results['file_upload'] = (success2, results2)
                        
                        # Test 3: JSON Upload
                        st.markdown("#### Test 3: JSON Output with File Upload")
                        success3, results3 = test_gemini_json_upload(gemini_key)
                        test_results['json_upload'] = (success3, results3)
                        
                        # Summary
                        st.markdown("### 📊 Test Summary")
                        all_passed = success1 and success2 and success3
                        
                        if all_passed:
                            st.success("🎉 All tests passed! Your Gemini API setup is working correctly.")
                        else:
                            st.warning("⚠️ Some tests failed. Check the details above.")
                        
                        # Store all results
                        st.session_state.gemini_all_tests = test_results
    
    # Main Content Area
    st.markdown("---")
    
    # File Upload Section
    st.subheader("📤 Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=['pdf'],
        accept_multiple_files=True,
        help="You can upload one or multiple PDF files"
    )
    
    if uploaded_files:
        st.markdown("#### Selected Files:")
        file_info = []
        for file in uploaded_files:
            size_mb = len(file.getvalue()) / (1024 * 1024)
            file_info.append({
                'Filename': file.name,
                'Size': format_file_size(size_mb)
            })
        st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)
        
        # Validate files
        invalid_files = []
        for file in uploaded_files:
            is_valid, error_msg = validate_pdf_file(file)
            if not is_valid:
                invalid_files.append((file.name, error_msg))
        
        if invalid_files:
            st.error("⚠️ Invalid files detected:")
            for filename, error in invalid_files:
                st.error(f"  - {filename}: {error}")
    
    # Process Button
    st.markdown("---")
    process_col1, process_col2 = st.columns([1, 4])
    with process_col1:
        process_button = st.button(
            "🚀 Extract Data",
            type="primary",
            use_container_width=True
        )
    
    # Processing and Results
    if process_button:
        if not uploaded_files:
            st.error("⚠️ Please upload at least one PDF file")
        elif not output_formats:
            st.error("⚠️ Please select at least one output format in the sidebar before processing")
        else:
            # Prepare API keys
            api_keys_dict = {
                'huggingface': hf_api_key or st.session_state.api_keys.get('huggingface', ''),
                'datalab': dl_api_key or st.session_state.api_keys.get('datalab', ''),
                'gemini': gem_api_key or st.session_state.api_keys.get('gemini', ''),
                'deepseek': ds_api_key or st.session_state.api_keys.get('deepseek', '')
            }
            
            # Map method names
            method_map = {
                'EasyOCR': 'easyocr',
                'PyTesseract': 'pytesseract',
                'Local Model': 'local',
                'HuggingFace': 'huggingface',
                'Datalab API': 'datalab',
                'Gemini AI': 'gemini',
                'Deepseek AI': 'deepseek'
            }
            selected_method = method_map[ocr_method]
            
            # Check API key requirements
            if selected_method in ['local', 'easyocr', 'pytesseract']:
                # No API key needed for local models or free OCR
                pass
            elif selected_method == 'huggingface' and not api_keys_dict.get('huggingface'):
                st.error("❌ HuggingFace API key is required for this method")
            elif selected_method == 'datalab' and not api_keys_dict.get('datalab'):
                st.error("❌ Datalab API key is required for this method")
            elif selected_method == 'gemini' and not api_keys_dict.get('gemini'):
                st.error("❌ Gemini API key is required for this method")
            elif selected_method == 'deepseek' and not api_keys_dict.get('deepseek'):
                st.error("❌ Deepseek API key is required for this method")
            else:
                # Process files
                st.markdown("### ⏳ Processing")
                
                # Create progress containers
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    details_text = st.empty()
                    
                    # Create metrics row for real-time stats
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                all_results = []
                temp_files = []
                
                try:
                    # Create temp directory
                    temp_dir = tempfile.mkdtemp()
                    
                    total_files = len(uploaded_files)
                    processed_files = 0
                    total_rows = 0
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        filename = uploaded_file.name
                        file_num = i + 1
                        
                        # Update progress
                        progress_percent = (file_num - 0.5) / total_files
                        progress_bar.progress(progress_percent)
                        status_text.markdown(f"**📄 Processing File {file_num}/{total_files}:** `{filename}`")
                        details_text.markdown(f"⏳ Extracting data using {ocr_method}...")
                        
                        # Update metrics
                        with metrics_col1:
                            st.metric("Files Processed", f"{processed_files}/{total_files}")
                        with metrics_col2:
                            st.metric("Rows Extracted", total_rows)
                        with metrics_col3:
                            st.metric("Current Status", "Processing...")
                        
                        try:
                            # Save uploaded file temporarily
                            details_text.markdown(f"💾 Saving file temporarily...")
                            temp_file_path = save_uploaded_file(uploaded_file, temp_dir)
                            temp_files.append(temp_file_path)
                            
                            # Get custom fields if specified
                            extraction_fields = None
                            if use_custom_fields and st.session_state.get('custom_fields'):
                                extraction_fields = st.session_state.custom_fields
                            elif not use_custom_fields and st.session_state.get('custom_fields'):
                                # Add custom fields to default selection
                                extraction_fields = list(st.session_state.selected_fields)
                            
                            # Extract data
                            if selected_method == 'local':
                                details_text.markdown(f"🔍 Extracting data using local model: {local_model} (first run may take 10-20 minutes to download model)...")
                                # For local models, we need to pass model name and options
                                from extraction_service import extract_with_local_model
                                rows = extract_with_local_model(
                                    temp_file_path,
                                    local_model or 'datalab-to/chandra',
                                    use_cpu=use_cpu_mode,
                                    use_pretty=use_pretty_output
                                )
                            else:
                                details_text.markdown(f"🔍 Extracting data from PDF (this may take 30-60 seconds)...")
                                rows = extract_data(
                                    temp_file_path,
                                    selected_method,
                                    api_keys_dict,
                                    hf_model if selected_method == 'huggingface' else None,
                                    extraction_fields
                                )
                            
                            all_results.extend(rows)
                            processed_files += 1
                            total_rows += len(rows)
                            
                            details_text.markdown(f"✅ Extracted {len(rows)} row(s) from `{filename}`")
                            
                            # Update progress
                            progress_percent = file_num / total_files
                            progress_bar.progress(progress_percent)
                            
                            # Update metrics
                            with metrics_col1:
                                st.metric("Files Processed", f"{processed_files}/{total_files}")
                            with metrics_col2:
                                st.metric("Rows Extracted", total_rows)
                            with metrics_col3:
                                st.metric("Current Status", "✅ Success")
                            
                            # Small delay for UI update
                            time.sleep(0.1)
                            
                        except Exception as e:
                            processed_files += 1
                            error_msg = str(e)
                            st.error(f"❌ Error processing `{filename}`: {error_msg}")
                            details_text.markdown(f"❌ Failed to process `{filename}`")
                            
                            # Update metrics
                            with metrics_col1:
                                st.metric("Files Processed", f"{processed_files}/{total_files}")
                            with metrics_col2:
                                st.metric("Rows Extracted", total_rows)
                            with metrics_col3:
                                st.metric("Current Status", "❌ Error")
                    
                    # Clean up temp files
                    details_text.markdown("🧹 Cleaning up temporary files...")
                    for temp_file in temp_files:
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    
                    # Final progress update
                    progress_bar.progress(1.0)
                    status_text.markdown("**✅ Processing Complete!**")
                    details_text.markdown(f"📊 Successfully processed {processed_files} file(s) with {total_rows} total row(s) extracted")
                    
                    # Final metrics
                    with metrics_col1:
                        st.metric("Files Processed", f"{processed_files}/{total_files}", delta=f"{processed_files} file(s)")
                    with metrics_col2:
                        st.metric("Rows Extracted", total_rows, delta=f"{total_rows} row(s)")
                    with metrics_col3:
                        st.metric("Success Rate", f"{100*processed_files/total_files:.0f}%")
                    
                    # Store results and selected formats
                    if all_results:
                        st.session_state.extraction_results = all_results
                        st.session_state.selected_output_formats = output_formats
                    
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                
                finally:
                    # Clean up temp directory
                    try:
                        import shutil
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except:
                        pass
    
    # Display Results
    if st.session_state.extraction_results:
        st.markdown("---")
        st.markdown("### 📊 Extraction Results")
        
        # Filter fields
        filtered_results = filter_fields(
            st.session_state.extraction_results,
            st.session_state.selected_fields
        )
        
        if filtered_results:
            # Create DataFrame
            df = pd.DataFrame(filtered_results)
            df = format_dataframe(df)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Files Processed", df['filename'].nunique() if 'filename' in df.columns else 0)
            with col3:
                st.metric("Fields Extracted", len(df.columns))
            with col4:
                st.metric("Unique Plot Nos", df['Plot No.'].nunique() if 'Plot No.' in df.columns else 0)
            
            # Display dataframe
            st.dataframe(df, use_container_width=True, height=400)
            
            # Download buttons for all selected formats
            st.markdown("### 💾 Download Results")
            
            format_map = {
                "CSV": ("csv", "text/csv", "extracted_data.csv"),
                "Excel (XLSX)": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "extracted_data.xlsx"),
                "JSON": ("json", "application/json", "extracted_data.json"),
                "Markdown (MD)": ("md", "text/markdown", "extracted_data.md")
            }
            
            # Get selected formats from session state (stored during processing)
            selected_formats = st.session_state.get('selected_output_formats', ["CSV"])
            
            # Create download buttons for each selected format
            if not selected_formats:
                st.warning("⚠️ No output formats selected. Please select at least one format in the sidebar.")
            else:
                import io
                
                # Arrange download buttons in columns (2 per row)
                num_formats = len(output_formats)
                cols_per_row = 2
                num_rows = (num_formats + cols_per_row - 1) // cols_per_row
                
                download_buttons = []
                
                for format_name in selected_formats:
                    file_format, mime_type, default_filename = format_map.get(format_name)
                    
                    # Prepare download data
                    download_data = None
                    
                    if file_format == "csv":
                        download_data = df.to_csv(index=False, encoding='utf-8-sig')
                    elif file_format == "xlsx":
                        excel_buffer = io.BytesIO()
                        df.to_excel(excel_buffer, index=False, engine='openpyxl')
                        excel_buffer.seek(0)
                        download_data = excel_buffer
                    elif file_format == "json":
                        from utils import dataframe_to_json_string
                        download_data = dataframe_to_json_string(df)
                    elif file_format == "md":
                        from utils import dataframe_to_markdown_string
                        download_data = dataframe_to_markdown_string(df)
                    
                    if download_data:
                        download_buttons.append((format_name, download_data, default_filename, mime_type))
                
                # Display download buttons in a grid
                for i in range(0, len(download_buttons), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, (format_name, data, filename, mime) in enumerate(download_buttons[i:i+cols_per_row]):
                        with cols[j]:
                            st.download_button(
                                label=f"📥 Download {format_name}",
                                data=data,
                                file_name=filename,
                                mime=mime,
                                use_container_width=True,
                                key=f"download_{format_name}_{i}_{j}"
                            )
        else:
            st.warning("⚠️ No data extracted. Please check your PDF files and extraction method.")


if __name__ == "__main__":
    main()

