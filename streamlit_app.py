#!/usr/bin/env python3
"""
Streamlit webapp for EC (Encumbrance Certificate) PDF extraction.
Modern, clean UI with multiple OCR options and API key management.
"""

import streamlit as st
import pandas as pd
import os
import tempfile
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
    st.markdown('<div class="main-header">📄 EC Data Extraction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Extract structured data from Encumbrance Certificate PDFs using multiple OCR methods</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OCR Method Selection
        st.subheader("🔧 Extraction Method")
        ocr_method = st.radio(
            "Select extraction method",
            options=["EasyOCR", "HuggingFace", "Datalab API", "Gemini AI", "Deepseek AI"],
            index=0,
            label_visibility="collapsed"
        )
        
        # HuggingFace Model Selection
        hf_model = None
        if ocr_method == "HuggingFace":
            st.markdown("---")
            st.subheader("🤗 Model Selection")
            hf_model = st.text_input(
                "HuggingFace Model",
                value="datalab-to/chandra",
                help="Enter the HuggingFace model name (e.g., datalab-to/chandra)"
            )
        
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
        output_format = st.selectbox(
            "Select output format",
            options=["CSV", "Excel (XLSX)", "JSON", "Markdown (MD)"],
            index=0,
            help="Choose the format for downloading results"
        )
    
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
                'HuggingFace': 'huggingface',
                'Datalab API': 'datalab',
                'Gemini AI': 'gemini',
                'Deepseek AI': 'deepseek'
            }
            selected_method = method_map[ocr_method]
            
            # Check API key requirements
            if selected_method == 'huggingface' and not api_keys_dict.get('huggingface'):
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
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_results = []
                temp_files = []
                
                try:
                    # Create temp directory
                    temp_dir = tempfile.mkdtemp()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        filename = uploaded_file.name
                        status_text.text(f"Processing {filename} ({i+1}/{len(uploaded_files)})...")
                        progress_bar.progress((i + 0.5) / len(uploaded_files))
                        
                        try:
                            # Save uploaded file temporarily
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
                            rows = extract_data(
                                temp_file_path,
                                selected_method,
                                api_keys_dict,
                                hf_model if selected_method == 'huggingface' else None,
                                extraction_fields
                            )
                            
                            all_results.extend(rows)
                            
                        except Exception as e:
                            st.error(f"❌ Error processing {filename}: {str(e)}")
                    
                    # Clean up temp files
                    for temp_file in temp_files:
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    # Store results
                    if all_results:
                        st.session_state.extraction_results = all_results
                    
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
            
            # Download buttons based on selected format
            st.markdown("### 💾 Download Results")
            
            format_map = {
                "CSV": ("csv", "text/csv", "extracted_data.csv"),
                "Excel (XLSX)": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "extracted_data.xlsx"),
                "JSON": ("json", "application/json", "extracted_data.json"),
                "Markdown (MD)": ("md", "text/markdown", "extracted_data.md")
            }
            
            file_format, mime_type, default_filename = format_map.get(output_format, format_map["CSV"])
            
            # Prepare download data
            import io
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
            
            # Download button
            if download_data:
                st.download_button(
                    label=f"📥 Download {output_format}",
                    data=download_data,
                    file_name=default_filename,
                    mime=mime_type,
                    use_container_width=True
                )
        else:
            st.warning("⚠️ No data extracted. Please check your PDF files and extraction method.")


if __name__ == "__main__":
    main()

