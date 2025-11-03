#!/usr/bin/env python3
"""
Streamlit webapp for EC (Encumbrance Certificate) PDF extraction.
Modern, clean UI with multiple OCR options and API key management.
"""

import os
import tempfile
import time

import pandas as pd
import streamlit as st

# Import our modules
from extraction_service import extract_data, save_uploaded_file
from logging_config import configure_logging
from streamlit_ui.results import render_footer, render_results_section
from streamlit_ui.sidebar import render_sidebar
from streamlit_ui.state import initialize_session_state, load_saved_api_keys
from utils import format_file_size, validate_pdf_file

# Configure logging before initializing UI
configure_logging()

# Page configuration
st.set_page_config(
    page_title="EC Data Extraction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI
st.markdown(
    """
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
    .footer {
        margin-top: 4rem;
        padding: 2rem 1rem;
        text-align: center;
        border-top: 1px solid #e0e0e0;
        color: #666;
    }
    .footer-content {
        margin-bottom: 1rem;
    }
    .footer-social {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .footer-social a {
        color: #666;
        text-decoration: none;
        transition: color 0.3s ease;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .footer-social a:hover {
        color: #1f77b4;
    }
    .footer-social svg {
        width: 20px;
        height: 20px;
        fill: currentColor;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    """Main application function."""
    initialize_session_state()
    load_saved_api_keys()

    # Header
    st.markdown(
        '<div class="main-header">📄 PDF Data Extraction</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Extract structured data from PDF documents using multiple OCR and AI methods</div>',
        unsafe_allow_html=True,
    )

    # Render sidebar and get configuration
    config = render_sidebar()
    ocr_method = config["ocr_method"]
    local_model = config["local_model"]
    hf_model = config["hf_model"]
    use_cpu_mode = config["use_cpu_mode"]
    use_pretty_output = config["use_pretty_output"]
    use_gpu_easyocr = config["use_gpu_easyocr"]
    api_keys_dict = config["api_keys"]
    auto_detect_fields = config["auto_detect_fields"]
    use_custom_fields = config["use_custom_fields"]
    output_formats = config["output_formats"]
    _ = config["json_format"]  # Reserved for future use

    # Main Content Area
    st.markdown("---")

    # File Upload Section
    st.subheader("📤 Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload one or multiple PDF files",
    )

    if uploaded_files:
        st.markdown("#### Selected Files:")
        file_info = []
        for file in uploaded_files:
            size_mb = len(file.getvalue()) / (1024 * 1024)
            file_info.append({"Filename": file.name, "Size": format_file_size(size_mb)})
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
            "🚀 Extract Data", type="primary", use_container_width=True
        )

    # Processing and Results
    if process_button:
        if not uploaded_files:
            st.error("⚠️ Please upload at least one PDF file")
        elif not output_formats:
            st.error(
                "⚠️ Please select at least one output format in the sidebar before processing"
            )
        else:
            # API keys already in config from sidebar

            # Map method names
            method_map = {
                "EasyOCR": "easyocr",
                "PyTesseract": "pytesseract",
                "Local Model": "local",
                "HuggingFace": "huggingface",
                "Datalab API": "datalab",
                "Gemini AI": "gemini",
                "Deepseek AI": "deepseek",
            }
            selected_method = method_map[ocr_method]

            # Check API key requirements
            if selected_method in ["local", "easyocr", "pytesseract"]:
                # No API key needed for local models or free OCR
                pass
            elif selected_method == "huggingface" and not api_keys_dict.get(
                "huggingface"
            ):
                st.error("❌ HuggingFace API key is required for this method")
            elif selected_method == "datalab" and not api_keys_dict.get("datalab"):
                st.error("❌ Datalab API key is required for this method")
            elif selected_method == "gemini" and not api_keys_dict.get("gemini"):
                st.error("❌ Gemini API key is required for this method")
            elif selected_method == "deepseek" and not api_keys_dict.get("deepseek"):
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
                detected_fields = None

                try:
                    # Create temp directory
                    temp_dir = tempfile.mkdtemp()

                    total_files = len(uploaded_files)
                    processed_files = 0
                    total_rows = 0

                    # Field Detection Step (if enabled)
                    if auto_detect_fields and uploaded_files:
                        st.markdown("#### 🔍 Field Detection")
                        with st.spinner("Detecting fields from PDF(s)..."):
                            try:
                                from field_detector import detect_fields_batch

                                # Save files temporarily for detection
                                temp_files_for_detection = []
                                for uploaded_file in uploaded_files:
                                    temp_path = save_uploaded_file(
                                        uploaded_file, temp_dir
                                    )
                                    temp_files_for_detection.append(temp_path)

                                detection_method = (
                                    "ai"
                                    if selected_method in ["gemini", "deepseek"]
                                    else "ocr"
                                )
                                detection_result = detect_fields_batch(
                                    temp_files_for_detection,
                                    method=detection_method,
                                    extraction_method=selected_method,
                                    api_keys=api_keys_dict,
                                    mode=st.session_state.get(
                                        "field_detection_mode", "unified"
                                    ),
                                )

                                detected_fields = detection_result.get("fields", [])
                                per_file_fields = detection_result.get("per_file", {})

                                if detected_fields:
                                    st.success(
                                        f"✅ Detected {len(detected_fields)} fields"
                                    )
                                    st.session_state.detected_fields = detected_fields

                                    # Display detected fields
                                    with st.expander(
                                        "📋 View Detected Fields", expanded=True
                                    ):
                                        st.write("**Detected Fields:**")
                                        for field in detected_fields:
                                            st.write(f"  • {field}")

                                        # Show per-file comparison if per_file mode
                                        if per_file_fields and len(per_file_fields) > 1:
                                            st.markdown("---")
                                            st.write("**Field Comparison by File:**")
                                            comparison_data = []
                                            for (
                                                filename,
                                                fields,
                                            ) in per_file_fields.items():
                                                comparison_data.append(
                                                    {
                                                        "Filename": filename,
                                                        "Fields Count": len(fields),
                                                        "Fields": ", ".join(fields[:5])
                                                        + (
                                                            "..."
                                                            if len(fields) > 5
                                                            else ""
                                                        ),
                                                    }
                                                )
                                            st.dataframe(
                                                pd.DataFrame(comparison_data),
                                                use_container_width=True,
                                                hide_index=True,
                                            )

                                    # Allow user to edit detected fields
                                    edit_fields = st.checkbox(
                                        "Edit Detected Fields", value=False
                                    )
                                    if edit_fields:
                                        edited_fields_text = st.text_area(
                                            "Edit Fields (one per line):",
                                            value="\n".join(detected_fields),
                                            height=150,
                                        )
                                        if edited_fields_text:
                                            detected_fields = [
                                                f.strip()
                                                for f in edited_fields_text.split("\n")
                                                if f.strip()
                                            ]
                                            st.session_state.detected_fields = (
                                                detected_fields
                                            )
                                            st.success(
                                                f"✅ Updated to {len(detected_fields)} fields"
                                            )
                                else:
                                    st.warning(
                                        "⚠️ No fields detected. Using default fields."
                                    )
                                    detected_fields = None

                            except Exception as e:
                                st.warning(
                                    f"⚠️ Field detection failed: {e}. Proceeding with default fields."
                                )
                                detected_fields = None

                    for i, uploaded_file in enumerate(uploaded_files):
                        filename = uploaded_file.name
                        file_num = i + 1

                        # Update progress
                        progress_percent = (file_num - 0.5) / total_files
                        progress_bar.progress(progress_percent)
                        status_text.markdown(
                            f"**📄 Processing File {file_num}/{total_files}:** `{filename}`"
                        )
                        details_text.markdown(
                            f"⏳ Extracting data using {ocr_method}..."
                        )

                        # Update metrics
                        with metrics_col1:
                            st.metric(
                                "Files Processed", f"{processed_files}/{total_files}"
                            )
                        with metrics_col2:
                            st.metric("Rows Extracted", total_rows)
                        with metrics_col3:
                            st.metric("Current Status", "Processing...")

                        try:
                            # Save uploaded file temporarily
                            details_text.markdown("💾 Saving file temporarily...")
                            temp_file_path = save_uploaded_file(uploaded_file, temp_dir)
                            temp_files.append(temp_file_path)

                            # Get custom fields if specified (prioritize auto-detected fields)
                            extraction_fields = None
                            if (
                                auto_detect_fields
                                and detected_fields
                                and selected_method in ["gemini", "deepseek"]
                            ):
                                # Use auto-detected fields
                                extraction_fields = detected_fields
                            elif use_custom_fields and st.session_state.get(
                                "custom_fields"
                            ):
                                extraction_fields = st.session_state.custom_fields
                            elif not use_custom_fields and st.session_state.get(
                                "custom_fields"
                            ):
                                # Add custom fields to default selection
                                extraction_fields = list(
                                    st.session_state.selected_fields
                                )

                            # Extract data
                            if selected_method == "local":
                                details_text.markdown(
                                    f"🔍 Extracting data using local model: {local_model} (first run may take 10-20 minutes to download model)..."
                                )
                                # For local models, we need to pass model name and options
                                from extraction_service import extract_with_local_model

                                rows = extract_with_local_model(
                                    temp_file_path,
                                    local_model or "datalab-to/chandra",
                                    use_cpu=use_cpu_mode,
                                    use_pretty=use_pretty_output,
                                )
                            else:
                                details_text.markdown(
                                    "🔍 Extracting data from PDF (this may take 30-60 seconds)..."
                                )
                                # Prepare options for extraction methods
                                extraction_options = None
                                if selected_method == "easyocr" and use_gpu_easyocr:
                                    extraction_options = {"use_gpu": True}
                                elif selected_method == "local":
                                    extraction_options = {
                                        "use_cpu": use_cpu_mode,
                                        "use_pretty": use_pretty_output,
                                    }
                                
                                rows = extract_data(
                                    temp_file_path,
                                    selected_method,
                                    api_keys_dict,
                                    (
                                        hf_model
                                        if selected_method == "huggingface"
                                        else None
                                    ),
                                    extraction_fields,
                                    auto_detect_fields=False,  # Already detected above
                                    detected_fields=(
                                        detected_fields if auto_detect_fields else None
                                    ),
                                    local_model_options=extraction_options,
                                )

                            all_results.extend(rows)
                            processed_files += 1
                            total_rows += len(rows)

                            details_text.markdown(
                                f"✅ Extracted {len(rows)} row(s) from `{filename}`"
                            )

                            # Update progress
                            progress_percent = file_num / total_files
                            progress_bar.progress(progress_percent)

                            # Update metrics
                            with metrics_col1:
                                st.metric(
                                    "Files Processed",
                                    f"{processed_files}/{total_files}",
                                )
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
                                st.metric(
                                    "Files Processed",
                                    f"{processed_files}/{total_files}",
                                )
                            with metrics_col2:
                                st.metric("Rows Extracted", total_rows)
                            with metrics_col3:
                                st.metric("Current Status", "❌ Error")

                    # Clean up temp files
                    details_text.markdown("🧹 Cleaning up temporary files...")
                    for temp_file in temp_files:
                        try:
                            os.remove(temp_file)
                        except OSError:
                            pass

                    # Final progress update
                    progress_bar.progress(1.0)
                    status_text.markdown("**✅ Processing Complete!**")
                    details_text.markdown(
                        f"📊 Successfully processed {processed_files} file(s) with {total_rows} total row(s) extracted"
                    )

                    # Final metrics
                    with metrics_col1:
                        st.metric(
                            "Files Processed",
                            f"{processed_files}/{total_files}",
                            delta=f"{processed_files} file(s)",
                        )
                    with metrics_col2:
                        st.metric(
                            "Rows Extracted", total_rows, delta=f"{total_rows} row(s)"
                        )
                    with metrics_col3:
                        st.metric(
                            "Success Rate", f"{100*processed_files/total_files:.0f}%"
                        )

                    # Store results and selected formats
                    if all_results:
                        st.session_state.extraction_results = all_results
                        st.session_state.selected_output_formats = output_formats
                        st.session_state.auto_detect_fields = auto_detect_fields

                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())

                finally:
                    # Clean up temp directory
                    try:
                        import shutil

                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except OSError:
                        pass

    # Display results and footer
    render_results_section(output_formats)
    render_footer()


if __name__ == "__main__":
    main()
