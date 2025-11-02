"""Result display helpers for the Streamlit UI."""

from __future__ import annotations

import io
from typing import Iterable

import pandas as pd
import streamlit as st

from utils import (
    dataframe_to_json_string,
    dataframe_to_markdown_string,
    filter_fields,
    format_dataframe,
)


def render_results_section(output_formats: Iterable[str]) -> None:
    """Render the extraction results table and download actions."""

    results = st.session_state.get("extraction_results")
    if not results:
        return

    st.markdown("---")
    st.markdown("### 📊 Extraction Results")

    filtered_results = filter_fields(results, st.session_state.get("selected_fields", set()))
    if not filtered_results:
        st.warning("⚠️ No data extracted. Please check your PDF files and extraction method.")
        return

    df = pd.DataFrame(filtered_results)
    df = format_dataframe(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Files Processed", df["filename"].nunique() if "filename" in df.columns else 0)
    with col3:
        st.metric("Fields Extracted", len(df.columns))
    with col4:
        st.metric("Unique Plot Nos", df["Plot No."].nunique() if "Plot No." in df.columns else 0)

    st.dataframe(df, use_container_width=True, height=400)

    st.markdown("### 💾 Download Results")

    format_map = {
        "CSV": ("csv", "text/csv", "extracted_data.csv"),
        "Excel (XLSX)": (
            "xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "extracted_data.xlsx",
        ),
        "JSON": ("json", "application/json", "extracted_data.json"),
        "Markdown (MD)": ("md", "text/markdown", "extracted_data.md"),
    }

    selected_formats = st.session_state.get("selected_output_formats", ["CSV"])
    if not selected_formats:
        st.warning("⚠️ No output formats selected. Please select at least one format in the sidebar.")
        return

    cols_per_row = 2
    download_buttons = []

    for format_name in selected_formats:
        file_format, mime_type, default_filename = format_map.get(format_name, (None, None, None))
        if not file_format:
            continue

        if file_format == "csv":
            download_data = df.to_csv(index=False, encoding="utf-8-sig")
        elif file_format == "xlsx":
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine="openpyxl")
            excel_buffer.seek(0)
            download_data = excel_buffer
        elif file_format == "json":
            json_format = st.session_state.get("json_format", "standard")
            metadata = None
            if json_format in {"structured", "unified"}:
                metadata = {
                    "detected_fields": st.session_state.get("detected_fields"),
                    "auto_detect_enabled": st.session_state.get("auto_detect_fields", False),
                }
            download_data = dataframe_to_json_string(df, format=json_format, metadata=metadata)
        elif file_format == "md":
            download_data = dataframe_to_markdown_string(df)
        else:
            download_data = None

        if download_data:
            download_buttons.append((format_name, download_data, default_filename, mime_type))

    if not download_buttons:
        st.warning("⚠️ No downloadable data generated.")
        return

    for i in range(0, len(download_buttons), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (format_name, data, filename, mime) in enumerate(download_buttons[i : i + cols_per_row]):
            with cols[j]:
                st.download_button(
                    label=f"📥 Download {format_name}",
                    data=data,
                    file_name=filename,
                    mime=mime,
                    use_container_width=True,
                    key=f"download_{format_name}_{i}_{j}",
                )


def render_footer() -> None:
    """Render the static footer section."""

    st.markdown("---")
    st.markdown(
        """
    <div class="footer">
        <div class="footer-content">
            <p style="margin-bottom: 0.5rem; font-size: 0.95rem;">Made by <strong>Prithiv Raj</strong></p>
        </div>
        <div class="footer-social">
            <a href="https://linkedin.com/in/prithivrajmu" target="_blank" rel="noopener noreferrer" title="LinkedIn">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
                <span>LinkedIn</span>
            </a>
            <a href="https://github.com/prithivrjamu" target="_blank" rel="noopener noreferrer" title="GitHub">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                <span>GitHub</span>
            </a>
            <a href="https://prithivraj.xyz" target="_blank" rel="noopener noreferrer" title="Website">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1.41 16.09V20h-2.67v-1.93c-1.71-.36-3.16-1.46-3.27-3.4h1.96c.1 1.05.82 1.87 2.65 1.87 1.96 0 2.4-.98 2.4-1.59 0-.83-.44-1.61-2.67-2.14-2.48-.6-4.18-1.62-4.18-3.67 0-1.72 1.39-2.84 3.11-3.21V4h2.67v1.95c1.86.45 2.79 1.86 2.85 3.39H14.3c-.05-1.11-.64-1.87-2.22-1.87-1.5 0-2.4.68-2.4 1.64 0 .84.65 1.39 2.67 1.91s4.18 1.39 4.18 3.91c-.01 1.83-1.38 2.83-3.12 3.16z"/>
                </svg>
                <span>Website</span>
            </a>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

