"""Unit tests for utility helper functions."""

from __future__ import annotations

import json
import pandas as pd
import pytest

from utils import (
    dataframe_to_json_string,
    dataframe_to_markdown_string,
    filter_fields,
    format_dataframe,
    validate_pdf_file,
)

from tests.fixtures.pdf_samples import make_corrupt_pdf, make_uploaded_pdf


def test_filter_fields_keeps_selected_and_filename():
    data = [
        {"filename": "file1.pdf", "Plot No.": "123", "Extra": "value"},
        {"filename": "file2.pdf", "Plot No.": "456", "Extra": "ignore"},
    ]

    filtered = filter_fields(data, {"Plot No."})

    assert all("filename" in row for row in filtered)
    assert all("Plot No." in row for row in filtered)
    assert all("Extra" not in row for row in filtered)


def test_format_dataframe_orders_columns():
    df = pd.DataFrame(
        [
            {
                "Plot No.": "P1",
                "filename": "file1.pdf",
                "Sr.No": "1",
                "Custom": "value",
            }
        ]
    )

    formatted = format_dataframe(df)

    columns = list(formatted.columns)
    assert columns[0] == "filename"
    assert columns[1] == "Sr.No"
    assert columns.index("Plot No.") < columns.index("Custom")


def test_validate_pdf_file_accepts_valid_pdf():
    uploaded = make_uploaded_pdf()

    is_valid, error = validate_pdf_file(uploaded)

    assert is_valid
    assert error == ""


def test_validate_pdf_file_rejects_wrong_extension():
    uploaded = make_uploaded_pdf(name="sample.txt")

    is_valid, error = validate_pdf_file(uploaded)

    assert not is_valid
    assert "not a PDF" in error


@pytest.mark.parametrize(
    "description, corrupt_file",
    [
        ("missing header", make_corrupt_pdf(missing_header=True)),
        ("missing EOF", make_corrupt_pdf(missing_eof=True)),
    ],
)
def test_validate_pdf_file_detects_corruption(description: str, corrupt_file):
    is_valid, error = validate_pdf_file(corrupt_file)

    assert not is_valid, description
    assert error, description


def test_dataframe_to_json_string_structured():
    df = pd.DataFrame(
        [
            {
                "filename": "file1.pdf",
                "Plot No.": "P1",
            }
        ]
    )

    json_output = dataframe_to_json_string(df, format="structured")
    payload = json.loads(json_output)

    assert "metadata" in payload
    assert payload["metadata"].get("total_rows") == 1
    assert payload["data"][0]["Plot No."] == "P1"


def test_dataframe_to_markdown_string_generates_table():
    df = pd.DataFrame(
        [
            {
                "filename": "file1.pdf",
                "Plot No.": "P1",
            }
        ]
    )

    markdown = dataframe_to_markdown_string(df)

    assert markdown.startswith("| filename | Plot No. |")
    assert "| file1.pdf | P1 |" in markdown
