"""Unit tests for extraction_service module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from extraction_service import extract_data


class TestExtractData:
    """Test cases for extract_data function."""

    @patch("extraction_service.extract_with_easyocr")
    def test_extract_data_easyocr(self, mock_extract):
        """Test extraction with EasyOCR method."""
        mock_extract.return_value = [
            {"filename": "test.pdf", "Plot No.": "123"}
        ]

        result = extract_data(
            "test.pdf",
            method="easyocr",
            api_keys={},
        )

        assert len(result) == 1
        assert result[0]["filename"] == "test.pdf"
        mock_extract.assert_called_once_with("test.pdf")

    def test_extract_data_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown extraction method"):
            extract_data(
                "test.pdf",
                method="invalid_method",
                api_keys={},
            )

    def test_extract_data_missing_api_key(self):
        """Test that missing required API key raises ValueError."""
        with pytest.raises(ValueError, match="Gemini API key is required"):
            extract_data(
                "test.pdf",
                method="gemini",
                api_keys={},
            )

    @patch("extraction_service.extract_with_local_model")
    def test_extract_data_local_with_options(self, mock_extract):
        """Test local model extraction with custom options."""
        mock_extract.return_value = [
            {"filename": "test.pdf", "Plot No.": "123"}
        ]

        result = extract_data(
            "test.pdf",
            method="local",
            api_keys={},
            local_model_options={"use_cpu": True, "use_pretty": False},
        )

        mock_extract.assert_called_once_with(
            "test.pdf",
            "datalab-to/chandra",
            use_cpu=True,
            use_pretty=False,
        )
        assert len(result) == 1

    @patch("extraction_service.extract_with_gemini")
    def test_extract_data_with_custom_fields(self, mock_extract):
        """Test extraction with custom fields."""
        mock_extract.return_value = [
            {
                "filename": "test.pdf",
                "Village Name": "Test Village",
                "Plot No.": "123",
            }
        ]

        result = extract_data(
            "test.pdf",
            method="gemini",
            api_keys={"gemini": "test-key"},
            custom_fields=["Village Name", "Plot No."],
        )

        mock_extract.assert_called_once_with(
            "test.pdf",
            "test-key",
            custom_fields=["Village Name", "Plot No."],
        )
        assert len(result) == 1
        assert "Village Name" in result[0]


class TestExtractDataMethodRouting:
    """Test that extract_data routes to correct extraction method."""

    @pytest.mark.parametrize(
        "method,expected_func",
        [
            ("local", "extract_with_local_model"),
            ("local_model", "extract_with_local_model"),
            ("pytesseract", "extract_with_pytesseract"),
            ("tesseract", "extract_with_pytesseract"),
            ("easyocr", "extract_with_easyocr"),
            ("huggingface", "extract_with_huggingface"),
            ("hf", "extract_with_huggingface"),
            ("datalab", "extract_with_datalab_api"),
            ("gemini", "extract_with_gemini"),
            ("deepseek", "extract_with_deepseek"),
        ],
    )
    def test_method_routing(self, method, expected_func):
        """Test that each method routes to correct function."""
        with patch(f"extraction_service.{expected_func}") as mock_func:
            mock_func.return_value = [{"filename": "test.pdf"}]

            api_keys = {}
            if method in ("huggingface", "hf"):
                api_keys["huggingface"] = "test-key"
            elif method == "datalab":
                api_keys["datalab"] = "test-key"
            elif method == "gemini":
                api_keys["gemini"] = "test-key"
            elif method == "deepseek":
                api_keys["deepseek"] = "test-key"

            extract_data(
                "test.pdf",
                method=method,
                api_keys=api_keys,
            )

            mock_func.assert_called_once()

