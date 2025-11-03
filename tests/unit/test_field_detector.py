"""Unit tests for field_detector module."""

from __future__ import annotations

from field_detector import (
    create_field_detection_prompt,
    normalize_field_names,
    parse_json_response,
)


class TestParseJsonResponse:
    """Test cases for parse_json_response function."""

    def test_parse_valid_json_array(self):
        """Test parsing valid JSON array."""
        response = '["Sr.No", "Plot No.", "Survey No."]'
        result = parse_json_response(response)
        assert result == ["Sr.No", "Plot No.", "Survey No."]

    def test_parse_json_with_markdown_code_block(self):
        """Test parsing JSON wrapped in markdown code block."""
        response = '```json\n["Plot No.", "Survey No."]\n```'
        result = parse_json_response(response)
        assert result == ["Plot No.", "Survey No."]

    def test_parse_json_with_whitespace(self):
        """Test parsing JSON with extra whitespace."""
        response = '  \n  ["Plot No."]  \n  '
        result = parse_json_response(response)
        assert result == ["Plot No."]

    def test_parse_invalid_json_returns_empty(self):
        """Test that invalid JSON returns empty list."""
        response = "not valid json"
        result = parse_json_response(response)
        assert result == []

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        result = parse_json_response("")
        assert result == []


class TestNormalizeFieldNames:
    """Test cases for normalize_field_names function."""

    def test_normalize_standard_fields(self):
        """Test normalizing standard field name variations."""
        fields = ["sr no", "plot no", "survey no"]
        result = normalize_field_names(fields)
        assert "Sr.No" in result
        assert "Plot No." in result
        assert "Survey No." in result

    def test_normalize_removes_duplicates(self):
        """Test that normalization removes duplicates."""
        fields = ["Plot No.", "plot no", "Plot Number"]
        result = normalize_field_names(fields)
        assert result.count("Plot No.") == 1

    def test_normalize_handles_empty_list(self):
        """Test normalizing empty list."""
        result = normalize_field_names([])
        assert result == []

    def test_normalize_preserves_unknown_fields(self):
        """Test that unknown fields are preserved."""
        fields = ["Plot No.", "Custom Field"]
        result = normalize_field_names(fields)
        assert "Plot No." in result
        assert "Custom Field" in result


class TestCreateFieldDetectionPrompt:
    """Test cases for create_field_detection_prompt function."""

    def test_prompt_contains_keywords(self):
        """Test that prompt contains important keywords."""
        prompt = create_field_detection_prompt()
        assert "column headers" in prompt.lower()
        assert "field names" in prompt.lower()
        assert "json" in prompt.lower()

    def test_prompt_returns_string(self):
        """Test that prompt returns a string."""
        prompt = create_field_detection_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_mentions_example_format(self):
        """Test that prompt mentions example output format."""
        prompt = create_field_detection_prompt()
        # Prompt should mention example format (now generic, not EC-specific)
        assert "example" in prompt.lower() or "format" in prompt.lower()
        # Should NOT contain EC-specific fields (generalized)
        assert "Sr.No" not in prompt
