"""Unit tests for field_presets module."""

from __future__ import annotations

import pytest

from field_presets import (
    get_available_presets,
    get_field_preset,
    get_preset_fields,
    get_preset_required_fields,
    list_preset_names,
    preset_exists,
    register_field_preset,
)


class TestGetFieldPreset:
    """Test cases for get_field_preset function."""

    def test_get_existing_preset(self):
        """Test getting an existing preset."""
        preset = get_field_preset("encumbrance")
        assert preset is not None
        assert preset["name"] == "Encumbrance Certificate"
        assert "fields" in preset
        assert "required_fields" in preset
        assert "description" in preset

    def test_get_nonexistent_preset(self):
        """Test getting a non-existent preset returns None."""
        preset = get_field_preset("nonexistent")
        assert preset is None

    def test_case_insensitive_preset(self):
        """Test that preset names are case-insensitive."""
        preset1 = get_field_preset("encumbrance")
        preset2 = get_field_preset("ENCUMBRANCE")
        preset3 = get_field_preset("Encumbrance")
        assert preset1 == preset2 == preset3


class TestGetAvailablePresets:
    """Test cases for get_available_presets function."""

    def test_get_all_presets(self):
        """Test getting all available presets."""
        presets = get_available_presets()
        assert isinstance(presets, dict)
        assert len(presets) > 0
        assert "encumbrance" in presets

    def test_presets_are_copies(self):
        """Test that returned presets are copies, not references."""
        presets1 = get_available_presets()
        presets2 = get_available_presets()
        assert presets1 is not presets2
        # Modifying one should not affect the other
        presets1["test"] = {"test": "value"}
        assert "test" not in presets2


class TestRegisterFieldPreset:
    """Test cases for register_field_preset function."""

    def test_register_new_preset(self):
        """Test registering a new preset."""
        register_field_preset(
            preset_name="test_preset",
            fields=["Field1", "Field2", "Field3"],
            required_fields=["Field1"],
            name="Test Preset",
            description="A test preset",
        )
        preset = get_field_preset("test_preset")
        assert preset is not None
        assert preset["name"] == "Test Preset"
        assert preset["fields"] == ["Field1", "Field2", "Field3"]
        assert preset["required_fields"] == ["Field1"]
        assert preset["description"] == "A test preset"

        # Cleanup
        from field_presets import PRESETS

        if "test_preset" in PRESETS:
            del PRESETS["test_preset"]

    def test_register_preset_without_required_fields(self):
        """Test registering a preset without required fields."""
        register_field_preset(
            preset_name="test_preset_no_required",
            fields=["Field1", "Field2"],
            name="Test Preset No Required",
        )
        preset = get_field_preset("test_preset_no_required")
        assert preset is not None
        assert preset["required_fields"] == []

        # Cleanup
        from field_presets import PRESETS

        if "test_preset_no_required" in PRESETS:
            del PRESETS["test_preset_no_required"]

    def test_register_preset_without_name(self):
        """Test registering a preset without display name."""
        register_field_preset(
            preset_name="test_preset_no_name",
            fields=["Field1"],
            description="Test description",
        )
        preset = get_field_preset("test_preset_no_name")
        assert preset is not None
        assert preset["name"] == "test_preset_no_name"

        # Cleanup
        from field_presets import PRESETS

        if "test_preset_no_name" in PRESETS:
            del PRESETS["test_preset_no_name"]

    def test_update_existing_preset(self):
        """Test updating an existing preset."""
        # Register initial preset
        register_field_preset(
            preset_name="test_update",
            fields=["Field1"],
            name="Original",
        )
        # Update it
        register_field_preset(
            preset_name="test_update",
            fields=["Field1", "Field2"],
            name="Updated",
        )
        preset = get_field_preset("test_update")
        assert preset["name"] == "Updated"
        assert preset["fields"] == ["Field1", "Field2"]

        # Cleanup
        from field_presets import PRESETS

        if "test_update" in PRESETS:
            del PRESETS["test_update"]


class TestGetPresetFields:
    """Test cases for get_preset_fields function."""

    def test_get_fields_for_existing_preset(self):
        """Test getting fields for an existing preset."""
        fields = get_preset_fields("encumbrance")
        assert fields is not None
        assert isinstance(fields, list)
        assert len(fields) > 0
        assert "Plot No." in fields

    def test_get_fields_for_nonexistent_preset(self):
        """Test getting fields for non-existent preset returns None."""
        fields = get_preset_fields("nonexistent")
        assert fields is None

    def test_case_insensitive_field_retrieval(self):
        """Test that field retrieval is case-insensitive."""
        fields1 = get_preset_fields("invoice")
        fields2 = get_preset_fields("INVOICE")
        assert fields1 == fields2


class TestGetPresetRequiredFields:
    """Test cases for get_preset_required_fields function."""

    def test_get_required_fields_for_existing_preset(self):
        """Test getting required fields for an existing preset."""
        required = get_preset_required_fields("encumbrance")
        assert required is not None
        assert isinstance(required, list)
        assert "Plot No." in required

    def test_get_required_fields_for_preset_without_required(self):
        """Test getting required fields for preset without required fields."""
        # Register a preset without required fields
        register_field_preset(
            preset_name="test_no_required",
            fields=["Field1"],
            required_fields=[],
        )
        required = get_preset_required_fields("test_no_required")
        assert required == []

        # Cleanup
        from field_presets import PRESETS

        if "test_no_required" in PRESETS:
            del PRESETS["test_no_required"]

    def test_get_required_fields_for_nonexistent_preset(self):
        """Test getting required fields for non-existent preset returns None."""
        required = get_preset_required_fields("nonexistent")
        assert required is None


class TestListPresetNames:
    """Test cases for list_preset_names function."""

    def test_list_all_preset_names(self):
        """Test listing all preset names."""
        names = list_preset_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "encumbrance" in names
        assert "invoice" in names
        assert "receipt" in names

    def test_list_returns_lowercase_names(self):
        """Test that listed names are lowercase."""
        names = list_preset_names()
        for name in names:
            assert name.islower()


class TestPresetExists:
    """Test cases for preset_exists function."""

    def test_existing_preset_exists(self):
        """Test that existing presets are detected."""
        assert preset_exists("encumbrance") is True
        assert preset_exists("invoice") is True
        assert preset_exists("receipt") is True

    def test_nonexistent_preset_does_not_exist(self):
        """Test that non-existent presets are not detected."""
        assert preset_exists("nonexistent") is False

    def test_case_insensitive_existence_check(self):
        """Test that existence check is case-insensitive."""
        assert preset_exists("ENCUMBRANCE") is True
        assert preset_exists("Encumbrance") is True
        assert preset_exists("encumbrance") is True


class TestIntegrationWithUtils:
    """Test integration with utils.get_default_fields()."""

    def test_get_default_fields_with_preset(self):
        """Test that get_default_fields uses presets correctly."""
        from utils import get_default_fields

        fields = get_default_fields("encumbrance")
        assert isinstance(fields, list)
        assert "filename" in fields
        assert len(fields) > 1

    def test_get_default_fields_with_different_preset(self):
        """Test get_default_fields with different presets."""
        from utils import get_default_fields

        encumbrance_fields = get_default_fields("encumbrance")
        invoice_fields = get_default_fields("invoice")
        assert encumbrance_fields != invoice_fields
        assert "filename" in encumbrance_fields
        assert "filename" in invoice_fields

