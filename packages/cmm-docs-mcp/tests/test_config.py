"""Tests for cmm_docs.config module."""

from __future__ import annotations

from pathlib import Path

from cmm_docs.config import (
    COMMODITIES,
    DATA_CATEGORIES,
    MAX_CSV_ROWS,
    MAX_PDF_CHARS,
    MAX_SEARCH_RESULTS,
    SUBDOMAINS,
)


class TestConfigConstants:
    """Tests for config-level constants and directory definitions."""

    def test_max_csv_rows_is_positive(self) -> None:
        """MAX_CSV_ROWS should be a positive integer."""
        assert isinstance(MAX_CSV_ROWS, int)
        assert MAX_CSV_ROWS > 0

    def test_max_pdf_chars_is_positive(self) -> None:
        """MAX_PDF_CHARS should be a positive integer."""
        assert isinstance(MAX_PDF_CHARS, int)
        assert MAX_PDF_CHARS > 0

    def test_max_search_results_is_positive(self) -> None:
        """MAX_SEARCH_RESULTS should be a positive integer."""
        assert isinstance(MAX_SEARCH_RESULTS, int)
        assert MAX_SEARCH_RESULTS > 0


class TestCommodities:
    """Tests for the COMMODITIES mapping."""

    def test_commodities_is_dict(self) -> None:
        """COMMODITIES should be a non-empty dictionary."""
        assert isinstance(COMMODITIES, dict)
        assert len(COMMODITIES) > 0

    def test_expected_commodity_codes_present(self) -> None:
        """Key commodity codes should be present."""
        expected_codes = {"HREE", "LREE", "CO", "LI", "GA", "GR", "NI", "CU", "GE"}
        assert expected_codes.issubset(set(COMMODITIES.keys()))

    def test_commodity_values_are_strings(self) -> None:
        """All commodity descriptions should be non-empty strings."""
        for code, description in COMMODITIES.items():
            assert isinstance(description, str), f"Description for {code} is not a string"
            assert len(description) > 0, f"Description for {code} is empty"


class TestSubdomains:
    """Tests for the SUBDOMAINS mapping."""

    def test_subdomains_is_dict(self) -> None:
        """SUBDOMAINS should be a non-empty dictionary."""
        assert isinstance(SUBDOMAINS, dict)
        assert len(SUBDOMAINS) > 0

    def test_expected_subdomain_codes_present(self) -> None:
        """Key subdomain codes should be present."""
        expected_codes = {"T-EC", "T-PM", "T-GO", "S-ST", "G-PR"}
        assert expected_codes.issubset(set(SUBDOMAINS.keys()))


class TestDataCategories:
    """Tests for the DATA_CATEGORIES mapping."""

    def test_data_categories_is_dict(self) -> None:
        """DATA_CATEGORIES should be a non-empty dictionary."""
        assert isinstance(DATA_CATEGORIES, dict)
        assert len(DATA_CATEGORIES) > 0

    def test_data_category_values_are_paths(self) -> None:
        """All data category values should be Path objects."""
        for cat_name, cat_path in DATA_CATEGORIES.items():
            assert isinstance(cat_path, Path), (
                f"Category '{cat_name}' value should be a Path, got {type(cat_path)}"
            )

    def test_expected_categories_present(self) -> None:
        """Key data category names should be present."""
        expected = {"LISA_Model", "USGS_Ore_Deposits"}
        assert expected.issubset(set(DATA_CATEGORIES.keys()))
