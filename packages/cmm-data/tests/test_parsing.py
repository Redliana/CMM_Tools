"""Tests for cmm_data.utils.parsing module."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from cmm_data.utils.parsing import (
    clean_numeric_column,
    extract_commodity_code,
    parse_numeric_value,
    parse_range,
    standardize_country_name,
)

# ---------------------------------------------------------------------------
# parse_numeric_value()
# ---------------------------------------------------------------------------


class TestParseNumericValue:
    """Tests for ``parse_numeric_value``."""

    # -- Normal numeric values -----------------------------------------------

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (42, 42.0),
            (3.14, 3.14),
            ("100", 100.0),
            ("1,000", 1000.0),
            ("1,234,567", 1234567.0),
            ("1.5e3", 1500.0),
        ],
    )
    def test_numeric_values(self, value: object, expected: float) -> None:
        """Plain numeric inputs should be parsed correctly."""
        result = parse_numeric_value(value)
        assert result == pytest.approx(expected)

    # -- Special codes returning NaN -----------------------------------------

    @pytest.mark.parametrize(
        "value",
        ["W", "w", "XX", "--", "\u2014", "NA", "N/A", "n.a.", "", None, float("nan")],
    )
    def test_special_codes_return_nan(self, value: object) -> None:
        """Special USGS codes and missing-value sentinels should return NaN."""
        result = parse_numeric_value(value)
        assert result is not None  # np.nan is a float, not None
        assert math.isnan(result)

    # -- Greater than / less than --------------------------------------------

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (">50", 50.0),
            ("<100", 100.0),
        ],
    )
    def test_comparison_operators_stripped(self, value: str, expected: float) -> None:
        """Leading > and < should be stripped and the number returned."""
        assert parse_numeric_value(value) == pytest.approx(expected)

    # -- Range values (midpoint) ---------------------------------------------

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("100-200", 150.0),
            ("1,000-2,000", 1500.0),
        ],
    )
    def test_range_returns_midpoint(self, value: str, expected: float) -> None:
        """A range like '100-200' should return the midpoint."""
        assert parse_numeric_value(value) == pytest.approx(expected)

    def test_negative_number_not_treated_as_range(self) -> None:
        """A leading negative sign should not be misinterpreted as a range."""
        assert parse_numeric_value("-42") == pytest.approx(-42.0)

    # -- Unparseable values --------------------------------------------------

    def test_garbage_returns_nan(self) -> None:
        """Completely unparseable strings should return NaN."""
        result = parse_numeric_value("hello")
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# parse_range()
# ---------------------------------------------------------------------------


class TestParseRange:
    """Tests for ``parse_range``."""

    def test_range_value(self) -> None:
        """'100-200' should return (100.0, 200.0)."""
        assert parse_range("100-200") == (100.0, 200.0)

    def test_greater_than(self) -> None:
        """'>50' should return (50.0, None)."""
        assert parse_range(">50") == (50.0, None)

    def test_less_than(self) -> None:
        """'<100' should return (None, 100.0)."""
        assert parse_range("<100") == (None, 100.0)

    def test_single_value(self) -> None:
        """A single number should return (value, value)."""
        assert parse_range("42") == (42.0, 42.0)

    def test_none_input(self) -> None:
        """None should return (None, None)."""
        assert parse_range(None) == (None, None)

    def test_nan_input(self) -> None:
        """NaN should return (None, None)."""
        assert parse_range(float("nan")) == (None, None)

    def test_garbage_input(self) -> None:
        """Unparseable strings should return (None, None)."""
        assert parse_range("hello") == (None, None)

    def test_comma_separated_range(self) -> None:
        """Commas inside range numbers should be handled."""
        assert parse_range("1,000-2,000") == (1000.0, 2000.0)


# ---------------------------------------------------------------------------
# clean_numeric_column()
# ---------------------------------------------------------------------------


class TestCleanNumericColumn:
    """Tests for ``clean_numeric_column``."""

    def test_cleans_series(self) -> None:
        """A Series of mixed values should be cleaned element-wise."""
        s = pd.Series(["100", "W", "1,000", "NA", "200-300"])
        result = clean_numeric_column(s)
        assert isinstance(result, pd.Series)
        assert result.iloc[0] == pytest.approx(100.0)
        assert math.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(1000.0)
        assert math.isnan(result.iloc[3])
        assert result.iloc[4] == pytest.approx(250.0)

    def test_keep_original_returns_dataframe(self) -> None:
        """With keep_original=True, a DataFrame with both columns is returned."""
        s = pd.Series(["100", "W"])
        result = clean_numeric_column(s, keep_original=True)
        assert isinstance(result, pd.DataFrame)
        assert "original" in result.columns
        assert "cleaned" in result.columns
        assert result["original"].iloc[0] == "100"
        assert result["cleaned"].iloc[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# standardize_country_name()
# ---------------------------------------------------------------------------


class TestStandardizeCountryName:
    """Tests for ``standardize_country_name``."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("USA", "United States"),
            ("U.S.", "United States"),
            ("United States of America", "United States"),
            ("UK", "United Kingdom"),
            ("Great Britain", "United Kingdom"),
            ("People's Republic of China", "China"),
            ("Republic of Korea", "South Korea"),
            ("Russian Federation", "Russia"),
            ("DRC", "Democratic Republic of the Congo"),
            ("Czechia", "Czech Republic"),
        ],
    )
    def test_known_mappings(self, raw: str, expected: str) -> None:
        """Known aliases should be resolved to the canonical name."""
        assert standardize_country_name(raw) == expected

    def test_unknown_name_returned_as_is(self) -> None:
        """An unmapped name should be returned unchanged (after stripping)."""
        assert standardize_country_name("  Brazil  ") == "Brazil"

    def test_already_canonical(self) -> None:
        """A name that is already canonical should pass through unchanged."""
        assert standardize_country_name("United States") == "United States"


# ---------------------------------------------------------------------------
# extract_commodity_code()
# ---------------------------------------------------------------------------


class TestExtractCommodityCode:
    """Tests for ``extract_commodity_code``."""

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("mcs2023-lithi_world.csv", "lithi"),
            ("mcs2024-cobal_salient.csv", "cobal"),
            ("mcs2023-zirco-hafni_world.csv", None),  # hyphen in code breaks simple \w+
            ("random_file.csv", None),
            ("mcs2023_lithi_world.csv", None),  # underscore instead of hyphen
        ],
    )
    def test_extraction(self, filename: str, expected: str | None) -> None:
        """Commodity codes should be extracted from well-formed USGS filenames."""
        assert extract_commodity_code(filename) == expected
