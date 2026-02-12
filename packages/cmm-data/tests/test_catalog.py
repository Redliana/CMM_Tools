"""Tests for cmm_data.catalog module."""

from __future__ import annotations

from cmm_data.catalog import (
    get_commodity_info,
    list_commodities,
    list_critical_minerals,
)
from cmm_data.loaders.usgs_commodity import COMMODITY_NAMES, CRITICAL_MINERALS

# ---------------------------------------------------------------------------
# list_commodities()
# ---------------------------------------------------------------------------


class TestListCommodities:
    """Tests for ``list_commodities``."""

    def test_returns_list(self) -> None:
        """list_commodities should return a list."""
        result = list_commodities()
        assert isinstance(result, list)

    def test_contains_known_codes(self) -> None:
        """Result should include well-known commodity codes."""
        result = list_commodities()
        for code in ["lithi", "cobal", "nicke", "graph", "raree"]:
            assert code in result, f"Expected commodity code '{code}' not found"

    def test_sorted_alphabetically(self) -> None:
        """list_commodities should return codes in sorted order."""
        result = list_commodities()
        assert result == sorted(result)

    def test_length_matches_commodity_names_dict(self) -> None:
        """The list length should match the COMMODITY_NAMES mapping."""
        assert len(list_commodities()) == len(COMMODITY_NAMES)


# ---------------------------------------------------------------------------
# list_critical_minerals()
# ---------------------------------------------------------------------------


class TestListCriticalMinerals:
    """Tests for ``list_critical_minerals``."""

    def test_returns_list(self) -> None:
        """list_critical_minerals should return a list."""
        result = list_critical_minerals()
        assert isinstance(result, list)

    def test_contains_known_critical_minerals(self) -> None:
        """Result should contain DOE-designated critical minerals."""
        result = list_critical_minerals()
        for code in ["lithi", "cobal", "galli", "germa"]:
            assert code in result, f"Expected critical mineral '{code}' not found"

    def test_is_a_copy(self) -> None:
        """The returned list should be a copy, not a reference to the module constant."""
        result = list_critical_minerals()
        result.append("fake_mineral")
        assert "fake_mineral" not in CRITICAL_MINERALS

    def test_all_codes_are_valid_commodities(self) -> None:
        """Every critical mineral code should also appear in COMMODITY_NAMES."""
        commodities = set(list_commodities())
        for mineral in list_critical_minerals():
            assert mineral in commodities, f"Critical mineral '{mineral}' not in commodity list"


# ---------------------------------------------------------------------------
# get_commodity_info()
# ---------------------------------------------------------------------------


class TestGetCommodityInfo:
    """Tests for ``get_commodity_info``."""

    def test_known_code_returns_correct_name(self) -> None:
        """Passing a known code should return the correct commodity name."""
        info = get_commodity_info("lithi")
        assert info["code"] == "lithi"
        assert info["name"] == "Lithium"

    def test_critical_mineral_flagged(self) -> None:
        """A critical mineral code should have is_critical_mineral == True."""
        info = get_commodity_info("cobal")
        assert info["is_critical_mineral"] is True

    def test_non_critical_mineral_flagged(self) -> None:
        """A non-critical commodity should have is_critical_mineral == False."""
        info = get_commodity_info("gold")
        assert info["is_critical_mineral"] is False

    def test_unknown_code_returns_titlecased_name(self) -> None:
        """An unrecognised code should fall back to title-cased code."""
        info = get_commodity_info("xyzzy")
        assert info["name"] == "Xyzzy"
        assert info["is_critical_mineral"] is False

    def test_data_types_present(self) -> None:
        """The info dict should always include a data_types list."""
        info = get_commodity_info("lithi")
        assert "data_types" in info
        assert isinstance(info["data_types"], list)
        assert "world" in info["data_types"]
        assert "salient" in info["data_types"]
