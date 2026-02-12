"""Tests for uncomtrade_mcp.models module.

Covers:
- TradeRecord construction from aliased API data
- TradeRecord property fallbacks (reporter_name, partner_name)
- TradeRecord model_dump roundtrip
- CountryReference and CommodityReference model construction
- CRITICAL_MINERAL_HS_CODES completeness and structure
- MINERAL_NAMES mapping coverage
"""

from __future__ import annotations

from typing import Any

import pytest

from uncomtrade_mcp.models import (
    CRITICAL_MINERAL_HS_CODES,
    MINERAL_NAMES,
    CommodityReference,
    CountryReference,
    TradeRecord,
)

# ---------------------------------------------------------------------------
# TradeRecord construction
# ---------------------------------------------------------------------------


class TestTradeRecordConstruction:
    """Tests for constructing TradeRecord from API response data."""

    def test_from_aliased_data(self, sample_trade_record_data: dict[str, Any]) -> None:
        """TradeRecord should parse aliased field names from the API response."""
        record = TradeRecord.model_validate(sample_trade_record_data)
        assert record.period == "2023"
        assert record.reporter_code == 842
        assert record.reporter == "United States of America"
        assert record.partner_code == 156
        assert record.partner == "China"
        assert record.flow_code == "M"
        assert record.flow == "Import"
        assert record.commodity_code == "2605"
        assert record.commodity == "Cobalt ores and concentrates"
        assert record.trade_value == 15_000_000.0
        assert record.net_weight == 500_000.0
        assert record.quantity == 500.0
        assert record.quantity_unit == "kg"

    def test_optional_fields_default_to_none(self) -> None:
        """Optional fields should default to None when not provided."""
        minimal = {
            "period": "2023",
            "reporterCode": 842,
            "partnerCode": 0,
            "flowCode": "M",
            "cmdCode": "TOTAL",
        }
        record = TradeRecord.model_validate(minimal)
        assert record.reporter is None
        assert record.partner is None
        assert record.flow is None
        assert record.commodity is None
        assert record.trade_value is None
        assert record.net_weight is None
        assert record.quantity is None
        assert record.quantity_unit is None

    def test_model_dump_roundtrip(self, sample_trade_record: TradeRecord) -> None:
        """model_dump followed by construction should yield an equivalent record."""
        data = sample_trade_record.model_dump()
        restored = TradeRecord(**data)
        assert restored.period == sample_trade_record.period
        assert restored.reporter_code == sample_trade_record.reporter_code
        assert restored.trade_value == sample_trade_record.trade_value


# ---------------------------------------------------------------------------
# TradeRecord properties
# ---------------------------------------------------------------------------


class TestTradeRecordProperties:
    """Tests for TradeRecord computed properties."""

    def test_reporter_name_with_desc(self, sample_trade_record: TradeRecord) -> None:
        """reporter_name should return the reporter description when available."""
        assert sample_trade_record.reporter_name == "United States of America"

    def test_reporter_name_fallback(self) -> None:
        """When reporter is None, reporter_name should fall back to code string."""
        record = TradeRecord.model_validate(
            {
                "period": "2023",
                "reporterCode": 999,
                "partnerCode": 0,
                "flowCode": "M",
                "cmdCode": "TOTAL",
            }
        )
        assert record.reporter_name == "Country 999"

    def test_partner_name_with_desc(self, sample_trade_record: TradeRecord) -> None:
        """partner_name should return the partner description when available."""
        assert sample_trade_record.partner_name == "China"

    def test_partner_name_world(self) -> None:
        """When partner_code is 0, partner_name should return 'World'."""
        record = TradeRecord.model_validate(
            {
                "period": "2023",
                "reporterCode": 842,
                "partnerCode": 0,
                "flowCode": "M",
                "cmdCode": "TOTAL",
            }
        )
        assert record.partner_name == "World"

    def test_partner_name_fallback(self) -> None:
        """When partner is None and code is non-zero, should fall back to code string."""
        record = TradeRecord.model_validate(
            {
                "period": "2023",
                "reporterCode": 842,
                "partnerCode": 999,
                "flowCode": "M",
                "cmdCode": "TOTAL",
            }
        )
        assert record.partner_name == "Country 999"


# ---------------------------------------------------------------------------
# CountryReference
# ---------------------------------------------------------------------------


class TestCountryReference:
    """Tests for the CountryReference model."""

    def test_construction(self, sample_country_reference: CountryReference) -> None:
        """Verify CountryReference creation with all fields."""
        assert sample_country_reference.id == 842
        assert sample_country_reference.text == "United States of America"
        assert sample_country_reference.iso3 == "USA"

    def test_iso3_optional(self) -> None:
        """iso3 should default to None when not provided."""
        ref = CountryReference(id=0, text="World")
        assert ref.iso3 is None


# ---------------------------------------------------------------------------
# CommodityReference
# ---------------------------------------------------------------------------


class TestCommodityReference:
    """Tests for the CommodityReference model."""

    def test_construction(self, sample_commodity_reference: CommodityReference) -> None:
        """Verify CommodityReference creation with all fields."""
        assert sample_commodity_reference.id == "2605"
        assert sample_commodity_reference.text == "Cobalt ores and concentrates"
        assert sample_commodity_reference.parent == "26"

    def test_parent_optional(self) -> None:
        """parent should default to None when not provided."""
        ref = CommodityReference(id="TOTAL", text="All Commodities")
        assert ref.parent is None


# ---------------------------------------------------------------------------
# CRITICAL_MINERAL_HS_CODES
# ---------------------------------------------------------------------------


class TestCriticalMineralHSCodes:
    """Tests for the CRITICAL_MINERAL_HS_CODES constant."""

    def test_has_expected_minerals(self) -> None:
        """All expected critical minerals should be present as keys."""
        expected = {
            "lithium",
            "cobalt",
            "hree",
            "lree",
            "rare_earth",
            "graphite",
            "nickel",
            "manganese",
            "gallium",
            "germanium",
            "copper",
        }
        assert expected.issubset(set(CRITICAL_MINERAL_HS_CODES.keys()))

    def test_minimum_mineral_count(self) -> None:
        """There should be at least 9 minerals defined."""
        assert len(CRITICAL_MINERAL_HS_CODES) >= 9

    def test_all_values_are_lists_of_strings(self) -> None:
        """Each mineral should map to a list of HS code strings."""
        for mineral, codes in CRITICAL_MINERAL_HS_CODES.items():
            assert isinstance(codes, list), f"{mineral} value is not a list"
            for code in codes:
                assert isinstance(code, str), f"{mineral} code {code} is not a string"

    def test_all_hs_codes_are_numeric(self) -> None:
        """All HS codes should be numeric strings (digits only)."""
        for mineral, codes in CRITICAL_MINERAL_HS_CODES.items():
            for code in codes:
                assert code.isdigit(), f"{mineral} code '{code}' contains non-digits"

    @pytest.mark.parametrize(
        ("mineral", "expected_code"),
        [
            ("lithium", "282520"),
            ("cobalt", "2605"),
            ("graphite", "250410"),
            ("nickel", "2604"),
            ("manganese", "2602"),
            ("gallium", "811292"),
            ("germanium", "811299"),
        ],
    )
    def test_specific_hs_codes(self, mineral: str, expected_code: str) -> None:
        """Verify specific HS codes are present for key minerals."""
        assert expected_code in CRITICAL_MINERAL_HS_CODES[mineral]


# ---------------------------------------------------------------------------
# MINERAL_NAMES
# ---------------------------------------------------------------------------


class TestMineralNames:
    """Tests for the MINERAL_NAMES display name mapping."""

    def test_covers_all_hs_code_minerals(self) -> None:
        """Every mineral in CRITICAL_MINERAL_HS_CODES should have a display name."""
        for mineral in CRITICAL_MINERAL_HS_CODES:
            assert mineral in MINERAL_NAMES, f"Missing display name for {mineral}"

    def test_all_values_are_strings(self) -> None:
        """All mineral display names should be non-empty strings."""
        for mineral, name in MINERAL_NAMES.items():
            assert isinstance(name, str)
            assert len(name) > 0, f"Empty display name for {mineral}"

    @pytest.mark.parametrize(
        ("mineral", "expected_name"),
        [
            ("lithium", "Lithium (Li)"),
            ("cobalt", "Cobalt (Co)"),
            ("nickel", "Nickel (Ni)"),
            ("graphite", "Graphite (Gr)"),
            ("manganese", "Manganese (Mn)"),
        ],
    )
    def test_specific_display_names(self, mineral: str, expected_name: str) -> None:
        """Verify specific mineral display names are correct."""
        assert MINERAL_NAMES[mineral] == expected_name
