"""Unit tests for usitc_mcp.models."""

from __future__ import annotations

from usitc_mcp.models import CMM_HTS_CODES, MINERAL_NAMES, TradeRecord


def test_cmm_hts_codes_nonempty():
    assert len(CMM_HTS_CODES) > 5
    assert "lithium" in CMM_HTS_CODES
    assert len(CMM_HTS_CODES["lithium"]) > 0


def test_mineral_names_match_hts_keys():
    for key in CMM_HTS_CODES:
        assert key in MINERAL_NAMES, f"Missing display name for {key}"


def test_trade_record_roundtrip():
    rec = TradeRecord(
        year=2023,
        month=12,
        hts_code="2836.91.00",
        hts_description="Lithium carbonates",
        country_code="CL",
        country_name="Chile",
        flow="import",
        data_type="general_imports",
        value_usd=1_234_567.89,
        quantity_1=500.0,
        quantity_1_unit="kg",
    )
    assert rec.hts_code == "2836.91.00"
    assert rec.value_usd == 1_234_567.89
    assert rec.model_dump()["flow"] == "import"
