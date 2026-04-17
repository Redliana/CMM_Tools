"""Unit tests for census_aes_mcp.models and client helpers."""

from __future__ import annotations

from census_aes_mcp.client import CensusClient, _safe_float
from census_aes_mcp.models import CMM_HS_CODES, MINERAL_NAMES, ExportRecord


def test_cmm_hs_codes_nonempty():
    assert len(CMM_HS_CODES) > 5
    assert "lithium" in CMM_HS_CODES


def test_mineral_names_match_hs_keys():
    for key in CMM_HS_CODES:
        assert key in MINERAL_NAMES


def test_safe_float_handles_missing():
    assert _safe_float(None) is None
    assert _safe_float("") is None
    assert _safe_float(".") is None
    assert _safe_float("N/A") is None
    assert _safe_float("1234.5") == 1234.5
    assert _safe_float("notanumber") is None


def test_rows_to_dicts_empty():
    assert CensusClient._rows_to_dicts([]) == []
    assert CensusClient._rows_to_dicts([["header"]]) == []


def test_rows_to_dicts_basic():
    data = [
        ["YEAR", "MONTH", "ALL_VAL_MO"],
        ["2023", "12", "1000000"],
        ["2023", "11", "950000"],
    ]
    result = CensusClient._rows_to_dicts(data)
    assert len(result) == 2
    assert result[0]["YEAR"] == "2023"
    assert result[0]["ALL_VAL_MO"] == "1000000"


def test_export_record_roundtrip():
    rec = ExportRecord(
        year=2023,
        month=12,
        hs_code="283691",
        hs_description="Lithium carbonates",
        country_code="3370",
        country_name="Chile",
        value_usd=1_234_567.89,
        quantity_1=500.0,
        quantity_1_unit="kg",
    )
    assert rec.hs_code == "283691"
    assert rec.model_dump()["value_usd"] == 1_234_567.89
