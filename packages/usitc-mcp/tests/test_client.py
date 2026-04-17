"""Unit tests for usitc_mcp.client."""

from __future__ import annotations

from usitc_mcp.client import USITCClient


def test_build_saved_query_uses_validated_list_mode_for_hts_filters():
    payload = USITCClient._build_saved_query(
        trade_type="Import",
        data_to_report="GEN_CUSTOMS_VALUE",
        years=[2023],
        hts_codes=["2836.91.00"],
        country_codes=None,
    )
    commodities = payload["searchOptions"]["commodities"]

    assert commodities["commodities"] == ["28369100"]
    assert commodities["commoditySelectType"] == "list"
    assert commodities["aggregation"] == "Aggregate Commodities"
    assert commodities["commoditiesExpanded"] == [
        {"name": "28369100", "value": "28369100", "hasChildren": None}
    ]
    assert commodities["commoditiesManual"] == "28369100"
    assert commodities["groupGranularity"] == "2"


def test_build_saved_query_uses_validated_list_mode_for_country_filters():
    payload = USITCClient._build_saved_query(
        trade_type="Import",
        data_to_report="GEN_CUSTOMS_VALUE",
        years=[2023],
        hts_codes=["28369100"],
        country_codes=["3370"],
    )
    countries = payload["searchOptions"]["countries"]

    assert countries["countries"] == ["3370"]
    assert countries["countriesSelectType"] == "list"
    assert countries["aggregation"] == "Aggregate countries"
    assert countries["countriesExpanded"] == [
        {"name": "3370", "value": "3370", "hasChildren": None}
    ]
