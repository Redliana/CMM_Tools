"""Shared fixtures for bgs-mcp tests."""

from __future__ import annotations

from typing import Any

import pytest

from bgs_mcp.bgs_client import BGSClient, MineralRecord


@pytest.fixture()
def client() -> BGSClient:
    """Create a BGSClient instance for testing."""
    return BGSClient()


@pytest.fixture()
def sample_mineral_record() -> MineralRecord:
    """Create a sample MineralRecord for testing."""
    return MineralRecord(
        commodity="lithium minerals",
        sub_commodity="Spodumene",
        statistic_type="Production",
        country="Australia",
        country_iso2="AU",
        country_iso3="AUS",
        year=2022,
        quantity=61000.0,
        units="Tonnes (metric)",
        yearbook_table="Lithium minerals",
        notes=None,
    )


@pytest.fixture()
def sample_api_response() -> dict[str, Any]:
    """Create a sample BGS OGC API response matching the real format.

    Returns:
        A dict mimicking the GeoJSON FeatureCollection structure from the
        BGS World Mineral Statistics API.
    """
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "bgs_commodity_trans": "lithium minerals",
                    "bgs_sub_commodity_trans": "Spodumene",
                    "bgs_statistic_type_trans": "Production",
                    "country_trans": "Australia",
                    "country_iso2_code": "AU",
                    "country_iso3_code": "AUS",
                    "year": "2022-01-01T00:00:00Z",
                    "quantity": 61000.0,
                    "units": "Tonnes (metric)",
                    "yearbook_table_trans": "Lithium minerals",
                    "concat_table_notes_text": None,
                },
            },
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "bgs_commodity_trans": "lithium minerals",
                    "bgs_sub_commodity_trans": None,
                    "bgs_statistic_type_trans": "Production",
                    "country_trans": "Chile",
                    "country_iso2_code": "CL",
                    "country_iso3_code": "CHL",
                    "year": "2022-01-01T00:00:00Z",
                    "quantity": 39000.0,
                    "units": "Tonnes (metric)",
                    "yearbook_table_trans": "Lithium minerals",
                    "concat_table_notes_text": "Estimated",
                },
            },
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "bgs_commodity_trans": "cobalt, mine",
                    "bgs_sub_commodity_trans": None,
                    "bgs_statistic_type_trans": "Production",
                    "country_trans": "Congo (Kinshasa)",
                    "country_iso2_code": "CD",
                    "country_iso3_code": "COD",
                    "year": "2021-01-01T00:00:00Z",
                    "quantity": 120000.0,
                    "units": "Tonnes (metal content)",
                    "yearbook_table_trans": "Cobalt",
                    "concat_table_notes_text": None,
                },
            },
        ],
        "numberMatched": 3,
        "numberReturned": 3,
    }


@pytest.fixture()
def sample_api_response_empty() -> dict[str, Any]:
    """Create an empty BGS OGC API response.

    Returns:
        A dict mimicking an empty GeoJSON FeatureCollection.
    """
    return {
        "type": "FeatureCollection",
        "features": [],
        "numberMatched": 0,
        "numberReturned": 0,
    }


@pytest.fixture()
def sample_multi_year_response() -> dict[str, Any]:
    """Create a multi-year API response for time series testing.

    Returns:
        A dict with records spanning multiple years for the same
        commodity/country combination.
    """
    features = []
    for year_val, qty in [(2019, 42000.0), (2020, 48000.0), (2021, 55000.0), (2022, 61000.0)]:
        features.append(
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "bgs_commodity_trans": "lithium minerals",
                    "bgs_sub_commodity_trans": None,
                    "bgs_statistic_type_trans": "Production",
                    "country_trans": "Australia",
                    "country_iso2_code": "AU",
                    "country_iso3_code": "AUS",
                    "year": f"{year_val}-01-01T00:00:00Z",
                    "quantity": qty,
                    "units": "Tonnes (metric)",
                    "yearbook_table_trans": "Lithium minerals",
                    "concat_table_notes_text": None,
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "numberMatched": len(features),
        "numberReturned": len(features),
    }
