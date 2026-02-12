"""Shared fixtures for uncomtrade-mcp tests."""

from __future__ import annotations

from typing import Any

import pytest

from uncomtrade_mcp.client import ComtradeClient
from uncomtrade_mcp.models import (
    CommodityReference,
    CountryReference,
    TradeRecord,
)

# ---------------------------------------------------------------------------
# Client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> ComtradeClient:
    """Create a ComtradeClient with a test API key.

    Returns:
        A ComtradeClient instance configured with a fake API key.
    """
    return ComtradeClient(api_key="test-api-key")


@pytest.fixture()
def client_no_key(monkeypatch: pytest.MonkeyPatch) -> ComtradeClient:
    """Create a ComtradeClient without an API key.

    Returns:
        A ComtradeClient instance with no API key configured.
    """
    monkeypatch.delenv("UNCOMTRADE_API_KEY", raising=False)
    return ComtradeClient(api_key=None)


# ---------------------------------------------------------------------------
# TradeRecord fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_trade_record_data() -> dict[str, Any]:
    """Return raw API response data for a single trade record.

    Returns:
        A dict matching the UN Comtrade API JSON structure with aliased keys.
    """
    return {
        "period": "2023",
        "reporterCode": 842,
        "reporterDesc": "United States of America",
        "partnerCode": 156,
        "partnerDesc": "China",
        "flowCode": "M",
        "flowDesc": "Import",
        "cmdCode": "2605",
        "cmdDesc": "Cobalt ores and concentrates",
        "primaryValue": 15_000_000.0,
        "netWgt": 500_000.0,
        "qty": 500.0,
        "qtyUnitAbbr": "kg",
    }


@pytest.fixture()
def sample_trade_record(sample_trade_record_data: dict[str, Any]) -> TradeRecord:
    """Create a validated TradeRecord from sample API data.

    Returns:
        A TradeRecord instance parsed from realistic API response data.
    """
    return TradeRecord.model_validate(sample_trade_record_data)


@pytest.fixture()
def sample_api_response(sample_trade_record_data: dict[str, Any]) -> dict[str, Any]:
    """Return a full API response containing trade data records.

    Returns:
        A dict mimicking the UN Comtrade data endpoint response envelope.
    """
    return {
        "data": [
            sample_trade_record_data,
            {
                "period": "2023",
                "reporterCode": 842,
                "reporterDesc": "United States of America",
                "partnerCode": 276,
                "partnerDesc": "Germany",
                "flowCode": "X",
                "flowDesc": "Export",
                "cmdCode": "2605",
                "cmdDesc": "Cobalt ores and concentrates",
                "primaryValue": 5_000_000.0,
                "netWgt": 100_000.0,
                "qty": 100.0,
                "qtyUnitAbbr": "kg",
            },
        ],
        "count": 2,
    }


@pytest.fixture()
def sample_api_response_empty() -> dict[str, Any]:
    """Return an empty API response.

    Returns:
        A dict mimicking an empty UN Comtrade data endpoint response.
    """
    return {"data": [], "count": 0}


@pytest.fixture()
def sample_reporters_response() -> dict[str, Any]:
    """Return a sample reporters reference response.

    Returns:
        A dict mimicking the UN Comtrade reporters endpoint response.
    """
    return {
        "results": [
            {"id": 842, "text": "United States of America", "iso3": "USA"},
            {"id": 156, "text": "China", "iso3": "CHN"},
            {"id": 276, "text": "Germany", "iso3": "DEU"},
            {"id": 392, "text": "Japan", "iso3": "JPN"},
        ]
    }


@pytest.fixture()
def sample_partners_response() -> dict[str, Any]:
    """Return a sample partners reference response.

    Returns:
        A dict mimicking the UN Comtrade partners endpoint response.
    """
    return {
        "results": [
            {"id": 0, "text": "World"},
            {"id": 842, "text": "United States of America"},
            {"id": 156, "text": "China"},
        ]
    }


@pytest.fixture()
def sample_commodities_response() -> dict[str, Any]:
    """Return a sample commodities reference response.

    Returns:
        A dict mimicking the UN Comtrade commodities endpoint response.
    """
    return {
        "results": [
            {"id": "2602", "text": "Manganese ores and concentrates", "parent": "26"},
            {"id": "2605", "text": "Cobalt ores and concentrates", "parent": "26"},
            {"id": "8105", "text": "Cobalt and articles thereof", "parent": "81"},
        ]
    }


# ---------------------------------------------------------------------------
# Reference model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_country_reference() -> CountryReference:
    """Create a sample CountryReference.

    Returns:
        A CountryReference for the United States.
    """
    return CountryReference(id=842, text="United States of America", iso3="USA")


@pytest.fixture()
def sample_commodity_reference() -> CommodityReference:
    """Create a sample CommodityReference.

    Returns:
        A CommodityReference for cobalt ores.
    """
    return CommodityReference(id="2605", text="Cobalt ores and concentrates", parent="26")
