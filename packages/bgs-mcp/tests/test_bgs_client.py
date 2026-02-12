"""Tests for bgs_mcp.bgs_client module.

Tests cover the BGSClient class including constructor configuration,
the critical minerals constant list, record parsing from API responses,
MineralRecord Pydantic model construction, and HTTP request handling
with mocked httpx responses.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from bgs_mcp.bgs_client import BGSClient, MineralRecord

# ---------------------------------------------------------------------------
# Helper to build a properly mocked httpx.AsyncClient context manager
# ---------------------------------------------------------------------------


def _make_mock_async_client(
    response_json: dict[str, Any],
    *,
    status_code: int = 200,
    raise_for_status_error: Exception | None = None,
) -> MagicMock:
    """Build a mock httpx.AsyncClient that works as an async context manager.

    Args:
        response_json: The JSON payload the mock response should return.
        status_code: HTTP status code for the response.
        raise_for_status_error: If set, raise_for_status will raise this error.

    Returns:
        A MagicMock suitable for patching ``httpx.AsyncClient``.
    """
    mock_response = MagicMock()
    mock_response.json.return_value = response_json
    mock_response.status_code = status_code
    if raise_for_status_error:
        mock_response.raise_for_status.side_effect = raise_for_status_error
    else:
        mock_response.raise_for_status.return_value = None

    mock_http_client = AsyncMock()
    mock_http_client.get.return_value = mock_response

    mock_cls = MagicMock()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    # Attach the inner client for call inspection
    mock_cls._inner_client = mock_http_client
    return mock_cls


# ---------------------------------------------------------------------------
# MineralRecord model tests
# ---------------------------------------------------------------------------


class TestMineralRecord:
    """Tests for the MineralRecord Pydantic model."""

    def test_construction_with_all_fields(self) -> None:
        """Verify that MineralRecord can be created with every field populated."""
        record = MineralRecord(
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
            notes="Estimated",
        )
        assert record.commodity == "lithium minerals"
        assert record.sub_commodity == "Spodumene"
        assert record.statistic_type == "Production"
        assert record.country == "Australia"
        assert record.country_iso2 == "AU"
        assert record.country_iso3 == "AUS"
        assert record.year == 2022
        assert record.quantity == 61000.0
        assert record.units == "Tonnes (metric)"
        assert record.yearbook_table == "Lithium minerals"
        assert record.notes == "Estimated"

    def test_construction_with_required_fields_only(self) -> None:
        """Verify defaults when only required fields are provided."""
        record = MineralRecord(
            commodity="cobalt, mine",
            statistic_type="Production",
            country="Congo (Kinshasa)",
        )
        assert record.commodity == "cobalt, mine"
        assert record.sub_commodity is None
        assert record.country_iso2 is None
        assert record.country_iso3 is None
        assert record.year is None
        assert record.quantity is None
        assert record.units is None
        assert record.yearbook_table is None
        assert record.notes is None

    def test_model_dump_roundtrip(self) -> None:
        """Verify dict serialization/deserialization roundtrip."""
        original = MineralRecord(
            commodity="nickel, mine",
            statistic_type="Production",
            country="Indonesia",
            country_iso3="IDN",
            year=2021,
            quantity=1000000.0,
            units="Tonnes (metric)",
        )
        data = original.model_dump()
        restored = MineralRecord(**data)
        assert restored == original

    def test_quantity_accepts_zero(self) -> None:
        """Verify that a zero quantity is accepted (not coerced to None)."""
        record = MineralRecord(
            commodity="gallium, primary",
            statistic_type="Production",
            country="Japan",
            quantity=0.0,
        )
        assert record.quantity == 0.0

    def test_quantity_accepts_none(self) -> None:
        """Verify that None quantity is accepted for records with no data."""
        record = MineralRecord(
            commodity="germanium metal",
            statistic_type="Production",
            country="Belgium",
            quantity=None,
        )
        assert record.quantity is None


# ---------------------------------------------------------------------------
# BGSClient constructor tests
# ---------------------------------------------------------------------------


class TestBGSClientInit:
    """Tests for BGSClient construction and configuration."""

    def test_default_timeout(self, client: BGSClient) -> None:
        """Verify the default request timeout is set."""
        assert client.timeout == 60.0

    def test_base_url_is_set(self, client: BGSClient) -> None:
        """Verify the BGS API base URL constant."""
        assert "ogcapi.bgs.ac.uk" in client.BASE_URL
        assert "world-mineral-statistics" in client.BASE_URL

    def test_critical_minerals_list_is_populated(self, client: BGSClient) -> None:
        """Verify the CRITICAL_MINERALS class constant is a non-empty list."""
        assert isinstance(client.CRITICAL_MINERALS, list)
        assert len(client.CRITICAL_MINERALS) > 0


# ---------------------------------------------------------------------------
# get_critical_minerals tests
# ---------------------------------------------------------------------------


class TestGetCriticalMinerals:
    """Tests for the get_critical_minerals method."""

    def test_returns_list_of_strings(self, client: BGSClient) -> None:
        """Verify the return type is a list of strings."""
        result = client.get_critical_minerals()
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)

    def test_returns_copy(self, client: BGSClient) -> None:
        """Verify that the returned list is a copy, not the class attribute."""
        result = client.get_critical_minerals()
        result.append("unobtanium")
        assert "unobtanium" not in client.CRITICAL_MINERALS

    def test_contains_expected_minerals(self, client: BGSClient) -> None:
        """Verify that key critical minerals are present in the list."""
        minerals = client.get_critical_minerals()
        expected_substrings = [
            "lithium",
            "cobalt",
            "nickel",
            "graphite",
            "rare earth",
            "gallium",
            "germanium",
            "copper",
        ]
        for substring in expected_substrings:
            matches = [m for m in minerals if substring in m.lower()]
            assert len(matches) > 0, f"No mineral containing '{substring}' found"

    def test_length_matches_class_constant(self, client: BGSClient) -> None:
        """Verify the returned list length matches the class constant."""
        result = client.get_critical_minerals()
        assert len(result) == len(BGSClient.CRITICAL_MINERALS)


# ---------------------------------------------------------------------------
# _parse_records tests
# ---------------------------------------------------------------------------


class TestParseRecords:
    """Tests for the _parse_records method."""

    def test_parses_valid_response(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify parsing a standard BGS API response with multiple features."""
        records = client._parse_records(sample_api_response)
        assert len(records) == 3
        assert all(isinstance(r, MineralRecord) for r in records)

    def test_first_record_fields(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify the first parsed record contains the expected field values."""
        records = client._parse_records(sample_api_response)
        rec = records[0]
        assert rec.commodity == "lithium minerals"
        assert rec.sub_commodity == "Spodumene"
        assert rec.statistic_type == "Production"
        assert rec.country == "Australia"
        assert rec.country_iso2 == "AU"
        assert rec.country_iso3 == "AUS"
        assert rec.year == 2022
        assert rec.quantity == 61000.0
        assert rec.units == "Tonnes (metric)"

    def test_parses_year_from_datetime_string(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify year extraction from ISO datetime strings (e.g., '2022-01-01T...')."""
        records = client._parse_records(sample_api_response)
        assert records[0].year == 2022
        assert records[2].year == 2021

    def test_handles_empty_response(
        self, client: BGSClient, sample_api_response_empty: dict[str, Any]
    ) -> None:
        """Verify an empty features list returns an empty records list."""
        records = client._parse_records(sample_api_response_empty)
        assert records == []

    def test_handles_missing_features_key(self, client: BGSClient) -> None:
        """Verify graceful handling when 'features' key is absent."""
        records = client._parse_records({})
        assert records == []

    def test_handles_missing_properties(self, client: BGSClient) -> None:
        """Verify parsing a feature with empty properties uses defaults."""
        data: dict[str, Any] = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": None,
                    "properties": {},
                }
            ]
        }
        records = client._parse_records(data)
        assert len(records) == 1
        assert records[0].commodity == ""
        assert records[0].year is None
        assert records[0].quantity is None

    def test_handles_partial_year_string(self, client: BGSClient) -> None:
        """Verify that a short or malformed year string results in year=None."""
        data: dict[str, Any] = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": None,
                    "properties": {
                        "bgs_commodity_trans": "cobalt, mine",
                        "bgs_statistic_type_trans": "Production",
                        "country_trans": "Test",
                        "year": "20",  # Too short to parse
                    },
                }
            ]
        }
        records = client._parse_records(data)
        assert len(records) == 1
        assert records[0].year is None

    def test_handles_empty_year_string(self, client: BGSClient) -> None:
        """Verify that an empty year string results in year=None."""
        data: dict[str, Any] = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": None,
                    "properties": {
                        "bgs_commodity_trans": "cobalt, mine",
                        "bgs_statistic_type_trans": "Production",
                        "country_trans": "Test",
                        "year": "",
                    },
                }
            ]
        }
        records = client._parse_records(data)
        assert records[0].year is None

    def test_notes_field_parsed(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify the notes field is parsed from concat_table_notes_text."""
        records = client._parse_records(sample_api_response)
        assert records[0].notes is None
        assert records[1].notes == "Estimated"


# ---------------------------------------------------------------------------
# Async method tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestSearchProduction:
    """Tests for the search_production async method with mocked HTTP."""

    @pytest.mark.asyncio()
    async def test_search_production_returns_records(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify search_production returns parsed MineralRecord objects."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            records = await client.search_production(
                commodity="lithium minerals",
                statistic_type="Production",
                limit=100,
            )

        assert len(records) > 0
        assert all(isinstance(r, MineralRecord) for r in records)

    @pytest.mark.asyncio()
    async def test_search_production_filters_by_year_from(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify year_from filtering excludes records before the start year."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            records = await client.search_production(
                commodity="lithium minerals",
                year_from=2022,
                limit=100,
            )

        # Only 2022 records should pass the filter (the 2021 cobalt record excluded)
        for record in records:
            if record.year is not None:
                assert record.year >= 2022

    @pytest.mark.asyncio()
    async def test_search_production_filters_by_year_to(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify year_to filtering excludes records after the end year."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            records = await client.search_production(
                commodity="lithium minerals",
                year_to=2021,
                limit=100,
            )

        for record in records:
            if record.year is not None:
                assert record.year <= 2021

    @pytest.mark.asyncio()
    async def test_search_production_empty_response(
        self, client: BGSClient, sample_api_response_empty: dict[str, Any]
    ) -> None:
        """Verify search_production returns an empty list for no matches."""
        mock_cls = _make_mock_async_client(sample_api_response_empty)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            records = await client.search_production(
                commodity="nonexistent mineral",
                limit=100,
            )

        assert records == []

    @pytest.mark.asyncio()
    async def test_search_production_uses_iso3_for_long_code(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify that a 3-character country_iso maps to country_iso3_code param."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            await client.search_production(
                commodity="lithium minerals",
                country_iso="AUS",
                limit=100,
            )

            inner = mock_cls._inner_client
            call_kwargs = inner.get.call_args
            params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
            assert params.get("country_iso3_code") == "AUS"

    @pytest.mark.asyncio()
    async def test_search_production_uses_iso2_for_short_code(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify that a 2-character country_iso maps to country_iso2_code param."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            await client.search_production(
                commodity="lithium minerals",
                country_iso="AU",
                limit=100,
            )

            inner = mock_cls._inner_client
            call_kwargs = inner.get.call_args
            params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
            assert params.get("country_iso2_code") == "AU"

    @pytest.mark.asyncio()
    async def test_search_production_sorted_descending_by_year(
        self, client: BGSClient, sample_multi_year_response: dict[str, Any]
    ) -> None:
        """Verify results are sorted by year in descending order."""
        mock_cls = _make_mock_async_client(sample_multi_year_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            records = await client.search_production(
                commodity="lithium minerals",
                limit=100,
            )

        years = [r.year for r in records if r.year is not None]
        assert years == sorted(years, reverse=True)


class TestGetCommodities:
    """Tests for the get_commodities async method with mocked HTTP."""

    @pytest.mark.asyncio()
    async def test_get_commodities_returns_sorted_list(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify get_commodities returns a sorted list of unique commodity names."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            commodities = await client.get_commodities()

        assert isinstance(commodities, list)
        assert commodities == sorted(commodities)
        # Should have unique entries from the sample: "lithium minerals" and "cobalt, mine"
        assert "lithium minerals" in commodities
        assert "cobalt, mine" in commodities

    @pytest.mark.asyncio()
    async def test_get_commodities_deduplicates(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify that duplicate commodity names are deduplicated."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            commodities = await client.get_commodities()

        # "lithium minerals" appears twice in sample, should appear once
        assert commodities.count("lithium minerals") == 1


class TestGetCountries:
    """Tests for the get_countries async method with mocked HTTP."""

    @pytest.mark.asyncio()
    async def test_get_countries_returns_sorted_dicts(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify get_countries returns country dicts sorted by name."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            countries = await client.get_countries()

        assert isinstance(countries, list)
        assert len(countries) == 3  # Australia, Chile, Congo
        names = [c["name"] for c in countries]
        assert names == sorted(names)

    @pytest.mark.asyncio()
    async def test_get_countries_has_expected_keys(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify each country dict has name, iso2, and iso3 keys."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            countries = await client.get_countries()

        for country in countries:
            assert "name" in country
            assert "iso2" in country
            assert "iso3" in country


class TestGetTimeSeries:
    """Tests for the get_time_series async method with mocked HTTP."""

    @pytest.mark.asyncio()
    async def test_get_time_series_sorted_ascending(
        self, client: BGSClient, sample_multi_year_response: dict[str, Any]
    ) -> None:
        """Verify time series results are sorted by year ascending."""
        mock_cls = _make_mock_async_client(sample_multi_year_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            records = await client.get_time_series(
                commodity="lithium minerals",
                country="Australia",
            )

        years = [r.year for r in records if r.year is not None]
        assert years == sorted(years)


class TestGetCommodityByCountry:
    """Tests for the get_commodity_by_country async method with mocked HTTP."""

    @pytest.mark.asyncio()
    async def test_returns_ranked_list(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify get_commodity_by_country returns a ranked list of dicts."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            ranked = await client.get_commodity_by_country(
                commodity="lithium minerals",
                year=2022,
            )

        assert isinstance(ranked, list)
        # Each item should have expected keys
        for item in ranked:
            assert "country" in item
            assert "quantity" in item
            assert "units" in item
            assert "year" in item
            assert "country_iso3" in item

    @pytest.mark.asyncio()
    async def test_returns_empty_for_no_data(
        self, client: BGSClient, sample_api_response_empty: dict[str, Any]
    ) -> None:
        """Verify returns empty list when no records match."""
        mock_cls = _make_mock_async_client(sample_api_response_empty)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            ranked = await client.get_commodity_by_country(
                commodity="nonexistent",
            )

        assert ranked == []

    @pytest.mark.asyncio()
    async def test_ranked_sorted_descending_by_quantity(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify results are sorted by quantity in descending order."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            ranked = await client.get_commodity_by_country(
                commodity="lithium minerals",
                year=2022,
            )

        if len(ranked) > 1:
            quantities = [r["quantity"] for r in ranked]
            assert quantities == sorted(quantities, reverse=True)


class TestRequest:
    """Tests for the _request private method with mocked HTTP."""

    @pytest.mark.asyncio()
    async def test_request_builds_correct_url(
        self, client: BGSClient, sample_api_response: dict[str, Any]
    ) -> None:
        """Verify _request calls the expected URL with correct parameters."""
        mock_cls = _make_mock_async_client(sample_api_response)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            await client._request(params={"bgs_commodity_trans": "lithium minerals"}, limit=50)

            inner = mock_cls._inner_client
            inner.get.assert_called_once()
            call_args = inner.get.call_args
            url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
            assert url.endswith("/items")

    @pytest.mark.asyncio()
    async def test_request_raises_on_http_error(self, client: BGSClient) -> None:
        """Verify _request propagates HTTP errors."""
        error = httpx.HTTPStatusError(
            "Server Error",
            request=httpx.Request("GET", "https://example.com"),
            response=httpx.Response(500),
        )
        mock_cls = _make_mock_async_client({}, raise_for_status_error=error)

        with patch("bgs_mcp.bgs_client.httpx.AsyncClient", mock_cls):
            with pytest.raises(httpx.HTTPStatusError):
                await client._request()
