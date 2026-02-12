"""Tests for uncomtrade_mcp.server MCP tool functions.

Covers:
- Server module attributes and configuration
- list_critical_minerals() tool output structure
- get_client() helper function
- get_trade_data() tool with mocked client
- get_critical_mineral_trade() tool with mocked client
- list_reporters(), list_partners(), list_commodities() tools with mocked client
- Error handling in get_critical_mineral_trade for unknown minerals
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from uncomtrade_mcp.client import ComtradeClient
from uncomtrade_mcp.models import CRITICAL_MINERAL_HS_CODES, MINERAL_NAMES, TradeRecord

# ---------------------------------------------------------------------------
# Server module attributes
# ---------------------------------------------------------------------------


class TestServerAttributes:
    """Verify the server module exposes expected attributes."""

    def test_mcp_name(self) -> None:
        """The MCP server should be named 'UN Comtrade'."""
        from uncomtrade_mcp.server import mcp

        assert mcp.name == "UN Comtrade"

    def test_server_instructions_content(self) -> None:
        """Server instructions should mention critical minerals and trade data."""
        from uncomtrade_mcp.server import mcp

        instructions = mcp.instructions or ""
        assert "critical" in instructions.lower()
        assert "trade" in instructions.lower()

    def test_main_is_callable(self) -> None:
        """main() should be defined and callable."""
        from uncomtrade_mcp.server import main

        assert callable(main)

    def test_get_client_returns_comtrade_client(self) -> None:
        """get_client() should return a ComtradeClient instance."""
        from uncomtrade_mcp.server import get_client

        client = get_client()
        assert isinstance(client, ComtradeClient)


# ---------------------------------------------------------------------------
# list_critical_minerals tool
# ---------------------------------------------------------------------------


class TestListCriticalMinerals:
    """Tests for the list_critical_minerals MCP tool."""

    @pytest.mark.asyncio()
    async def test_returns_dict_with_count_and_minerals(self) -> None:
        """Response should include 'count', 'minerals', and 'usage' keys."""
        from uncomtrade_mcp.server import list_critical_minerals

        result = await list_critical_minerals()
        assert "count" in result
        assert "minerals" in result
        assert "usage" in result

    @pytest.mark.asyncio()
    async def test_mineral_count_matches_hs_codes(self) -> None:
        """The count should match the number of CRITICAL_MINERAL_HS_CODES entries."""
        from uncomtrade_mcp.server import list_critical_minerals

        result = await list_critical_minerals()
        assert result["count"] == len(CRITICAL_MINERAL_HS_CODES)

    @pytest.mark.asyncio()
    async def test_each_mineral_has_required_fields(self) -> None:
        """Each mineral entry should have 'id', 'name', and 'hs_codes' fields."""
        from uncomtrade_mcp.server import list_critical_minerals

        result = await list_critical_minerals()
        for mineral in result["minerals"]:
            assert "id" in mineral
            assert "name" in mineral
            assert "hs_codes" in mineral
            assert isinstance(mineral["hs_codes"], list)

    @pytest.mark.asyncio()
    async def test_mineral_names_from_lookup(self) -> None:
        """Mineral names should come from the MINERAL_NAMES mapping."""
        from uncomtrade_mcp.server import list_critical_minerals

        result = await list_critical_minerals()
        minerals_by_id = {m["id"]: m["name"] for m in result["minerals"]}
        for mineral_id, expected_name in MINERAL_NAMES.items():
            if mineral_id in minerals_by_id:
                assert minerals_by_id[mineral_id] == expected_name


# ---------------------------------------------------------------------------
# get_trade_data tool
# ---------------------------------------------------------------------------


class TestGetTradeData:
    """Tests for the get_trade_data MCP tool with mocked client."""

    @pytest.mark.asyncio()
    async def test_returns_dict_with_count_and_records(
        self,
        sample_trade_record: TradeRecord,
    ) -> None:
        """Response should include 'count', 'query', and 'records' keys."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_trade_data = AsyncMock(return_value=[sample_trade_record])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import get_trade_data

            result = await get_trade_data(reporter="842", commodity="2605")

        assert "count" in result
        assert "query" in result
        assert "records" in result
        assert result["count"] == 1

    @pytest.mark.asyncio()
    async def test_query_parameters_in_response(
        self,
        sample_trade_record: TradeRecord,
    ) -> None:
        """The response should echo back the query parameters."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_trade_data = AsyncMock(return_value=[sample_trade_record])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import get_trade_data

            result = await get_trade_data(
                reporter="842",
                commodity="2605",
                partner="156",
                flow="M",
                year="2023",
            )

        assert result["query"]["reporter"] == "842"
        assert result["query"]["commodity"] == "2605"
        assert result["query"]["partner"] == "156"
        assert result["query"]["flow"] == "M"
        assert result["query"]["year"] == "2023"

    @pytest.mark.asyncio()
    async def test_max_records_capped_at_500(
        self,
        sample_trade_record: TradeRecord,
    ) -> None:
        """max_records should be capped at 500 even if a larger value is passed."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_trade_data = AsyncMock(return_value=[])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import get_trade_data

            await get_trade_data(reporter="842", commodity="2605", max_records=1000)

        call_kwargs = mock_client.get_trade_data.call_args.kwargs
        assert call_kwargs["max_records"] == 500

    @pytest.mark.asyncio()
    async def test_records_are_serialized_dicts(
        self,
        sample_trade_record: TradeRecord,
    ) -> None:
        """Each record in the response should be a serialized dict."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_trade_data = AsyncMock(return_value=[sample_trade_record])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import get_trade_data

            result = await get_trade_data(reporter="842", commodity="2605")

        assert isinstance(result["records"][0], dict)
        assert "period" in result["records"][0]

    @pytest.mark.asyncio()
    async def test_empty_results(self) -> None:
        """When no records are returned, count should be 0."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_trade_data = AsyncMock(return_value=[])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import get_trade_data

            result = await get_trade_data(reporter="842", commodity="9999")

        assert result["count"] == 0
        assert result["records"] == []


# ---------------------------------------------------------------------------
# get_critical_mineral_trade tool
# ---------------------------------------------------------------------------


class TestGetCriticalMineralTrade:
    """Tests for the get_critical_mineral_trade MCP tool with mocked client."""

    @pytest.mark.asyncio()
    async def test_returns_mineral_info(
        self,
        sample_trade_record: TradeRecord,
    ) -> None:
        """Response should include mineral name and HS codes queried."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_critical_mineral_trade = AsyncMock(return_value=[sample_trade_record])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import get_critical_mineral_trade

            result = await get_critical_mineral_trade(mineral="cobalt")

        assert "mineral" in result
        assert "hs_codes_queried" in result
        assert result["mineral"] == "Cobalt (Co)"

    @pytest.mark.asyncio()
    async def test_returns_error_for_unknown_mineral(self) -> None:
        """An unknown mineral name should return an error dict."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_critical_mineral_trade = AsyncMock(
            side_effect=ValueError("Unknown mineral: unobtanium")
        )

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import get_critical_mineral_trade

            result = await get_critical_mineral_trade(mineral="unobtanium")

        assert "error" in result

    @pytest.mark.asyncio()
    async def test_passes_correct_parameters(
        self,
        sample_trade_record: TradeRecord,
    ) -> None:
        """The tool should forward parameters to the client method."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_critical_mineral_trade = AsyncMock(return_value=[])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import get_critical_mineral_trade

            await get_critical_mineral_trade(
                mineral="lithium",
                reporter="842",
                partner="156",
                flow="M",
                year="2022",
                max_records=50,
            )

        call_kwargs = mock_client.get_critical_mineral_trade.call_args.kwargs
        assert call_kwargs["mineral"] == "lithium"
        assert call_kwargs["reporter"] == "842"
        assert call_kwargs["partner"] == "156"
        assert call_kwargs["flow"] == "M"
        assert call_kwargs["period"] == "2022"
        assert call_kwargs["max_records"] == 50


# ---------------------------------------------------------------------------
# list_reporters tool
# ---------------------------------------------------------------------------


class TestListReporters:
    """Tests for the list_reporters MCP tool with mocked client."""

    @pytest.mark.asyncio()
    async def test_returns_dict_with_count_and_reporters(
        self,
        sample_reporters_response: dict[str, Any],
    ) -> None:
        """Response should include 'count', 'reporters', and 'note'."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_reporters = AsyncMock(return_value=sample_reporters_response["results"])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import list_reporters

            result = await list_reporters()

        assert "count" in result
        assert "reporters" in result
        assert "note" in result

    @pytest.mark.asyncio()
    async def test_search_filters_reporters(
        self,
        sample_reporters_response: dict[str, Any],
    ) -> None:
        """A search term should filter reporters by name."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_reporters = AsyncMock(return_value=sample_reporters_response["results"])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import list_reporters

            result = await list_reporters(search="china")

        assert result["count"] == 1
        assert result["reporters"][0]["text"] == "China"

    @pytest.mark.asyncio()
    async def test_limit_parameter(
        self,
        sample_reporters_response: dict[str, Any],
    ) -> None:
        """The limit parameter should cap the number of results."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_reporters = AsyncMock(return_value=sample_reporters_response["results"])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import list_reporters

            result = await list_reporters(limit=2)

        assert result["count"] <= 2


# ---------------------------------------------------------------------------
# list_partners tool
# ---------------------------------------------------------------------------


class TestListPartners:
    """Tests for the list_partners MCP tool with mocked client."""

    @pytest.mark.asyncio()
    async def test_returns_dict_with_count_and_partners(
        self,
        sample_partners_response: dict[str, Any],
    ) -> None:
        """Response should include 'count', 'partners', and 'note'."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_partners = AsyncMock(return_value=sample_partners_response["results"])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import list_partners

            result = await list_partners()

        assert "count" in result
        assert "partners" in result
        assert "note" in result

    @pytest.mark.asyncio()
    async def test_search_filters_partners(
        self,
        sample_partners_response: dict[str, Any],
    ) -> None:
        """A search term should filter partners by name."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_partners = AsyncMock(return_value=sample_partners_response["results"])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import list_partners

            result = await list_partners(search="world")

        assert result["count"] == 1
        assert result["partners"][0]["text"] == "World"


# ---------------------------------------------------------------------------
# list_commodities tool
# ---------------------------------------------------------------------------


class TestListCommodities:
    """Tests for the list_commodities MCP tool with mocked client."""

    @pytest.mark.asyncio()
    async def test_returns_dict_with_count_and_commodities(
        self,
        sample_commodities_response: dict[str, Any],
    ) -> None:
        """Response should include 'count', 'commodities', and 'note'."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_commodities = AsyncMock(return_value=sample_commodities_response["results"])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import list_commodities

            result = await list_commodities()

        assert "count" in result
        assert "commodities" in result
        assert "note" in result

    @pytest.mark.asyncio()
    async def test_search_filters_commodities(
        self,
        sample_commodities_response: dict[str, Any],
    ) -> None:
        """A search term should filter commodities by text or code."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_commodities = AsyncMock(return_value=sample_commodities_response["results"])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import list_commodities

            result = await list_commodities(search="cobalt")

        assert result["count"] >= 1
        assert any("cobalt" in c["text"].lower() for c in result["commodities"])

    @pytest.mark.asyncio()
    async def test_hs_level_filter(
        self,
        sample_commodities_response: dict[str, Any],
    ) -> None:
        """The hs_level parameter should filter by code length."""
        mock_client = AsyncMock(spec=ComtradeClient)
        mock_client.get_commodities = AsyncMock(return_value=sample_commodities_response["results"])

        with patch("uncomtrade_mcp.server.get_client", return_value=mock_client):
            from uncomtrade_mcp.server import list_commodities

            result = await list_commodities(hs_level=4)

        for commodity in result["commodities"]:
            assert len(str(commodity["id"])) == 4
