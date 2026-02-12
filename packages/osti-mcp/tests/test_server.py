"""Tests for osti_mcp.server MCP tool functions.

Covers:
- get_osti_overview() tool wrapper
- list_commodities() tool wrapper
- search_osti_documents() tool wrapper with various filter combinations
- get_osti_document() tool wrapper for existing and missing documents
- get_documents_by_commodity() tool wrapper
- get_recent_documents() tool wrapper
- Server module attributes and configuration
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osti_mcp.client import OSTIClient

# ---------------------------------------------------------------------------
# Server module attributes
# ---------------------------------------------------------------------------


class TestServerAttributes:
    """Verify the server module exposes expected attributes."""

    def test_mcp_name(self) -> None:
        """The MCP server should be named 'OSTI'."""
        from osti_mcp.server import mcp

        assert mcp.name == "OSTI"

    def test_server_instructions_content(self) -> None:
        """The server instructions should mention critical minerals topics."""
        from osti_mcp.server import mcp

        instructions = mcp.instructions or ""
        assert "critical minerals" in instructions.lower()

    def test_main_is_callable(self) -> None:
        """main() should be defined and callable."""
        from osti_mcp.server import main

        assert callable(main)


# ---------------------------------------------------------------------------
# get_osti_overview tool
# ---------------------------------------------------------------------------


class TestGetOstiOverview:
    """Tests for the get_osti_overview MCP tool."""

    @pytest.mark.asyncio()
    async def test_returns_statistics_dict(self, client: OSTIClient) -> None:
        """get_osti_overview should return a dict with statistics keys."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_osti_overview

            result = await get_osti_overview()

        assert isinstance(result, dict)
        assert "total_documents" in result
        assert "commodities" in result
        assert "product_types" in result
        assert "year_range" in result

    @pytest.mark.asyncio()
    async def test_total_documents_count(self, client: OSTIClient) -> None:
        """Total document count should match the fixture data size."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_osti_overview

            result = await get_osti_overview()

        assert result["total_documents"] == 6


# ---------------------------------------------------------------------------
# list_commodities tool
# ---------------------------------------------------------------------------


class TestListCommoditiesTool:
    """Tests for the list_commodities MCP tool."""

    @pytest.mark.asyncio()
    async def test_returns_dict(self, client: OSTIClient) -> None:
        """list_commodities should return a dict mapping codes to names."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import list_commodities

            result = await list_commodities()

        assert isinstance(result, dict)
        assert "HREE" in result
        assert "CO" in result

    @pytest.mark.asyncio()
    async def test_contains_all_commodity_codes(self, client: OSTIClient) -> None:
        """All expected commodity codes should be present."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import list_commodities

            result = await list_commodities()

        expected = {"HREE", "LREE", "CO", "LI", "GA", "GR", "NI", "CU", "GE", "OTH"}
        assert expected.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# search_osti_documents tool
# ---------------------------------------------------------------------------


class TestSearchOstiDocuments:
    """Tests for the search_osti_documents MCP tool."""

    @pytest.mark.asyncio()
    async def test_returns_dict_with_count_and_documents(self, client: OSTIClient) -> None:
        """Response should include 'count' and 'documents' keys."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import search_osti_documents

            result = await search_osti_documents()

        assert "count" in result
        assert "documents" in result
        assert isinstance(result["documents"], list)

    @pytest.mark.asyncio()
    async def test_no_filters_returns_all(self, client: OSTIClient) -> None:
        """Searching with no filters should return all documents."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import search_osti_documents

            result = await search_osti_documents(limit=100)

        assert result["count"] == 6

    @pytest.mark.asyncio()
    async def test_query_filter(self, client: OSTIClient) -> None:
        """A text query should filter matching documents."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import search_osti_documents

            result = await search_osti_documents(query="rare earth")

        assert result["count"] >= 1
        titles = [doc["title"] for doc in result["documents"]]
        assert any("Rare Earth" in t for t in titles)

    @pytest.mark.asyncio()
    async def test_commodity_filter(self, client: OSTIClient) -> None:
        """Filtering by commodity code should narrow results."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import search_osti_documents

            result = await search_osti_documents(commodity="CO")

        assert result["count"] == 1
        assert result["documents"][0]["commodity_category"] == "CO"

    @pytest.mark.asyncio()
    async def test_year_range_filter(self, client: OSTIClient) -> None:
        """Year filters should restrict results to the specified range."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import search_osti_documents

            result = await search_osti_documents(year_from=2023, year_to=2024)

        assert result["count"] >= 2
        for doc in result["documents"]:
            year = int(doc["publication_date"][:4])
            assert 2023 <= year <= 2024

    @pytest.mark.asyncio()
    async def test_limit_parameter(self, client: OSTIClient) -> None:
        """The limit parameter should cap the number of results."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import search_osti_documents

            result = await search_osti_documents(limit=2)

        assert result["count"] <= 2

    @pytest.mark.asyncio()
    async def test_documents_are_dicts(self, client: OSTIClient) -> None:
        """Each document in the response should be a serialized dict."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import search_osti_documents

            result = await search_osti_documents(limit=1)

        if result["count"] > 0:
            doc = result["documents"][0]
            assert isinstance(doc, dict)
            assert "osti_id" in doc
            assert "title" in doc


# ---------------------------------------------------------------------------
# get_osti_document tool
# ---------------------------------------------------------------------------


class TestGetOstiDocument:
    """Tests for the get_osti_document MCP tool."""

    @pytest.mark.asyncio()
    async def test_returns_document_for_valid_id(self, client: OSTIClient) -> None:
        """A valid OSTI ID should return the document as a dict."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_osti_document

            result = await get_osti_document("2342032")

        assert result is not None
        assert result["osti_id"] == "2342032"
        assert "title" in result

    @pytest.mark.asyncio()
    async def test_returns_none_for_invalid_id(self, client: OSTIClient) -> None:
        """A non-existent OSTI ID should return None."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_osti_document

            result = await get_osti_document("0000000")

        assert result is None

    @pytest.mark.asyncio()
    async def test_document_has_all_expected_fields(self, client: OSTIClient) -> None:
        """The returned document dict should contain all expected metadata fields."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_osti_document

            result = await get_osti_document("2342032")

        assert result is not None
        expected_keys = {
            "osti_id",
            "title",
            "authors",
            "publication_date",
            "description",
            "subjects",
            "commodity_category",
            "doi",
            "product_type",
            "research_orgs",
            "sponsor_orgs",
        }
        assert expected_keys.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# get_documents_by_commodity tool
# ---------------------------------------------------------------------------


class TestGetDocumentsByCommodity:
    """Tests for the get_documents_by_commodity MCP tool."""

    @pytest.mark.asyncio()
    async def test_returns_dict_with_expected_keys(self, client: OSTIClient) -> None:
        """Response should include commodity info, count, and documents."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_documents_by_commodity

            result = await get_documents_by_commodity(commodity="LI")

        assert "commodity" in result
        assert "commodity_name" in result
        assert "count" in result
        assert "documents" in result

    @pytest.mark.asyncio()
    async def test_commodity_code_uppercased(self, client: OSTIClient) -> None:
        """The commodity code in the response should be uppercased."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_documents_by_commodity

            result = await get_documents_by_commodity(commodity="li")

        assert result["commodity"] == "LI"

    @pytest.mark.asyncio()
    async def test_includes_commodity_name(self, client: OSTIClient) -> None:
        """The response should include the human-readable commodity name."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_documents_by_commodity

            result = await get_documents_by_commodity(commodity="HREE")

        assert result["commodity_name"] == "Heavy Rare Earth Elements"

    @pytest.mark.asyncio()
    async def test_filters_correct_documents(self, client: OSTIClient) -> None:
        """Only documents matching the commodity should be returned."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_documents_by_commodity

            result = await get_documents_by_commodity(commodity="CO")

        assert result["count"] == 1
        assert result["documents"][0]["commodity_category"] == "CO"


# ---------------------------------------------------------------------------
# get_recent_documents tool
# ---------------------------------------------------------------------------


class TestGetRecentDocuments:
    """Tests for the get_recent_documents MCP tool."""

    @pytest.mark.asyncio()
    async def test_returns_dict_with_count_and_documents(self, client: OSTIClient) -> None:
        """Response should include 'count' and 'documents' keys."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_recent_documents

            result = await get_recent_documents(limit=5)

        assert "count" in result
        assert "documents" in result

    @pytest.mark.asyncio()
    async def test_sorted_by_date_descending(self, client: OSTIClient) -> None:
        """Documents should be sorted by publication date newest first."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_recent_documents

            result = await get_recent_documents(limit=10)

        dates = [doc["publication_date"] for doc in result["documents"]]
        assert dates == sorted(dates, reverse=True)

    @pytest.mark.asyncio()
    async def test_limit_parameter(self, client: OSTIClient) -> None:
        """The limit parameter should cap results."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_recent_documents

            result = await get_recent_documents(limit=2)

        assert result["count"] <= 2

    @pytest.mark.asyncio()
    async def test_most_recent_is_first(self, client: OSTIClient) -> None:
        """The first document should be the most recently published."""
        with patch("osti_mcp.server.client", client):
            from osti_mcp.server import get_recent_documents

            result = await get_recent_documents(limit=1)

        assert result["count"] == 1
        assert result["documents"][0]["publication_date"] == "2024-01-20"
