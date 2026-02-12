"""Tests for arxiv_mcp.server module.

Covers:
- Module import smoke test
- parse_arxiv_entry() with various XML inputs
- format_paper_result() with sample dictionaries
- make_arxiv_request() with mocked httpx responses
- search_arxiv() integration with mocked network
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from arxiv_mcp.server import (
    ARXIV_API_BASE,
    ARXIV_NAMESPACE,
    USER_AGENT,
    format_paper_result,
    make_arxiv_request,
    parse_arxiv_entry,
    search_arxiv,
)

# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestModuleImport:
    """Verify the module and its key symbols are importable."""

    def test_import_server_module(self) -> None:
        """Importing arxiv_mcp.server should succeed without errors."""
        import arxiv_mcp.server  # noqa: F401

    def test_constants_are_defined(self) -> None:
        """Module-level constants should have sensible values."""
        assert ARXIV_API_BASE.startswith("https://")
        assert ARXIV_NAMESPACE["atom"] == "http://www.w3.org/2005/Atom"
        assert isinstance(USER_AGENT, str)
        assert len(USER_AGENT) > 0


# ---------------------------------------------------------------------------
# parse_arxiv_entry
# ---------------------------------------------------------------------------


class TestParseArxivEntry:
    """Tests for parse_arxiv_entry()."""

    def test_parse_full_entry(self, sample_entry_element: ET.Element) -> None:
        """A well-formed entry should yield all expected fields."""
        result: dict[str, Any] = parse_arxiv_entry(sample_entry_element)

        assert result["id"] == "2301.07041v1"
        assert result["title"] == "Attention Is All You Need"
        assert result["authors"] == [
            "Ashish Vaswani",
            "Noam Shazeer",
            "Niki Parmar",
            "Jakob Uszkoreit",
        ]
        assert "Transformer" in result["summary"]
        assert result["published"] == "2023-01-17T18:00:00Z"
        assert result["categories"] == ["cs.CL", "cs.AI"]
        assert result["pdf_url"] == "http://arxiv.org/pdf/2301.07041v1"

    def test_title_whitespace_normalized(self, sample_entry_element: ET.Element) -> None:
        """Leading/trailing whitespace and newlines in titles should be stripped."""
        result = parse_arxiv_entry(sample_entry_element)
        assert result["title"] == "Attention Is All You Need"
        assert "\n" not in result["title"]

    def test_summary_whitespace_normalized(self, sample_entry_element: ET.Element) -> None:
        """Leading/trailing whitespace and newlines in summaries should be stripped."""
        result = parse_arxiv_entry(sample_entry_element)
        assert not result["summary"].startswith(" ")
        assert not result["summary"].endswith(" ")

    def test_parse_entry_without_pdf_link(self, sample_entry_no_pdf_element: ET.Element) -> None:
        """When no explicit PDF link exists, a fallback URL should be generated."""
        result = parse_arxiv_entry(sample_entry_no_pdf_element)
        assert result["id"] == "9901.00001v1"
        assert result["pdf_url"] == "http://arxiv.org/pdf/9901.00001v1.pdf"

    def test_parse_entry_single_author(self, sample_entry_no_pdf_element: ET.Element) -> None:
        """An entry with one author should return a single-element list."""
        result = parse_arxiv_entry(sample_entry_no_pdf_element)
        assert result["authors"] == ["Jane Doe"]

    def test_parse_empty_entry(self, sample_entry_empty_element: ET.Element) -> None:
        """An entry missing all child elements should use safe defaults."""
        result = parse_arxiv_entry(sample_entry_empty_element)
        assert result["title"] == "Unknown"
        assert result["id"] == "Unknown"
        assert result["authors"] == []
        assert result["summary"] == ""
        assert result["published"] == "Unknown"
        assert result["categories"] == []


# ---------------------------------------------------------------------------
# format_paper_result
# ---------------------------------------------------------------------------


class TestFormatPaperResult:
    """Tests for format_paper_result()."""

    def test_format_includes_title(self, sample_paper_dict: dict[str, Any]) -> None:
        """The formatted string should contain the paper title."""
        output: str = format_paper_result(sample_paper_dict)
        assert "Attention Is All You Need" in output

    def test_format_includes_arxiv_id(self, sample_paper_dict: dict[str, Any]) -> None:
        """The formatted string should contain the ArXiv ID."""
        output = format_paper_result(sample_paper_dict)
        assert "2301.07041v1" in output

    def test_format_truncates_authors_over_three(self, sample_paper_dict: dict[str, Any]) -> None:
        """When there are more than three authors, the output should show 'et al.'."""
        output = format_paper_result(sample_paper_dict)
        assert "et al." in output
        assert "4 total" in output

    def test_format_does_not_truncate_few_authors(
        self, sample_paper_dict_few_authors: dict[str, Any]
    ) -> None:
        """When there are three or fewer authors, 'et al.' should not appear."""
        output = format_paper_result(sample_paper_dict_few_authors)
        assert "et al." not in output
        assert "Jane Doe" in output

    def test_format_includes_published_date(self, sample_paper_dict: dict[str, Any]) -> None:
        """Published date should appear in the output."""
        output = format_paper_result(sample_paper_dict)
        assert "2023-01-17" in output

    def test_format_includes_categories(self, sample_paper_dict: dict[str, Any]) -> None:
        """Categories should appear in the output."""
        output = format_paper_result(sample_paper_dict)
        assert "cs.CL" in output

    def test_format_includes_pdf_url(self, sample_paper_dict: dict[str, Any]) -> None:
        """PDF URL should appear in the output."""
        output = format_paper_result(sample_paper_dict)
        assert "http://arxiv.org/pdf/2301.07041v1" in output

    def test_format_truncates_abstract(self, sample_paper_dict: dict[str, Any]) -> None:
        """The abstract portion should be truncated to 300 characters with ellipsis."""
        output = format_paper_result(sample_paper_dict)
        assert output.rstrip().endswith("...")


# ---------------------------------------------------------------------------
# make_arxiv_request (async, mocked httpx)
# ---------------------------------------------------------------------------


class TestMakeArxivRequest:
    """Tests for make_arxiv_request() with mocked httpx.AsyncClient."""

    @pytest.mark.asyncio()
    async def test_successful_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 200 response should return the response text."""
        mock_response = MagicMock()
        mock_response.text = "<feed>ok</feed>"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await make_arxiv_request("https://export.arxiv.org/api/query?search_query=test")
        assert result == "<feed>ok</feed>"
        mock_client.get.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_http_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An HTTP error should be caught and None should be returned."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500 Server Error",
                request=MagicMock(),
                response=MagicMock(),
            )
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await make_arxiv_request("https://export.arxiv.org/api/query?search_query=test")
        assert result is None

    @pytest.mark.asyncio()
    async def test_sends_correct_user_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The request should include the correct User-Agent header."""
        captured_kwargs: dict[str, Any] = {}

        async def mock_get(url: str, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.text = "<feed/>"
            resp.raise_for_status = MagicMock()
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        await make_arxiv_request("https://export.arxiv.org/api/query?search_query=test")
        assert captured_kwargs["headers"]["User-Agent"] == USER_AGENT


# ---------------------------------------------------------------------------
# search_arxiv (async, end-to-end with mocked network)
# ---------------------------------------------------------------------------


class TestSearchArxiv:
    """Integration tests for the search_arxiv MCP tool with mocked network."""

    @pytest.mark.asyncio()
    async def test_search_returns_results(
        self, sample_feed_xml: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid feed should produce formatted output with paper titles."""
        mock_response = MagicMock()
        mock_response.text = sample_feed_xml
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        output: str = await search_arxiv("transformer", max_results=10)
        assert "Attention Is All You Need" in output
        assert "Minimal Entry" in output
        assert "Found 2 papers" in output

    @pytest.mark.asyncio()
    async def test_search_empty_results(
        self, sample_empty_feed_xml: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty feed should produce a 'no papers found' message."""
        mock_response = MagicMock()
        mock_response.text = sample_empty_feed_xml
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        output = await search_arxiv("nonexistent_topic_xyz")
        assert "No papers found" in output

    @pytest.mark.asyncio()
    async def test_search_network_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A network failure should return an error message, not raise."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        output = await search_arxiv("anything")
        assert "Error" in output

    @pytest.mark.asyncio()
    async def test_search_caps_max_results(
        self, sample_feed_xml: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_results should be capped at 100 even if a larger value is passed."""
        captured_urls: list[str] = []

        async def mock_get(url: str, **kwargs: Any) -> MagicMock:
            captured_urls.append(url)
            resp = MagicMock()
            resp.text = sample_feed_xml
            resp.raise_for_status = MagicMock()
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        await search_arxiv("test", max_results=500)
        assert "max_results=100" in captured_urls[0]

    @pytest.mark.asyncio()
    async def test_search_invalid_sort_falls_back(
        self, sample_feed_xml: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An invalid sort_by value should fall back to 'relevance'."""
        captured_urls: list[str] = []

        async def mock_get(url: str, **kwargs: Any) -> MagicMock:
            captured_urls.append(url)
            resp = MagicMock()
            resp.text = sample_feed_xml
            resp.raise_for_status = MagicMock()
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        await search_arxiv("test", sort_by="bogus_sort")
        assert "sortBy=relevance" in captured_urls[0]
