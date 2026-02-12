"""Tests for scholar_mcp.server module.

Covers:
- Module import smoke test
- SERVER_INSTRUCTIONS content validation
- MCP tool wrapper functions return valid JSON strings
- Wrapper functions delegate to the underlying scholar library
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

# Note: scholar_mcp.server imports from 'scholar' directly (not scholar.search),
# and it relies on having the scholar package importable. We mock the SerpAPI
# calls so no real network or API key is needed.


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestModuleImport:
    """Verify the module and its key symbols are importable."""

    def test_import_server_module(self) -> None:
        """Importing scholar_mcp.server should succeed without errors."""
        import scholar_mcp.server  # noqa: F401

    def test_import_package_init(self) -> None:
        """Importing the scholar_mcp package should succeed."""
        import scholar_mcp  # noqa: F401

    def test_mcp_instance_exists(self) -> None:
        """The module should expose a FastMCP instance named 'mcp'."""
        from scholar_mcp.server import mcp

        assert mcp is not None
        assert hasattr(mcp, "run")

    def test_server_instructions_defined(self) -> None:
        """SERVER_INSTRUCTIONS should be a non-empty string."""
        from scholar_mcp.server import SERVER_INSTRUCTIONS

        assert isinstance(SERVER_INSTRUCTIONS, str)
        assert len(SERVER_INSTRUCTIONS) > 0

    def test_server_instructions_mentions_key_topics(self) -> None:
        """SERVER_INSTRUCTIONS should mention important usage guidance."""
        from scholar_mcp.server import SERVER_INSTRUCTIONS

        assert "Google Scholar" in SERVER_INSTRUCTIONS
        assert "arXiv" in SERVER_INSTRUCTIONS or "arxiv" in SERVER_INSTRUCTIONS
        assert "preprint" in SERVER_INSTRUCTIONS.lower()


# ---------------------------------------------------------------------------
# search_scholar wrapper
# ---------------------------------------------------------------------------


class TestSearchScholarWrapper:
    """Tests for the search_scholar() MCP tool wrapper."""

    @patch("scholar_mcp.server._search_scholar")
    def test_returns_json_string(self, mock_search: MagicMock) -> None:
        """search_scholar() should return a valid JSON string."""
        from scholar.search import ScholarResult

        mock_search.return_value = ScholarResult(query="test", total_results=0)

        from scholar_mcp.server import search_scholar

        result: str = search_scholar(query="test")
        parsed: dict[str, Any] = json.loads(result)
        assert parsed["query"] == "test"
        assert parsed["total_results"] == 0

    @patch("scholar_mcp.server._search_scholar")
    def test_passes_all_arguments(self, mock_search: MagicMock) -> None:
        """search_scholar() should forward all parameters to the underlying function."""
        from scholar.search import ScholarResult

        mock_search.return_value = ScholarResult(query="minerals", total_results=0)

        from scholar_mcp.server import search_scholar

        search_scholar(query="minerals", year_from=2020, year_to=2024, num_results=5)
        mock_search.assert_called_once_with(
            query="minerals",
            year_from=2020,
            year_to=2024,
            num_results=5,
        )

    @patch("scholar_mcp.server._search_scholar")
    def test_includes_papers_in_output(self, mock_search: MagicMock) -> None:
        """When papers are found, they should appear in the JSON output."""
        from scholar.search import Paper, ScholarResult

        mock_search.return_value = ScholarResult(
            query="lithium",
            total_results=1,
            papers=[
                Paper(
                    title="Lithium Mining",
                    authors="A Smith",
                    venue="Nature",
                    year="2023",
                    snippet="Study of lithium...",
                    citations=50,
                    url="https://example.com",
                )
            ],
        )

        from scholar_mcp.server import search_scholar

        result = search_scholar(query="lithium")
        parsed = json.loads(result)
        assert len(parsed["papers"]) == 1
        assert parsed["papers"][0]["title"] == "Lithium Mining"


# ---------------------------------------------------------------------------
# get_paper_citations wrapper
# ---------------------------------------------------------------------------


class TestGetPaperCitationsWrapper:
    """Tests for the get_paper_citations() MCP tool wrapper."""

    @patch("scholar_mcp.server._get_paper_citations")
    def test_returns_json_string(self, mock_citations: MagicMock) -> None:
        """get_paper_citations() should return a valid JSON string."""
        from scholar.search import CitationResult

        mock_citations.return_value = CitationResult(citation_id="CIT123", total_citations=0)

        from scholar_mcp.server import get_paper_citations

        result: str = get_paper_citations(citation_id="CIT123")
        parsed: dict[str, Any] = json.loads(result)
        assert parsed["citation_id"] == "CIT123"
        assert parsed["total_citations"] == 0

    @patch("scholar_mcp.server._get_paper_citations")
    def test_passes_arguments(self, mock_citations: MagicMock) -> None:
        """get_paper_citations() should forward parameters to the underlying function."""
        from scholar.search import CitationResult

        mock_citations.return_value = CitationResult(citation_id="CIT456", total_citations=0)

        from scholar_mcp.server import get_paper_citations

        get_paper_citations(citation_id="CIT456", num_results=15)
        mock_citations.assert_called_once_with(citation_id="CIT456", num_results=15)


# ---------------------------------------------------------------------------
# get_author_profile wrapper
# ---------------------------------------------------------------------------


class TestGetAuthorProfileWrapper:
    """Tests for the get_author_profile() MCP tool wrapper."""

    @patch("scholar_mcp.server._get_author_profile")
    def test_returns_json_string(self, mock_profile: MagicMock) -> None:
        """get_author_profile() should return a valid JSON string."""
        from scholar.search import AuthorResult

        mock_profile.return_value = AuthorResult(query="AID123")

        from scholar_mcp.server import get_author_profile

        result: str = get_author_profile(author_id="AID123")
        parsed: dict[str, Any] = json.loads(result)
        assert parsed["query"] == "AID123"

    @patch("scholar_mcp.server._get_author_profile")
    def test_passes_author_id(self, mock_profile: MagicMock) -> None:
        """get_author_profile() should forward author_id to the underlying function."""
        from scholar.search import AuthorResult

        mock_profile.return_value = AuthorResult(query="AID789")

        from scholar_mcp.server import get_author_profile

        get_author_profile(author_id="AID789")
        mock_profile.assert_called_once_with(author_id="AID789")


# ---------------------------------------------------------------------------
# search_author wrapper
# ---------------------------------------------------------------------------


class TestSearchAuthorWrapper:
    """Tests for the search_author() MCP tool wrapper."""

    @patch("scholar_mcp.server._search_author")
    def test_returns_json_string(self, mock_search_author: MagicMock) -> None:
        """search_author() should return a valid JSON string."""
        from scholar.search import AuthorResult

        mock_search_author.return_value = AuthorResult(query="Hinton")

        from scholar_mcp.server import search_author

        result: str = search_author(author_name="Hinton")
        parsed: dict[str, Any] = json.loads(result)
        assert parsed["query"] == "Hinton"

    @patch("scholar_mcp.server._search_author")
    def test_passes_author_name(self, mock_search_author: MagicMock) -> None:
        """search_author() should forward author_name to the underlying function."""
        from scholar.search import AuthorResult

        mock_search_author.return_value = AuthorResult(query="LeCun")

        from scholar_mcp.server import search_author

        search_author(author_name="LeCun")
        mock_search_author.assert_called_once_with(author_name="LeCun")


# ---------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() entry point."""

    def test_main_is_callable(self) -> None:
        """main() should be defined and callable."""
        from scholar_mcp.server import main

        assert callable(main)
