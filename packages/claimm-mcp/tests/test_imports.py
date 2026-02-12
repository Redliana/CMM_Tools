"""Smoke tests verifying that all claimm-mcp modules are importable."""

from __future__ import annotations


class TestImports:
    """Verify that package modules can be imported without errors."""

    def test_import_package(self) -> None:
        """The top-level claimm_mcp package should be importable."""
        import claimm_mcp

        assert hasattr(claimm_mcp, "__version__")

    def test_import_config(self) -> None:
        """claimm_mcp.config should be importable."""
        from claimm_mcp.config import Settings, get_settings

        assert Settings is not None
        assert callable(get_settings)

    def test_import_header_detector(self) -> None:
        """claimm_mcp.header_detector should be importable."""
        from claimm_mcp.header_detector import HeaderDetector

        assert HeaderDetector is not None

    def test_import_edx_client(self) -> None:
        """claimm_mcp.edx_client should be importable."""
        from claimm_mcp.edx_client import EDXClient, Resource, SearchResult, Submission

        assert EDXClient is not None
        assert Resource is not None
        assert SearchResult is not None
        assert Submission is not None

    def test_import_llm_client(self) -> None:
        """claimm_mcp.llm_client should be importable."""
        from claimm_mcp.llm_client import LLMClient

        assert LLMClient is not None

    def test_import_server(self) -> None:
        """claimm_mcp.server should be importable."""
        from claimm_mcp import server

        assert hasattr(server, "mcp")
        assert hasattr(server, "main")
