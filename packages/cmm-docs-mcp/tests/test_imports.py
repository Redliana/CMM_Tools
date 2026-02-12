"""Smoke tests verifying that all cmm-docs-mcp modules are importable."""

from __future__ import annotations


class TestImports:
    """Verify that package modules can be imported without errors."""

    def test_import_package(self) -> None:
        """The top-level cmm_docs package should be importable."""
        import cmm_docs

        assert cmm_docs is not None

    def test_import_config(self) -> None:
        """cmm_docs.config should be importable with expected constants."""
        from cmm_docs.config import (
            COMMODITIES,
            DATA_CATEGORIES,
            MAX_CSV_ROWS,
            MAX_PDF_CHARS,
            SUBDOMAINS,
        )

        assert isinstance(COMMODITIES, dict)
        assert isinstance(SUBDOMAINS, dict)
        assert isinstance(DATA_CATEGORIES, dict)
        assert isinstance(MAX_CSV_ROWS, int)
        assert isinstance(MAX_PDF_CHARS, int)

    def test_import_data_tools(self) -> None:
        """cmm_docs.data_tools should be importable."""
        from cmm_docs.data_tools import DataManager, get_data_manager

        assert DataManager is not None
        assert callable(get_data_manager)

    def test_import_document_tools(self) -> None:
        """cmm_docs.document_tools should be importable."""
        from cmm_docs.document_tools import DocumentManager, get_document_manager

        assert DocumentManager is not None
        assert callable(get_document_manager)
