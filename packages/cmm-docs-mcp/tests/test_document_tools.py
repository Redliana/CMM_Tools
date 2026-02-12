"""Tests for cmm_docs.document_tools module."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from cmm_docs.document_tools import DocumentManager


def _make_document_manager(
    catalog: list[dict[str, Any]],
) -> DocumentManager:
    """Create a DocumentManager with a given catalog, bypassing file I/O.

    Args:
        catalog: The catalog list to inject.

    Returns:
        A DocumentManager instance with the provided catalog.
    """
    with patch.object(DocumentManager, "_load_catalog", return_value=catalog):
        return DocumentManager()


class TestBuildIndex:
    """Tests for DocumentManager._build_index()."""

    def test_by_id_index(self, sample_catalog: list[dict[str, Any]]) -> None:
        """by_id should map osti_id to the corresponding document dict."""
        dm = _make_document_manager(sample_catalog)
        assert "3004920" in dm.by_id
        assert dm.by_id["3004920"]["title"] == "Lithium Recovery from Coal Combustion Residuals"

    def test_by_id_all_docs_indexed(self, sample_catalog: list[dict[str, Any]]) -> None:
        """All documents in the catalog should be indexed by osti_id."""
        dm = _make_document_manager(sample_catalog)
        assert len(dm.by_id) == len(sample_catalog)

    def test_by_commodity_index(self, sample_catalog: list[dict[str, Any]]) -> None:
        """by_commodity should group documents by their commodity_category."""
        dm = _make_document_manager(sample_catalog)
        assert "LI" in dm.by_commodity
        assert len(dm.by_commodity["LI"]) == 2
        assert "CO" in dm.by_commodity
        assert len(dm.by_commodity["CO"]) == 1
        assert "HREE" in dm.by_commodity
        assert len(dm.by_commodity["HREE"]) == 1

    def test_empty_catalog_builds_empty_indices(self) -> None:
        """An empty catalog should produce empty indices."""
        dm = _make_document_manager([])
        assert dm.by_id == {}
        assert dm.by_commodity == {}


class TestListDocuments:
    """Tests for DocumentManager.list_documents()."""

    def test_list_all_documents(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Without filter, all documents should be returned."""
        dm = _make_document_manager(sample_catalog)
        results = dm.list_documents()
        assert len(results) == 4

    def test_list_documents_by_commodity(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Filtering by commodity should return only matching documents."""
        dm = _make_document_manager(sample_catalog)
        results = dm.list_documents(commodity="LI")
        assert len(results) == 2
        assert all(r["commodity_category"] == "LI" for r in results)

    def test_list_documents_limit(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Limit parameter should cap the number of returned documents."""
        dm = _make_document_manager(sample_catalog)
        results = dm.list_documents(limit=2)
        assert len(results) == 2

    def test_list_documents_nonexistent_commodity(
        self, sample_catalog: list[dict[str, Any]]
    ) -> None:
        """Filtering by a non-existent commodity should return empty list."""
        dm = _make_document_manager(sample_catalog)
        results = dm.list_documents(commodity="NONEXISTENT")
        assert results == []

    def test_list_documents_result_keys(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Each result should contain expected summary fields."""
        dm = _make_document_manager(sample_catalog)
        results = dm.list_documents()
        for result in results:
            assert "osti_id" in result
            assert "title" in result
            assert "authors" in result
            assert "publication_date" in result
            assert "commodity_category" in result
            assert "product_type" in result

    def test_list_documents_authors_limited_to_three(
        self, sample_catalog: list[dict[str, Any]]
    ) -> None:
        """Authors should be limited to at most 3 entries."""
        dm = _make_document_manager(sample_catalog)
        results = dm.list_documents()
        for result in results:
            assert len(result["authors"]) <= 3

    def test_list_documents_commodity_case_insensitive(
        self, sample_catalog: list[dict[str, Any]]
    ) -> None:
        """Commodity filter should be converted to uppercase."""
        dm = _make_document_manager(sample_catalog)
        results = dm.list_documents(commodity="li")
        assert len(results) == 2


class TestGetMetadata:
    """Tests for DocumentManager.get_metadata()."""

    def test_get_existing_metadata(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Should return the full metadata dict for a known osti_id."""
        dm = _make_document_manager(sample_catalog)
        result = dm.get_metadata("3004920")
        assert result is not None
        assert result["title"] == "Lithium Recovery from Coal Combustion Residuals"
        assert result["doi"] == "10.1234/test.2024.001"

    def test_get_nonexistent_metadata(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Should return None for an unknown osti_id."""
        dm = _make_document_manager(sample_catalog)
        result = dm.get_metadata("9999999")
        assert result is None


class TestExportCitation:
    """Tests for DocumentManager.export_citation()."""

    def test_bibtex_journal_article(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Journal article citations should use @article entry type."""
        dm = _make_document_manager(sample_catalog)
        bibtex = dm.export_citation("3004920")
        assert "@article{osti_3004920" in bibtex
        assert "Smith, John and Doe, Jane and Johnson, Alice" in bibtex
        assert "2024" in bibtex
        assert "10.1234/test.2024.001" in bibtex
        assert "Environmental Science & Technology" in bibtex
        assert "https://www.osti.gov/biblio/3004920" in bibtex

    def test_bibtex_tech_report(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Technical reports should use @techreport entry type."""
        dm = _make_document_manager(sample_catalog)
        bibtex = dm.export_citation("3005100")
        assert "@techreport{osti_3005100" in bibtex
        assert "Brown, Bob" in bibtex
        assert "2023" in bibtex
        assert "Pacific Northwest National Laboratory" in bibtex

    def test_bibtex_unknown_document(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Citation for unknown osti_id should return an error string."""
        dm = _make_document_manager(sample_catalog)
        result = dm.export_citation("9999999")
        assert "Error" in result
        assert "9999999" in result

    def test_bibtex_includes_url(self, sample_catalog: list[dict[str, Any]]) -> None:
        """All BibTeX entries should include a url field pointing to OSTI."""
        dm = _make_document_manager(sample_catalog)
        for doc in sample_catalog:
            bibtex = dm.export_citation(doc["osti_id"])
            if "Error" not in bibtex:
                assert f"https://www.osti.gov/biblio/{doc['osti_id']}" in bibtex

    def test_bibtex_volume_field_for_journal(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Journal articles with volume info should include the volume field."""
        dm = _make_document_manager(sample_catalog)
        bibtex = dm.export_citation("3004920")
        assert "volume = {58}" in bibtex


class TestGetStatistics:
    """Tests for DocumentManager.get_statistics()."""

    def test_statistics_structure(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Statistics should have total count and breakdowns."""
        dm = _make_document_manager(sample_catalog)
        stats = dm.get_statistics()
        assert "total_documents" in stats
        assert "by_commodity" in stats
        assert "by_product_type" in stats

    def test_statistics_total_documents(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Total documents should match catalog length."""
        dm = _make_document_manager(sample_catalog)
        stats = dm.get_statistics()
        assert stats["total_documents"] == 4

    def test_statistics_by_commodity(self, sample_catalog: list[dict[str, Any]]) -> None:
        """by_commodity counts should match the catalog."""
        dm = _make_document_manager(sample_catalog)
        stats = dm.get_statistics()
        assert stats["by_commodity"]["LI"] == 2
        assert stats["by_commodity"]["CO"] == 1
        assert stats["by_commodity"]["HREE"] == 1

    def test_statistics_by_product_type(self, sample_catalog: list[dict[str, Any]]) -> None:
        """by_product_type counts should match the catalog."""
        dm = _make_document_manager(sample_catalog)
        stats = dm.get_statistics()
        assert stats["by_product_type"]["Journal Article"] == 2
        assert stats["by_product_type"]["Technical Report"] == 2


class TestSearchByCommodity:
    """Tests for DocumentManager.search_by_commodity()."""

    def test_search_known_commodity(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Searching a known commodity should return matching documents."""
        dm = _make_document_manager(sample_catalog)
        result = dm.search_by_commodity("LI")
        assert result["commodity_code"] == "LI"
        assert result["document_count"] == 2
        assert len(result["documents"]) == 2

    def test_search_commodity_case_insensitive(self, sample_catalog: list[dict[str, Any]]) -> None:
        """search_by_commodity should be case-insensitive."""
        dm = _make_document_manager(sample_catalog)
        result = dm.search_by_commodity("li")
        assert result["commodity_code"] == "LI"
        assert result["document_count"] == 2

    def test_search_unknown_commodity(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Unknown commodity should return zero documents."""
        dm = _make_document_manager(sample_catalog)
        result = dm.search_by_commodity("NONEXISTENT")
        assert result["document_count"] == 0
        assert result["documents"] == []

    def test_search_includes_description(self, sample_catalog: list[dict[str, Any]]) -> None:
        """Result should include the commodity description from config."""
        dm = _make_document_manager(sample_catalog)
        result = dm.search_by_commodity("LI")
        assert result["description"] == "Lithium"
