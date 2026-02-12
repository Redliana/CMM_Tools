"""Tests for osti_mcp.client module.

Tests cover the OSTIClient class including constructor configuration,
list_commodities(), OSTIDocument Pydantic model construction, get_statistics()
with mock data, search_documents() filtering logic, get_document() lookup,
get_documents_by_commodity(), get_recent_documents(), and a smoke test for
the server module import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osti_mcp.client import OSTIClient, OSTIDocument

# ---------------------------------------------------------------------------
# OSTIDocument model tests
# ---------------------------------------------------------------------------


class TestOSTIDocument:
    """Tests for the OSTIDocument Pydantic model."""

    def test_construction_with_all_fields(self, sample_osti_document: OSTIDocument) -> None:
        """Verify OSTIDocument creation with every field populated."""
        doc = sample_osti_document
        assert doc.osti_id == "2342032"
        assert doc.title == "Recovery of Rare Earth Elements from Coal Fly Ash"
        assert len(doc.authors) == 2
        assert doc.publication_date == "2023-06-15"
        assert doc.description is not None
        assert len(doc.subjects) == 3
        assert doc.commodity_category == "HREE"
        assert doc.doi == "10.2172/2342032"
        assert doc.product_type == "Technical Report"
        assert len(doc.research_orgs) == 1
        assert len(doc.sponsor_orgs) == 1

    def test_construction_with_required_fields_only(self) -> None:
        """Verify defaults when only required fields are provided."""
        doc = OSTIDocument(
            osti_id="9999999",
            title="Minimal Document",
        )
        assert doc.osti_id == "9999999"
        assert doc.title == "Minimal Document"
        assert doc.authors == []
        assert doc.publication_date is None
        assert doc.description is None
        assert doc.subjects == []
        assert doc.commodity_category is None
        assert doc.doi is None
        assert doc.product_type is None
        assert doc.research_orgs == []
        assert doc.sponsor_orgs == []

    def test_model_dump_roundtrip(self, sample_osti_document: OSTIDocument) -> None:
        """Verify dict serialization/deserialization roundtrip."""
        data = sample_osti_document.model_dump()
        restored = OSTIDocument(**data)
        assert restored == sample_osti_document

    def test_model_dump_contains_all_keys(self, sample_osti_document: OSTIDocument) -> None:
        """Verify model_dump includes all expected keys."""
        data = sample_osti_document.model_dump()
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
        assert set(data.keys()) == expected_keys

    def test_empty_list_defaults(self) -> None:
        """Verify list fields default to empty lists, not None."""
        doc = OSTIDocument(osti_id="0001", title="Test")
        assert isinstance(doc.authors, list)
        assert isinstance(doc.subjects, list)
        assert isinstance(doc.research_orgs, list)
        assert isinstance(doc.sponsor_orgs, list)


# ---------------------------------------------------------------------------
# OSTIClient constructor tests
# ---------------------------------------------------------------------------


class TestOSTIClientInit:
    """Tests for OSTIClient construction and configuration."""

    def test_init_with_explicit_path(self, tmp_catalog_dir: Path) -> None:
        """Verify data_path is set when provided explicitly."""
        client = OSTIClient(data_path=str(tmp_catalog_dir))
        assert client.data_path == tmp_catalog_dir

    def test_init_from_environment_variable(
        self,
        tmp_catalog_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify data_path is read from OSTI_DATA_PATH env variable."""
        monkeypatch.setenv("OSTI_DATA_PATH", str(tmp_catalog_dir))
        client = OSTIClient()
        assert client.data_path == tmp_catalog_dir

    def test_init_default_path_without_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify a default path is computed when no env or arg is given."""
        monkeypatch.delenv("OSTI_DATA_PATH", raising=False)
        client = OSTIClient()
        # Should use the relative path computation
        assert isinstance(client.data_path, Path)
        assert "Globus_Sharing" in str(client.data_path)

    def test_catalog_lazy_loading(self, client: OSTIClient) -> None:
        """Verify the catalog is not loaded until first access."""
        assert client._catalog is None
        # Access the catalog property
        _ = client.catalog
        assert client._catalog is not None

    def test_catalog_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError when catalog JSON is missing."""
        client = OSTIClient(data_path=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Document catalog not found"):
            _ = client.catalog


# ---------------------------------------------------------------------------
# list_commodities tests
# ---------------------------------------------------------------------------


class TestListCommodities:
    """Tests for the list_commodities method."""

    def test_returns_dict(self, client: OSTIClient) -> None:
        """Verify list_commodities returns a dict of code to name mappings."""
        result = client.list_commodities()
        assert isinstance(result, dict)

    def test_returns_copy(self, client: OSTIClient) -> None:
        """Verify the returned dict is a copy, not the class attribute."""
        result = client.list_commodities()
        result["FAKE"] = "Fake Mineral"
        assert "FAKE" not in OSTIClient.COMMODITIES

    def test_contains_expected_commodities(self, client: OSTIClient) -> None:
        """Verify that key commodity codes are present."""
        result = client.list_commodities()
        expected_codes = ["HREE", "LREE", "CO", "LI", "GA", "GR", "NI", "CU", "GE", "OTH"]
        for code in expected_codes:
            assert code in result, f"Commodity code '{code}' not found"

    def test_commodity_values_are_strings(self, client: OSTIClient) -> None:
        """Verify all commodity descriptions are strings."""
        result = client.list_commodities()
        assert all(isinstance(v, str) for v in result.values())

    @pytest.mark.parametrize(
        ("code", "expected_name"),
        [
            ("HREE", "Heavy Rare Earth Elements"),
            ("LREE", "Light Rare Earth Elements"),
            ("CO", "Cobalt"),
            ("LI", "Lithium"),
            ("GA", "Gallium"),
            ("GR", "Graphite"),
            ("NI", "Nickel"),
            ("CU", "Copper"),
            ("GE", "Germanium"),
            ("OTH", "Other Critical Materials"),
        ],
    )
    def test_commodity_name_matches(
        self, client: OSTIClient, code: str, expected_name: str
    ) -> None:
        """Verify each commodity code maps to the correct full name."""
        result = client.list_commodities()
        assert result[code] == expected_name


# ---------------------------------------------------------------------------
# get_statistics tests
# ---------------------------------------------------------------------------


class TestGetStatistics:
    """Tests for the get_statistics method with test catalog data."""

    def test_returns_dict(self, client: OSTIClient) -> None:
        """Verify get_statistics returns a dict."""
        stats = client.get_statistics()
        assert isinstance(stats, dict)

    def test_total_documents(self, client: OSTIClient) -> None:
        """Verify the total document count matches the fixture data."""
        stats = client.get_statistics()
        assert stats["total_documents"] == 6

    def test_commodity_counts(self, client: OSTIClient) -> None:
        """Verify commodity category counts are populated."""
        stats = client.get_statistics()
        commodities = stats["commodities"]
        assert "HREE" in commodities
        assert commodities["HREE"]["count"] == 1
        assert commodities["HREE"]["name"] == "Heavy Rare Earth Elements"
        assert "LI" in commodities
        assert commodities["LI"]["count"] == 1

    def test_product_type_counts(self, client: OSTIClient) -> None:
        """Verify product type counts are populated."""
        stats = client.get_statistics()
        product_types = stats["product_types"]
        assert "Technical Report" in product_types
        assert product_types["Technical Report"] == 4
        assert "Journal Article" in product_types
        assert product_types["Journal Article"] == 2

    def test_year_range(self, client: OSTIClient) -> None:
        """Verify the year range is extracted from publication dates."""
        stats = client.get_statistics()
        year_range = stats["year_range"]
        assert "min" in year_range
        assert "max" in year_range
        assert year_range["min"] == 2020
        assert year_range["max"] == 2024

    def test_statistics_keys(self, client: OSTIClient) -> None:
        """Verify all expected top-level keys are present."""
        stats = client.get_statistics()
        expected_keys = {"total_documents", "commodities", "product_types", "year_range"}
        assert set(stats.keys()) == expected_keys


# ---------------------------------------------------------------------------
# search_documents tests
# ---------------------------------------------------------------------------


class TestSearchDocuments:
    """Tests for the search_documents method with test catalog data."""

    def test_returns_list_of_osti_documents(self, client: OSTIClient) -> None:
        """Verify search returns a list of OSTIDocument instances."""
        results = client.search_documents()
        assert isinstance(results, list)
        assert all(isinstance(doc, OSTIDocument) for doc in results)

    def test_no_filters_returns_all(self, client: OSTIClient) -> None:
        """Verify searching with no filters returns all documents."""
        results = client.search_documents(limit=100)
        assert len(results) == 6

    def test_text_query_filters_by_title(self, client: OSTIClient) -> None:
        """Verify text query matches against document titles."""
        results = client.search_documents(query="rare earth")
        assert len(results) >= 1
        assert any("Rare Earth" in doc.title for doc in results)

    def test_text_query_filters_by_description(self, client: OSTIClient) -> None:
        """Verify text query matches against document descriptions."""
        results = client.search_documents(query="geothermal")
        assert len(results) >= 1
        assert results[0].osti_id == "2342033"

    def test_text_query_case_insensitive(self, client: OSTIClient) -> None:
        """Verify text search is case-insensitive."""
        results_lower = client.search_documents(query="lithium")
        results_upper = client.search_documents(query="LITHIUM")
        assert len(results_lower) == len(results_upper)

    def test_commodity_filter(self, client: OSTIClient) -> None:
        """Verify filtering by commodity category code."""
        results = client.search_documents(commodity="CO")
        assert len(results) == 1
        assert results[0].commodity_category == "CO"
        assert results[0].osti_id == "2342034"

    def test_commodity_filter_case_insensitive(self, client: OSTIClient) -> None:
        """Verify commodity filter handles lowercase input."""
        results = client.search_documents(commodity="co")
        assert len(results) == 1
        assert results[0].commodity_category == "CO"

    def test_product_type_filter(self, client: OSTIClient) -> None:
        """Verify filtering by product type."""
        results = client.search_documents(product_type="Journal Article")
        assert len(results) == 2
        assert all(doc.product_type == "Journal Article" for doc in results)

    def test_product_type_filter_case_insensitive(self, client: OSTIClient) -> None:
        """Verify product type filter is case-insensitive."""
        results = client.search_documents(product_type="journal article")
        assert len(results) == 2

    def test_year_from_filter(self, client: OSTIClient) -> None:
        """Verify year_from excludes documents published before the cutoff."""
        results = client.search_documents(year_from=2023)
        assert all(
            doc.publication_date is not None and doc.publication_date >= "2023" for doc in results
        )
        assert len(results) >= 2  # 2023 and 2024 documents

    def test_year_to_filter(self, client: OSTIClient) -> None:
        """Verify year_to excludes documents published after the cutoff."""
        results = client.search_documents(year_to=2021)
        assert len(results) >= 1
        for doc in results:
            assert doc.publication_date is not None
            year = int(doc.publication_date[:4])
            assert year <= 2021

    def test_combined_filters(self, client: OSTIClient) -> None:
        """Verify combining commodity and year filters narrows results."""
        results = client.search_documents(
            commodity="HREE",
            year_from=2023,
        )
        assert len(results) == 1
        assert results[0].commodity_category == "HREE"

    def test_limit_parameter(self, client: OSTIClient) -> None:
        """Verify the limit parameter caps the number of results."""
        results = client.search_documents(limit=2)
        assert len(results) <= 2

    def test_no_matching_results(self, client: OSTIClient) -> None:
        """Verify empty list is returned when no documents match."""
        results = client.search_documents(query="completely unique nonexistent term xyz123")
        assert results == []

    def test_query_matches_subjects(self, client: OSTIClient) -> None:
        """Verify text query matches against subjects field."""
        results = client.search_documents(query="hydrometallurgy")
        assert len(results) >= 1
        assert results[0].osti_id == "2342036"


# ---------------------------------------------------------------------------
# get_document tests
# ---------------------------------------------------------------------------


class TestGetDocument:
    """Tests for the get_document method."""

    def test_found_document(self, client: OSTIClient) -> None:
        """Verify retrieving an existing document by OSTI ID."""
        doc = client.get_document("2342032")
        assert doc is not None
        assert doc.osti_id == "2342032"
        assert doc.title == "Recovery of Rare Earth Elements from Coal Fly Ash"

    def test_not_found_returns_none(self, client: OSTIClient) -> None:
        """Verify None is returned for a non-existent OSTI ID."""
        doc = client.get_document("9999999")
        assert doc is None

    def test_string_coercion(self, client: OSTIClient) -> None:
        """Verify OSTI ID is matched as a string regardless of input type."""
        doc = client.get_document("2342034")
        assert doc is not None
        assert doc.commodity_category == "CO"


# ---------------------------------------------------------------------------
# get_documents_by_commodity tests
# ---------------------------------------------------------------------------


class TestGetDocumentsByCommodity:
    """Tests for the get_documents_by_commodity method."""

    def test_returns_documents_for_commodity(self, client: OSTIClient) -> None:
        """Verify documents are returned for a valid commodity code."""
        docs = client.get_documents_by_commodity("LI")
        assert len(docs) == 1
        assert docs[0].commodity_category == "LI"

    def test_empty_for_unused_commodity(self, client: OSTIClient) -> None:
        """Verify empty list for a commodity with no documents in test data."""
        docs = client.get_documents_by_commodity("OTH")
        assert docs == []

    def test_limit_parameter(self, client: OSTIClient) -> None:
        """Verify the limit parameter is respected."""
        docs = client.get_documents_by_commodity("HREE", limit=1)
        assert len(docs) <= 1


# ---------------------------------------------------------------------------
# get_recent_documents tests
# ---------------------------------------------------------------------------


class TestGetRecentDocuments:
    """Tests for the get_recent_documents method."""

    def test_returns_sorted_by_date_descending(self, client: OSTIClient) -> None:
        """Verify documents are sorted by publication date newest first."""
        docs = client.get_recent_documents(limit=10)
        dates = [doc.publication_date for doc in docs if doc.publication_date]
        assert dates == sorted(dates, reverse=True)

    def test_limit_parameter(self, client: OSTIClient) -> None:
        """Verify the limit parameter caps results."""
        docs = client.get_recent_documents(limit=3)
        assert len(docs) <= 3

    def test_most_recent_first(self, client: OSTIClient) -> None:
        """Verify the first document has the most recent date."""
        docs = client.get_recent_documents(limit=1)
        assert len(docs) == 1
        # 2024-01-20 is the most recent in our fixture data
        assert docs[0].publication_date == "2024-01-20"
        assert docs[0].osti_id == "2342035"


# ---------------------------------------------------------------------------
# Server import smoke tests
# ---------------------------------------------------------------------------


class TestServerImport:
    """Smoke tests verifying the server module can be imported."""

    def test_server_module_imports(self) -> None:
        """Verify the osti_mcp.server module can be imported without error."""
        import osti_mcp.server

        assert hasattr(osti_mcp.server, "mcp")

    def test_server_has_mcp_instance(self) -> None:
        """Verify the server exposes a FastMCP instance."""
        from osti_mcp.server import mcp

        assert mcp is not None
        assert mcp.name == "OSTI"

    def test_server_has_main_function(self) -> None:
        """Verify the server has a main() entry point."""
        from osti_mcp.server import main

        assert callable(main)

    def test_client_module_imports(self) -> None:
        """Verify the osti_mcp.client module can be imported without error."""
        import osti_mcp.client

        assert hasattr(osti_mcp.client, "OSTIClient")
        assert hasattr(osti_mcp.client, "OSTIDocument")
