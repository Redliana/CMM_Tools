"""Tests for claimm_mcp.edx_client module."""

from __future__ import annotations

from typing import Any

from claimm_mcp.edx_client import EDXClient, Resource, SearchResult, Submission


class TestResourceModel:
    """Tests for the Resource Pydantic model."""

    def test_resource_minimal_fields(self) -> None:
        """Resource should be constructable with only id and name."""
        r = Resource(id="abc-123", name="test.csv")
        assert r.id == "abc-123"
        assert r.name == "test.csv"
        assert r.format is None
        assert r.size is None

    def test_resource_all_fields(self) -> None:
        """Resource should accept all optional fields."""
        r = Resource(
            id="abc-123",
            name="test.csv",
            description="A test CSV",
            format="CSV",
            size=2048,
            url="https://example.com/test.csv",
            created="2024-01-01",
            last_modified="2024-06-01",
            package_id="pkg-456",
        )
        assert r.format == "CSV"
        assert r.size == 2048
        assert r.package_id == "pkg-456"


class TestSearchResultModel:
    """Tests for the SearchResult Pydantic model."""

    def test_empty_search_result(self) -> None:
        """SearchResult should work with zero results."""
        sr = SearchResult(count=0, resources=[])
        assert sr.count == 0
        assert sr.resources == []

    def test_search_result_with_resources(self) -> None:
        """SearchResult should hold a list of Resource objects."""
        resources = [
            Resource(id="r1", name="file1.csv"),
            Resource(id="r2", name="file2.json"),
        ]
        sr = SearchResult(count=2, resources=resources)
        assert sr.count == 2
        assert len(sr.resources) == 2
        assert sr.resources[0].name == "file1.csv"


class TestSubmissionModel:
    """Tests for the Submission Pydantic model."""

    def test_submission_minimal(self) -> None:
        """Submission should be constructable with only id and name."""
        s = Submission(id="sub-1", name="test-submission")
        assert s.id == "sub-1"
        assert s.tags == []
        assert s.resources == []

    def test_submission_with_resources_and_tags(self) -> None:
        """Submission should hold resources and tags."""
        s = Submission(
            id="sub-1",
            name="test-submission",
            title="Test Submission",
            tags=["lithium", "critical-minerals"],
            resources=[Resource(id="r1", name="data.csv")],
        )
        assert len(s.tags) == 2
        assert len(s.resources) == 1
        assert s.title == "Test Submission"


class TestEDXClientConstruction:
    """Tests for EDXClient instantiation."""

    def test_client_uses_settings(self, settings_with_reset: Any) -> None:
        """EDXClient should read base_url and headers from Settings."""
        client = EDXClient()
        assert client.base_url == "https://edx.netl.doe.gov/api/3/action"
        assert "X-CKAN-API-Key" in client.headers
        assert client.headers["X-CKAN-API-Key"] == settings_with_reset.edx_api_key


class TestGetDownloadUrl:
    """Tests for EDXClient.get_download_url()."""

    def test_download_url_format(self, settings_with_reset: Any) -> None:
        """Download URL should follow the EDX pattern."""
        client = EDXClient()
        url = client.get_download_url("my-resource-id")
        assert url == "https://edx.netl.doe.gov/resource/my-resource-id/download"

    def test_download_url_with_special_chars(self, settings_with_reset: Any) -> None:
        """Download URL should handle resource IDs with special characters."""
        client = EDXClient()
        url = client.get_download_url("abc-def-123-456")
        assert "abc-def-123-456" in url
