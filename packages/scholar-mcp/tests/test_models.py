"""Tests for scholar data models and serialization.

Covers:
- Paper dataclass construction and serialization
- ScholarResult.to_dict() output structure
- CitationResult.to_dict() output structure
- AuthorResult.to_dict() output structure (with and without profile data)
- Author dataclass construction and defaults
- Edge cases: empty results, missing fields, error states
"""

from __future__ import annotations

import json

import pytest

from scholar.search import (
    Author,
    AuthorResult,
    CitationResult,
    Paper,
    ScholarResult,
    _parse_venue_year,
)

# ---------------------------------------------------------------------------
# Paper dataclass
# ---------------------------------------------------------------------------


class TestPaper:
    """Tests for the Paper dataclass."""

    def test_construction_with_all_fields(self) -> None:
        """Verify Paper creation with all fields populated."""
        paper = Paper(
            title="Test Paper",
            authors="A Smith, B Jones",
            venue="Nature",
            year="2023",
            snippet="This paper tests...",
            citations=42,
            url="https://example.com/paper",
            pdf_url="https://example.com/paper.pdf",
        )
        assert paper.title == "Test Paper"
        assert paper.authors == "A Smith, B Jones"
        assert paper.venue == "Nature"
        assert paper.year == "2023"
        assert paper.snippet == "This paper tests..."
        assert paper.citations == 42
        assert paper.url == "https://example.com/paper"
        assert paper.pdf_url == "https://example.com/paper.pdf"

    def test_pdf_url_defaults_to_empty_string(self) -> None:
        """When pdf_url is not provided, it should default to an empty string."""
        paper = Paper(
            title="No PDF",
            authors="Author",
            venue="Venue",
            year="2024",
            snippet="",
            citations=0,
            url="https://example.com",
        )
        assert paper.pdf_url == ""


# ---------------------------------------------------------------------------
# ScholarResult
# ---------------------------------------------------------------------------


class TestScholarResult:
    """Tests for the ScholarResult dataclass and its serialization."""

    def test_empty_result_to_dict(self) -> None:
        """An empty ScholarResult should serialize with zero papers."""
        result = ScholarResult(query="test", total_results=0)
        d = result.to_dict()
        assert d["query"] == "test"
        assert d["total_results"] == 0
        assert d["papers"] == []
        assert d["error"] is None

    def test_result_with_papers_to_dict(self) -> None:
        """A ScholarResult with papers should serialize all paper fields."""
        paper = Paper(
            title="Critical Minerals Review",
            authors="A Researcher",
            venue="Science",
            year="2023",
            snippet="A review of critical...",
            citations=100,
            url="https://example.com",
            pdf_url="https://example.com/pdf",
        )
        result = ScholarResult(query="critical minerals", total_results=1, papers=[paper])
        d = result.to_dict()
        assert len(d["papers"]) == 1
        assert d["papers"][0]["title"] == "Critical Minerals Review"
        assert d["papers"][0]["pdf_url"] == "https://example.com/pdf"

    def test_error_result_to_dict(self) -> None:
        """A ScholarResult with an error should include the error message."""
        result = ScholarResult(query="test", total_results=0, error="API key not found")
        d = result.to_dict()
        assert d["error"] == "API key not found"
        assert d["total_results"] == 0

    def test_to_dict_is_json_serializable(self) -> None:
        """The output of to_dict should be JSON serializable."""
        result = ScholarResult(
            query="test",
            total_results=1,
            papers=[
                Paper(
                    title="Test",
                    authors="Auth",
                    venue="V",
                    year="2024",
                    snippet="S",
                    citations=0,
                    url="http://example.com",
                )
            ],
        )
        json_str = json.dumps(result.to_dict())
        parsed = json.loads(json_str)
        assert parsed["query"] == "test"

    def test_papers_default_to_empty_list(self) -> None:
        """The papers field should default to an empty list."""
        result = ScholarResult(query="q", total_results=0)
        assert result.papers == []


# ---------------------------------------------------------------------------
# CitationResult
# ---------------------------------------------------------------------------


class TestCitationResult:
    """Tests for the CitationResult dataclass and its serialization."""

    def test_empty_citation_result(self) -> None:
        """An empty CitationResult should serialize correctly."""
        result = CitationResult(citation_id="CIT123", total_citations=0)
        d = result.to_dict()
        assert d["citation_id"] == "CIT123"
        assert d["total_citations"] == 0
        assert d["citing_papers"] == []
        assert d["error"] is None

    def test_citation_result_with_papers(self) -> None:
        """A CitationResult with citing papers should serialize all fields."""
        paper = Paper(
            title="Citing Paper",
            authors="B Author",
            venue="ICML",
            year="2024",
            snippet="This work builds on...",
            citations=5,
            url="https://example.com/citing",
        )
        result = CitationResult(
            citation_id="CIT456",
            total_citations=1,
            citing_papers=[paper],
        )
        d = result.to_dict()
        assert len(d["citing_papers"]) == 1
        assert d["citing_papers"][0]["title"] == "Citing Paper"
        # Note: citing_papers serialization does not include pdf_url or citations
        assert "url" in d["citing_papers"][0]

    def test_error_citation_result(self) -> None:
        """A CitationResult with an error should include the error message."""
        result = CitationResult(citation_id="BAD", total_citations=0, error="Invalid citation ID")
        d = result.to_dict()
        assert d["error"] == "Invalid citation ID"

    def test_to_dict_is_json_serializable(self) -> None:
        """The output of to_dict should be JSON serializable."""
        result = CitationResult(citation_id="CIT", total_citations=0)
        json_str = json.dumps(result.to_dict())
        parsed = json.loads(json_str)
        assert parsed["citation_id"] == "CIT"


# ---------------------------------------------------------------------------
# Author and AuthorResult
# ---------------------------------------------------------------------------


class TestAuthor:
    """Tests for the Author dataclass."""

    def test_construction(self) -> None:
        """Verify Author creation with all fields."""
        author = Author(
            name="Jane Doe",
            author_id="ABC123",
            affiliation="MIT",
            email_domain="mit.edu",
            citations=5000,
            interests=["materials science", "energy"],
        )
        assert author.name == "Jane Doe"
        assert author.author_id == "ABC123"
        assert author.citations == 5000
        assert len(author.interests) == 2

    def test_interests_default_to_empty_list(self) -> None:
        """The interests field should default to an empty list."""
        author = Author(
            name="Test",
            author_id="X",
            affiliation="U",
            email_domain="",
            citations=0,
        )
        assert author.interests == []


class TestAuthorResult:
    """Tests for the AuthorResult dataclass and its serialization."""

    def test_empty_author_result(self) -> None:
        """An empty AuthorResult should serialize correctly."""
        result = AuthorResult(query="Nobody")
        d = result.to_dict()
        assert d["query"] == "Nobody"
        assert d["authors"] == []
        assert d["error"] is None

    def test_author_result_with_authors(self) -> None:
        """An AuthorResult with authors should serialize author details."""
        author = Author(
            name="Jane Doe",
            author_id="ABC123",
            affiliation="MIT",
            email_domain="mit.edu",
            citations=5000,
            interests=["AI", "ML"],
        )
        result = AuthorResult(query="Jane Doe", authors=[author])
        d = result.to_dict()
        assert len(d["authors"]) == 1
        assert d["authors"][0]["name"] == "Jane Doe"
        assert d["authors"][0]["interests"] == ["AI", "ML"]

    def test_profile_result_includes_h_index(self) -> None:
        """When h_index is set, it should appear in the serialized output."""
        result = AuthorResult(
            query="ID123",
            h_index=42,
            i10_index=100,
            publications=[{"title": "Paper 1", "year": "2023", "citations": 10}],
        )
        d = result.to_dict()
        assert d["h_index"] == 42
        assert d["i10_index"] == 100
        assert len(d["publications"]) == 1

    def test_non_profile_result_excludes_h_index(self) -> None:
        """When h_index is None, it should not appear in the serialized output."""
        result = AuthorResult(query="search")
        d = result.to_dict()
        assert "h_index" not in d

    def test_error_author_result(self) -> None:
        """An AuthorResult with an error should include the error message."""
        result = AuthorResult(query="test", error="API failure")
        d = result.to_dict()
        assert d["error"] == "API failure"

    def test_to_dict_is_json_serializable(self) -> None:
        """The output of to_dict should be JSON serializable."""
        author = Author(
            name="Test",
            author_id="X",
            affiliation="U",
            email_domain="u.edu",
            citations=0,
        )
        result = AuthorResult(query="Test", authors=[author])
        json_str = json.dumps(result.to_dict())
        parsed = json.loads(json_str)
        assert parsed["query"] == "Test"


# ---------------------------------------------------------------------------
# _parse_venue_year helper
# ---------------------------------------------------------------------------


class TestParseVenueYear:
    """Tests for the _parse_venue_year helper function."""

    @pytest.mark.parametrize(
        ("summary", "expected_venue", "expected_year"),
        [
            (
                "A Smith, B Jones - Nature, 2023",
                "Nature",
                "2023",
            ),
            (
                "C Author - arXiv preprint arXiv:2301.07041, 2023",
                "arXiv preprint arXiv:2301.07041",
                "2023",
            ),
            (
                "D Researcher - Proceedings of NeurIPS, 2022",
                "Proceedings of NeurIPS",
                "2022",
            ),
            (
                "",
                "Unknown",
                "Unknown",
            ),
        ],
    )
    def test_parses_venue_and_year(
        self,
        summary: str,
        expected_venue: str,
        expected_year: str,
    ) -> None:
        """Verify venue and year are correctly extracted from publication summaries."""
        venue, year = _parse_venue_year(summary)
        assert venue == expected_venue
        assert year == expected_year

    def test_no_year_in_summary(self) -> None:
        """When no year is present, year should be 'Unknown'."""
        venue, year = _parse_venue_year("A Smith - Some Journal")
        assert year == "Unknown"
        assert venue != "Unknown"

    def test_old_year_format(self) -> None:
        """Years from the 1900s should also be parsed correctly."""
        _venue, year = _parse_venue_year("Author - Journal of Physics, 1999")
        assert year == "1999"
