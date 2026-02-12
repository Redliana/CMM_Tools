"""Tests for scholar.search module.

Covers:
- Dataclass construction (Paper, ScholarResult, AuthorResult, CitationResult)
- to_dict() serialization for all result types
- _parse_venue_year() with various input formats
- Module import smoke test
"""

from __future__ import annotations

from typing import Any

import pytest

from scholar.search import (
    AuthorResult,
    CitationResult,
    Paper,
    ScholarResult,
    _parse_venue_year,
)

# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestModuleImport:
    """Verify the module and its public symbols are importable."""

    def test_import_search_module(self) -> None:
        """Importing scholar.search should succeed without errors."""
        import scholar.search  # noqa: F401

    def test_import_package_init(self) -> None:
        """Importing the scholar package should succeed and expose key symbols."""
        import scholar

        assert hasattr(scholar, "ScholarResult")
        assert hasattr(scholar, "search_scholar")
        assert hasattr(scholar, "__version__")


# ---------------------------------------------------------------------------
# Paper dataclass
# ---------------------------------------------------------------------------


class TestPaperDataclass:
    """Tests for the Paper dataclass."""

    def test_construction_all_fields(self, sample_paper: Paper) -> None:
        """A Paper with all fields should store them correctly."""
        assert sample_paper.title == "Attention Is All You Need"
        assert sample_paper.citations == 90000
        assert sample_paper.pdf_url == "https://arxiv.org/pdf/1706.03762.pdf"

    def test_construction_default_pdf_url(self) -> None:
        """The pdf_url field should default to an empty string."""
        paper = Paper(
            title="Test",
            authors="A Author",
            venue="Test Venue",
            year="2023",
            snippet="Snippet",
            citations=0,
            url="https://example.com",
        )
        assert paper.pdf_url == ""

    def test_construction_minimal(self, sample_paper_minimal: Paper) -> None:
        """A minimal Paper should have empty defaults."""
        assert sample_paper_minimal.pdf_url == ""
        assert sample_paper_minimal.snippet == ""
        assert sample_paper_minimal.citations == 0


# ---------------------------------------------------------------------------
# ScholarResult
# ---------------------------------------------------------------------------


class TestScholarResult:
    """Tests for the ScholarResult dataclass and to_dict()."""

    def test_construction_with_papers(self, sample_scholar_result: ScholarResult) -> None:
        """ScholarResult should store query, total_results, and papers."""
        assert sample_scholar_result.query == "transformer architecture"
        assert sample_scholar_result.total_results == 1
        assert len(sample_scholar_result.papers) == 1
        assert sample_scholar_result.error is None

    def test_construction_with_error(self, sample_scholar_result_with_error: ScholarResult) -> None:
        """ScholarResult with an error should store the error string."""
        assert sample_scholar_result_with_error.error == "API rate limit exceeded"
        assert sample_scholar_result_with_error.total_results == 0
        assert sample_scholar_result_with_error.papers == []

    def test_to_dict_structure(self, sample_scholar_result: ScholarResult) -> None:
        """to_dict() should return a dict with the expected top-level keys."""
        d: dict[str, Any] = sample_scholar_result.to_dict()
        assert set(d.keys()) == {"query", "total_results", "papers", "error"}

    def test_to_dict_papers_serialization(self, sample_scholar_result: ScholarResult) -> None:
        """Each paper in to_dict() should be a dict with standard keys."""
        d = sample_scholar_result.to_dict()
        paper_dict = d["papers"][0]
        expected_keys = {
            "title",
            "authors",
            "venue",
            "year",
            "snippet",
            "citations",
            "url",
            "pdf_url",
        }
        assert set(paper_dict.keys()) == expected_keys
        assert paper_dict["title"] == "Attention Is All You Need"

    def test_to_dict_empty_papers(self) -> None:
        """to_dict() with no papers should return an empty list."""
        result = ScholarResult(query="empty", total_results=0)
        d = result.to_dict()
        assert d["papers"] == []
        assert d["error"] is None

    def test_to_dict_error_field(self, sample_scholar_result_with_error: ScholarResult) -> None:
        """to_dict() should include the error string when present."""
        d = sample_scholar_result_with_error.to_dict()
        assert d["error"] == "API rate limit exceeded"


# ---------------------------------------------------------------------------
# AuthorResult
# ---------------------------------------------------------------------------


class TestAuthorResult:
    """Tests for the AuthorResult dataclass and to_dict()."""

    def test_construction_search(self, sample_author_result: AuthorResult) -> None:
        """AuthorResult from a search should have authors but no h_index."""
        assert sample_author_result.query == "Geoffrey Hinton"
        assert len(sample_author_result.authors) == 1
        assert sample_author_result.h_index is None
        assert sample_author_result.error is None

    def test_construction_profile(self, sample_author_result_with_profile: AuthorResult) -> None:
        """AuthorResult from a profile lookup should include h_index and publications."""
        r = sample_author_result_with_profile
        assert r.h_index == 180
        assert r.i10_index == 500
        assert len(r.publications) == 2

    def test_to_dict_search_structure(self, sample_author_result: AuthorResult) -> None:
        """to_dict() for a search should have query, authors, and error keys."""
        d = sample_author_result.to_dict()
        assert "query" in d
        assert "authors" in d
        assert "error" in d
        # h_index should NOT be present for search-only results
        assert "h_index" not in d

    def test_to_dict_profile_structure(
        self, sample_author_result_with_profile: AuthorResult
    ) -> None:
        """to_dict() for a profile lookup should include h_index and publications."""
        d = sample_author_result_with_profile.to_dict()
        assert d["h_index"] == 180
        assert d["i10_index"] == 500
        assert len(d["publications"]) == 2

    def test_to_dict_author_fields(self, sample_author_result: AuthorResult) -> None:
        """Each author dict should have standard keys."""
        d = sample_author_result.to_dict()
        author = d["authors"][0]
        expected_keys = {
            "name",
            "author_id",
            "affiliation",
            "email_domain",
            "citations",
            "interests",
        }
        assert set(author.keys()) == expected_keys
        assert author["name"] == "Geoffrey Hinton"


# ---------------------------------------------------------------------------
# CitationResult
# ---------------------------------------------------------------------------


class TestCitationResult:
    """Tests for the CitationResult dataclass and to_dict()."""

    def test_construction(self, sample_citation_result: CitationResult) -> None:
        """CitationResult should store citation_id and citing_papers."""
        assert sample_citation_result.citation_id == "ABC123"
        assert sample_citation_result.total_citations == 1
        assert len(sample_citation_result.citing_papers) == 1
        assert sample_citation_result.error is None

    def test_to_dict_structure(self, sample_citation_result: CitationResult) -> None:
        """to_dict() should have citation_id, total_citations, citing_papers, error."""
        d = sample_citation_result.to_dict()
        assert set(d.keys()) == {"citation_id", "total_citations", "citing_papers", "error"}

    def test_to_dict_paper_fields(self, sample_citation_result: CitationResult) -> None:
        """Citing papers in to_dict() should have expected keys (no pdf_url)."""
        d = sample_citation_result.to_dict()
        paper = d["citing_papers"][0]
        # CitationResult.to_dict() omits pdf_url for citing papers
        expected_keys = {"title", "authors", "venue", "year", "snippet", "url"}
        assert set(paper.keys()) == expected_keys

    def test_construction_with_error(self) -> None:
        """CitationResult with an error should store it."""
        r = CitationResult(citation_id="BAD", total_citations=0, error="Not found")
        assert r.error == "Not found"
        d = r.to_dict()
        assert d["error"] == "Not found"
        assert d["citing_papers"] == []


# ---------------------------------------------------------------------------
# _parse_venue_year
# ---------------------------------------------------------------------------


class TestParseVenueYear:
    """Tests for the _parse_venue_year() helper."""

    def test_standard_format(self) -> None:
        """A standard 'Authors - Venue, Year' string should be parsed correctly."""
        summary = "J Devlin, MW Chang - arXiv preprint arXiv:1810.04805, 2018"
        venue, year = _parse_venue_year(summary)
        assert year == "2018"
        assert "arXiv" in venue

    def test_journal_with_year(self) -> None:
        """A journal-style summary should extract venue and year."""
        summary = "A Smith - Nature, 2021"
        venue, year = _parse_venue_year(summary)
        assert year == "2021"
        assert "Nature" in venue

    def test_conference_format(self) -> None:
        """A conference proceedings summary should extract the venue."""
        summary = "B Jones, C Lee - Proceedings of NeurIPS, 2023"
        venue, year = _parse_venue_year(summary)
        assert year == "2023"
        assert "NeurIPS" in venue

    def test_empty_string(self) -> None:
        """An empty string should return 'Unknown' for both venue and year."""
        venue, year = _parse_venue_year("")
        assert venue == "Unknown"
        assert year == "Unknown"

    def test_no_separator(self) -> None:
        """A string without ' - ' separator should return defaults."""
        venue, year = _parse_venue_year("just some text without separator")
        assert venue == "Unknown"
        assert year == "Unknown"

    def test_year_only(self) -> None:
        """A string with a year but no comma should still extract the year."""
        summary = "Author - Venue 2020"
        _venue, year = _parse_venue_year(summary)
        assert year == "2020"

    def test_old_year(self) -> None:
        """Years starting with 19xx should be recognized."""
        summary = "Author - Vintage Journal, 1998"
        _venue, year = _parse_venue_year(summary)
        assert year == "1998"

    @pytest.mark.parametrize(
        ("summary", "expected_year"),
        [
            ("A - B, 2000", "2000"),
            ("A - B, 2099", "2099"),
            ("A - B, 1900", "1900"),
            ("A - B, 1899", "Unknown"),  # 1899 does not match (19|20)XX
        ],
        ids=["year-2000", "year-2099", "year-1900", "year-1899-no-match"],
    )
    def test_year_boundary_values(self, summary: str, expected_year: str) -> None:
        """Boundary years should be handled according to the regex (19|20)XX."""
        _, year = _parse_venue_year(summary)
        assert year == expected_year
