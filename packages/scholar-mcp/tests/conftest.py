"""Shared fixtures for google-scholar-mcp tests."""

from __future__ import annotations

from typing import Any

import pytest

from scholar.search import AuthorResult, CitationResult, Paper, ScholarResult


@pytest.fixture()
def sample_scholar_result_dict() -> dict[str, Any]:
    """Return a ScholarResult.to_dict() output for testing JSON formatting."""
    return ScholarResult(
        query="critical minerals supply chain",
        total_results=1,
        papers=[
            Paper(
                title="Critical Minerals in the Energy Transition",
                authors="A Researcher, B Scientist",
                venue="Nature Energy",
                year="2023",
                snippet="This study examines critical mineral supply chains...",
                citations=150,
                url="https://example.com/paper",
                pdf_url="https://example.com/paper.pdf",
            )
        ],
    ).to_dict()


@pytest.fixture()
def sample_citation_result_dict() -> dict[str, Any]:
    """Return a CitationResult.to_dict() output for testing JSON formatting."""
    return CitationResult(
        citation_id="TEST123",
        total_citations=1,
        citing_papers=[
            Paper(
                title="Follow-up Study on Minerals",
                authors="C Author",
                venue="Science",
                year="2024",
                snippet="Building on prior work...",
                citations=10,
                url="https://example.com/citing",
            )
        ],
    ).to_dict()


@pytest.fixture()
def sample_author_result_dict() -> dict[str, Any]:
    """Return an AuthorResult.to_dict() output for testing JSON formatting."""
    from scholar.search import Author

    return AuthorResult(
        query="Jane Doe",
        authors=[
            Author(
                name="Jane Doe",
                author_id="ABCDEF123",
                affiliation="MIT",
                email_domain="mit.edu",
                citations=5000,
                interests=["materials science", "energy"],
            )
        ],
    ).to_dict()
