"""Shared fixtures for google-scholar-api tests."""

from __future__ import annotations

from typing import Any

import pytest

from scholar.search import Author, AuthorResult, CitationResult, Paper, ScholarResult


@pytest.fixture()
def sample_paper() -> Paper:
    """Return a fully populated Paper dataclass instance."""
    return Paper(
        title="Attention Is All You Need",
        authors="A Vaswani, N Shazeer, N Parmar",
        venue="Advances in Neural Information Processing Systems",
        year="2017",
        snippet="We propose a new simple network architecture, the Transformer...",
        citations=90000,
        url="https://proceedings.neurips.cc/paper/2017/hash/attention",
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
    )


@pytest.fixture()
def sample_paper_minimal() -> Paper:
    """Return a Paper with only required fields and empty optional fields."""
    return Paper(
        title="Minimal Paper",
        authors="J Doe",
        venue="Unknown",
        year="Unknown",
        snippet="",
        citations=0,
        url="",
    )


@pytest.fixture()
def sample_author() -> Author:
    """Return a fully populated Author dataclass instance."""
    return Author(
        name="Geoffrey Hinton",
        author_id="JicYPdAAAAAJ",
        affiliation="University of Toronto",
        email_domain="utoronto.ca",
        citations=500000,
        interests=["neural networks", "deep learning", "machine learning"],
    )


@pytest.fixture()
def sample_scholar_result(sample_paper: Paper) -> ScholarResult:
    """Return a ScholarResult with one paper."""
    return ScholarResult(
        query="transformer architecture",
        total_results=1,
        papers=[sample_paper],
    )


@pytest.fixture()
def sample_scholar_result_with_error() -> ScholarResult:
    """Return a ScholarResult representing an error."""
    return ScholarResult(
        query="bad query",
        total_results=0,
        error="API rate limit exceeded",
    )


@pytest.fixture()
def sample_author_result(sample_author: Author) -> AuthorResult:
    """Return an AuthorResult from a name search."""
    return AuthorResult(
        query="Geoffrey Hinton",
        authors=[sample_author],
    )


@pytest.fixture()
def sample_author_result_with_profile(sample_author: Author) -> AuthorResult:
    """Return an AuthorResult from a profile lookup with h-index data."""
    return AuthorResult(
        query="JicYPdAAAAAJ",
        authors=[sample_author],
        h_index=180,
        i10_index=500,
        publications=[
            {"title": "Deep Learning", "year": "2015", "citations": 60000},
            {"title": "Backpropagation", "year": "1986", "citations": 40000},
        ],
    )


@pytest.fixture()
def sample_citation_result(sample_paper: Paper) -> CitationResult:
    """Return a CitationResult with one citing paper."""
    return CitationResult(
        citation_id="ABC123",
        total_citations=1,
        citing_papers=[sample_paper],
    )


@pytest.fixture()
def sample_serpapi_organic_result() -> dict[str, Any]:
    """Return a dict matching the structure of a SerpAPI organic_results item."""
    return {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "link": "https://arxiv.org/abs/1810.04805",
        "snippet": "We introduce a new language representation model...",
        "publication_info": {
            "summary": "J Devlin, MW Chang - arXiv preprint arXiv:1810.04805, 2018",
        },
        "inline_links": {
            "cited_by": {"total": 75000},
        },
        "resources": [
            {"link": "https://arxiv.org/pdf/1810.04805.pdf"},
        ],
    }
