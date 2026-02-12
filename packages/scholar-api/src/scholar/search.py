"""
Core Google Scholar search functions.

These functions can be called directly or used as tools with any LLM.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from serpapi import GoogleScholarSearch

# Load .env file from the package directory or current directory
_package_dir = Path(__file__).parent.parent
_env_file = _package_dir / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    load_dotenv()  # Try current directory

# Module-level API key
_api_key: str | None = None


def set_api_key(key: str) -> None:
    """Set the SerpAPI key for all subsequent requests."""
    global _api_key
    _api_key = key


def _get_api_key() -> str:
    """Get the API key from module state, .env file, or environment."""
    key = _api_key or os.environ.get("SERPAPI_KEY", "")
    if not key:
        raise ValueError(
            "SerpAPI key not set. Either:\n"
            "  1. Create a .env file with SERPAPI_KEY=your-key\n"
            "  2. Call set_api_key('your-key')\n"
            "  3. set SERPAPI_KEY environment variable\n"
            "Get a free key at https://serpapi.com"
        )
    return key


@dataclass
class Paper:
    """A single paper result."""

    title: str
    authors: str
    venue: str
    year: str
    snippet: str
    citations: int
    url: str
    pdf_url: str = ""


@dataclass
class ScholarResult:
    """Results from a Google Scholar search."""

    query: str
    total_results: int
    papers: list[Paper] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "total_results": self.total_results,
            "papers": [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "venue": p.venue,
                    "year": p.year,
                    "snippet": p.snippet,
                    "citations": p.citations,
                    "url": p.url,
                    "pdf_url": p.pdf_url,
                }
                for p in self.papers
            ],
            "error": self.error,
        }


@dataclass
class Author:
    """A single author result."""

    name: str
    author_id: str
    affiliation: str
    email_domain: str
    citations: int
    interests: list[str] = field(default_factory=list)


@dataclass
class AuthorResult:
    """Results from an author search or profile lookup."""

    query: str
    authors: list[Author] = field(default_factory=list)
    # For profile lookups
    h_index: int | None = None
    i10_index: int | None = None
    publications: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "query": self.query,
            "authors": [
                {
                    "name": a.name,
                    "author_id": a.author_id,
                    "affiliation": a.affiliation,
                    "email_domain": a.email_domain,
                    "citations": a.citations,
                    "interests": a.interests,
                }
                for a in self.authors
            ],
            "error": self.error,
        }
        if self.h_index is not None:
            result["h_index"] = self.h_index
            result["i10_index"] = self.i10_index
            result["publications"] = self.publications
        return result


@dataclass
class CitationResult:
    """Results from a citation lookup."""

    citation_id: str
    total_citations: int
    citing_papers: list[Paper] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "citation_id": self.citation_id,
            "total_citations": self.total_citations,
            "citing_papers": [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "venue": p.venue,
                    "year": p.year,
                    "snippet": p.snippet,
                    "url": p.url,
                }
                for p in self.citing_papers
            ],
            "error": self.error,
        }


def _parse_venue_year(summary: str) -> tuple[str, str]:
    """Parse venue and year from publication summary."""
    venue = "Unknown"
    year = "Unknown"
    if summary:
        parts = summary.split(" - ")
        if len(parts) > 1:
            venue_year = parts[-1]
            year_match = re.search(r"\b(19|20)\d{2}\b", venue_year)
            if year_match:
                year = year_match.group()
            venue = (
                venue_year.rsplit(",", 1)[0].strip() if "," in venue_year else venue_year.strip()
            )
    return venue, year


def search_scholar(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    num_results: int = 10,
) -> ScholarResult:
    """
    Search Google Scholar for academic papers.

    Searches comprehensively across all publication types:
    - Peer-reviewed journal articles
    - Conference proceedings (NeurIPS, ICML, ACL, CVPR, etc.)
    - Preprints (arXiv, bioRxiv, medRxiv, SSRN, etc.)
    - Technical reports, theses, and books

    Args:
        query: Search query. Tips:
            - Add "arxiv" to find preprints
            - Add conference names like "NeurIPS" for proceedings
        year_from: Filter papers from this year (inclusive)
        year_to: Filter papers until this year (inclusive)
        num_results: Maximum results to return (1-20)

    Returns:
        ScholarResult with list of papers
    """
    try:
        api_key = _get_api_key()
        num_results = max(1, min(num_results, 20))

        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": api_key,
            "num": num_results,
        }
        if year_from:
            params["as_ylo"] = year_from
        if year_to:
            params["as_yhi"] = year_to

        search = GoogleScholarSearch(params)
        results = search.get_dict()

        if "error" in results:
            return ScholarResult(query=query, total_results=0, error=results["error"])

        papers = []
        for result in results.get("organic_results", [])[:num_results]:
            pub_info = result.get("publication_info", {})
            summary = pub_info.get("summary", "")
            venue, year = _parse_venue_year(summary)

            papers.append(
                Paper(
                    title=result.get("title", "Unknown"),
                    authors=summary.split(" - ")[0] if " - " in summary else summary,
                    venue=venue,
                    year=year,
                    snippet=result.get("snippet", ""),
                    citations=result.get("inline_links", {}).get("cited_by", {}).get("total", 0),
                    url=result.get("link", ""),
                    pdf_url=result.get("resources", [{}])[0].get("link", "")
                    if result.get("resources")
                    else "",
                )
            )

        return ScholarResult(query=query, total_results=len(papers), papers=papers)

    except Exception as e:
        return ScholarResult(query=query, total_results=0, error=str(e))


def search_author(author_name: str) -> AuthorResult:
    """
    Search for an author on Google Scholar.

    Args:
        author_name: Name of the author (e.g., "Geoffrey Hinton")

    Returns:
        AuthorResult with matching authors and their IDs
    """
    try:
        api_key = _get_api_key()

        params = {
            "engine": "google_scholar_profiles",
            "mauthors": author_name,
            "api_key": api_key,
        }

        search = GoogleScholarSearch(params)
        results = search.get_dict()

        if "error" in results:
            return AuthorResult(query=author_name, error=results["error"])

        authors = []
        for profile in results.get("profiles", [])[:5]:
            authors.append(
                Author(
                    name=profile.get("name", "Unknown"),
                    author_id=profile.get("author_id", ""),
                    affiliation=profile.get("affiliations", "Unknown"),
                    email_domain=profile.get("email", ""),
                    citations=profile.get("cited_by", 0),
                    interests=[i.get("title", "") for i in profile.get("interests", [])],
                )
            )

        return AuthorResult(query=author_name, authors=authors)

    except Exception as e:
        return AuthorResult(query=author_name, error=str(e))


def get_author_profile(author_id: str) -> AuthorResult:
    """
    Get detailed author profile by Google Scholar author ID.

    Args:
        author_id: Google Scholar author ID (e.g., "JicYPdAAAAAJ")

    Returns:
        AuthorResult with author details, h-index, and publications
    """
    try:
        api_key = _get_api_key()

        params = {
            "engine": "google_scholar_author",
            "author_id": author_id,
            "api_key": api_key,
        }

        search = GoogleScholarSearch(params)
        results = search.get_dict()

        if "error" in results:
            return AuthorResult(query=author_id, error=results["error"])

        author_data = results.get("author", {})
        cited_by = results.get("cited_by", {})
        articles = results.get("articles", [])

        author = Author(
            name=author_data.get("name", "Unknown"),
            author_id=author_id,
            affiliation=author_data.get("affiliations", "Unknown"),
            email_domain=author_data.get("email", ""),
            citations=cited_by.get("table", [{}])[0].get("citations", {}).get("all", 0)
            if cited_by.get("table")
            else 0,
            interests=[i.get("title", "") for i in author_data.get("interests", [])],
        )

        publications = []
        for article in articles[:10]:
            publications.append(
                {
                    "title": article.get("title", "Unknown"),
                    "year": article.get("year", "Unknown"),
                    "citations": article.get("cited_by", {}).get("value", 0),
                }
            )

        h_index = (
            cited_by.get("table", [{}])[0].get("h_index", {}).get("all", 0)
            if cited_by.get("table")
            else 0
        )
        i10_index = (
            cited_by.get("table", [{}])[0].get("i10_index", {}).get("all", 0)
            if cited_by.get("table")
            else 0
        )

        return AuthorResult(
            query=author_id,
            authors=[author],
            h_index=h_index,
            i10_index=i10_index,
            publications=publications,
        )

    except Exception as e:
        return AuthorResult(query=author_id, error=str(e))


def get_paper_citations(citation_id: str, num_results: int = 10) -> CitationResult:
    """
    Get papers that cite a given paper.

    Args:
        citation_id: The citation ID from a search result
        num_results: Maximum citing papers to return (1-20)

    Returns:
        CitationResult with list of citing papers
    """
    try:
        api_key = _get_api_key()
        num_results = max(1, min(num_results, 20))

        params = {
            "engine": "google_scholar",
            "cites": citation_id,
            "api_key": api_key,
            "num": num_results,
        }

        search = GoogleScholarSearch(params)
        results = search.get_dict()

        if "error" in results:
            return CitationResult(
                citation_id=citation_id, total_citations=0, error=results["error"]
            )

        papers = []
        for result in results.get("organic_results", [])[:num_results]:
            pub_info = result.get("publication_info", {})
            summary = pub_info.get("summary", "")
            venue, year = _parse_venue_year(summary)

            papers.append(
                Paper(
                    title=result.get("title", "Unknown"),
                    authors=summary.split(" - ")[0] if " - " in summary else summary,
                    venue=venue,
                    year=year,
                    snippet=result.get("snippet", ""),
                    citations=0,
                    url=result.get("link", ""),
                )
            )

        return CitationResult(
            citation_id=citation_id,
            total_citations=len(papers),
            citing_papers=papers,
        )

    except Exception as e:
        return CitationResult(citation_id=citation_id, total_citations=0, error=str(e))
