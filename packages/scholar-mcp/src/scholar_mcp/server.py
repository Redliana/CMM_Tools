"""
Google Scholar MCP Server

Provides tools for searching academic papers, getting citations,
and retrieving author profiles from Google Scholar via SerpAPI.

This server wraps the google-scholar-api library as an MCP tool server
for use with Claude Code and Claude Desktop.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

from scholar import (
    get_author_profile as _get_author_profile,
)
from scholar import (
    get_paper_citations as _get_paper_citations,
)
from scholar import (
    search_author as _search_author,
)
from scholar import (
    search_scholar as _search_scholar,
)
from scholar import (
    set_api_key,
)

# Configure logging to stderr (NEVER stdout - it corrupts JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("google-scholar-mcp")

# Initialize the MCP server with instructions
SERVER_INSTRUCTIONS = """
Google Scholar MCP Server - Academic Literature Search

IMPORTANT: Always search comprehensively across ALL publication types:
- Peer-reviewed journal articles
- Conference proceedings (NeurIPS, ICML, ACL, CVPR, etc.)
- Preprints (arXiv, bioRxiv, medRxiv, SSRN, etc.)
- Technical reports and working papers
- Theses and dissertations
- Books and book chapters

Search Tips for Comprehensive Results:
1. Use "source:arxiv" in query to specifically find arXiv preprints
2. Include conference names (e.g., "NeurIPS 2023") to find proceedings
3. Search without year filters first to get the broadest results
4. Recent preprints may have fewer citations but contain cutting-edge research
5. For emerging topics, prioritize recent preprints over older journal articles

When presenting results, always note the publication venue/source so users
can identify preprints vs peer-reviewed articles.
"""

mcp = FastMCP("google-scholar", instructions=SERVER_INSTRUCTIONS)

# Get API key from environment and configure the library
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
if SERPAPI_KEY:
    set_api_key(SERPAPI_KEY)


@mcp.tool()
def search_scholar(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    num_results: int = 10,
) -> str:
    """
    Search Google Scholar for academic literature across ALL publication types.

    This searches comprehensively across:
    - Peer-reviewed journal articles
    - Conference proceedings (NeurIPS, ICML, ACL, CVPR, AAAI, etc.)
    - Preprints (arXiv, bioRxiv, medRxiv, SSRN, etc.)
    - Technical reports and working papers
    - Theses and dissertations
    - Books and book chapters

    Search Tips:
    - Add "arxiv" or "source:arxiv" to query to find arXiv preprints
    - Add conference names like "NeurIPS" or "ICML" to find proceedings
    - For cutting-edge research, check recent preprints (may have low citations)
    - The venue/source in results indicates publication type

    Args:
        query: Search query. Examples:
            - "retrieval augmented generation" (general search)
            - "transformer architecture arxiv" (preprints)
            - "deep learning NeurIPS 2023" (conference proceedings)
        year_from: Filter papers published from this year (inclusive)
        year_to: Filter papers published until this year (inclusive)
        num_results: Maximum number of results to return (1-20, default 10)

    Returns:
        JSON string with search results including titles, authors, venue/source, and citations
    """
    logger.info(f"Searching Google Scholar for: {query}")
    result = _search_scholar(
        query=query,
        year_from=year_from,
        year_to=year_to,
        num_results=num_results,
    )
    logger.info(f"Found {result.total_results} results for query: {query}")
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def get_paper_citations(
    citation_id: str,
    num_results: int = 10,
) -> str:
    """
    Get papers that cite a given paper using its citation ID.

    Args:
        citation_id: The citation ID from a previous search result (e.g., "1234567890")
        num_results: Maximum number of citing papers to return (1-20, default 10)

    Returns:
        JSON string with list of papers that cite the given paper
    """
    logger.info(f"Getting citations for ID: {citation_id}")
    result = _get_paper_citations(citation_id=citation_id, num_results=num_results)
    logger.info(f"Found {result.total_citations} citing papers")
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def get_author_profile(author_id: str) -> str:
    """
    Get an author's profile from Google Scholar using their author ID.

    To find an author ID, search for papers by the author and look for
    author links in the results, or search Google Scholar directly.

    Args:
        author_id: Google Scholar author ID (e.g., "JicYPdAAAAAJ")

    Returns:
        JSON string with author profile including name, affiliation, citations, and publications
    """
    logger.info(f"Getting author profile for ID: {author_id}")
    result = _get_author_profile(author_id=author_id)
    if result.authors:
        logger.info(f"Found author: {result.authors[0].name}")
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def search_author(
    author_name: str,
) -> str:
    """
    Search for an author on Google Scholar to find their author ID and basic info.

    Args:
        author_name: Name of the author to search for (e.g., "Geoffrey Hinton")

    Returns:
        JSON string with matching authors and their IDs
    """
    logger.info(f"Searching for author: {author_name}")
    result = _search_author(author_name=author_name)
    logger.info(f"Found {len(result.authors)} matching authors")
    return json.dumps(result.to_dict(), indent=2)


def main():
    """Run the MCP server."""
    logger.info("Starting Google Scholar MCP Server (using google-scholar-api library)")
    if not SERPAPI_KEY:
        logger.warning("SERPAPI_KEY not set - tools will fail until configured")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
