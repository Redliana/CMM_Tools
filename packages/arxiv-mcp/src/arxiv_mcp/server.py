"""
ArXiv MCP Server with LLM Integration

This MCP server provides tools for searching ArXiv papers and summarizing them
using commercial LLM APIs (OpenAI GPT, Anthropic Claude, etc.).

Author: MCP Tutorial
License: MIT
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# Configure logging to stderr (NEVER use print() for STDIO servers!)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Writes to stderr by default
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("arxiv-search")

# Constants
ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "arxiv-mcp-server/1.0"

# LLM API Configuration (read from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


async def make_arxiv_request(url: str) -> str | None:
    """
    Make a request to the ArXiv API with proper error handling.

    Args:
        url: The full URL to request from ArXiv API

    Returns:
        The XML response as a string, or None if the request failed
    """
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Making request to ArXiv API: {url}")
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            return None
        except (httpx.InvalidURL, httpx.StreamError, ValueError) as e:
            logger.error(f"Unexpected error occurred: {e}")
            return None


def parse_arxiv_entry(entry: ET.Element) -> dict[str, Any]:
    """
    Parse a single ArXiv entry from the Atom feed.

    Args:
        entry: An XML Element representing a single paper entry

    Returns:
        A dictionary containing parsed paper metadata
    """
    # Extract basic metadata
    title = entry.find("atom:title", ARXIV_NAMESPACE)
    title_text = title.text.strip().replace("\n", " ") if title is not None else "Unknown"

    # Extract ArXiv ID from the entry ID
    entry_id = entry.find("atom:id", ARXIV_NAMESPACE)
    arxiv_id = entry_id.text.split("/abs/")[-1] if entry_id is not None else "Unknown"

    # Extract authors
    authors = []
    for author in entry.findall("atom:author", ARXIV_NAMESPACE):
        name = author.find("atom:name", ARXIV_NAMESPACE)
        if name is not None:
            authors.append(name.text)

    # Extract summary (abstract)
    summary = entry.find("atom:summary", ARXIV_NAMESPACE)
    summary_text = summary.text.strip().replace("\n", " ") if summary is not None else ""

    # Extract published date
    published = entry.find("atom:published", ARXIV_NAMESPACE)
    published_date = published.text if published is not None else "Unknown"

    # Extract categories
    categories = []
    for category in entry.findall("atom:category", ARXIV_NAMESPACE):
        term = category.get("term")
        if term:
            categories.append(term)

    # Extract PDF link
    pdf_link = None
    for link in entry.findall("atom:link", ARXIV_NAMESPACE):
        if link.get("title") == "pdf":
            pdf_link = link.get("href")
            break

    return {
        "id": arxiv_id,
        "title": title_text,
        "authors": authors,
        "summary": summary_text,
        "published": published_date,
        "categories": categories,
        "pdf_url": pdf_link or f"http://arxiv.org/pdf/{arxiv_id}.pdf",
    }


def format_paper_result(paper: dict[str, Any]) -> str:
    """
    Format a paper dictionary into a readable string.

    Args:
        paper: Dictionary containing paper metadata

    Returns:
        Formatted string representation of the paper
    """
    authors_str = ", ".join(paper["authors"][:3])
    if len(paper["authors"]) > 3:
        authors_str += f" et al. ({len(paper['authors'])} total)"

    categories_str = ", ".join(paper["categories"][:3])

    return f"""
Title: {paper["title"]}
ArXiv ID: {paper["id"]}
Authors: {authors_str}
Published: {paper["published"]}
Categories: {categories_str}
PDF: {paper["pdf_url"]}
Abstract: {paper["summary"][:300]}...
"""


async def call_openai_api(prompt: str, model: str = "gpt-4") -> str | None:
    """
    Call OpenAI API to generate text completions.

    Args:
        prompt: The prompt to send to the LLM
        model: The model to use (default: gpt-4)

    Returns:
        The generated text, or None if the request failed
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set in environment variables")
        return None

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-type": "application/json"}

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful research assistant that summarizes academic papers clearly and concisely.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
    }

    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Calling OpenAI API with model: {model}")
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            logger.error(f"OpenAI API error: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            return None


async def call_anthropic_api(prompt: str, model: str = "claude-3-5-sonnet-20241022") -> str | None:
    """
    Call Anthropic Claude API to generate text completions.

    Args:
        prompt: The prompt to send to Claude
        model: The model to use (default: claude-3-5-sonnet)

    Returns:
        The generated text, or None if the request failed
    """
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set in environment variables")
        return None

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-type": "application/json",
    }

    payload = {
        "model": model,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Calling Anthropic API with model: {model}")
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
        except httpx.HTTPError as e:
            logger.error(f"Anthropic API error: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Unexpected error calling Anthropic: {e}")
            return None


@mcp.tool()
async def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance") -> str:
    """
    Search ArXiv for papers matching a query.

    This tool searches the ArXiv repository for academic papers matching the provided
    query string. It returns paper metadata including title, authors, abstract, and PDF links.

    Args:
        query: Search query string (e.g., "machine learning", "quantum computing")
               Supports field prefixes like "ti:" (title), "au:" (author), "abs:" (abstract)
               Example: "ti:transformer AND au:vaswani"
        max_results: Maximum number of results to return (default: 10, max: 100)
        sort_by: Sort order - "relevance", "lastUpdatedDate", or "submittedDate" (default: "relevance")

    Returns:
        A formatted string containing paper metadata for all matching papers
    """
    # Validate inputs
    max_results = min(max_results, 100)  # Cap at 100 results

    valid_sort_options = ["relevance", "lastUpdatedDate", "submittedDate"]
    if sort_by not in valid_sort_options:
        sort_by = "relevance"

    # Construct the ArXiv API URL
    params = {
        "search_query": f"all:{query}" if ":" not in query else query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    # Build URL with query parameters
    param_str = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{ARXIV_API_BASE}?{param_str}"

    # Make the request
    xml_response = await make_arxiv_request(url)
    if not xml_response:
        return "Error: Failed to fetch results from ArXiv API. Please try again later."

    # Parse the XML response
    try:
        root = ET.fromstring(xml_response)
        entries = root.findall("atom:entry", ARXIV_NAMESPACE)

        if not entries:
            return f"No papers found matching query: '{query}'"

        # Parse each entry
        papers = [parse_arxiv_entry(entry) for entry in entries]

        # Format results
        result_lines = [f"Found {len(papers)} papers matching '{query}':\n"]
        result_lines.append("=" * 80)

        for i, paper in enumerate(papers, 1):
            result_lines.append(f"\n{i}. {format_paper_result(paper)}")
            result_lines.append("-" * 80)

        return "\n".join(result_lines)

    except ET.ParseError as e:
        logger.error(f"XML parsing error: {e}")
        return "Error: Failed to parse ArXiv API response."
    except (ValueError, KeyError, AttributeError) as e:
        logger.error(f"Unexpected error processing results: {e}")
        return "Error: An unexpected error occurred while processing results."


@mcp.tool()
async def get_arxiv_paper(arxiv_id: str) -> str:
    """
    Get detailed information about a specific ArXiv paper by its ID.

    Args:
        arxiv_id: The ArXiv ID of the paper (e.g., "2301.07041" or "cs.AI/0001001")

    Returns:
        Detailed information about the paper including full abstract
    """
    # Clean the ArXiv ID (remove version if present)
    clean_id = arxiv_id.split("v")[0]

    # Construct URL to fetch specific paper
    url = f"{ARXIV_API_BASE}?id_list={clean_id}"

    # Make the request
    xml_response = await make_arxiv_request(url)
    if not xml_response:
        return f"Error: Failed to fetch paper {arxiv_id} from ArXiv API."

    # Parse the response
    try:
        root = ET.fromstring(xml_response)
        entries = root.findall("atom:entry", ARXIV_NAMESPACE)

        if not entries:
            return f"Paper not found: {arxiv_id}"

        paper = parse_arxiv_entry(entries[0])

        # Format detailed result
        authors_str = ", ".join(paper["authors"])
        categories_str = ", ".join(paper["categories"])

        return f"""
Title: {paper["title"]}

ArXiv ID: {paper["id"]}

Authors: {authors_str}

Published: {paper["published"]}

Categories: {categories_str}

PDF URL: {paper["pdf_url"]}

Abstract:
{paper["summary"]}
"""

    except (ET.ParseError, ValueError, KeyError, AttributeError) as e:
        logger.error(f"Error fetching paper {arxiv_id}: {e}")
        return f"Error: Failed to retrieve paper {arxiv_id}."


@mcp.tool()
async def summarize_paper_with_llm(
    arxiv_id: str, llm_provider: str = "openai", model: str = "auto"
) -> str:
    """
    Fetch an ArXiv paper and generate a summary using a commercial LLM.

    This tool retrieves a paper from ArXiv and uses a commercial LLM API
    (OpenAI GPT or Anthropic Claude) to generate a concise summary of the paper's
    key contributions, methodology, and findings.

    Args:
        arxiv_id: The ArXiv ID of the paper to summarize (e.g., "2301.07041")
        llm_provider: Which LLM to use - "openai" or "anthropic" (default: "openai")
        model: Specific model to use, or "auto" for default (default: "auto")
               OpenAI: "gpt-4", "gpt-3.5-turbo"
               Anthropic: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"

    Returns:
        A concise summary of the paper generated by the LLM
    """
    # First, fetch the paper details
    paper_info = await get_arxiv_paper(arxiv_id)

    if paper_info.startswith("Error:") or paper_info.startswith("Paper not found:"):
        return paper_info

    # Construct the prompt for the LLM
    prompt = f"""Please provide a concise summary of the following research paper. 
Focus on:
1. The main research question or problem
2. The key methodology or approach
3. The main findings or contributions
4. The significance of the work

Paper details:
{paper_info}

Please provide a clear, structured summary in 3-4 paragraphs."""

    # Call the appropriate LLM API
    summary = None

    if llm_provider.lower() == "openai":
        if model == "auto":
            model = "gpt-4"
        summary = await call_openai_api(prompt, model)
    elif llm_provider.lower() == "anthropic":
        if model == "auto":
            model = "claude-3-5-sonnet-20241022"
        summary = await call_anthropic_api(prompt, model)
    else:
        return f"Error: Unknown LLM provider '{llm_provider}'. Use 'openai' or 'anthropic'."

    if not summary:
        return (
            f"Error: Failed to generate summary using {llm_provider}. Check API key and try again."
        )

    return f"""
Summary of ArXiv Paper: {arxiv_id}
Generated using: {llm_provider} ({model})

{summary}

---
Original paper: http://arxiv.org/abs/{arxiv_id}
"""


@mcp.tool()
async def search_and_summarize(
    query: str, max_papers: int = 3, llm_provider: str = "openai"
) -> str:
    """
    Search ArXiv and automatically summarize the top papers using an LLM.

    This is a convenience tool that combines search_arxiv and summarize_paper_with_llm
    to find relevant papers and generate summaries in one step.

    Args:
        query: Search query string
        max_papers: Number of top papers to summarize (default: 3, max: 5)
        llm_provider: Which LLM to use - "openai" or "anthropic" (default: "openai")

    Returns:
        Summaries of the top matching papers
    """
    # Cap at 5 papers to avoid excessive API calls
    max_papers = min(max_papers, 5)

    # First, search for papers
    search_results = await search_arxiv(query, max_results=max_papers)

    if search_results.startswith("Error:") or search_results.startswith("No papers found"):
        return search_results

    # Extract ArXiv IDs from search results (simple parsing)
    import re

    arxiv_ids = re.findall(r"ArXiv ID: ([^\s]+)", search_results)

    if not arxiv_ids:
        return "Error: Could not extract paper IDs from search results."

    # Summarize each paper
    summaries = [
        f"Search Query: '{query}'\nGenerating summaries for top {len(arxiv_ids)} papers...\n"
    ]
    summaries.append("=" * 80)

    for i, arxiv_id in enumerate(arxiv_ids, 1):
        logger.info(f"Summarizing paper {i}/{len(arxiv_ids)}: {arxiv_id}")
        summary = await summarize_paper_with_llm(arxiv_id, llm_provider)
        summaries.append(f"\n{i}. {summary}")
        summaries.append("=" * 80)

    return "\n".join(summaries)


def main():
    """
    Main entry point for the MCP server.

    This initializes and runs the MCP server using STDIO transport.
    The server will listen for JSON-RPC messages from MCP clients.
    """
    logger.info("Starting ArXiv MCP Server...")
    logger.info(f"OpenAI API Key configured: {bool(OPENAI_API_KEY)}")
    logger.info(f"Anthropic API Key configured: {bool(ANTHROPIC_API_KEY)}")

    # Run the server with STDIO transport
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
