"""MCP Server for OSTI (Office of Scientific and Technical Information).

Provides access to DOE technical reports and publications on critical minerals
and materials science.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .client import OSTIClient

# Initialize MCP server
mcp = FastMCP(
    "OSTI",
    instructions="""OSTI (Office of Scientific and Technical Information) document server.

Provides access to DOE technical reports and publications related to:
- Critical minerals (rare earths, lithium, cobalt, nickel, copper, graphite, gallium, germanium)
- Materials science and processing
- Extraction chemistry and metallurgy
- Geological occurrence and resources
- Supply chain analysis

Use these tools to search research literature, find technical reports, and discover
publications relevant to critical minerals and materials research.""",
)

# Initialize client
client = OSTIClient()


# ============================================================================
# Overview Tools
# ============================================================================


@mcp.tool()
async def get_osti_overview() -> dict:
    """Get overview of the OSTI document collection.

    Returns statistics including:
    - Total document count
    - Documents by commodity category
    - Documents by product type
    - Publication year range
    """
    return client.get_statistics()


@mcp.tool()
async def list_commodities() -> dict[str, str]:
    """List available commodity categories.

    Returns dictionary mapping commodity codes to full names:
    - HREE: Heavy Rare Earth Elements
    - LREE: Light Rare Earth Elements
    - CO: Cobalt
    - LI: Lithium
    - GA: Gallium
    - GR: Graphite
    - NI: Nickel
    - CU: Copper
    - GE: Germanium
    - OTH: Other Critical Materials
    """
    return client.list_commodities()


# ============================================================================
# Search Tools
# ============================================================================


@mcp.tool()
async def search_osti_documents(
    query: str | None = None,
    commodity: str | None = None,
    product_type: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 50,
) -> dict:
    """Search OSTI documents with filters.

    Args:
        query: Text search in title and description (e.g., "rare earth extraction")
        commodity: Commodity category code (HREE, LREE, CO, LI, GA, GR, NI, CU, GE, OTH)
        product_type: Filter by type ("Technical Report" or "Journal Article")
        year_from: Minimum publication year (e.g., 2020)
        year_to: Maximum publication year (e.g., 2024)
        limit: Maximum results to return (default 50)

    Returns dict with count and list of matching documents including:
    - osti_id, title, authors, publication_date
    - description, subjects, doi
    - commodity_category, product_type
    - research_orgs, sponsor_orgs
    """
    documents = client.search_documents(
        query=query,
        commodity=commodity,
        product_type=product_type,
        year_from=year_from,
        year_to=year_to,
        limit=limit,
    )
    return {
        "count": len(documents),
        "documents": [doc.model_dump() for doc in documents],
    }


@mcp.tool()
async def get_osti_document(osti_id: str) -> dict | None:
    """Get detailed information about a specific OSTI document.

    Args:
        osti_id: The OSTI document identifier (e.g., "2342032")

    Returns full document metadata or None if not found.
    """
    doc = client.get_document(osti_id)
    return doc.model_dump() if doc else None


# ============================================================================
# Browse Tools
# ============================================================================


@mcp.tool()
async def get_documents_by_commodity(
    commodity: str,
    limit: int = 50,
) -> dict:
    """Get documents for a specific commodity category.

    Args:
        commodity: Commodity code (HREE, LREE, CO, LI, GA, GR, NI, CU, GE, OTH)
        limit: Maximum results to return

    Returns dict with count and list of documents for the specified commodity.
    Useful for exploring research on specific critical materials.
    """
    documents = client.get_documents_by_commodity(commodity=commodity, limit=limit)
    commodity_name = client.COMMODITIES.get(commodity.upper(), commodity)
    return {
        "commodity": commodity.upper(),
        "commodity_name": commodity_name,
        "count": len(documents),
        "documents": [doc.model_dump() for doc in documents],
    }


@mcp.tool()
async def get_recent_documents(limit: int = 20) -> dict:
    """Get most recently published OSTI documents.

    Args:
        limit: Maximum results to return (default 20)

    Returns dict with list of documents sorted by publication date (newest first).
    Useful for staying current with recent research.
    """
    documents = client.get_recent_documents(limit=limit)
    return {
        "count": len(documents),
        "documents": [doc.model_dump() for doc in documents],
    }


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
