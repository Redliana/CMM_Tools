"""MCP server for FRED (Federal Reserve Economic Data)."""

from __future__ import annotations

import asyncio

import httpx
from mcp.server.fastmcp import FastMCP

from .client import FredClient
from .models import CMM_FRED_SERIES

mcp = FastMCP(
    name="FRED",
    instructions="""FRED MCP Server provides access to the Federal Reserve Economic
Data (FRED) time series API. Useful for macroeconomic context: commodity price
indices, industrial production, import/export price indices, exchange rates,
and freight-cost proxies relevant to critical-minerals supply chains.

Key capabilities:
- Fetch observations for any FRED series by ID
- Search series catalogue by keyword
- Retrieve series metadata (units, frequency, revision dates)
- Browse categorical taxonomy
- Curated CMM series bundle for rapid profiling

Requires FRED_API_KEY. Register at https://fred.stlouisfed.org/docs/api/api_key.html.""",
)


def get_client() -> FredClient:
    """Return a fresh FredClient instance."""
    return FredClient()


# =============================================================================
# Overview Tools
# =============================================================================


@mcp.tool()
async def get_api_status() -> dict:
    """Check FRED API connectivity and API key validity."""
    client = get_client()
    return await client.check_status()


@mcp.tool()
async def list_cmm_series() -> dict:
    """List the curated CMM-relevant FRED series (commodity prices, production, trade prices, FX, rates)."""
    series = [
        {"id": series_id, "title": meta["title"], "category": meta["category"]}
        for series_id, meta in CMM_FRED_SERIES.items()
    ]
    categories: dict[str, int] = {}
    for s in series:
        categories[s["category"]] = categories.get(s["category"], 0) + 1
    return {
        "count": len(series),
        "series": series,
        "by_category": categories,
        "usage": "Use get_observations(series_id='PCOPPUSDM', observation_start='2020-01-01')",
    }


# =============================================================================
# Search and Metadata
# =============================================================================


@mcp.tool()
async def search_series(query: str, limit: int = 25, order_by: str = "popularity") -> dict:
    """Search the FRED series catalogue by keyword.

    Args:
        query: Search text (e.g., "copper price", "lithium", "industrial production").
        limit: Max results (1-1000).
        order_by: Sort order ("popularity", "search_rank", "last_updated", "frequency").

    Returns:
        List of matching series with IDs, titles, units, and temporal coverage.
    """
    client = get_client()
    results = await client.search_series(query, limit=limit, order_by=order_by)
    return {"query": query, "count": len(results), "series": results}


@mcp.tool()
async def get_series_metadata(series_id: str) -> dict:
    """Fetch metadata for a single FRED series.

    Args:
        series_id: FRED series identifier (e.g., "GDPC1", "PCOPPUSDM").
    """
    client = get_client()
    meta = await client.get_series_metadata(series_id)
    if meta is None:
        return {"error": f"Series not found: {series_id}"}
    return meta.model_dump()


# =============================================================================
# Observations
# =============================================================================


@mcp.tool()
async def get_observations(
    series_id: str,
    observation_start: str | None = None,
    observation_end: str | None = None,
    frequency: str | None = None,
    aggregation_method: str | None = None,
    max_records: int = 1000,
) -> dict:
    """Fetch observations for a FRED series.

    Args:
        series_id: FRED series ID.
        observation_start: Start date (YYYY-MM-DD); omit for earliest.
        observation_end: End date (YYYY-MM-DD); omit for latest.
        frequency: Optional aggregation frequency: d, w, bw, m, q, sa, a.
        aggregation_method: avg | sum | eop.
        max_records: Maximum observations to return.

    Returns:
        Dict with query, count, and observations list.
    """
    client = get_client()
    records = await client.get_observations(
        series_id=series_id,
        observation_start=observation_start,
        observation_end=observation_end,
        frequency=frequency,
        aggregation_method=aggregation_method,
        limit=max_records,
    )
    return {
        "query": {
            "series_id": series_id,
            "observation_start": observation_start,
            "observation_end": observation_end,
            "frequency": frequency,
        },
        "count": len(records),
        "observations": [r.model_dump() for r in records],
    }


@mcp.tool()
async def get_cmm_dashboard(
    observation_start: str = "2015-01-01",
    observation_end: str | None = None,
    category: str | None = None,
) -> dict:
    """Fetch the curated CMM dashboard: all curated series, most recent observation per series.

    Args:
        observation_start: Start date for backfill (default 2015-01-01).
        observation_end: Optional end date.
        category: Optional category filter (commodity_prices, production, trade_prices,
                  logistics, fx, rates).

    Returns:
        Dict keyed by series_id with latest observation, units, frequency, and timestamp.
    """
    client = get_client()
    target_series = (
        {k: v for k, v in CMM_FRED_SERIES.items() if v["category"] == category}
        if category
        else CMM_FRED_SERIES
    )

    # Rate-limit concurrent FRED calls; pairs obs + metadata fetches per series.
    sem = asyncio.Semaphore(5)

    async def fetch_one(series_id: str, meta: dict) -> tuple[str, dict]:
        async with sem:
            try:
                records, metadata = await asyncio.gather(
                    client.get_observations(
                        series_id=series_id,
                        observation_start=observation_start,
                        observation_end=observation_end,
                        limit=5000,
                    ),
                    client.get_series_metadata(series_id),
                )
            except (httpx.HTTPError, OSError, ValueError) as e:
                return series_id, {"error": str(e), "title": meta["title"]}
            latest = None
            for r in reversed(records):
                if r.value is not None:
                    latest = {"date": r.date, "value": r.value}
                    break
            return series_id, {
                "title": meta["title"],
                "category": meta["category"],
                "latest_observation": latest,
                "units": metadata.units if metadata else None,
                "frequency": metadata.frequency if metadata else None,
                "observation_count": len(records),
            }

    results = await asyncio.gather(*(fetch_one(sid, meta) for sid, meta in target_series.items()))
    dashboard: dict[str, dict] = dict(results)

    return {
        "observation_start": observation_start,
        "observation_end": observation_end,
        "category_filter": category,
        "series_count": len(dashboard),
        "dashboard": dashboard,
    }


# =============================================================================
# Taxonomy
# =============================================================================


@mcp.tool()
async def list_categories(category_id: int = 0) -> dict:
    """List children of a FRED category (0 = root)."""
    client = get_client()
    categories = await client.list_categories(category_id)
    return {"parent_id": category_id, "count": len(categories), "categories": categories}


@mcp.tool()
async def list_category_series(category_id: int, limit: int = 100) -> dict:
    """List series belonging to a specific category."""
    client = get_client()
    series = await client.list_category_series(category_id, limit=limit)
    return {"category_id": category_id, "count": len(series), "series": series}


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
