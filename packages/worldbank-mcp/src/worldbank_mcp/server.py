"""MCP server for World Bank WDI indicators."""

from __future__ import annotations

import asyncio

import httpx
from mcp.server.fastmcp import FastMCP

from .client import WorldBankClient
from .models import CMM_KEY_ECONOMIES, CMM_WDI_INDICATORS

mcp = FastMCP(
    name="World Bank",
    instructions="""World Bank MCP Server provides access to the World Development
Indicators (WDI) API. Useful for sovereign and macroeconomic context surrounding
critical-minerals supply chains, trade balances, manufacturing value-added,
natural-resource rents, and governance proxies.

Key capabilities:
- Fetch indicator observations for any country/year combination
- Search the indicator catalogue by keyword
- List countries with regional and income classification
- Pre-configured CMM-relevant indicator set for rapid profiling

Country codes use ISO3 (e.g., USA, CHN, DEU). Date ranges accept single years
(\"2022\") or colon-separated ranges (\"2000:2023\"). No API key is required.""",
)


def get_client() -> WorldBankClient:
    """Return a fresh WorldBankClient instance."""
    return WorldBankClient()


# =============================================================================
# Overview Tools
# =============================================================================


@mcp.tool()
async def get_api_status() -> dict:
    """Check World Bank WDI API connectivity.

    Returns a dict with `status`, `api_key_configured`, and diagnostic fields.
    """
    client = get_client()
    return await client.check_status()


@mcp.tool()
async def list_cmm_indicators() -> dict:
    """List the curated CMM-relevant WDI indicators.

    Returns the pre-configured indicators most useful for critical-minerals
    supply-chain, trade, and governance analysis.
    """
    return {
        "count": len(CMM_WDI_INDICATORS),
        "indicators": [{"code": code, "name": name} for code, name in CMM_WDI_INDICATORS.items()],
        "usage": "Use get_indicator(country='USA', indicator='NY.GDP.MKTP.CD', date='2000:2023')",
    }


@mcp.tool()
async def list_cmm_key_economies() -> dict:
    """List ISO3 codes for key economies in critical-minerals supply chains."""
    return {
        "count": len(CMM_KEY_ECONOMIES),
        "economies": [{"iso3": iso3, "name": name} for iso3, name in CMM_KEY_ECONOMIES.items()],
    }


@mcp.tool()
async def search_indicators(query: str, limit: int = 25) -> dict:
    """Search the WDI indicator catalogue by keyword.

    Args:
        query: Keyword (e.g., "manufacturing", "mineral rents", "renewable").
        limit: Maximum number of matches to return.

    Returns:
        List of matching indicator metadata.
    """
    client = get_client()
    matches = await client.search_indicators(query)
    return {
        "query": query,
        "count": len(matches[:limit]),
        "total_matches": len(matches),
        "indicators": matches[:limit],
    }


@mcp.tool()
async def list_countries(search: str | None = None, limit: int = 100) -> dict:
    """List World Bank countries with region and income classification.

    Args:
        search: Optional name fragment to filter.
        limit: Maximum results.
    """
    client = get_client()
    countries = await client.list_countries()
    # Remove aggregate regions (no ISO2) unless explicitly requested
    if search:
        q = search.lower()
        countries = [c for c in countries if q in (c.get("name") or "").lower()]
    return {
        "count": len(countries[:limit]),
        "total": len(countries),
        "countries": countries[:limit],
    }


# =============================================================================
# Data Query Tools
# =============================================================================


@mcp.tool()
async def get_indicator(
    country: str,
    indicator: str,
    date: str | None = None,
    max_records: int = 500,
) -> dict:
    """Fetch WDI indicator observations for a country or country group.

    Args:
        country: ISO3 code or semicolon-separated list (e.g., "USA;CHN;DEU").
                 Also accepts "all" or "WLD" for the world aggregate.
        indicator: WDI indicator code (e.g., "NY.GDP.MKTP.CD").
        date: Year ("2022") or range ("2000:2023"). Omit for all available years.
        max_records: Maximum records to return (paginated up to 1000 upstream).

    Returns:
        Dict with `query`, `count`, and `records` list.
    """
    client = get_client()
    records = await client.get_indicator_observations(
        country=country, indicator=indicator, date=date, per_page=min(max_records, 1000)
    )
    return {
        "query": {"country": country, "indicator": indicator, "date": date},
        "count": len(records),
        "records": [r.model_dump() for r in records],
    }


@mcp.tool()
async def get_cmm_profile(
    country: str,
    date: str = "2015:2023",
) -> dict:
    """Fetch the curated CMM indicator bundle for a single country.

    Args:
        country: ISO3 country code (e.g., "USA", "CHN", "CHL").
        date: Year or range (default: "2015:2023").

    Returns:
        Country profile keyed by indicator, each with a list of (year, value) pairs.
    """
    client = get_client()
    profile: dict[str, list[dict]] = {}

    for indicator_code, indicator_name in CMM_WDI_INDICATORS.items():
        try:
            await asyncio.sleep(0.1)  # Gentle pacing
            records = await client.get_indicator_observations(
                country=country, indicator=indicator_code, date=date, per_page=100
            )
            profile[indicator_code] = [
                {"year": r.year, "value": r.value, "name": indicator_name}
                for r in records
                if r.value is not None
            ]
        except (httpx.HTTPError, OSError, ValueError):
            profile[indicator_code] = []

    return {
        "country": country,
        "date_range": date,
        "indicator_count": len(profile),
        "profile": profile,
    }


@mcp.tool()
async def compare_countries(
    countries: str,
    indicator: str,
    date: str = "2015:2023",
) -> str:
    """Produce a markdown comparison table of one indicator across multiple countries.

    Args:
        countries: Semicolon-separated ISO3 codes (e.g., "USA;CHN;DEU;JPN").
        indicator: WDI indicator code (e.g., "NV.IND.MANF.ZS").
        date: Year or range.

    Returns:
        Markdown-formatted pivot table (countries by years).
    """
    client = get_client()
    records = await client.get_indicator_observations(
        country=countries, indicator=indicator, date=date, per_page=1000
    )
    if not records:
        return f"No observations found for {indicator} across {countries} in {date}"

    # Pivot: {country: {year: value}}
    pivot: dict[str, dict[int, float | None]] = {}
    indicator_name = records[0].indicator_name or indicator
    years_set: set[int] = set()
    for r in records:
        pivot.setdefault(r.country_name or r.country_code, {})[r.year] = r.value
        years_set.add(r.year)
    years = sorted(years_set)

    header = "| Country | " + " | ".join(str(y) for y in years) + " |\n"
    sep = "|---|" + "|".join(["---"] * len(years)) + "|\n"
    body = ""
    for country, year_vals in sorted(pivot.items()):
        row = f"| {country} |"
        for y in years:
            v = year_vals.get(y)
            row += f" {v:,.3g} |" if v is not None else " — |"
        body += row + "\n"

    return f"**{indicator_name} ({indicator})**\n\n{header}{sep}{body}"


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
