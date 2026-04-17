"""MCP server for USITC DataWeb."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .client import USITCAPIError, USITCClient
from .models import CMM_HTS_CODES, MINERAL_NAMES

mcp = FastMCP(
    name="USITC DataWeb",
    instructions="""USITC DataWeb MCP Server provides access to US import/export
trade statistics at HTS (Harmonized Tariff Schedule of the United States)
granularity. Complements uncomtrade-mcp by offering US-specific HTS codes
(8-10 digit) and trade-measure granularity (general imports, imports for
consumption, domestic exports, total exports) not available in UN Comtrade.

Key capabilities:
- Fetch US import or export values by HTS code, partner, and year
- Search the HTS classification catalogue
- Query curated CMM-focused HTS codes (lithium, cobalt, REE, graphite, Ni, Mn, Ga, Ge, Cu)
- Switch between general imports and imports for consumption
- Retrieve both value (USD) and quantity (multi-unit)

Requires USITC_API_TOKEN. Register at https://dataweb.usitc.gov and request
API access from your account page.""",
)


def get_client() -> USITCClient:
    """Return a fresh USITCClient instance."""
    return USITCClient()


# =============================================================================
# Overview
# =============================================================================


@mcp.tool()
async def get_api_status() -> dict:
    """Check USITC DataWeb API connectivity and token validity."""
    client = get_client()
    return await client.check_status()


@mcp.tool()
async def list_cmm_hts_codes() -> dict:
    """List curated CMM-focused HTS codes for each critical mineral."""
    items = []
    for key, codes in CMM_HTS_CODES.items():
        items.append(
            {
                "id": key,
                "name": MINERAL_NAMES.get(key, key),
                "hts_codes": codes,
                "count": len(codes),
            }
        )
    return {
        "count": len(items),
        "minerals": items,
        "usage": "Use get_critical_mineral_trade(mineral='lithium', years=[2022,2023])",
    }


# =============================================================================
# Search
# =============================================================================


@mcp.tool()
async def search_hts(query: str, limit: int = 25) -> dict:
    """Search the HTS classification by keyword or code prefix.

    Args:
        query: Search string (e.g., "lithium", "8507", "battery").
        limit: Max results.
    """
    client = get_client()
    try:
        results = await client.search_hts(query, limit=limit)
    except USITCAPIError as e:
        return {"error": str(e)}
    return {"query": query, "count": len(results), "results": results}


# =============================================================================
# Data Queries
# =============================================================================


@mcp.tool()
async def get_trade_data(
    hts_codes: list[str],
    years: list[int],
    flow: str = "import",
    data_type: str = "general_imports",
    country_codes: list[str] | None = None,
    aggregate_by: str = "year",
) -> dict:
    """Fetch US import or export data by HTS code, partner, and year.

    Args:
        hts_codes: List of HTS codes (e.g., ["2836.91.00", "2825.20.00"]).
        years: List of years to query (e.g., [2020, 2021, 2022, 2023]).
        flow: "import" or "export".
        data_type: "general_imports" | "imports_for_consumption" |
                   "domestic_exports" | "total_exports".
        country_codes: Optional list of partner codes; omit for all partners.
        aggregate_by: "year" | "month" | "quarter".

    Returns:
        Dict with query and list of TradeRecord.
    """
    client = get_client()
    try:
        records = await client.get_trade_data(
            hts_codes=hts_codes,
            years=years,
            flow=flow,
            data_type=data_type,
            country_codes=country_codes,
            aggregate_by=aggregate_by,
        )
    except USITCAPIError as e:
        return {"error": str(e)}
    return {
        "query": {
            "hts_codes": hts_codes,
            "years": years,
            "flow": flow,
            "data_type": data_type,
            "country_codes": country_codes,
            "aggregate_by": aggregate_by,
        },
        "count": len(records),
        "records": [r.model_dump() for r in records],
    }


@mcp.tool()
async def get_critical_mineral_trade(
    mineral: str,
    years: list[int],
    flow: str = "import",
    country_codes: list[str] | None = None,
) -> dict:
    """Get US trade data for a critical mineral using curated HTS codes.

    Available minerals: lithium, cobalt, rare_earth, graphite, nickel,
    manganese, gallium, germanium, copper.

    Args:
        mineral: Mineral key (e.g., "lithium").
        years: List of years.
        flow: "import" or "export".
        country_codes: Optional partner filter.

    Returns:
        Dict with records and HTS codes queried.
    """
    client = get_client()
    try:
        records = await client.get_critical_mineral_trade(
            mineral=mineral, years=years, flow=flow, country_codes=country_codes
        )
    except USITCAPIError as e:
        return {"error": str(e)}
    key = mineral.lower().replace(" ", "_")
    return {
        "mineral": MINERAL_NAMES.get(key, mineral),
        "hts_codes_queried": CMM_HTS_CODES.get(key, []),
        "flow": flow,
        "years": years,
        "count": len(records),
        "records": [r.model_dump() for r in records],
    }


@mcp.tool()
async def compare_import_types(
    hts_codes: list[str],
    years: list[int],
    country_codes: list[str] | None = None,
) -> dict:
    """Compare 'general imports' vs. 'imports for consumption' for the same HTS set.

    Useful for identifying goods entering bonded warehouses / FTZs versus those
    cleared for domestic consumption — relevant to CMM stockpile and FTZ analysis.

    Args:
        hts_codes: List of HTS codes.
        years: List of years.
        country_codes: Optional partner filter.
    """
    client = get_client()
    result: dict[str, list[dict]] = {}
    try:
        for dt in ("general_imports", "imports_for_consumption"):
            records = await client.get_trade_data(
                hts_codes=hts_codes,
                years=years,
                flow="import",
                data_type=dt,
                country_codes=country_codes,
            )
            result[dt] = [r.model_dump() for r in records]
    except USITCAPIError as e:
        return {"error": str(e)}
    return {
        "query": {"hts_codes": hts_codes, "years": years, "country_codes": country_codes},
        "general_imports_count": len(result.get("general_imports", [])),
        "imports_for_consumption_count": len(result.get("imports_for_consumption", [])),
        "records": result,
    }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
