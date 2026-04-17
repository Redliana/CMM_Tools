"""MCP server for US Census International Trade (AES-derived) data."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .client import CensusAPIError, CensusClient
from .models import CMM_HS_CODES, MINERAL_NAMES

mcp = FastMCP(
    name="Census AES",
    instructions="""Census AES MCP Server provides access to US export and import
data published by the US Census Bureau's International Trade API. Export
records are derived from the Automated Export System (AES); import records
are derived from Customs entry filings. Data are monthly at HS classification
granularity (6- and 10-digit).

Key capabilities:
- Fetch US exports or imports by HS code, year, month, and partner country
- Query pre-configured critical-minerals HS sets
- Detailed vs. country-level summary levels
- Both FAS (export) and CIF/customs (import) valuations

Requires CENSUS_API_KEY. Register at https://api.census.gov/data/key_signup.html.

Scope clarification: AES itself is a filing system (exporters submit EEI
records); this server queries the publicly available Census International
Trade data feed, which is AES-derived for exports.""",
)


def get_client() -> CensusClient:
    """Return a fresh CensusClient instance."""
    return CensusClient()


# =============================================================================
# Overview
# =============================================================================


@mcp.tool()
async def get_api_status() -> dict:
    """Check Census API connectivity and key validity."""
    client = get_client()
    return await client.check_status()


@mcp.tool()
async def list_cmm_hs_codes() -> dict:
    """List curated CMM-focused HS codes for each critical mineral."""
    items = [
        {
            "id": key,
            "name": MINERAL_NAMES.get(key, key),
            "hs_codes": codes,
            "count": len(codes),
        }
        for key, codes in CMM_HS_CODES.items()
    ]
    return {
        "count": len(items),
        "minerals": items,
        "usage": "Use get_cmm_mineral_trade(mineral='lithium', year=2023, flow='export')",
    }


# =============================================================================
# Exports
# =============================================================================


@mcp.tool()
async def get_exports(
    hs_codes: list[str],
    year: int,
    month: int | None = None,
    country_codes: list[str] | None = None,
    summary_level: str = "DET",
) -> dict:
    """Fetch US export records (AES-derived) for given HS codes.

    Args:
        hs_codes: HS codes (6 or 10 digit), e.g., ["283691", "282520"].
        year: Calendar year (e.g., 2023).
        month: Optional month 1-12; omit for full year.
        country_codes: Optional Schedule C codes to filter destinations.
        summary_level: "DET" (detailed) or "CTY" (country-level aggregate).
    """
    client = get_client()
    try:
        records = await client.get_exports(
            hs_codes=hs_codes,
            year=year,
            month=month,
            country_codes=country_codes,
            summary_level=summary_level,
        )
    except CensusAPIError as e:
        return {"error": str(e)}
    return {
        "query": {
            "hs_codes": hs_codes,
            "year": year,
            "month": month,
            "country_codes": country_codes,
            "summary_level": summary_level,
        },
        "count": len(records),
        "records": [r.model_dump() for r in records],
    }


# =============================================================================
# Imports
# =============================================================================


@mcp.tool()
async def get_imports(
    hs_codes: list[str],
    year: int,
    month: int | None = None,
    country_codes: list[str] | None = None,
    summary_level: str = "DET",
) -> dict:
    """Fetch US import records for given HS codes.

    Args:
        hs_codes: HS codes (6 or 10 digit).
        year: Calendar year.
        month: Optional month 1-12.
        country_codes: Optional Schedule C codes to filter origins.
        summary_level: "DET" or "CTY".
    """
    client = get_client()
    try:
        records = await client.get_imports(
            hs_codes=hs_codes,
            year=year,
            month=month,
            country_codes=country_codes,
            summary_level=summary_level,
        )
    except CensusAPIError as e:
        return {"error": str(e)}
    return {
        "query": {
            "hs_codes": hs_codes,
            "year": year,
            "month": month,
            "country_codes": country_codes,
            "summary_level": summary_level,
        },
        "count": len(records),
        "records": [r.model_dump() for r in records],
    }


# =============================================================================
# Critical Minerals Convenience
# =============================================================================


@mcp.tool()
async def get_cmm_mineral_trade(
    mineral: str,
    year: int,
    flow: str = "export",
    month: int | None = None,
    country_codes: list[str] | None = None,
) -> dict:
    """Fetch US trade data for a critical mineral using curated HS codes.

    Available minerals: lithium, cobalt, rare_earth, graphite, nickel,
    manganese, gallium, germanium, copper.

    Args:
        mineral: Mineral key.
        year: Calendar year.
        flow: "export" or "import".
        month: Optional month 1-12.
        country_codes: Optional Schedule C country codes.
    """
    client = get_client()
    try:
        records = await client.get_critical_mineral_trade(
            mineral=mineral,
            year=year,
            month=month,
            flow=flow,
            country_codes=country_codes,
        )
    except CensusAPIError as e:
        return {"error": str(e)}
    key = mineral.lower().replace(" ", "_")
    return {
        "mineral": MINERAL_NAMES.get(key, mineral),
        "hs_codes_queried": CMM_HS_CODES.get(key, []),
        "flow": flow,
        "year": year,
        "month": month,
        "count": len(records),
        "records": [r.model_dump() for r in records],
    }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
