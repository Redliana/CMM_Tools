"""REST API for BGS World Mineral Statistics - Works with any LLM."""

from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .bgs_client import BGSClient

# Initialize FastAPI app
app = FastAPI(
    title="BGS World Mineral Statistics API",
    description="""
REST API for accessing British Geological Survey World Mineral Statistics.

**Data Coverage:**
- Time Range: 1970 - 2022+
- Commodities: 70+ minerals including critical minerals
- Statistics: Production, Imports, Exports
- Geographic: Country-level data worldwide

**Use with any LLM:**
- OpenAI function calling
- Anthropic tool use
- Google Gemini
- Local LLMs (Ollama, etc.)
- Any HTTP client

**Data Source:** British Geological Survey (Open Government Licence)
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for browser-based LLM applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class CommodityList(BaseModel):
    """List of available commodities."""

    total: int
    commodities: list[str]
    categories: dict[str, list[str]] | None = None


class CountryInfo(BaseModel):
    """Country information."""

    name: str
    iso2: str | None
    iso3: str | None


class CountryList(BaseModel):
    """List of countries."""

    total: int
    countries: list[CountryInfo]


class ProductionRecord(BaseModel):
    """Single production record."""

    commodity: str
    country: str
    country_iso3: str | None
    year: int | None
    quantity: float | None
    units: str | None


class ProductionResponse(BaseModel):
    """Production search response."""

    query: dict[str, Any]
    total: int
    records: list[ProductionRecord]


class RankedCountry(BaseModel):
    """Country ranking entry."""

    rank: int
    country: str
    country_iso3: str | None
    quantity: float
    share_percent: float


class RankingResponse(BaseModel):
    """Commodity ranking response."""

    commodity: str
    year: int
    statistic_type: str
    units: str | None
    total_quantity: float
    rankings: list[RankedCountry]


class TimeSeriesPoint(BaseModel):
    """Single time series data point."""

    year: int
    quantity: float
    yoy_change_percent: float | None = None


class TimeSeriesResponse(BaseModel):
    """Time series response."""

    commodity: str
    country: str | None
    statistic_type: str
    units: str | None
    data: list[TimeSeriesPoint]


class ComparisonResponse(BaseModel):
    """Country comparison response."""

    commodity: str
    statistic_type: str
    units: str | None
    countries: dict[str, list[dict[str, Any]]]


class CountryProfile(BaseModel):
    """Country production profile."""

    country: str
    year: int
    statistic_type: str
    commodities: list[dict[str, Any]]


class OpenAIFunction(BaseModel):
    """OpenAI function definition."""

    name: str
    description: str
    parameters: dict[str, Any]


# Client instance
def get_client() -> BGSClient:
    return BGSClient()


# Endpoints
@app.get("/", tags=["Info"])
async def root():
    """API root - returns basic info and links."""
    return {
        "name": "BGS World Mineral Statistics API",
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "openai_functions": "/openai/functions",
        "endpoints": {
            "commodities": "/commodities",
            "countries": "/countries",
            "production": "/production/search",
            "ranking": "/production/ranking",
            "time_series": "/production/timeseries",
            "compare": "/production/compare",
            "country_profile": "/countries/{country}/profile",
        },
    }


@app.get("/commodities", response_model=CommodityList, tags=["Commodities"])
async def list_commodities(
    critical_only: bool = Query(
        False,
        description="Return only critical minerals (faster, pre-defined list)",
    ),
    categorize: bool = Query(
        True,
        description="Group commodities by category",
    ),
):
    """
    List all available mineral commodities.

    Use `critical_only=true` for strategic minerals relevant to supply chain analysis.
    """
    client = get_client()

    if critical_only:
        commodities = client.get_critical_minerals()
    else:
        commodities = await client.get_commodities()

    categories = None
    if categorize:
        categories = {
            "battery": [],
            "rare_earth": [],
            "strategic": [],
            "technology": [],
            "base_metals": [],
            "precious": [],
            "industrial": [],
            "other": [],
        }

        for c in commodities:
            cl = c.lower()
            if any(x in cl for x in ["lithium", "cobalt", "nickel", "graphite", "manganese"]):
                categories["battery"].append(c)
            elif "rare earth" in cl:
                categories["rare_earth"].append(c)
            elif any(
                x in cl
                for x in [
                    "platinum",
                    "vanadium",
                    "tungsten",
                    "chromium",
                    "tantalum",
                    "niobium",
                    "titanium",
                ]
            ):
                categories["strategic"].append(c)
            elif any(
                x in cl for x in ["gallium", "germanium", "indium", "beryl", "selenium", "rhenium"]
            ):
                categories["technology"].append(c)
            elif any(
                x in cl
                for x in [
                    "copper",
                    "zinc",
                    "lead",
                    "tin",
                    "aluminium",
                    "bauxite",
                    "alumina",
                    "iron",
                ]
            ):
                categories["base_metals"].append(c)
            elif any(x in cl for x in ["gold", "silver"]):
                categories["precious"].append(c)
            elif any(x in cl for x in ["fluorspar", "magnesite", "phosphate", "barytes", "borate"]):
                categories["industrial"].append(c)
            else:
                categories["other"].append(c)

        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}

    return CommodityList(
        total=len(commodities),
        commodities=commodities,
        categories=categories,
    )


@app.get("/countries", response_model=CountryList, tags=["Countries"])
async def list_countries(
    commodity: str | None = Query(
        None,
        description="Filter to countries that produce this commodity",
    ),
):
    """
    List all countries with mineral production data.

    Optionally filter to countries that produce a specific commodity.
    """
    client = get_client()
    countries = await client.get_countries(commodity=commodity)

    return CountryList(
        total=len(countries),
        countries=[CountryInfo(**c) for c in countries],
    )


@app.get("/production/search", response_model=ProductionResponse, tags=["Production"])
async def search_production(
    commodity: str = Query(
        ..., description="Commodity name (e.g., 'lithium minerals', 'cobalt, mine')"
    ),
    country: str | None = Query(None, description="Country name or ISO code"),
    year_from: int | None = Query(None, description="Start year (inclusive)"),
    year_to: int | None = Query(None, description="End year (inclusive)"),
    statistic_type: str = Query("Production", description="Production, Imports, or Exports"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
):
    """
    Search for mineral production or trade data.

    **Examples:**
    - Lithium production: `?commodity=lithium minerals`
    - Cobalt in DRC: `?commodity=cobalt, mine&country=COD`
    - Recent copper: `?commodity=copper, mine&year_from=2020`
    """
    client = get_client()

    country_iso = None
    country_name = country
    if country and len(country) <= 3:
        country_iso = country
        country_name = None

    records = await client.search_production(
        commodity=commodity,
        country=country_name,
        country_iso=country_iso,
        year_from=year_from,
        year_to=year_to,
        statistic_type=statistic_type,
        limit=limit,
    )

    return ProductionResponse(
        query={
            "commodity": commodity,
            "country": country,
            "year_from": year_from,
            "year_to": year_to,
            "statistic_type": statistic_type,
        },
        total=len(records),
        records=[
            ProductionRecord(
                commodity=r.commodity,
                country=r.country,
                country_iso3=r.country_iso3,
                year=r.year,
                quantity=r.quantity,
                units=r.units,
            )
            for r in records
        ],
    )


@app.get("/production/ranking", response_model=RankingResponse, tags=["Production"])
async def get_commodity_ranking(
    commodity: str = Query(..., description="Commodity name"),
    year: int | None = Query(None, description="Year (defaults to most recent)"),
    statistic_type: str = Query("Production", description="Production, Imports, or Exports"),
    top_n: int = Query(15, ge=1, le=50, description="Number of top countries"),
):
    """
    Get top producing countries for a commodity.

    Returns countries ranked by production quantity with market share percentages.

    **Examples:**
    - Top lithium producers: `?commodity=lithium minerals`
    - Cobalt in 2022: `?commodity=cobalt, mine&year=2022`
    """
    client = get_client()

    ranked = await client.get_commodity_by_country(
        commodity=commodity,
        year=year,
        statistic_type=statistic_type,
        top_n=top_n,
    )

    if not ranked:
        raise HTTPException(status_code=404, detail=f"No data found for {commodity}")

    actual_year = ranked[0]["year"]
    units = ranked[0]["units"]
    total = sum(r["quantity"] for r in ranked if r["quantity"])

    rankings = []
    for i, r in enumerate(ranked, 1):
        share = (r["quantity"] / total * 100) if total > 0 else 0
        rankings.append(
            RankedCountry(
                rank=i,
                country=r["country"],
                country_iso3=r["country_iso3"],
                quantity=r["quantity"],
                share_percent=round(share, 2),
            )
        )

    return RankingResponse(
        commodity=commodity,
        year=actual_year,
        statistic_type=statistic_type,
        units=units,
        total_quantity=total,
        rankings=rankings,
    )


@app.get("/production/timeseries", response_model=TimeSeriesResponse, tags=["Production"])
async def get_time_series(
    commodity: str = Query(..., description="Commodity name"),
    country: str | None = Query(None, description="Country name or ISO code (omit for global)"),
    statistic_type: str = Query("Production", description="Production, Imports, or Exports"),
):
    """
    Get historical time series data for a commodity.

    Omit `country` to get global aggregated production.

    **Examples:**
    - Global lithium: `?commodity=lithium minerals`
    - China rare earths: `?commodity=rare earth minerals&country=CHN`
    """
    client = get_client()

    country_iso = None
    country_name = country
    if country and len(country) <= 3:
        country_iso = country
        country_name = None

    records = await client.get_time_series(
        commodity=commodity,
        country=country_name,
        country_iso=country_iso,
        statistic_type=statistic_type,
    )

    if not records:
        raise HTTPException(status_code=404, detail="No time series data found")

    # Aggregate by year if no country specified
    if not country:
        year_totals = {}
        units = None
        for r in records:
            if r.year and r.quantity is not None:
                if r.year not in year_totals:
                    year_totals[r.year] = 0
                year_totals[r.year] += r.quantity
                units = r.units

        data = []
        prev_qty = None
        for year in sorted(year_totals.keys()):
            qty = year_totals[year]
            yoy = None
            if prev_qty and prev_qty > 0:
                yoy = round(((qty - prev_qty) / prev_qty) * 100, 2)
            data.append(TimeSeriesPoint(year=year, quantity=qty, yoy_change_percent=yoy))
            prev_qty = qty

        return TimeSeriesResponse(
            commodity=commodity,
            country=None,
            statistic_type=statistic_type,
            units=units,
            data=data,
        )
    else:
        actual_country = records[0].country if records else country
        units = records[0].units if records else None

        data = []
        prev_qty = None
        for r in records:
            if r.year and r.quantity is not None:
                yoy = None
                if prev_qty and prev_qty > 0:
                    yoy = round(((r.quantity - prev_qty) / prev_qty) * 100, 2)
                data.append(
                    TimeSeriesPoint(year=r.year, quantity=r.quantity, yoy_change_percent=yoy)
                )
                prev_qty = r.quantity

        return TimeSeriesResponse(
            commodity=commodity,
            country=actual_country,
            statistic_type=statistic_type,
            units=units,
            data=data,
        )


@app.get("/production/compare", response_model=ComparisonResponse, tags=["Production"])
async def compare_countries(
    commodity: str = Query(..., description="Commodity name"),
    countries: str = Query(..., description="Comma-separated country names or ISO codes"),
    year_from: int | None = Query(None, description="Start year"),
    year_to: int | None = Query(None, description="End year"),
    statistic_type: str = Query("Production", description="Production, Imports, or Exports"),
):
    """
    Compare mineral production between multiple countries.

    **Example:**
    `?commodity=lithium minerals&countries=AUS,CHL,CHN,ARG`
    """
    client = get_client()

    country_list = [c.strip() for c in countries.split(",")]

    comparison = await client.compare_countries(
        commodity=commodity,
        countries=country_list,
        year_from=year_from,
        year_to=year_to,
        statistic_type=statistic_type,
    )

    if not comparison:
        raise HTTPException(status_code=404, detail="No comparison data found")

    # Get units from first record
    units = None
    for country_data in comparison.values():
        if country_data and country_data[0].get("units"):
            units = country_data[0]["units"]
            break

    return ComparisonResponse(
        commodity=commodity,
        statistic_type=statistic_type,
        units=units,
        countries=comparison,
    )


@app.get("/countries/{country}/profile", response_model=CountryProfile, tags=["Countries"])
async def get_country_profile(
    country: str,
    year: int | None = Query(None, description="Year (defaults to most recent)"),
    statistic_type: str = Query("Production", description="Production, Imports, or Exports"),
):
    """
    Get all commodities produced by a specific country.

    **Examples:**
    - `/countries/AUS/profile` - Australia's production
    - `/countries/COD/profile?year=2022` - DRC in 2022
    """
    client = get_client()

    country_iso = None
    country_name = country
    if len(country) <= 3:
        country_iso = country
        country_name = None

    records = await client.search_production(
        country=country_name,
        country_iso=country_iso,
        statistic_type=statistic_type,
        limit=5000,
    )

    if not records:
        raise HTTPException(status_code=404, detail=f"No data found for {country}")

    actual_country = records[0].country
    available_years = set(r.year for r in records if r.year)

    if year is None and available_years:
        year = max(available_years)

    # Aggregate by commodity
    commodity_data = {}
    for r in records:
        if r.year != year:
            continue
        if r.quantity is None:
            continue

        key = r.commodity
        if key not in commodity_data:
            commodity_data[key] = {"commodity": key, "quantity": 0, "units": r.units}
        commodity_data[key]["quantity"] += r.quantity

    # Sort by quantity
    commodities = sorted(
        commodity_data.values(),
        key=lambda x: x["quantity"],
        reverse=True,
    )

    return CountryProfile(
        country=actual_country,
        year=year,
        statistic_type=statistic_type,
        commodities=commodities,
    )


# OpenAI-compatible function definitions
@app.get("/openai/functions", response_model=list[OpenAIFunction], tags=["LLM Integration"])
async def get_openai_functions():
    """
    Get OpenAI-compatible function definitions for use with GPT models.

    Copy these into your OpenAI `tools` parameter for function calling.
    """
    return [
        OpenAIFunction(
            name="search_mineral_production",
            description="Search for mineral production data by commodity, country, and year range",
            parameters={
                "type": "object",
                "properties": {
                    "commodity": {
                        "type": "string",
                        "description": "Mineral commodity name (e.g., 'lithium minerals', 'cobalt, mine', 'rare earth minerals')",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country name or ISO code (e.g., 'Australia', 'AUS', 'CHN')",
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "Start year (e.g., 2015)",
                    },
                    "year_to": {
                        "type": "integer",
                        "description": "End year (e.g., 2022)",
                    },
                },
                "required": ["commodity"],
            },
        ),
        OpenAIFunction(
            name="get_top_producers",
            description="Get top producing countries for a mineral commodity with market share percentages",
            parameters={
                "type": "object",
                "properties": {
                    "commodity": {
                        "type": "string",
                        "description": "Mineral commodity name",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Year to query (defaults to most recent)",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top countries to return (default 15)",
                    },
                },
                "required": ["commodity"],
            },
        ),
        OpenAIFunction(
            name="get_production_time_series",
            description="Get historical time series of mineral production for trend analysis",
            parameters={
                "type": "object",
                "properties": {
                    "commodity": {
                        "type": "string",
                        "description": "Mineral commodity name",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country name or ISO code (omit for global total)",
                    },
                },
                "required": ["commodity"],
            },
        ),
        OpenAIFunction(
            name="compare_country_production",
            description="Compare mineral production between multiple countries over time",
            parameters={
                "type": "object",
                "properties": {
                    "commodity": {
                        "type": "string",
                        "description": "Mineral commodity name",
                    },
                    "countries": {
                        "type": "string",
                        "description": "Comma-separated country names or ISO codes (e.g., 'AUS,CHL,CHN')",
                    },
                },
                "required": ["commodity", "countries"],
            },
        ),
        OpenAIFunction(
            name="list_critical_minerals",
            description="List all available critical minerals in the database",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


def main():
    """Run the REST API server."""
    uvicorn.run(
        "bgs_mcp.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
