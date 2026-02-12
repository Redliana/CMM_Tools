"""BGS World Mineral Statistics MCP Server."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .bgs_client import BGSClient

# Initialize the MCP server
mcp = FastMCP(
    name="BGS World Mineral Statistics",
    instructions="Access British Geological Survey World Mineral Statistics for critical minerals production and trade data (1970-2022+)",
)


def get_client() -> BGSClient:
    """Get BGS client instance."""
    return BGSClient()


@mcp.tool()
async def list_commodities(critical_only: bool = False) -> str:
    """
    List all available mineral commodities in the BGS database.

    Args:
        critical_only: If True, show only critical minerals. If False, fetch all from API.

    Returns:
        List of commodity names
    """
    client = get_client()

    if critical_only:
        commodities = client.get_critical_minerals()
        output = "**Critical Minerals (Pre-defined List)**\n\n"
    else:
        commodities = await client.get_commodities()
        output = "**All Available Commodities**\n\n"

    output += f"Total: {len(commodities)} commodities\n\n"

    # Group by category
    battery = []
    rare_earth = []
    strategic = []
    technology = []
    base = []
    precious = []
    industrial = []
    other = []

    for c in commodities:
        cl = c.lower()
        if any(x in cl for x in ["lithium", "cobalt", "nickel", "graphite", "manganese"]):
            battery.append(c)
        elif "rare earth" in cl:
            rare_earth.append(c)
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
            strategic.append(c)
        elif any(
            x in cl for x in ["gallium", "germanium", "indium", "beryl", "selenium", "rhenium"]
        ):
            technology.append(c)
        elif any(
            x in cl
            for x in ["copper", "zinc", "lead", "tin", "aluminium", "bauxite", "alumina", "iron"]
        ):
            base.append(c)
        elif any(x in cl for x in ["gold", "silver"]):
            precious.append(c)
        elif any(x in cl for x in ["fluorspar", "magnesite", "phosphate", "barytes", "borate"]):
            industrial.append(c)
        else:
            other.append(c)

    if battery:
        output += "**Battery Minerals:**\n" + "\n".join(f"- {c}" for c in battery) + "\n\n"
    if rare_earth:
        output += "**Rare Earth Elements:**\n" + "\n".join(f"- {c}" for c in rare_earth) + "\n\n"
    if strategic:
        output += "**Strategic Metals:**\n" + "\n".join(f"- {c}" for c in strategic) + "\n\n"
    if technology:
        output += "**Technology Minerals:**\n" + "\n".join(f"- {c}" for c in technology) + "\n\n"
    if base:
        output += "**Base Metals:**\n" + "\n".join(f"- {c}" for c in base) + "\n\n"
    if precious:
        output += "**Precious Metals:**\n" + "\n".join(f"- {c}" for c in precious) + "\n\n"
    if industrial:
        output += "**Industrial Minerals:**\n" + "\n".join(f"- {c}" for c in industrial) + "\n\n"
    if other:
        output += "**Other:**\n" + "\n".join(f"- {c}" for c in other) + "\n\n"

    return output


@mcp.tool()
async def list_countries(commodity: str | None = None) -> str:
    """
    List all countries with mineral production data.

    Args:
        commodity: Optional commodity to filter countries that produce it

    Returns:
        List of countries with ISO codes
    """
    client = get_client()
    countries = await client.get_countries(commodity=commodity)

    if commodity:
        output = f"**Countries Producing: {commodity}**\n\n"
    else:
        output = "**Countries in BGS Database**\n\n"

    output += f"Total: {len(countries)} countries\n\n"
    output += "| Country | ISO2 | ISO3 |\n"
    output += "|---------|------|------|\n"

    for c in countries:
        output += f"| {c['name']} | {c['iso2'] or '-'} | {c['iso3'] or '-'} |\n"

    return output


@mcp.tool()
async def search_production(
    commodity: str,
    country: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    statistic_type: str = "Production",
    limit: int = 50,
) -> str:
    """
    Search for mineral production or trade data.

    Args:
        commodity: Commodity name (e.g., "lithium minerals", "cobalt, mine")
        country: Country name or ISO code (optional)
        year_from: Start year (optional)
        year_to: End year (optional)
        statistic_type: "Production", "Imports", or "Exports" (default: Production)
        limit: Maximum records to return (default: 50)

    Returns:
        Table of production data with country, year, quantity, units
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

    if not records:
        return f"No {statistic_type.lower()} data found for {commodity}" + (
            f" in {country}" if country else ""
        )

    output = f"**{commodity} - {statistic_type}**\n\n"
    if country:
        output += f"Country: {country}\n"
    if year_from or year_to:
        year_range = f"{year_from or 'start'} - {year_to or 'present'}"
        output += f"Years: {year_range}\n"
    output += f"Records: {len(records)}\n\n"

    output += "| Country | Year | Quantity | Units |\n"
    output += "|---------|------|----------|-------|\n"

    for r in records:
        qty = f"{r.quantity:,.1f}" if r.quantity is not None else "N/A"
        output += f"| {r.country} | {r.year or 'N/A'} | {qty} | {r.units or 'N/A'} |\n"

    return output


@mcp.tool()
async def get_commodity_ranking(
    commodity: str,
    year: int | None = None,
    statistic_type: str = "Production",
    top_n: int = 15,
) -> str:
    """
    Get top producing countries for a commodity in a specific year.

    Args:
        commodity: Commodity name (e.g., "lithium minerals")
        year: Year to query (defaults to most recent available)
        statistic_type: "Production", "Imports", or "Exports"
        top_n: Number of top countries to show (default: 15)

    Returns:
        Ranked table of countries by production quantity
    """
    client = get_client()

    ranked = await client.get_commodity_by_country(
        commodity=commodity,
        year=year,
        statistic_type=statistic_type,
        top_n=top_n,
    )

    if not ranked:
        return f"No data found for {commodity}" + (f" in {year}" if year else "")

    actual_year = ranked[0]["year"] if ranked else year
    units = ranked[0]["units"] if ranked else "N/A"

    output = f"**{commodity} - Top {statistic_type} Countries ({actual_year})**\n\n"
    output += f"Units: {units}\n\n"

    output += "| Rank | Country | Quantity | Share |\n"
    output += "|------|---------|----------|-------|\n"

    total = sum(r["quantity"] for r in ranked if r["quantity"])

    for i, r in enumerate(ranked, 1):
        qty = r["quantity"]
        share = (qty / total * 100) if total > 0 else 0
        output += f"| {i} | {r['country']} | {qty:,.1f} | {share:.1f}% |\n"

    output += f"\n**Total (top {len(ranked)}): {total:,.1f} {units}**\n"

    return output


@mcp.tool()
async def get_time_series(
    commodity: str,
    country: str | None = None,
    statistic_type: str = "Production",
) -> str:
    """
    Get historical time series data for a commodity.

    Args:
        commodity: Commodity name
        country: Country name or ISO code (optional, for single country)
        statistic_type: "Production", "Imports", or "Exports"

    Returns:
        Time series data showing year-over-year values
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
        return f"No time series data found for {commodity}" + (f" in {country}" if country else "")

    # If no country specified, aggregate by year
    if not country:
        year_totals = {}
        units = None
        for r in records:
            if r.year and r.quantity is not None:
                if r.year not in year_totals:
                    year_totals[r.year] = 0
                year_totals[r.year] += r.quantity
                units = r.units

        output = f"**{commodity} - Global {statistic_type} Time Series**\n\n"
        output += f"Units: {units}\n\n"
        output += "| Year | Total Quantity | YoY Change |\n"
        output += "|------|----------------|------------|\n"

        sorted_years = sorted(year_totals.keys())
        prev_qty = None

        for year in sorted_years:
            qty = year_totals[year]
            if prev_qty and prev_qty > 0:
                change = ((qty - prev_qty) / prev_qty) * 100
                change_str = f"{change:+.1f}%"
            else:
                change_str = "-"
            output += f"| {year} | {qty:,.1f} | {change_str} |\n"
            prev_qty = qty

    else:
        actual_country = records[0].country if records else country
        units = records[0].units if records else "N/A"

        output = f"**{commodity} - {actual_country} {statistic_type} Time Series**\n\n"
        output += f"Units: {units}\n\n"
        output += "| Year | Quantity | YoY Change |\n"
        output += "|------|----------|------------|\n"

        prev_qty = None
        for r in records:
            if r.year and r.quantity is not None:
                if prev_qty and prev_qty > 0:
                    change = ((r.quantity - prev_qty) / prev_qty) * 100
                    change_str = f"{change:+.1f}%"
                else:
                    change_str = "-"
                output += f"| {r.year} | {r.quantity:,.1f} | {change_str} |\n"
                prev_qty = r.quantity

    return output


@mcp.tool()
async def compare_countries(
    commodity: str,
    countries: str,
    year_from: int | None = None,
    year_to: int | None = None,
    statistic_type: str = "Production",
) -> str:
    """
    Compare mineral production between multiple countries.

    Args:
        commodity: Commodity name
        countries: Comma-separated list of country names or ISO codes
        year_from: Start year (optional)
        year_to: End year (optional)
        statistic_type: "Production", "Imports", or "Exports"

    Returns:
        Comparison table showing production by country and year
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

    if not comparison or all(len(v) == 0 for v in comparison.values()):
        return f"No comparison data found for {commodity} in specified countries"

    # Get all years across all countries
    all_years = set()
    units = None
    for country_data in comparison.values():
        for record in country_data:
            if record["year"]:
                all_years.add(record["year"])
            if record["units"]:
                units = record["units"]

    sorted_years = sorted(all_years)

    output = f"**{commodity} - Country Comparison ({statistic_type})**\n\n"
    output += f"Units: {units}\n\n"

    # Build header
    header = "| Year |"
    separator = "|------|"
    for country in comparison:
        header += f" {country[:15]} |"
        separator += "----------|"

    output += header + "\n" + separator + "\n"

    # Build data rows
    for year in sorted_years:
        row = f"| {year} |"
        for country, data in comparison.items():
            qty = next((r["quantity"] for r in data if r["year"] == year), None)
            if qty is not None:
                row += f" {qty:,.0f} |"
            else:
                row += " - |"
        output += row + "\n"

    return output


@mcp.tool()
async def get_country_profile(
    country: str,
    year: int | None = None,
    statistic_type: str = "Production",
) -> str:
    """
    Get a production profile for a specific country showing all commodities.

    Args:
        country: Country name or ISO code
        year: Year to query (defaults to most recent)
        statistic_type: "Production", "Imports", or "Exports"

    Returns:
        Table of all commodities produced by the country
    """
    client = get_client()

    country_iso = None
    country_name = country
    if len(country) <= 3:
        country_iso = country
        country_name = None

    # Search for all commodities for this country
    records = await client.search_production(
        country=country_name,
        country_iso=country_iso,
        statistic_type=statistic_type,
        limit=5000,
    )

    if not records:
        return f"No {statistic_type.lower()} data found for {country}"

    actual_country = records[0].country
    available_years = set(r.year for r in records if r.year)

    if year is None and available_years:
        year = max(available_years)

    # Filter to target year and aggregate by commodity
    commodity_data = {}
    for r in records:
        if r.year != year:
            continue
        if r.quantity is None:
            continue

        key = r.commodity
        if key not in commodity_data:
            commodity_data[key] = {"quantity": 0, "units": r.units}
        commodity_data[key]["quantity"] += r.quantity

    output = f"**{actual_country} - {statistic_type} Profile ({year})**\n\n"
    output += f"Commodities: {len(commodity_data)}\n\n"

    output += "| Commodity | Quantity | Units |\n"
    output += "|-----------|----------|-------|\n"

    # Sort by quantity descending
    sorted_commodities = sorted(
        commodity_data.items(),
        key=lambda x: x[1]["quantity"],
        reverse=True,
    )

    for commodity, data in sorted_commodities:
        output += f"| {commodity} | {data['quantity']:,.1f} | {data['units'] or 'N/A'} |\n"

    return output


@mcp.tool()
def get_api_info() -> str:
    """
    Get information about the BGS World Mineral Statistics API and data coverage.

    Returns:
        API documentation and data coverage information
    """
    return """**BGS World Mineral Statistics API**

**Data Source:** British Geological Survey
**License:** Open Government Licence

**API Endpoint:** https://ogcapi.bgs.ac.uk/collections/world-mineral-statistics

**Time Coverage:**
- Full data: 1970 - 2022+
- Historical archives: 1913 - 1969 (PDF only)

**Statistics Types:**
- Production (mine output)
- Imports
- Exports

**Data Fields:**
- `commodity`: Mineral/metal name
- `country`: Country name
- `country_iso2/iso3`: ISO country codes
- `year`: Data year
- `quantity`: Numeric value
- `units`: Measurement units (tonnes, kg, etc.)

**Critical Minerals Covered:**
- Battery: Lithium, Cobalt, Nickel, Graphite, Manganese
- Rare Earths: REE minerals, REE oxides
- Strategic: PGMs, Tungsten, Vanadium, Chromium, Tantalum/Niobium
- Technology: Gallium, Germanium, Indium, Beryllium
- Base Metals: Copper, Zinc, Lead, Tin, Aluminium
- Precious: Gold, Silver

**Usage Tips:**
1. Use `list_commodities(critical_only=True)` for strategic minerals
2. Use `get_commodity_ranking` for top producers
3. Use `get_time_series` for trend analysis
4. Use `compare_countries` for supply chain analysis

**Website:** https://www.bgs.ac.uk/mineralsuk/statistics/world-mineral-statistics/
"""


def main():
    """Run the BGS MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
