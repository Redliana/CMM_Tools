"""BGS World Mineral Statistics API client."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel


class MineralRecord(BaseModel):
    """A single mineral production/trade record."""

    commodity: str
    sub_commodity: str | None = None
    statistic_type: str
    country: str
    country_iso2: str | None = None
    country_iso3: str | None = None
    year: int | None = None
    quantity: float | None = None
    units: str | None = None
    yearbook_table: str | None = None
    notes: str | None = None


class BGSClient:
    """Client for BGS World Mineral Statistics OGC API."""

    BASE_URL = "https://ogcapi.bgs.ac.uk/collections/world-mineral-statistics"

    # Critical minerals list
    CRITICAL_MINERALS = [
        # Battery minerals
        "lithium minerals",
        "cobalt, mine",
        "cobalt, refined",
        "nickel, mine",
        "nickel, smelter/refinery",
        "graphite",
        "manganese ore",
        # Rare earths
        "rare earth minerals",
        "rare earth oxides",
        # Strategic metals
        "platinum group metals, mine",
        "vanadium, mine",
        "tungsten, mine",
        "chromium ores and concentrates",
        "tantalum and niobium minerals",
        "titanium minerals",
        # Technology minerals
        "gallium, primary",
        "germanium metal",
        "indium, refinery",
        "beryl",
        "bismuth, mine",
        "selenium, refined",
        "rhenium",
        "strontium minerals",
        # Base metals
        "copper, mine",
        "copper, refined",
        "zinc, mine",
        "lead, mine",
        "tin, mine",
        "aluminium, primary",
        "bauxite",
        # Industrial minerals
        "fluorspar",
        "magnesite",
        "phosphate rock",
        "barytes",
        "borates",
        # Precious metals
        "gold, mine",
        "silver, mine",
        # Other critical
        "antimony, mine",
        "molybdenum, mine",
        "iron ore",
    ]

    def __init__(self):
        self.timeout = 60.0

    async def _request(
        self,
        params: dict[str, Any] | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Make a request to the BGS API."""
        url = f"{self.BASE_URL}/items"

        query_params = {"limit": limit, "offset": offset}
        if params:
            query_params.update(params)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                url,
                params=query_params,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            return response.json()

    def _parse_records(self, data: dict[str, Any]) -> list[MineralRecord]:
        """Parse API response into MineralRecord objects."""
        records = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            year_str = props.get("year", "")
            year = int(year_str[:4]) if year_str and len(year_str) >= 4 else None

            record = MineralRecord(
                commodity=props.get("bgs_commodity_trans", ""),
                sub_commodity=props.get("bgs_sub_commodity_trans"),
                statistic_type=props.get("bgs_statistic_type_trans", ""),
                country=props.get("country_trans", ""),
                country_iso2=props.get("country_iso2_code"),
                country_iso3=props.get("country_iso3_code"),
                year=year,
                quantity=props.get("quantity"),
                units=props.get("units"),
                yearbook_table=props.get("yearbook_table_trans"),
                notes=props.get("concat_table_notes_text"),
            )
            records.append(record)
        return records

    async def get_commodities(self) -> list[str]:
        """Get list of all available commodities."""
        # Fetch a large sample to get commodity list
        commodities = set()

        for offset in [0, 5000, 10000, 20000]:
            try:
                data = await self._request(limit=5000, offset=offset)
                for feature in data.get("features", []):
                    commodity = feature.get("properties", {}).get("bgs_commodity_trans")
                    if commodity:
                        commodities.add(commodity)
            except (httpx.HTTPError, ValueError, KeyError):
                break

        return sorted(commodities)

    async def get_countries(self, commodity: str | None = None) -> list[dict[str, str]]:
        """Get list of countries, optionally filtered by commodity."""
        countries = {}

        params = {}
        if commodity:
            params["bgs_commodity_trans"] = commodity

        data = await self._request(params=params, limit=5000)

        for feature in data.get("features", []):
            props = feature.get("properties", {})
            country = props.get("country_trans")
            iso2 = props.get("country_iso2_code")
            iso3 = props.get("country_iso3_code")

            if country and country not in countries:
                countries[country] = {"name": country, "iso2": iso2, "iso3": iso3}

        return sorted(countries.values(), key=lambda x: x["name"])

    async def search_production(
        self,
        commodity: str | None = None,
        country: str | None = None,
        country_iso: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        statistic_type: str = "Production",
        limit: int = 1000,
    ) -> list[MineralRecord]:
        """
        Search for mineral production data.

        Args:
            commodity: Commodity name (e.g., "lithium minerals")
            country: Country name
            country_iso: ISO2 or ISO3 country code
            year_from: Start year (inclusive)
            year_to: End year (inclusive)
            statistic_type: "Production", "Imports", or "Exports"
            limit: Maximum records to return

        Returns:
            List of MineralRecord objects
        """
        params = {"bgs_statistic_type_trans": statistic_type}

        if commodity:
            params["bgs_commodity_trans"] = commodity
        if country:
            params["country_trans"] = country
        if country_iso:
            if len(country_iso) == 2:
                params["country_iso2_code"] = country_iso.upper()
            else:
                params["country_iso3_code"] = country_iso.upper()

        # Fetch data
        all_records = []
        offset = 0

        while len(all_records) < limit:
            fetch_limit = min(1000, limit - len(all_records))
            data = await self._request(params=params, limit=fetch_limit, offset=offset)
            records = self._parse_records(data)

            if not records:
                break

            # Filter by year if specified
            for record in records:
                if year_from and record.year and record.year < year_from:
                    continue
                if year_to and record.year and record.year > year_to:
                    continue
                all_records.append(record)

                if len(all_records) >= limit:
                    break

            if len(records) < fetch_limit:
                break

            offset += fetch_limit

        # Sort by year descending
        all_records.sort(key=lambda x: x.year or 0, reverse=True)

        return all_records[:limit]

    async def get_time_series(
        self,
        commodity: str,
        country: str | None = None,
        country_iso: str | None = None,
        statistic_type: str = "Production",
    ) -> list[MineralRecord]:
        """
        Get time series data for a commodity.

        Args:
            commodity: Commodity name
            country: Country name (optional, for single country series)
            country_iso: ISO country code (optional)
            statistic_type: "Production", "Imports", or "Exports"

        Returns:
            List of records sorted by year ascending
        """
        records = await self.search_production(
            commodity=commodity,
            country=country,
            country_iso=country_iso,
            statistic_type=statistic_type,
            limit=5000,
        )

        # Sort by year ascending for time series
        records.sort(key=lambda x: x.year or 0)

        return records

    async def get_commodity_by_country(
        self,
        commodity: str,
        year: int | None = None,
        statistic_type: str = "Production",
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Get commodity production ranked by country.

        Args:
            commodity: Commodity name
            year: Specific year (defaults to most recent)
            statistic_type: "Production", "Imports", or "Exports"
            top_n: Number of top countries to return

        Returns:
            List of dicts with country, quantity, units, year
        """
        records = await self.search_production(
            commodity=commodity,
            statistic_type=statistic_type,
            limit=5000,
        )

        if not records:
            return []

        # Find the target year
        if year is None:
            year = max(r.year for r in records if r.year)

        # Filter to target year and aggregate by country
        country_totals = {}

        for record in records:
            if record.year != year:
                continue
            if record.quantity is None:
                continue

            country = record.country
            if country not in country_totals:
                country_totals[country] = {
                    "country": country,
                    "country_iso3": record.country_iso3,
                    "quantity": 0,
                    "units": record.units,
                    "year": year,
                }
            country_totals[country]["quantity"] += record.quantity

        # Sort by quantity descending
        ranked = sorted(
            country_totals.values(),
            key=lambda x: x["quantity"],
            reverse=True,
        )

        return ranked[:top_n]

    async def compare_countries(
        self,
        commodity: str,
        countries: list[str],
        year_from: int | None = None,
        year_to: int | None = None,
        statistic_type: str = "Production",
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Compare production between multiple countries.

        Args:
            commodity: Commodity name
            countries: List of country names or ISO codes
            year_from: Start year
            year_to: End year
            statistic_type: "Production", "Imports", or "Exports"

        Returns:
            Dict mapping country to list of {year, quantity, units}
        """
        result = {}

        for country in countries:
            # Try as country name first, then as ISO code
            records = await self.search_production(
                commodity=commodity,
                country=country if len(country) > 3 else None,
                country_iso=country if len(country) <= 3 else None,
                year_from=year_from,
                year_to=year_to,
                statistic_type=statistic_type,
                limit=1000,
            )

            country_name = country
            if records:
                country_name = records[0].country

            result[country_name] = [
                {
                    "year": r.year,
                    "quantity": r.quantity,
                    "units": r.units,
                }
                for r in sorted(records, key=lambda x: x.year or 0)
            ]

        return result

    def get_critical_minerals(self) -> list[str]:
        """Get the list of pre-defined critical minerals."""
        return self.CRITICAL_MINERALS.copy()
