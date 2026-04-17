"""Async HTTP client for World Bank WDI API.

The World Bank Indicators API (v2) is public and does not require authentication
for the WDI endpoints used here. An optional WORLDBANK_API_KEY env var is
retained for future extension (e.g., WITS authenticated endpoints).

API reference: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

from .models import IndicatorObservation

load_dotenv()


class WorldBankClient:
    """Client for the World Bank WDI API (v2)."""

    BASE_URL = "https://api.worldbank.org/v2"

    def __init__(self, api_key: str | None = None):
        """Initialize client. API key is optional for WDI."""
        self.api_key = api_key or os.getenv("WORLDBANK_API_KEY")
        self.timeout = 60.0

    def is_available(self) -> bool:
        """WDI is publicly accessible; always returns True."""
        return True

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> list[Any]:
        """GET against the WDI API. Responses are a two-element list: [meta, data]."""
        params = dict(params or {})
        params.setdefault("format", "json")
        params.setdefault("per_page", 1000)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"Unexpected WDI response shape: {type(payload).__name__}")
            return payload

    async def check_status(self) -> dict[str, Any]:
        """Check connectivity via a minimal WDI query."""
        try:
            payload = await self._request(
                "country/USA/indicator/NY.GDP.MKTP.CD",
                params={"date": "2022", "per_page": 1},
            )
            meta = payload[0] if payload else {}
            data = payload[1] if len(payload) > 1 else []
            return {
                "status": "connected",
                "api_key_configured": bool(self.api_key),
                "total_observations_sample": meta.get("total"),
                "sample_record_count": len(data),
                "message": "World Bank WDI API reachable",
            }
        except httpx.TimeoutException:
            return {"status": "timeout", "message": "Request timed out"}
        except (httpx.HTTPError, OSError, ValueError) as e:
            return {"status": "error", "message": str(e)}

    async def get_indicator_observations(
        self,
        country: str,
        indicator: str,
        date: str | None = None,
        per_page: int = 100,
    ) -> list[IndicatorObservation]:
        """
        Fetch indicator observations for a country or country group.

        Args:
            country: ISO3 code or semicolon-separated list (e.g., "USA;CHN;DEU").
                     Special groups: "all", "WLD" (world).
            indicator: WDI indicator code (e.g., "NY.GDP.MKTP.CD").
            date: Year or range, e.g., "2022" or "2000:2023".
            per_page: Page size (max 1000).

        Returns:
            List of IndicatorObservation, newest first.
        """
        endpoint = f"country/{country}/indicator/{indicator}"
        params: dict[str, Any] = {"per_page": min(per_page, 1000)}
        if date:
            params["date"] = date

        payload = await self._request(endpoint, params=params)
        data = payload[1] if len(payload) > 1 else []
        records: list[IndicatorObservation] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                country_meta = item.get("country") or {}
                indicator_meta = item.get("indicator") or {}
                raw_date = item.get("date")
                try:
                    year = int(raw_date) if raw_date is not None else 0
                except (TypeError, ValueError):
                    year = 0
                records.append(
                    IndicatorObservation(
                        country_code=item.get("countryiso3code") or country_meta.get("id", ""),
                        country_name=country_meta.get("value"),
                        indicator_code=indicator_meta.get("id", indicator),
                        indicator_name=indicator_meta.get("value"),
                        year=year,
                        value=item.get("value"),
                        unit=item.get("unit") or None,
                    )
                )
            except (ValueError, TypeError):
                continue
        return records

    async def search_indicators(self, query: str, per_page: int = 50) -> list[dict[str, Any]]:
        """Search the WDI indicator catalogue by free-text keyword."""
        params = {"per_page": min(per_page, 500)}
        payload = await self._request("indicator", params=params)
        data = payload[1] if len(payload) > 1 else []
        q = query.lower()
        matches: list[dict[str, Any]] = []
        for ind in data:
            if not isinstance(ind, dict):
                continue
            name = (ind.get("name") or "").lower()
            note = (ind.get("sourceNote") or "").lower()
            if q in name or q in note:
                matches.append(
                    {
                        "id": ind.get("id"),
                        "name": ind.get("name"),
                        "source_note": ind.get("sourceNote"),
                        "topics": [t.get("value") for t in (ind.get("topics") or [])],
                    }
                )
        return matches

    async def list_countries(self) -> list[dict[str, Any]]:
        """List all World Bank country/region reference entries."""
        payload = await self._request("country", params={"per_page": 500})
        data = payload[1] if len(payload) > 1 else []
        result: list[dict[str, Any]] = []
        for c in data:
            if not isinstance(c, dict):
                continue
            result.append(
                {
                    "iso3": c.get("id"),
                    "iso2": c.get("iso2Code"),
                    "name": c.get("name"),
                    "region": (c.get("region") or {}).get("value"),
                    "income_level": (c.get("incomeLevel") or {}).get("value"),
                }
            )
        return result
