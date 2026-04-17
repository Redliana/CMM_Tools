"""Async HTTP client for FRED (Federal Reserve Economic Data).

API reference: https://fred.stlouisfed.org/docs/api/fred/
Authentication: FRED_API_KEY env var (register at https://fred.stlouisfed.org/docs/api/api_key.html).
"""

from __future__ import annotations

import os
from typing import Any, cast

import httpx
from dotenv import load_dotenv

from .models import Observation, SeriesMetadata

load_dotenv()


class FredAPIError(Exception):
    """Raised for FRED API errors."""


class FredClient:
    """Client for the FRED API (v2 JSON endpoints)."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str | None = None):
        """Initialize client; api_key may be passed explicitly or via FRED_API_KEY."""
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.timeout = 60.0

    def is_available(self) -> bool:
        """Return True if an API key is configured."""
        return bool(self.api_key)

    def _require_key(self) -> None:
        if not self.api_key:
            raise FredAPIError(
                "FRED_API_KEY is not set. Register at "
                "https://fred.stlouisfed.org/docs/api/api_key.html and export FRED_API_KEY."
            )

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET against a FRED JSON endpoint."""
        self._require_key()
        query = dict(params or {})
        query["api_key"] = self.api_key
        query.setdefault("file_type", "json")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.BASE_URL}/{endpoint}", params=query)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data

    async def check_status(self) -> dict[str, Any]:
        """Check API connectivity with a lightweight series lookup."""
        if not self.api_key:
            return {
                "status": "unauthorized",
                "api_key_configured": False,
                "message": "FRED_API_KEY not set",
            }
        try:
            await self._request("series", params={"series_id": "GDPC1"})
            return {
                "status": "connected",
                "api_key_configured": True,
                "message": "FRED API is accessible",
            }
        except httpx.TimeoutException:
            return {"status": "timeout", "message": "Request timed out"}
        except (httpx.HTTPError, OSError, ValueError) as e:
            return {"status": "error", "message": str(e)}

    async def get_series_metadata(self, series_id: str) -> SeriesMetadata | None:
        """Fetch metadata for a single series."""
        data = await self._request("series", params={"series_id": series_id})
        items = data.get("seriess") or data.get("series") or []
        if not items:
            return None
        raw = items[0]
        return SeriesMetadata(
            id=raw.get("id"),
            title=raw.get("title", ""),
            units=raw.get("units"),
            frequency=raw.get("frequency"),
            seasonalAdjustment=raw.get("seasonal_adjustment"),
            observation_start=raw.get("observation_start"),
            observation_end=raw.get("observation_end"),
            last_updated=raw.get("last_updated"),
            notes=raw.get("notes"),
        )

    async def search_series(
        self, query: str, limit: int = 25, order_by: str = "popularity"
    ) -> list[dict[str, Any]]:
        """Search series catalogue by keyword."""
        data = await self._request(
            "series/search",
            params={
                "search_text": query,
                "limit": max(1, min(limit, 1000)),
                "order_by": order_by,
            },
        )
        series = data.get("seriess") or []
        return [
            {
                "id": s.get("id"),
                "title": s.get("title"),
                "units": s.get("units"),
                "frequency": s.get("frequency"),
                "observation_start": s.get("observation_start"),
                "observation_end": s.get("observation_end"),
                "popularity": s.get("popularity"),
                "notes": s.get("notes"),
            }
            for s in series
        ]

    async def get_observations(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
        frequency: str | None = None,
        aggregation_method: str | None = None,
        limit: int = 1000,
    ) -> list[Observation]:
        """
        Fetch observations for a series.

        Args:
            series_id: FRED series ID (e.g., "PCOPPUSDM").
            observation_start: Start date (YYYY-MM-DD).
            observation_end: End date (YYYY-MM-DD).
            frequency: Optional aggregation target ("m","q","a", etc.)
            aggregation_method: avg | sum | eop (end of period).
            limit: Max observations (FRED max 100000).
        """
        params: dict[str, Any] = {"series_id": series_id, "limit": min(limit, 100000)}
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        if frequency:
            params["frequency"] = frequency
        if aggregation_method:
            params["aggregation_method"] = aggregation_method

        data = await self._request("series/observations", params=params)
        records: list[Observation] = []
        for obs in data.get("observations", []):
            raw_val = obs.get("value")
            try:
                val: float | None = float(raw_val) if raw_val not in (".", "", None) else None
            except (ValueError, TypeError):
                val = None
            records.append(
                Observation(
                    date=obs.get("date", ""),
                    value=val,
                    realtime_start=obs.get("realtime_start"),
                    realtime_end=obs.get("realtime_end"),
                )
            )
        return records

    async def list_categories(self, category_id: int = 0) -> list[dict[str, Any]]:
        """List children of a FRED category (0 = root)."""
        data = await self._request("category/children", params={"category_id": category_id})
        categories = data.get("categories", [])
        return cast(list[dict[str, Any]], categories)

    async def list_category_series(
        self, category_id: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List series under a specific category."""
        data = await self._request(
            "category/series",
            params={"category_id": category_id, "limit": min(limit, 1000)},
        )
        seriess = data.get("seriess", [])
        return cast(list[dict[str, Any]], seriess)
