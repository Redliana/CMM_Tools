"""Async HTTP client for UN Comtrade API."""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

from .models import CRITICAL_MINERAL_HS_CODES, TradeRecord

# Load environment variables
load_dotenv()


class ComtradeClient:
    """Client for UN Comtrade API v1."""

    BASE_URL = "https://comtradeapi.un.org"
    DATA_URL = f"{BASE_URL}/data/v1/get/C/A/HS"  # Commodities, Annual, HS classification
    REFS_URL = f"{BASE_URL}/files/v1/app/reference"

    def __init__(self, api_key: str | None = None):
        """Initialize the client with an API key."""
        self.api_key = api_key or os.getenv("UNCOMTRADE_API_KEY")
        self.timeout = 60.0

    def is_available(self) -> bool:
        """Check if the API key is configured."""
        return bool(self.api_key)

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with API key."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Ocp-Apim-Subscription-Key"] = self.api_key
        return headers

    async def _request(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make an async request to the API."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params, headers=self._get_headers())
            response.raise_for_status()
            return response.json()

    async def check_status(self) -> dict[str, Any]:
        """Check API connectivity and key validity."""
        try:
            # Try a minimal query to check connectivity
            params = {
                "reporterCode": "842",  # USA
                "period": "2023",
                "partnerCode": "0",  # World
                "cmdCode": "TOTAL",
                "flowCode": "M",
                "maxRecords": 1,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    self.DATA_URL, params=params, headers=self._get_headers()
                )
                if response.status_code == 200:
                    return {
                        "status": "connected",
                        "api_key_configured": self.is_available(),
                        "message": "UN Comtrade API is accessible",
                    }
                elif response.status_code == 401:
                    return {
                        "status": "unauthorized",
                        "api_key_configured": self.is_available(),
                        "message": "Invalid or missing API key",
                    }
                else:
                    return {
                        "status": "error",
                        "api_key_configured": self.is_available(),
                        "message": f"API returned status {response.status_code}",
                    }
        except httpx.TimeoutException:
            return {
                "status": "timeout",
                "api_key_configured": self.is_available(),
                "message": "Request timed out",
            }
        except (httpx.HTTPError, OSError) as e:
            return {
                "status": "error",
                "api_key_configured": self.is_available(),
                "message": str(e),
            }

    async def get_trade_data(
        self,
        reporter: str,
        partner: str = "0",
        commodity: str = "TOTAL",
        flow: str = "M",
        period: str = "2023",
        max_records: int = 500,
    ) -> list[TradeRecord]:
        """
        Get trade data from UN Comtrade.

        Args:
            reporter: Reporter country code (e.g., "842" for USA)
            partner: Partner country code or "0" for world
            commodity: HS commodity code (e.g., "2602" for manganese ores)
            flow: Trade flow - "M" (imports), "X" (exports), or "M,X" (both)
            period: Year(s) - single year or comma-separated
            max_records: Maximum records to return

        Returns:
            List of TradeRecord objects
        """
        params = {
            "reporterCode": reporter,
            "partnerCode": partner,
            "cmdCode": commodity,
            "flowCode": flow,
            "period": period,
            "maxRecords": max_records,
        }

        data = await self._request(self.DATA_URL, params)
        records = []

        for item in data.get("data", []):
            try:
                record = TradeRecord.model_validate(item)
                records.append(record)
            except (ValueError, KeyError):
                # Skip malformed records
                continue

        return records

    async def get_critical_mineral_trade(
        self,
        mineral: str,
        reporter: str = "0",
        partner: str = "0",
        flow: str = "M,X",
        period: str = "2023",
        max_records: int = 500,
    ) -> list[TradeRecord]:
        """
        Get trade data for a critical mineral using preset HS codes.

        Args:
            mineral: Mineral name (lithium, cobalt, rare_earth, graphite, etc.)
            reporter: Reporter country code or "0" for all
            partner: Partner country code or "0" for world
            flow: Trade flow - "M", "X", or "M,X"
            period: Year(s)
            max_records: Maximum records to return

        Returns:
            List of TradeRecord objects
        """
        mineral_lower = mineral.lower().replace(" ", "_")
        hs_codes = CRITICAL_MINERAL_HS_CODES.get(mineral_lower, [])

        if not hs_codes:
            available = ", ".join(CRITICAL_MINERAL_HS_CODES.keys())
            raise ValueError(f"Unknown mineral: {mineral}. Available: {available}")

        # Query with comma-separated HS codes
        commodity = ",".join(hs_codes)
        return await self.get_trade_data(
            reporter=reporter,
            partner=partner,
            commodity=commodity,
            flow=flow,
            period=period,
            max_records=max_records,
        )

    async def get_reporters(self) -> list[dict[str, Any]]:
        """Get list of available reporter countries."""
        url = f"{self.REFS_URL}/Reporters.json"
        data = await self._request(url)
        return data.get("results", [])

    async def get_partners(self) -> list[dict[str, Any]]:
        """Get list of available partner countries."""
        url = f"{self.REFS_URL}/partnerAreas.json"
        data = await self._request(url)
        return data.get("results", [])

    async def get_commodities(self, classification: str = "HS") -> list[dict[str, Any]]:
        """
        Get list of commodity codes.

        Args:
            classification: Trade classification (HS, S1, S2, etc.)

        Returns:
            List of commodity reference data
        """
        url = f"{self.REFS_URL}/{classification}.json"
        data = await self._request(url)
        return data.get("results", [])
