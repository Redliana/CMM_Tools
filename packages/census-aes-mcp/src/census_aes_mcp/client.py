"""Async HTTP client for Census International Trade API (AES-derived).

The Census International Trade time-series API publishes monthly trade data at
HS classification granularity. Export records are derived from the Automated
Export System (AES); import records are derived from Customs entry filings.

Response format: Census returns a JSON array-of-arrays, first row is column
headers. We normalize to list[dict].

API reference: https://api.census.gov/data/timeseries/intltrade.html
Key signup: https://api.census.gov/data/key_signup.html
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

from .models import CMM_HS_CODES, ExportRecord, ImportRecord

load_dotenv()


class CensusAPIError(Exception):
    """Raised for Census API errors."""


class CensusClient:
    """Client for Census International Trade time-series API."""

    BASE_URL = "https://api.census.gov/data/timeseries/intltrade"

    # Default export variable set (HS classification)
    DEFAULT_EXPORT_VARS = [
        "E_COMMODITY",
        "E_COMMODITY_LDESC",
        "ALL_VAL_MO",
        "ALL_VAL_YR",
        "CTY_CODE",
        "CTY_NAME",
        "UNIT_QY1",
        "QTY_1_MO",
        "UNIT_QY2",
        "QTY_2_MO",
    ]

    # Default import variable set (HS classification)
    DEFAULT_IMPORT_VARS = [
        "I_COMMODITY",
        "I_COMMODITY_LDESC",
        "GEN_VAL_MO",
        "GEN_VAL_YR",
        "CON_VAL_MO",
        "CIF_VAL_MO",
        "CTY_CODE",
        "CTY_NAME",
        "UNIT_QY1",
        "GEN_QY1_MO",
    ]

    def __init__(self, api_key: str | None = None):
        """Initialize client; api_key may be passed explicitly or via CENSUS_API_KEY."""
        self.api_key = api_key or os.getenv("CENSUS_API_KEY")
        self.timeout = 90.0

    def is_available(self) -> bool:
        """Return True if an API key is configured."""
        return bool(self.api_key)

    def _require_key(self) -> None:
        if not self.api_key:
            raise CensusAPIError(
                "CENSUS_API_KEY is not set. Register at "
                "https://api.census.gov/data/key_signup.html and export CENSUS_API_KEY."
            )

    async def _get(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> list[list[Any]]:
        """GET against Census API; returns raw array-of-arrays."""
        self._require_key()
        query = dict(params or {})
        query["key"] = self.api_key

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.BASE_URL}/{endpoint}", params=query)
            if response.status_code == 204:
                return []
            response.raise_for_status()
            try:
                data: list[list[Any]] = response.json()
            except ValueError as e:
                raise CensusAPIError(
                    f"Non-JSON response from Census API: {response.text[:300]}"
                ) from e
            return data

    @staticmethod
    def _rows_to_dicts(data: list[list[Any]]) -> list[dict[str, Any]]:
        """Convert Census array-of-arrays (header + rows) to list[dict]."""
        if not data or len(data) < 2:
            return []
        header = [str(c) for c in data[0]]
        return [dict(zip(header, row)) for row in data[1:]]

    async def check_status(self) -> dict[str, Any]:
        """Check Census API connectivity and key validity."""
        if not self.api_key:
            return {
                "status": "unauthorized",
                "api_key_configured": False,
                "message": "CENSUS_API_KEY not set",
            }
        try:
            # Minimal query: one month of lithium carbonate exports
            data = await self._get(
                "exports/hs",
                params={
                    "get": "E_COMMODITY,E_COMMODITY_LDESC,ALL_VAL_MO",
                    "E_COMMODITY": "283691",
                    "YEAR": "2023",
                    "MONTH": "12",
                    "SUMMARY_LVL": "DET",
                },
            )
            return {
                "status": "connected",
                "api_key_configured": True,
                "sample_rows": max(0, len(data) - 1),
                "message": "Census International Trade API reachable",
            }
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                return {
                    "status": "unauthorized",
                    "api_key_configured": True,
                    "message": "API key rejected",
                }
            return {
                "status": "error",
                "api_key_configured": True,
                "message": f"HTTP {e.response.status_code}",
            }
        except (httpx.HTTPError, CensusAPIError, OSError) as e:
            return {"status": "error", "message": str(e)}

    async def get_exports(
        self,
        hs_codes: list[str],
        year: int,
        month: str | int | None = None,
        country_codes: list[str] | None = None,
        summary_level: str = "DET",
    ) -> list[ExportRecord]:
        """Fetch US export records (AES-derived) for given HS codes.

        Args:
            hs_codes: List of HS codes (6 or 10 digit).
            year: Calendar year.
            month: "01"-"12" or int 1-12; None for all months (YTD).
            country_codes: Optional Schedule C country codes.
            summary_level: "DET" (detailed) or "CTY" (country-level).
        """
        params: dict[str, Any] = {
            "get": ",".join(self.DEFAULT_EXPORT_VARS),
            "YEAR": str(year),
            "E_COMMODITY": ",".join(hs_codes),
            "SUMMARY_LVL": summary_level,
        }
        if month is not None:
            params["MONTH"] = f"{int(month):02d}"
        if country_codes:
            params["CTY_CODE"] = ",".join(country_codes)

        raw = await self._get("exports/hs", params=params)
        rows = self._rows_to_dicts(raw)
        records: list[ExportRecord] = []
        month_fallback = int(month) if month is not None else 0
        for row in rows:
            try:
                records.append(
                    ExportRecord(
                        year=int(row.get("YEAR") or year),
                        month=int(row.get("MONTH") or month_fallback),
                        hs_code=str(row.get("E_COMMODITY", "")),
                        hs_description=row.get("E_COMMODITY_LDESC"),
                        country_code=row.get("CTY_CODE"),
                        country_name=row.get("CTY_NAME"),
                        value_usd=_safe_float(row.get("ALL_VAL_MO")),
                        quantity_1=_safe_float(row.get("QTY_1_MO")),
                        quantity_1_unit=row.get("UNIT_QY1"),
                        quantity_2=_safe_float(row.get("QTY_2_MO")),
                        quantity_2_unit=row.get("UNIT_QY2"),
                    )
                )
            except (ValueError, TypeError):
                continue
        return records

    async def get_imports(
        self,
        hs_codes: list[str],
        year: int,
        month: str | int | None = None,
        country_codes: list[str] | None = None,
        summary_level: str = "DET",
    ) -> list[ImportRecord]:
        """Fetch US import records for given HS codes.

        Args:
            hs_codes: List of HS codes (6 or 10 digit).
            year: Calendar year.
            month: Optional month.
            country_codes: Optional Schedule C country codes.
            summary_level: "DET" or "CTY".
        """
        params: dict[str, Any] = {
            "get": ",".join(self.DEFAULT_IMPORT_VARS),
            "YEAR": str(year),
            "I_COMMODITY": ",".join(hs_codes),
            "SUMMARY_LVL": summary_level,
        }
        if month is not None:
            params["MONTH"] = f"{int(month):02d}"
        if country_codes:
            params["CTY_CODE"] = ",".join(country_codes)

        raw = await self._get("imports/hs", params=params)
        rows = self._rows_to_dicts(raw)
        records: list[ImportRecord] = []
        month_fallback = int(month) if month is not None else 0
        for row in rows:
            try:
                records.append(
                    ImportRecord(
                        year=int(row.get("YEAR") or year),
                        month=int(row.get("MONTH") or month_fallback),
                        hs_code=str(row.get("I_COMMODITY", "")),
                        hs_description=row.get("I_COMMODITY_LDESC"),
                        country_code=row.get("CTY_CODE"),
                        country_name=row.get("CTY_NAME"),
                        value_usd=_safe_float(row.get("GEN_VAL_MO")),
                        value_cif_usd=_safe_float(row.get("CIF_VAL_MO")),
                        quantity_1=_safe_float(row.get("GEN_QY1_MO")),
                        quantity_1_unit=row.get("UNIT_QY1"),
                    )
                )
            except (ValueError, TypeError):
                continue
        return records

    async def get_critical_mineral_trade(
        self,
        mineral: str,
        year: int,
        month: str | int | None = None,
        flow: str = "export",
        country_codes: list[str] | None = None,
    ) -> list[ExportRecord] | list[ImportRecord]:
        """Fetch US trade data for a critical mineral using curated HS codes."""
        key = mineral.lower().replace(" ", "_")
        codes = CMM_HS_CODES.get(key)
        if not codes:
            raise CensusAPIError(
                f"Unknown mineral: {mineral}. Available: {', '.join(CMM_HS_CODES.keys())}"
            )
        if flow == "export":
            return await self.get_exports(codes, year, month=month, country_codes=country_codes)
        return await self.get_imports(codes, year, month=month, country_codes=country_codes)


def _safe_float(value: Any) -> float | None:
    """Convert Census string values to float, returning None for missing markers."""
    if value is None or value in ("", ".", "N/A", "null"):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
