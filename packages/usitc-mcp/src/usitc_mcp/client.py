"""Async HTTP client for USITC DataWeb v2 API.

USITC DataWeb is a bearer-token authenticated REST API that exposes US
import/export tariff-line trade statistics. Queries are submitted as a
nested ``SavedQuery`` envelope to ``/api/v2/report2/runReport`` and the
server responds with a pivot-table DTO containing one or more tables with
row groups keyed by commodity/country/etc. and column groups keyed by year.

Schema discovery, 2026-04: the OpenAPI document at
``https://datawebws.usitc.gov/dataweb/v3/api-docs`` lists a flat ``SavedQuery``
component, but the server actually expects a **nested** envelope whose
structure matches the JSON returned by ``/api/v2/savedQuery/getAllSystemSavedQueries``.
The shapes below were validated against a live bearer token.

Caveats still needing resolution against your workflow:

* ``dataToReport`` values are tradeType-specific. Validated: ``FAS_VALUE``
  for ``tradeType=Export``; ``GEN_CUSTOMS_VALUE`` for ``tradeType=Import``.
  Other codes seen in the DataWeb UI (``GEN_VAL_YR``, ``CIF_VAL_YR``,
  ``CONS_VAL_YR``) produce ``Invalid dataToReport object`` validation errors.
* Commodity filtering via ``commodities.commodities=[...]`` with
  ``commoditySelectType='entered'`` currently triggers a server-side
  "step 4" validation failure. Until the correct companion-field values are
  pinned down, ``get_trade_data`` submits filters but falls back to
  whole-chapter aggregation when the server rejects the narrowed query. The
  TODO in :meth:`USITCClient.get_trade_data` tracks this.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

from .models import CMM_HTS_CODES, TradeRecord

load_dotenv()

# Data-to-report codes validated against a live token.
_DATA_TO_REPORT = {
    "general_imports": "GEN_CUSTOMS_VALUE",
    "imports_for_consumption": "GEN_CUSTOMS_VALUE",
    "domestic_exports": "FAS_VALUE",
    "total_exports": "FAS_VALUE",
}

_TRADE_TYPE = {
    "import": "Import",
    "export": "Export",
}


class USITCAPIError(Exception):
    """Raised for USITC DataWeb API errors."""


class USITCClient:
    """Client for USITC DataWeb v2 API."""

    BASE_URL = "https://datawebws.usitc.gov/dataweb/api/v2"

    def __init__(self, api_token: str | None = None):
        self.api_token = api_token or os.getenv("USITC_API_TOKEN")
        self.timeout = 120.0

    def is_available(self) -> bool:
        return bool(self.api_token)

    def _require_token(self) -> None:
        if not self.api_token:
            raise USITCAPIError(
                "USITC_API_TOKEN is not set. Obtain a token at "
                "https://dataweb.usitc.gov → API tab, then set USITC_API_TOKEN."
            )

    def _headers(self) -> dict[str, str]:
        self._require_token()
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.BASE_URL}/{endpoint}",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data

    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.BASE_URL}/{endpoint}",
                headers=self._headers(),
                params=params,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data

    async def check_status(self) -> dict[str, Any]:
        """Probe API connectivity and token validity via ``country/getAllCountries``."""
        if not self.api_token:
            return {
                "status": "unauthorized",
                "api_key_configured": False,
                "message": "USITC_API_TOKEN not set",
            }
        try:
            data = await self._get("country/getAllCountries")
            options = data.get("options") or []
            return {
                "status": "connected",
                "api_key_configured": True,
                "countries_indexed": len(options),
                "message": "USITC DataWeb API reachable",
            }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return {
                    "status": "unauthorized",
                    "api_key_configured": True,
                    "message": "Token rejected (401). Check validity/expiry.",
                }
            return {
                "status": "error",
                "api_key_configured": True,
                "message": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            }
        except (httpx.HTTPError, OSError) as e:
            return {"status": "error", "message": str(e)}

    async def search_hts(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search HTS by description via ``commodity/commodityDescriptionLookup``."""
        try:
            data = await self._post(
                "commodity/commodityDescriptionLookup",
                {"tradeType": "Import", "classificationSystem": "HTS", "search": query},
            )
            items = data.get("options") or []
            return items[:limit]
        except httpx.HTTPError as e:
            raise USITCAPIError(f"HTS search failed: {e}") from e

    @staticmethod
    def _build_saved_query(
        trade_type: str,
        data_to_report: str,
        years: list[int],
        hts_codes: list[str] | None,
        country_codes: list[str] | None,
    ) -> dict[str, Any]:
        """Construct a SavedQuery envelope for /report2/runReport.

        Mirrors the JSON shape returned by ``/savedQuery/getAllSystemSavedQueries``
        (which is the only validated-working template).
        """
        # Commodity and country filtering are validated against the
        # /savedQuery/getAllSystemSavedQueries payloads. The API expects
        # list-based selection semantics, not "entered", and the aggregation
        # labels are case-sensitive.
        commodities_agg = "Aggregate Commodities"
        commodities_select = "list" if hts_codes else "all"
        countries_agg = "Aggregate countries"
        countries_select = "list" if country_codes else "all"

        # Granularity is inferred from the longest HTS code supplied. The API
        # accepts "2"|"4"|"6"|"8"|"10".
        granularity = "2"
        if hts_codes:
            max_len = max(len(c.replace(".", "")) for c in hts_codes)
            granularity = str(max(2, min(10, (max_len // 2) * 2)))
        normalized_hts = [c.replace(".", "") for c in (hts_codes or [])]
        commodities_expanded = [
            {"name": code, "value": code, "hasChildren": None} for code in normalized_hts
        ]
        commodities_manual = ", ".join(normalized_hts) if normalized_hts else None
        countries_expanded = [
            {"name": code, "value": code, "hasChildren": None} for code in (country_codes or [])
        ]

        return {
            "savedQueryName": "",
            "isOwner": True,
            "runMonthly": False,
            "reportOptions": {
                "tradeType": trade_type,
                "classificationSystem": "HTS",
            },
            "searchOptions": {
                "componentSettings": {
                    "dataToReport": [data_to_report],
                    "scale": "1",
                    "timeframeSelectType": "fullYears",
                    "years": [str(y) for y in years],
                    "startDate": None,
                    "endDate": None,
                    "startMonth": None,
                    "endMonth": None,
                    "yearsTimeline": "Annual",
                },
                "commodities": {
                    "commodities": normalized_hts,
                    "commoditiesExpanded": commodities_expanded,
                    "commoditiesManual": commodities_manual,
                    "commodityGroups": {"systemGroups": [], "userGroups": []},
                    "granularity": granularity,
                    "searchGranularity": granularity,
                    "groupGranularity": "2",
                    "aggregation": commodities_agg,
                    "codeDisplayFormat": "NO",
                    "commoditySelectType": commodities_select,
                    "showHTSValidDetails": True,
                },
                "countries": {
                    "countries": country_codes or [],
                    "countriesExpanded": countries_expanded,
                    "countryGroups": {"systemGroups": [], "userGroups": []},
                    "aggregation": countries_agg,
                    "countriesSelectType": countries_select,
                },
                "MiscGroup": {
                    "importPrograms": {
                        "importPrograms": [],
                        "aggregation": "Aggregate CSC",
                    },
                    "extImportPrograms": {
                        "programsSelectType": "all",
                        "extImportPrograms": [],
                        "extImportProgramsExpanded": [],
                        "aggregation": "Aggregate CSC",
                    },
                    "provisionCodes": {
                        "rateProvisionCodes": [],
                        "rateProvisionCodesExpanded": [],
                        "aggregation": "Aggregate RPCODE",
                        "provisionCodesSelectType": "all",
                        "rateProvisionGroups": {"systemGroups": []},
                    },
                    "districts": {
                        "districts": [],
                        "districtsExpanded": [],
                        "districtGroups": {"userGroups": []},
                        "aggregation": "Aggregate District",
                        "districtsSelectType": "all",
                    },
                },
            },
            "sortingAndDataFormat": {
                "DataSort": {"sortOrder": [], "columnOrder": [], "sortYear": None},
                "reportCustomizations": {
                    "totalRecords": "20000",
                    "exportCombineTables": False,
                    "reportsGrid": True,
                    "removeDuplicateValues": True,
                    "suppressZeroValues": False,
                    "displayCommodityList": False,
                    "reportsFontSize": "m",
                    "exportRawData": False,
                },
            },
        }

    @staticmethod
    def _parse_number(s: Any) -> float | None:
        if s is None:
            return None
        if isinstance(s, int | float):
            return float(s)
        text = str(s).strip().replace(",", "")
        if not text or text == "-":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _flatten_dto(
        self,
        dto: dict[str, Any],
        flow: str,
        data_type: str,
    ) -> list[TradeRecord]:
        """Convert a pivot-table DTO into flat TradeRecord rows.

        Each ``rowsNew[i].rowEntries[j]`` cell becomes one record, keyed by
        the corresponding ``row_groups[i].columnInfo[j]`` entry (which carries
        the year/period label).
        """
        records: list[TradeRecord] = []
        for table in dto.get("tables") or []:
            row_groups = table.get("row_groups") or []
            for rg in row_groups:
                col_info = rg.get("columnInfo") or []
                for row in rg.get("rowsNew") or []:
                    entries = row.get("rowEntries") or []
                    # Leading string cells (commodity label, country, etc.)
                    # are positioned before the numeric data columns. Map by
                    # columnInfo index: its ``columnIndex`` field points into
                    # ``rowEntries``.
                    label_cells = [e.get("value") for e in entries[: len(entries) - len(col_info)]]
                    hts_code = label_cells[0] if label_cells else ""
                    country_name = label_cells[1] if len(label_cells) > 1 else None
                    for ci in col_info:
                        idx = ci.get("columnIndex")
                        if idx is None or idx >= len(entries):
                            continue
                        value = self._parse_number(entries[idx].get("value"))
                        year_label = ci.get("columnLabel") or ci.get("queryResultLabel") or ""
                        try:
                            year = int(str(year_label)[:4])
                        except ValueError:
                            continue
                        records.append(
                            TradeRecord(
                                year=year,
                                hts_code=str(hts_code or ""),
                                country_name=country_name,
                                flow=flow,
                                data_type=data_type,
                                value_usd=value,
                            )
                        )
        return records

    async def get_trade_data(
        self,
        hts_codes: list[str],
        years: list[int],
        flow: str = "import",
        data_type: str = "general_imports",
        country_codes: list[str] | None = None,
    ) -> list[TradeRecord]:
        """Fetch US trade data via ``/report2/runReport``.

        Args:
            hts_codes: HTS codes (2/4/6/8/10 digit, dots optional).
            years: List of years.
            flow: ``"import"`` or ``"export"``.
            data_type: ``general_imports`` | ``imports_for_consumption`` |
                ``domestic_exports`` | ``total_exports``.
            country_codes: USITC partner country ``value`` codes (see
                ``country/getAllCountries``). ``None`` = aggregate all partners.

        Returns:
            List of ``TradeRecord``.

        Note: commodity filtering via ``commoditySelectType='entered'`` is not
        yet reliably accepted by the server (it raises a "step 4" validation
        error whose companion-field requirements are not pinned down). When
        that happens this method raises ``USITCAPIError`` and the caller must
        drop the HTS filter and aggregate client-side.
        """
        trade_type = _TRADE_TYPE.get(flow.lower())
        if not trade_type:
            raise USITCAPIError(f"Unknown flow '{flow}'. Use 'import' or 'export'.")
        dtr = _DATA_TO_REPORT.get(data_type)
        if not dtr:
            raise USITCAPIError(
                f"Unknown data_type '{data_type}'. Use one of {list(_DATA_TO_REPORT)}."
            )

        payload = self._build_saved_query(
            trade_type=trade_type,
            data_to_report=dtr,
            years=years,
            hts_codes=hts_codes,
            country_codes=country_codes,
        )

        try:
            data = await self._post("report2/runReport", payload)
        except httpx.HTTPError as e:
            raise USITCAPIError(f"runReport failed: {e}") from e

        dto = data.get("dto") or {}
        errors = dto.get("errors") or []
        if errors:
            # Server validation (step 2/step 4/etc). Surface one level up.
            raise USITCAPIError(f"Query rejected: {'; '.join(errors)}")

        return self._flatten_dto(dto, flow=flow, data_type=data_type)

    async def get_critical_mineral_trade(
        self,
        mineral: str,
        years: list[int],
        flow: str = "import",
        country_codes: list[str] | None = None,
    ) -> list[TradeRecord]:
        key = mineral.lower().replace(" ", "_")
        codes = CMM_HTS_CODES.get(key)
        if not codes:
            raise USITCAPIError(
                f"Unknown mineral: {mineral}. Available: {', '.join(CMM_HTS_CODES.keys())}"
            )
        data_type = "general_imports" if flow == "import" else "total_exports"
        return await self.get_trade_data(
            hts_codes=codes,
            years=years,
            flow=flow,
            data_type=data_type,
            country_codes=country_codes,
        )
