"""Pydantic models and reference data for Census International Trade (AES-derived) API.

Scope: US Census Bureau's International Trade time-series API, which publishes
export records originating from the Automated Export System (AES) and import
records from Customs entry filings. This is the publicly queryable face of AES
data; the AES filing system itself is not a query interface.

API reference: https://www.census.gov/foreign-trade/reference/guides/Guide%20to%20International%20Trade%20Datasets.pdf
Endpoint base: https://api.census.gov/data/timeseries/intltrade/
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExportRecord(BaseModel):
    """Single monthly export observation (AES-derived) at HS granularity."""

    year: int = Field(description="Calendar year")
    month: int = Field(description="Calendar month (1-12)")
    hs_code: str = Field(description="HS commodity code (2/4/6/10-digit)")
    hs_description: str | None = Field(default=None, description="HS description")
    country_code: str | None = Field(default=None, description="Destination country Schedule C code")
    country_name: str | None = Field(default=None, description="Destination country name")
    value_usd: float | None = Field(default=None, description="Free-alongside-ship (FAS) value USD")
    quantity_1: float | None = Field(default=None, description="First quantity measure")
    quantity_1_unit: str | None = Field(default=None, description="First quantity unit")
    quantity_2: float | None = Field(default=None, description="Second quantity measure")
    quantity_2_unit: str | None = Field(default=None, description="Second quantity unit")


class ImportRecord(BaseModel):
    """Single monthly import observation at HS granularity."""

    year: int = Field(description="Calendar year")
    month: int = Field(description="Calendar month (1-12)")
    hs_code: str = Field(description="HS commodity code")
    hs_description: str | None = Field(default=None, description="HS description")
    country_code: str | None = Field(default=None, description="Source country Schedule C code")
    country_name: str | None = Field(default=None, description="Source country name")
    value_usd: float | None = Field(
        default=None, description="General imports customs value USD"
    )
    value_cif_usd: float | None = Field(
        default=None, description="CIF value USD where available"
    )
    quantity_1: float | None = Field(default=None, description="First quantity measure")
    quantity_1_unit: str | None = Field(default=None, description="First quantity unit")


# ── CMM-focused HS codes for Census (uses 6-digit HS matching UN Comtrade) ───
# Same HS codes as uncomtrade-mcp for cross-source reconciliation.
CMM_HS_CODES: dict[str, list[str]] = {
    "lithium": ["253090", "282520", "283691", "850650"],
    "cobalt": ["260500", "282200", "810520", "810590"],
    "rare_earth": ["280530", "284610", "284690"],
    "graphite": ["250410", "250490", "380110"],
    "nickel": ["260400", "750110", "750210", "281122"],
    "manganese": ["260200", "811100"],
    "gallium": ["811292"],
    "germanium": ["811299"],
    "copper": ["740200", "740311"],
}


MINERAL_NAMES: dict[str, str] = {
    "lithium": "Lithium (Li)",
    "cobalt": "Cobalt (Co)",
    "rare_earth": "Rare Earth Elements (all)",
    "graphite": "Graphite (Gr)",
    "nickel": "Nickel (Ni)",
    "manganese": "Manganese (Mn)",
    "gallium": "Gallium (Ga)",
    "germanium": "Germanium (Ge)",
    "copper": "Copper (Cu)",
}
