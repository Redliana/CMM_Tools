"""Pydantic models and curated series lists for FRED."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SeriesMetadata(BaseModel):
    """Metadata for a FRED time series."""

    id: str = Field(description="FRED series ID (e.g., 'GDPC1')")
    title: str = Field(description="Series title")
    units: str | None = Field(default=None, description="Units of measurement")
    frequency: str | None = Field(default=None, description="Reporting frequency")
    seasonal_adjustment: str | None = Field(
        default=None, alias="seasonalAdjustment", description="Seasonal adjustment code"
    )
    observation_start: str | None = Field(
        default=None, alias="observation_start", description="First observation date"
    )
    observation_end: str | None = Field(
        default=None, alias="observation_end", description="Latest observation date"
    )
    last_updated: str | None = Field(
        default=None, alias="last_updated", description="Last revision timestamp"
    )
    notes: str | None = Field(default=None, description="Source/notes")

    class Config:
        populate_by_name = True


class Observation(BaseModel):
    """A single observation from a FRED series."""

    date: str = Field(description="Observation date (YYYY-MM-DD)")
    value: float | None = Field(default=None, description="Observation value; None if missing")
    realtime_start: str | None = Field(default=None, description="Real-time start date")
    realtime_end: str | None = Field(default=None, description="Real-time end date")


# ── CMM-relevant curated series ──────────────────────────────────────────────
# Series IDs that support critical-minerals supply-chain analysis: commodity
# price indices, industrial production, manufacturing PMI proxies, import/export
# price indices, and freight/shipping cost proxies.
CMM_FRED_SERIES: dict[str, dict[str, str]] = {
    # Commodity price indices (Global Commodity Price Index family - PALLFNFINDEXM)
    "PALLFNFINDEXM": {
        "title": "Global Price Index of All Commodities",
        "category": "commodity_prices",
    },
    "PMETAINDEXM": {
        "title": "Global Price Index of Industrial Materials (Metals)",
        "category": "commodity_prices",
    },
    "PCOPPUSDM": {
        "title": "Global Price of Copper (USD/metric ton)",
        "category": "commodity_prices",
    },
    "PNICKUSDM": {
        "title": "Global Price of Nickel (USD/metric ton)",
        "category": "commodity_prices",
    },
    "PALUMINUMUSDM": {
        "title": "Global Price of Aluminum (USD/metric ton)",
        "category": "commodity_prices",
    },
    "PIORECRUSDM": {
        "title": "Global Price of Iron Ore (USD/metric ton)",
        "category": "commodity_prices",
    },
    # Industrial production
    "INDPRO": {
        "title": "Industrial Production: Total Index",
        "category": "production",
    },
    "IPMINE": {
        "title": "Industrial Production: Mining",
        "category": "production",
    },
    "IPMANSICS": {
        "title": "Industrial Production: Manufacturing (SIC)",
        "category": "production",
    },
    # Trade / prices
    "IR": {
        "title": "Import Price Index (All Commodities)",
        "category": "trade_prices",
    },
    "IQ": {
        "title": "Export Price Index (All Commodities)",
        "category": "trade_prices",
    },
    # Freight / shipping proxies
    "WPU301": {
        "title": "Producer Price Index: Transportation Services",
        "category": "logistics",
    },
    # FX (for trade valuation)
    "DEXCHUS": {
        "title": "China / U.S. Foreign Exchange Rate (CNY per USD)",
        "category": "fx",
    },
    "DEXUSEU": {
        "title": "U.S. / Euro Foreign Exchange Rate (USD per EUR)",
        "category": "fx",
    },
    # Interest rates (capital cost for CMM projects)
    "DGS10": {
        "title": "10-Year Treasury Constant Maturity Rate",
        "category": "rates",
    },
}
