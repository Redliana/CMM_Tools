"""Pydantic models and reference data for World Bank WDI / WITS."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndicatorObservation(BaseModel):
    """A single WDI indicator observation for a country-year."""

    country_code: str = Field(description="ISO3 country code")
    country_name: str | None = Field(default=None, description="Country name")
    indicator_code: str = Field(description="WDI indicator code")
    indicator_name: str | None = Field(default=None, description="Indicator description")
    year: int = Field(description="Observation year")
    value: float | None = Field(default=None, description="Observation value (None if missing)")
    unit: str | None = Field(default=None, description="Unit of measurement where available")

    class Config:
        populate_by_name = True


class IndicatorMetadata(BaseModel):
    """Metadata record for a WDI indicator."""

    id: str = Field(description="Indicator code (e.g., NY.GDP.MKTP.CD)")
    name: str = Field(description="Indicator name")
    source_note: str | None = Field(default=None, description="Description / source note")
    topic: str | None = Field(default=None, description="Primary topic classification")


class CountryReference(BaseModel):
    """World Bank country reference entry."""

    iso3: str = Field(description="ISO 3-letter code")
    iso2: str | None = Field(default=None, description="ISO 2-letter code")
    name: str = Field(description="Country name")
    region: str | None = Field(default=None, description="Region")
    income_level: str | None = Field(default=None, description="Income classification")


# ── CMM-relevant WDI indicators ──────────────────────────────────────────────
# Curated set of indicators useful for critical-minerals supply-chain analysis.
CMM_WDI_INDICATORS: dict[str, str] = {
    # Macro & trade context
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "NE.EXP.GNFS.CD": "Exports of goods and services (current US$)",
    "NE.IMP.GNFS.CD": "Imports of goods and services (current US$)",
    "BX.KLT.DINV.CD.WD": "Foreign direct investment, net inflows (BoP, current US$)",
    # Manufacturing & industry
    "NV.IND.MANF.ZS": "Manufacturing, value added (% of GDP)",
    "NV.IND.TOTL.ZS": "Industry (including construction), value added (% of GDP)",
    # Natural resources
    "NY.GDP.TOTL.RT.ZS": "Total natural resources rents (% of GDP)",
    "NY.GDP.MINR.RT.ZS": "Mineral rents (% of GDP)",
    "TX.VAL.MRCH.CD.WT": "Merchandise exports (current US$)",
    "TM.VAL.MRCH.CD.WT": "Merchandise imports (current US$)",
    # Energy & transition
    "EG.USE.ELEC.KH.PC": "Electric power consumption (kWh per capita)",
    "EG.ELC.RNEW.ZS": "Renewable electricity output (% of total)",
    # Governance / risk proxies
    "IQ.CPA.ECON.XQ": "CPIA economic management cluster avg (1=low to 6=high)",
    "GE.EST": "Government Effectiveness: Estimate (WGI)",
    "RL.EST": "Rule of Law: Estimate (WGI)",
}


# ISO3 codes for major CMM-relevant economies
CMM_KEY_ECONOMIES: dict[str, str] = {
    "USA": "United States",
    "CHN": "China",
    "AUS": "Australia",
    "CHL": "Chile",
    "COD": "Democratic Republic of Congo",
    "ARG": "Argentina",
    "BOL": "Bolivia",
    "IDN": "Indonesia",
    "PHL": "Philippines",
    "ZAF": "South Africa",
    "CAN": "Canada",
    "MEX": "Mexico",
    "RUS": "Russia",
    "KAZ": "Kazakhstan",
    "MMR": "Myanmar",
    "VNM": "Vietnam",
    "JPN": "Japan",
    "KOR": "South Korea",
    "DEU": "Germany",
    "BRA": "Brazil",
}
