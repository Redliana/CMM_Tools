"""Pydantic models for UN Comtrade API data structures."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TradeRecord(BaseModel):
    """A single trade record from UN Comtrade."""

    period: str = Field(description="Year of the trade record")
    reporter_code: int = Field(alias="reporterCode", description="Reporter country code")
    reporter: str | None = Field(
        alias="reporterDesc", default=None, description="Reporter country name"
    )
    partner_code: int = Field(alias="partnerCode", description="Partner country code")
    partner: str | None = Field(
        alias="partnerDesc", default=None, description="Partner country name"
    )
    flow_code: str = Field(alias="flowCode", description="Trade flow code (M/X)")
    flow: str | None = Field(alias="flowDesc", default=None, description="Trade flow description")
    commodity_code: str = Field(alias="cmdCode", description="HS commodity code")
    commodity: str | None = Field(
        alias="cmdDesc", default=None, description="Commodity description"
    )
    trade_value: float | None = Field(
        alias="primaryValue", default=None, description="Trade value in USD"
    )
    net_weight: float | None = Field(alias="netWgt", default=None, description="Net weight in kg")
    quantity: float | None = Field(
        alias="qty", default=None, description="Quantity in reported units"
    )
    quantity_unit: str | None = Field(
        alias="qtyUnitAbbr", default=None, description="Quantity unit abbreviation"
    )

    @property
    def reporter_name(self) -> str:
        """Get reporter name with fallback to code."""
        return self.reporter or f"Country {self.reporter_code}"

    @property
    def partner_name(self) -> str:
        """Get partner name with fallback to code."""
        if self.partner_code == 0:
            return "World"
        return self.partner or f"Country {self.partner_code}"

    class Config:
        populate_by_name = True


class CountryReference(BaseModel):
    """Country reference data."""

    id: int = Field(description="Country code")
    text: str = Field(description="Country name")
    iso3: str | None = Field(default=None, description="ISO 3-letter code")


class CommodityReference(BaseModel):
    """HS commodity code reference data."""

    id: str = Field(description="HS code")
    text: str = Field(description="Commodity description")
    parent: str | None = Field(default=None, description="Parent HS code")


# Critical Minerals HS Code Mapping (CMM Focus)
CRITICAL_MINERAL_HS_CODES: dict[str, list[str]] = {
    "lithium": [
        "253090",
        "282520",
        "283691",
        "850650",
    ],  # Ores, oxide/hydroxide, carbonate, batteries
    "cobalt": ["2605", "282200", "810520", "810590"],  # Ores, oxides, unwrought, articles
    "hree": ["284690"],  # Heavy REE compounds (Dy, Tb, etc.)
    "lree": ["284610"],  # Light REE compounds (Nd, Pr, etc.)
    "rare_earth": ["2846", "280530"],  # REE compounds, REE metals
    "graphite": ["250410", "250490", "380110"],  # Natural (amorphous, crystalline), artificial
    "nickel": [
        "2604",
        "7501",
        "750210",
        "750220",
        "281122",
    ],  # Ores, matte, unwrought, alloys, oxides
    "manganese": ["2602", "811100"],  # Manganese ores, unwrought
    "gallium": ["811292"],  # Gallium unwrought
    "germanium": ["811299"],  # Germanium (other base metals)
    "copper": ["7402", "7403"],  # Refined copper, unrefined copper
}

# Friendly names for display
MINERAL_NAMES: dict[str, str] = {
    "lithium": "Lithium (Li)",
    "cobalt": "Cobalt (Co)",
    "hree": "Heavy Rare Earth Elements",
    "lree": "Light Rare Earth Elements",
    "rare_earth": "Rare Earth Elements (all)",
    "graphite": "Graphite (Gr)",
    "nickel": "Nickel (Ni)",
    "manganese": "Manganese (Mn)",
    "gallium": "Gallium (Ga)",
    "germanium": "Germanium (Ge)",
    "copper": "Copper (Cu)",
}
