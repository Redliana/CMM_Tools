"""Pydantic models and reference data for USITC DataWeb.

USITC DataWeb v2 API reference: https://datawebws.usitc.gov/dataweb
(Production documentation is accessible after registering an account at
https://dataweb.usitc.gov and obtaining a bearer token.)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TradeRecord(BaseModel):
    """A single USITC trade-data record (aggregated import or export)."""

    year: int = Field(description="Calendar year")
    month: int | None = Field(default=None, description="Calendar month (1-12); None for annual")
    hts_code: str = Field(description="HTS commodity code (2-10 digits)")
    hts_description: str | None = Field(default=None, description="HTS description")
    country_code: str | None = Field(default=None, description="Partner country ISO/USITC code")
    country_name: str | None = Field(default=None, description="Partner country name")
    flow: str = Field(description="Trade flow: 'import' or 'export'")
    data_type: str = Field(
        description="Data type: 'general_imports', 'imports_for_consumption', "
        "'domestic_exports', 'total_exports'"
    )
    value_usd: float | None = Field(default=None, description="Customs value in USD")
    quantity_1: float | None = Field(default=None, description="First reported quantity")
    quantity_1_unit: str | None = Field(default=None, description="Units of first quantity")
    quantity_2: float | None = Field(default=None, description="Second reported quantity")
    quantity_2_unit: str | None = Field(default=None, description="Units of second quantity")


class HTSReference(BaseModel):
    """HTS (Harmonized Tariff Schedule of the United States) classification entry."""

    code: str = Field(description="HTS code (2/4/6/8/10 digits)")
    description: str = Field(description="HTS description")
    parent: str | None = Field(default=None, description="Parent HTS code")


# ── CMM-focused HTS codes ────────────────────────────────────────────────────
# US HTS codes for critical minerals. These extend the UN HS codes with US-specific
# 8-10 digit granularity where appropriate. Keep this list aligned with the
# uncomtrade-mcp HS codes but at US-HTS precision.
CMM_HTS_CODES: dict[str, list[str]] = {
    "lithium": [
        "2530.90.00",  # Other mineral substances
        "2825.20.00",  # Lithium oxide and hydroxide
        "2836.91.00",  # Lithium carbonates
        "8507.60.00",  # Lithium-ion batteries
    ],
    "cobalt": [
        "2605.00.00",  # Cobalt ores and concentrates
        "2822.00.00",  # Cobalt oxides and hydroxides
        "8105.20.30",  # Cobalt, unwrought; powders
        "8105.90.00",  # Cobalt articles
    ],
    "rare_earth": [
        "2805.30.00",  # Rare-earth metals
        "2846.10.00",  # Cerium compounds
        "2846.90.00",  # Other rare-earth compounds
    ],
    "graphite": [
        "2504.10.00",  # Natural graphite, powder/flakes
        "2504.90.00",  # Natural graphite, other
        "3801.10.00",  # Artificial graphite
    ],
    "nickel": [
        "2604.00.00",  # Nickel ores and concentrates
        "7501.10.00",  # Nickel mattes
        "7502.10.00",  # Unwrought nickel, not alloyed
        "2811.22.00",  # Nickel oxides
    ],
    "manganese": [
        "2602.00.00",  # Manganese ores and concentrates
        "8111.00.30",  # Manganese, unwrought
    ],
    "gallium": [
        "8112.92.00",  # Gallium, unwrought
    ],
    "germanium": [
        "8112.92.06",  # Germanium, unwrought (US-specific)
    ],
    "copper": [
        "7402.00.00",  # Unrefined copper; copper anodes
        "7403.11.00",  # Refined copper, cathodes
    ],
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
