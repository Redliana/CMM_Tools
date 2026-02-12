"""Load and normalize heterogeneous CSVs from API_Scripts/."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from cmm_fine_tune.constants import COMMODITY_CONFIG, TRADE_DATA_DIR, USGS_DATA_DIR
from cmm_fine_tune.models import SalientRecord, TradeFlowRecord, WorldProductionRecord

logger = logging.getLogger(__name__)

# Values that should be treated as missing / non-numeric
_WITHHELD = {"W", "w", "XX", "--", "—", "NA", "N/A", ""}
_GT_PATTERN = re.compile(r"^[><]\s*(\d+\.?\d*)$")  # ">25", ">50", "<1"
_LESS_THAN_HALF = re.compile(r"(?i)less\s+than\s+1/2\s+unit")


def _parse_numeric(value: str | float | int | None) -> float | None:
    """Parse a potentially messy value into a float or None.

    Handles withheld markers ("W"), inequality markers (">25"), empty strings,
    and the special "Less than 1/2 unit" text.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().replace(",", "")
    if s in _WITHHELD:
        return None
    if _LESS_THAN_HALF.search(s):
        return None
    m = _GT_PATTERN.match(s)
    if m:
        return float(m.group(1))  # store the numeric part
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Trade data
# ---------------------------------------------------------------------------


def load_trade_data(csv_path: Path) -> list[TradeFlowRecord]:
    """Load a UN Comtrade trade CSV into normalized TradeFlowRecord objects."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    records: list[TradeFlowRecord] = []
    for _, row in df.iterrows():
        records.append(
            TradeFlowRecord(
                commodity=row.get("commodity_name", ""),
                hs_code=row.get("hs_code", row.get("cmdCode", "")),
                reporter_code=str(row.get("reporterCode", "")),
                reporter_desc=row.get("reporterDesc", ""),
                partner_code=str(row.get("partnerCode", "")),
                partner_desc=row.get("partnerDesc", ""),
                flow_code=row.get("flowCode", ""),
                year=int(row.get("query_year", row.get("refYear", 0))),
                primary_value=_parse_numeric(row.get("primaryValue")),
                net_weight=_parse_numeric(row.get("netWgt")),
                quantity=_parse_numeric(row.get("qty")),
                qty_unit=row.get("qtyUnitAbbr", ""),
            )
        )
    logger.info("Loaded %d trade records from %s", len(records), csv_path.name)
    return records


def load_all_trade_data() -> dict[str, list[TradeFlowRecord]]:
    """Load trade CSVs for all configured commodities."""
    result: dict[str, list[TradeFlowRecord]] = {}
    for key, cfg in COMMODITY_CONFIG.items():
        csv_path = TRADE_DATA_DIR / cfg["trade_csv"]
        if csv_path.exists():
            result[key] = load_trade_data(csv_path)
        else:
            logger.warning("Trade CSV not found: %s", csv_path)
    return result


# ---------------------------------------------------------------------------
# Salient statistics
# ---------------------------------------------------------------------------

# Columns that are always present and promoted to typed fields
_SALIENT_FIXED_COLS = {"DataSource", "Commodity", "Year"}


def load_salient_data(csv_path: Path, commodity_key: str = "") -> list[SalientRecord]:
    """Load a USGS MCS salient statistics CSV.

    Each commodity has a different schema, so variable columns are stored in
    ``fields`` as a flat dict.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    records: list[SalientRecord] = []
    variable_cols = [c for c in df.columns if c not in _SALIENT_FIXED_COLS]

    for _, row in df.iterrows():
        year_raw = str(row.get("Year", "")).strip()
        if not year_raw:
            continue  # skip trailing/empty rows
        try:
            year = int(float(year_raw))
        except (ValueError, TypeError):
            continue

        fields: dict[str, str | float | None] = {}
        for col in variable_cols:
            raw = row.get(col, "")
            num = _parse_numeric(raw)
            # Store numeric if parseable, otherwise the raw string (for "W", ">50", etc.)
            fields[col] = num if num is not None else (raw if raw else None)
        records.append(
            SalientRecord(
                data_source=row.get("DataSource", ""),
                commodity=row.get("Commodity", commodity_key),
                year=year,
                fields=fields,
            )
        )
    logger.info("Loaded %d salient records from %s", len(records), csv_path.name)
    return records


def load_all_salient_data() -> dict[str, list[SalientRecord]]:
    """Load salient CSVs for all configured commodities."""
    result: dict[str, list[SalientRecord]] = {}
    for key, cfg in COMMODITY_CONFIG.items():
        all_records: list[SalientRecord] = []
        for fname in cfg.get("salient_csvs", []):
            csv_path = USGS_DATA_DIR / fname
            if csv_path.exists():
                all_records.extend(load_salient_data(csv_path, commodity_key=key))
            else:
                logger.warning("Salient CSV not found: %s", csv_path)
        result[key] = all_records
    return result


# ---------------------------------------------------------------------------
# World production
# ---------------------------------------------------------------------------


def load_world_production_data(
    csv_path: Path, commodity_key: str = ""
) -> list[WorldProductionRecord]:
    """Load a USGS MCS world mine production CSV."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    records: list[WorldProductionRecord] = []

    # Detect production year columns dynamically (e.g. Prod_t_2021, Prod_t_est_2022)
    prod_cols = [c for c in df.columns if c.startswith("Prod_")]
    prod_col_1 = prod_cols[0] if len(prod_cols) > 0 else ""
    prod_col_2 = prod_cols[1] if len(prod_cols) > 1 else ""

    # Extract year labels from column names
    def _extract_year(col: str) -> str:
        m = re.search(r"(\d{4})", col)
        if not m:
            return ""
        year = m.group(1)
        return f"{year} (est.)" if "est" in col.lower() else year

    year1_label = _extract_year(prod_col_1)
    year2_label = _extract_year(prod_col_2)

    for _, row in df.iterrows():
        records.append(
            WorldProductionRecord(
                source=row.get("Source", ""),
                commodity=commodity_key or row.get("Commodity", ""),
                country=row.get("Country", ""),
                production_type=row.get("Type", ""),
                production_year1=_parse_numeric(row.get(prod_col_1)) if prod_col_1 else None,
                production_year1_label=year1_label,
                production_year2=_parse_numeric(row.get(prod_col_2)) if prod_col_2 else None,
                production_year2_label=year2_label,
                production_notes=row.get("Prod_notes", ""),
                reserves=_parse_numeric(row.get("Reserves_t")),
                reserves_notes=row.get("Reserves_notes", ""),
            )
        )
    logger.info("Loaded %d world production records from %s", len(records), csv_path.name)
    return records


def load_all_world_production_data() -> dict[str, list[WorldProductionRecord]]:
    """Load world production CSVs for all configured commodities."""
    result: dict[str, list[WorldProductionRecord]] = {}
    for key, cfg in COMMODITY_CONFIG.items():
        all_records: list[WorldProductionRecord] = []
        for fname in cfg.get("world_csvs", []):
            csv_path = USGS_DATA_DIR / fname
            if csv_path.exists():
                all_records.extend(load_world_production_data(csv_path, commodity_key=key))
            else:
                logger.warning("World production CSV not found: %s", csv_path)
        result[key] = all_records
    return result


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------


def load_all_data() -> dict:
    """Load all available data from both source directories.

    Returns a dict with keys ``trade``, ``salient``, ``world``, each mapping
    commodity key to a list of records.
    """
    return {
        "trade": load_all_trade_data(),
        "salient": load_all_salient_data(),
        "world": load_all_world_production_data(),
    }
