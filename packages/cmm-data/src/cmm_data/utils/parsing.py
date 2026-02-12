"""Value parsing utilities for CMM data."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def parse_numeric_value(value: Any) -> float | None:
    """
    Parse a numeric value from USGS data, handling special codes.

    Handles:
    - 'W' or 'w': Withheld (returns NaN)
    - '--' or '—': Not available (returns NaN)
    - 'NA', 'N/A', 'n.a.': Not applicable (returns NaN)
    - '>50', '<100': Range indicators (returns numeric part)
    - '1,000': Comma-separated numbers
    - '1.5e3': Scientific notation
    - Ranges like '100-200': Returns midpoint

    Args:
        value: Value to parse

    Returns:
        float or None if value cannot be parsed
    """
    if value is None or pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    # Convert to string and clean
    s = str(value).strip()

    # Handle special codes
    if s.upper() in ("W", "XX", "--", "—", "NA", "N/A", "N.A.", ""):
        return np.nan

    # Handle greater/less than
    if s.startswith(">") or s.startswith("<"):
        s = s[1:]

    # Handle range (return midpoint)
    if "-" in s and not s.startswith("-"):
        parts = s.split("-")
        if len(parts) == 2:
            try:
                low = float(parts[0].replace(",", ""))
                high = float(parts[1].replace(",", ""))
                return (low + high) / 2
            except ValueError:
                pass

    # Remove commas and try to parse
    s = s.replace(",", "")

    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_range(value: Any) -> tuple[float | None, float | None]:
    """
    Parse a range value, returning (low, high) bounds.

    Args:
        value: Value containing a range (e.g., '100-200', '>50', '<100')

    Returns:
        tuple of (low_bound, high_bound), with None for unbounded sides
    """
    if value is None or pd.isna(value):
        return (None, None)

    s = str(value).strip()

    # Greater than
    if s.startswith(">"):
        try:
            v = float(s[1:].replace(",", ""))
            return (v, None)
        except ValueError:
            return (None, None)

    # Less than
    if s.startswith("<"):
        try:
            v = float(s[1:].replace(",", ""))
            return (None, v)
        except ValueError:
            return (None, None)

    # Range
    if "-" in s and not s.startswith("-"):
        parts = s.split("-")
        if len(parts) == 2:
            try:
                low = float(parts[0].replace(",", ""))
                high = float(parts[1].replace(",", ""))
                return (low, high)
            except ValueError:
                pass

    # Single value
    try:
        v = float(s.replace(",", ""))
        return (v, v)
    except ValueError:
        return (None, None)


def clean_numeric_column(
    series: pd.Series, keep_original: bool = False
) -> pd.Series | pd.DataFrame:
    """
    Clean a pandas Series containing numeric values with special codes.

    Args:
        series: pandas Series to clean
        keep_original: If True, return DataFrame with original and cleaned columns

    Returns:
        Cleaned Series, or DataFrame with 'original' and 'cleaned' columns
    """
    cleaned = series.apply(parse_numeric_value)

    if keep_original:
        return pd.DataFrame({"original": series, "cleaned": cleaned})

    return cleaned


def standardize_country_name(name: str) -> str:
    """
    Standardize country names for consistent merging.

    Args:
        name: Country name to standardize

    Returns:
        Standardized country name
    """
    if pd.isna(name):
        return name

    # Common standardizations
    mappings = {
        "United States": "United States",
        "USA": "United States",
        "U.S.": "United States",
        "United States of America": "United States",
        "UK": "United Kingdom",
        "Great Britain": "United Kingdom",
        "People's Republic of China": "China",
        "China, People's Republic of": "China",
        "Republic of Korea": "South Korea",
        "Korea, Republic of": "South Korea",
        "Korea (South)": "South Korea",
        "Russian Federation": "Russia",
        "Congo (Kinshasa)": "Democratic Republic of the Congo",
        "DRC": "Democratic Republic of the Congo",
        "Czechia": "Czech Republic",
    }

    name_clean = str(name).strip()

    # Check direct mapping
    if name_clean in mappings:
        return mappings[name_clean]

    return name_clean


def extract_commodity_code(filename: str) -> str | None:
    """
    Extract commodity code from USGS filename.

    Args:
        filename: Filename like 'mcs2023-lithi_world.csv'

    Returns:
        Commodity code like 'lithi' or None
    """
    match = re.search(r"mcs\d{4}-(\w+)_(?:world|salient)", filename)
    if match:
        return match.group(1)
    return None
