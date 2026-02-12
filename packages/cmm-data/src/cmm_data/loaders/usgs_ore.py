"""USGS Ore Deposits database loader."""

from __future__ import annotations

import pandas as pd

from ..exceptions import DataNotFoundError
from .base import BaseLoader

# REE elements for filtering
REE_ELEMENTS = [
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Y",
    "Sc",
]


class USGSOreDepositsLoader(BaseLoader):
    """
    Loader for USGS Ore Deposits geochemistry database.

    Provides access to geochemical analyses from ore deposits worldwide,
    including 356 data fields and multiple tables.
    """

    dataset_name = "usgs_ore"

    # Table metadata
    TABLES = {
        "AnalyticMethod": "Analytical method descriptions",
        "AnalyticMethod_Biblio": "Analytical method bibliographic references",
        "BV_Ag_Mo": "Best values for Ag through Mo elements",
        "BV_Na_Zr": "Best values for Na through Zr elements",
        "ChemData1": "Chemical data part 1",
        "ChemData2": "Chemical data part 2",
        "DataDictionary": "Field definitions and metadata",
        "Geochem_BV": "Geochemistry best values summary",
        "Geology": "Geological context and deposit information",
        "LabName": "Laboratory names and codes",
        "Parameter": "Parameter definitions",
        "Reference": "Reference/citation data",
    }

    def list_available(self) -> list[str]:
        """List available tables in the database."""
        if not self.data_path.exists():
            return []

        return [f.stem for f in self.data_path.glob("*.csv")]

    def load(self, table: str = "Geology") -> pd.DataFrame:
        """
        Load a table from the ore deposits database.

        Args:
            table: Table name (e.g., 'Geology', 'BV_Ag_Mo', 'DataDictionary')

        Returns:
            pandas.DataFrame with table data
        """
        cache_key = self._cache_key("table", table)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._validate_path(self.data_path, "Ore deposits directory")

        file_path = self.data_path / f"{table}.csv"
        if not file_path.exists():
            # Try case-insensitive match
            for f in self.data_path.glob("*.csv"):
                if f.stem.lower() == table.lower():
                    file_path = f
                    break
            else:
                available = self.list_available()
                raise DataNotFoundError(f"Table '{table}' not found. Available: {available}")

        df = self._read_csv(file_path)

        self._set_cached(cache_key, df)
        return df

    def load_data_dictionary(self) -> pd.DataFrame:
        """
        Load the data dictionary with field definitions.

        Returns:
            DataFrame with field metadata
        """
        return self.load("DataDictionary")

    def load_geology(self) -> pd.DataFrame:
        """
        Load geology/deposit information.

        Returns:
            DataFrame with deposit locations and geological context
        """
        return self.load("Geology")

    def load_geochemistry(self, elements: list[str] | None = None) -> pd.DataFrame:
        """
        Load combined geochemistry data.

        Args:
            elements: Optional list of element symbols to include

        Returns:
            DataFrame with best-value geochemistry data
        """
        cache_key = self._cache_key("geochem", elements)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Load both BV tables
        df_ag_mo = self.load("BV_Ag_Mo")
        df_na_zr = self.load("BV_Na_Zr")

        # Merge on common key (likely SAMPLE_ID or similar)
        common_cols = list(set(df_ag_mo.columns) & set(df_na_zr.columns))
        if common_cols:
            key_col = common_cols[0]
            df = pd.merge(df_ag_mo, df_na_zr, on=key_col, how="outer", suffixes=("", "_dup"))
            # Remove duplicate columns
            df = df[[c for c in df.columns if not c.endswith("_dup")]]
        else:
            # Concatenate if no common key
            df = pd.concat([df_ag_mo, df_na_zr], axis=1)

        # Filter to specific elements if requested
        if elements:
            element_cols = []
            for col in df.columns:
                for elem in elements:
                    if col.startswith(f"{elem}_") or col == elem:
                        element_cols.append(col)
                        break
            # Always keep ID/key columns
            id_cols = [c for c in df.columns if "ID" in c.upper() or "SAMPLE" in c.upper()]
            keep_cols = list(set(id_cols + element_cols))
            df = df[keep_cols]

        self._set_cached(cache_key, df)
        return df

    def get_ree_samples(self) -> pd.DataFrame:
        """
        Get samples with REE (Rare Earth Element) data.

        Returns:
            DataFrame with REE geochemistry data
        """
        return self.load_geochemistry(elements=REE_ELEMENTS)

    def get_element_statistics(self, element: str) -> dict:
        """
        Get statistics for a specific element.

        Args:
            element: Element symbol (e.g., 'Li', 'Co', 'La')

        Returns:
            Dictionary with min, max, mean, median, count statistics
        """
        df = self.load_geochemistry(elements=[element])

        # Find the main concentration column
        conc_cols = [c for c in df.columns if element in c and ("ppm" in c or "pct" in c)]
        if not conc_cols:
            raise DataNotFoundError(f"No concentration data found for element: {element}")

        col = conc_cols[0]
        series = pd.to_numeric(df[col], errors="coerce")

        # Filter out below-detection values (negative)
        valid = series[series >= 0]

        return {
            "element": element,
            "column": col,
            "total_samples": len(series),
            "valid_samples": len(valid),
            "below_detection": len(series) - len(valid),
            "min": valid.min() if len(valid) > 0 else None,
            "max": valid.max() if len(valid) > 0 else None,
            "mean": valid.mean() if len(valid) > 0 else None,
            "median": valid.median() if len(valid) > 0 else None,
            "std": valid.std() if len(valid) > 0 else None,
        }

    def search_deposits(
        self,
        deposit_type: str | None = None,
        commodity: str | None = None,
        country: str | None = None,
    ) -> pd.DataFrame:
        """
        Search deposits by type, commodity, or country.

        Args:
            deposit_type: Deposit type to filter
            commodity: Commodity to filter
            country: Country to filter

        Returns:
            Filtered geology DataFrame
        """
        df = self.load_geology()

        if deposit_type:
            type_cols = [c for c in df.columns if "TYPE" in c.upper() or "CLASS" in c.upper()]
            for col in type_cols:
                mask = df[col].str.contains(deposit_type, case=False, na=False)
                df = df[mask]

        if commodity:
            comm_cols = [
                c for c in df.columns if "COMMODITY" in c.upper() or "MINERAL" in c.upper()
            ]
            for col in comm_cols:
                mask = df[col].str.contains(commodity, case=False, na=False)
                df = df[mask]

        if country:
            country_cols = [
                c for c in df.columns if "COUNTRY" in c.upper() or "NATION" in c.upper()
            ]
            for col in country_cols:
                mask = df[col].str.contains(country, case=False, na=False)
                df = df[mask]

        return df

    def describe(self) -> dict:
        """Describe the ore deposits dataset."""
        base = super().describe()
        base["tables"] = self.TABLES
        base["total_fields"] = 356

        # Try to get sample counts
        try:
            geology = self.load_geology()
            base["deposit_count"] = len(geology)
        except (OSError, ValueError):
            base["deposit_count"] = "Unknown"

        return base
