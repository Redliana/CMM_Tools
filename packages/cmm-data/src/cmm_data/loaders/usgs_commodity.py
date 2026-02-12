"""USGS Mineral Commodity Summaries data loader."""

from __future__ import annotations

import re

import pandas as pd

from ..exceptions import DataNotFoundError
from ..utils.parsing import clean_numeric_column, parse_numeric_value
from .base import BaseLoader

# Mapping of commodity codes to full names
COMMODITY_NAMES = {
    "abras": "Abrasives",
    "alumi": "Aluminum",
    "antim": "Antimony",
    "arsen": "Arsenic",
    "asbes": "Asbestos",
    "barit": "Barite",
    "bauxi": "Bauxite",
    "beryl": "Beryllium",
    "bismu": "Bismuth",
    "boron": "Boron",
    "bromi": "Bromine",
    "cadmi": "Cadmium",
    "cemen": "Cement",
    "chrom": "Chromium",
    "clays": "Clays",
    "cobal": "Cobalt",
    "coppe": "Copper",
    "diamo": "Diamond",
    "diato": "Diatomite",
    "felds": "Feldspar",
    "feore": "Iron Ore",
    "fepig": "Iron Oxide Pigments",
    "feste": "Iron and Steel",
    "fluor": "Fluorspar",
    "galli": "Gallium",
    "garne": "Garnet",
    "gemst": "Gemstones",
    "germa": "Germanium",
    "gold": "Gold",
    "graph": "Graphite",
    "gypsu": "Gypsum",
    "heliu": "Helium",
    "indiu": "Indium",
    "iodin": "Iodine",
    "kyani": "Kyanite",
    "lead": "Lead",
    "lime": "Lime",
    "lithi": "Lithium",
    "manga": "Manganese",
    "mercu": "Mercury",
    "mgcomp": "Magnesium Compounds",
    "mgmet": "Magnesium Metal",
    "mica": "Mica",
    "molyb": "Molybdenum",
    "nicke": "Nickel",
    "niobi": "Niobium",
    "nitro": "Nitrogen",
    "peat": "Peat",
    "perli": "Perlite",
    "phosp": "Phosphate Rock",
    "plati": "Platinum Group",
    "potas": "Potash",
    "pumic": "Pumice",
    "raree": "Rare Earths",
    "rheni": "Rhenium",
    "salt": "Salt",
    "sandi": "Sand and Gravel (Industrial)",
    "selen": "Selenium",
    "silve": "Silver",
    "simet": "Silicon",
    "sodaa": "Soda Ash",
    "stond": "Stone (Dimension)",
    "stron": "Strontium",
    "sulfu": "Sulfur",
    "talc": "Talc",
    "tanta": "Tantalum",
    "tellu": "Tellurium",
    "timin": "Titanium Mineral Concentrates",
    "tin": "Tin",
    "titan": "Titanium Metal",
    "tungs": "Tungsten",
    "vanad": "Vanadium",
    "vermi": "Vermiculite",
    "wolla": "Wollastonite",
    "zeoli": "Zeolites",
    "zinc": "Zinc",
    "zirco-hafni": "Zirconium and Hafnium",
}

# DOE Critical Minerals list (2023)
CRITICAL_MINERALS = [
    "alumi",
    "antim",
    "arsen",
    "barit",
    "beryl",
    "bismu",
    "chrom",
    "cobal",
    "fluor",
    "galli",
    "germa",
    "graph",
    "indiu",
    "lithi",
    "manga",
    "nicke",
    "niobi",
    "plati",
    "raree",
    "tanta",
    "tellu",
    "tin",
    "titan",
    "tungs",
    "vanad",
    "zinc",
    "zirco-hafni",
]


class USGSCommodityLoader(BaseLoader):
    """
    Loader for USGS Mineral Commodity Summaries data.

    Provides access to world production data and salient statistics
    for 80+ mineral commodities.
    """

    dataset_name = "usgs_commodity"

    def list_available(self) -> list[str]:
        """List available commodity codes."""
        if not self.data_path.exists():
            return []

        codes = set()
        for pattern in ["world/mcs*_world.csv", "salient/mcs*_salient.csv"]:
            for f in self.data_path.glob(pattern):
                match = re.search(r"mcs\d{4}-(\w+)_", f.name)
                if match:
                    codes.add(match.group(1))

        return sorted(codes)

    def list_critical_minerals(self) -> list[str]:
        """List commodity codes for DOE critical minerals."""
        available = set(self.list_available())
        return [c for c in CRITICAL_MINERALS if c in available]

    def get_commodity_name(self, code: str) -> str:
        """Get full commodity name from code."""
        return COMMODITY_NAMES.get(code, code.title())

    def load(self, commodity: str | None = None, data_type: str = "world") -> pd.DataFrame:
        """
        Load USGS commodity data.

        Args:
            commodity: Commodity code (e.g., 'lithi'). If None, loads all commodities.
            data_type: 'world' for world production or 'salient' for salient statistics

        Returns:
            pandas.DataFrame with commodity data
        """
        if commodity:
            if data_type == "world":
                return self.load_world_production(commodity)
            else:
                return self.load_salient_statistics(commodity)
        else:
            return self._load_all(data_type)

    def load_world_production(self, commodity: str) -> pd.DataFrame:
        """
        Load world production data for a commodity.

        Args:
            commodity: Commodity code (e.g., 'lithi', 'cobal', 'raree')

        Returns:
            DataFrame with columns: Source, Country, type, Prod_t_2021,
            Prod_t_est_2022, Prod_notes, Reserves_t, Reserves_notes
        """
        cache_key = self._cache_key("world", commodity)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        world_dir = self.data_path / "world"
        self._validate_path(world_dir, "World production directory")

        # Find matching file
        pattern = f"mcs*-{commodity}_world.csv"
        try:
            file_path = self._find_file(pattern, world_dir)
        except DataNotFoundError:
            # Try fuzzy match
            available = self.list_available()
            matches = [c for c in available if commodity.lower() in c.lower()]
            if matches:
                raise DataNotFoundError(
                    f"Commodity '{commodity}' not found. Did you mean: {matches}?"
                )
            raise DataNotFoundError(
                f"Commodity '{commodity}' not found. Available: {available[:10]}..."
            )

        df = self._read_csv(file_path)

        # Clean numeric columns
        numeric_cols = [c for c in df.columns if "Prod_t" in c or "Reserves" in c]
        for col in numeric_cols:
            if col in df.columns and not col.endswith("_notes"):
                df[f"{col}_clean"] = clean_numeric_column(df[col])

        # Add commodity metadata
        df["commodity_code"] = commodity
        df["commodity_name"] = self.get_commodity_name(commodity)

        self._set_cached(cache_key, df)
        return df

    def load_salient_statistics(self, commodity: str) -> pd.DataFrame:
        """
        Load salient statistics for a commodity.

        Args:
            commodity: Commodity code

        Returns:
            DataFrame with columns: DataSource, Commodity, Year, USprod_t,
            Imports_t, Exports_t, Consump_t, Price_dt, Employment_num, NIR_pct
        """
        cache_key = self._cache_key("salient", commodity)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        salient_dir = self.data_path / "salient"
        self._validate_path(salient_dir, "Salient statistics directory")

        pattern = f"mcs*-{commodity}_salient.csv"
        try:
            file_path = self._find_file(pattern, salient_dir)
        except DataNotFoundError:
            available = self.list_available()
            raise DataNotFoundError(
                f"Salient statistics for '{commodity}' not found. Available: {available[:10]}..."
            )

        df = self._read_csv(file_path)

        # Clean numeric columns
        numeric_cols = [
            "USprod_t",
            "Imports_t",
            "Exports_t",
            "Consump_t",
            "Price_dt",
            "Employment_num",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[f"{col}_clean"] = clean_numeric_column(df[col])

        # Parse NIR (Net Import Reliance) percentage
        if "NIR_pct" in df.columns:
            df["NIR_pct_clean"] = df["NIR_pct"].apply(
                lambda x: parse_numeric_value(str(x).replace(">", "").replace("<", ""))
            )

        df["commodity_code"] = commodity
        df["commodity_name"] = self.get_commodity_name(commodity)

        self._set_cached(cache_key, df)
        return df

    def _load_all(self, data_type: str) -> pd.DataFrame:
        """Load all commodities of a given type."""
        dfs = []
        for commodity in self.list_available():
            try:
                if data_type == "world":
                    df = self.load_world_production(commodity)
                else:
                    df = self.load_salient_statistics(commodity)
                dfs.append(df)
            except DataNotFoundError:
                continue

        if not dfs:
            raise DataNotFoundError(f"No {data_type} data found")

        return pd.concat(dfs, ignore_index=True)

    def get_top_producers(
        self, commodity: str, year_col: str = "Prod_t_est_2022", top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top producing countries for a commodity.

        Args:
            commodity: Commodity code
            year_col: Column name for production year
            top_n: Number of top producers to return

        Returns:
            DataFrame sorted by production
        """
        df = self.load_world_production(commodity)

        # Use cleaned column if available
        clean_col = f"{year_col}_clean"
        if clean_col in df.columns:
            year_col = clean_col

        # Filter out world totals and sort
        df = df[~df["Country"].str.contains("World|total", case=False, na=False)]
        df = df.sort_values(year_col, ascending=False)

        return df.head(top_n)

    def describe(self) -> dict:
        """Describe the USGS commodity dataset."""
        base = super().describe()
        base["commodity_count"] = len(self.list_available())
        base["critical_minerals_available"] = len(self.list_critical_minerals())
        base["data_types"] = ["world", "salient"]
        return base
