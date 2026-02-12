"""NETL REE and Coal geodatabase loader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ..exceptions import ConfigurationError, DataNotFoundError
from .base import BaseLoader

if TYPE_CHECKING:
    from pathlib import Path


class NETLREECoalLoader(BaseLoader):
    """
    Loader for NETL REE and Coal Open Geodatabase.

    Provides access to rare earth element data from coal and coal-related
    resources compiled by the National Energy Technology Laboratory.
    """

    dataset_name = "netl_ree"

    def __init__(self, config=None):
        super().__init__(config)
        self._gdb_path = None
        self._layers = None

    @property
    def gdb_path(self) -> Path:
        """Get path to the geodatabase."""
        if self._gdb_path is None:
            gdb_files = list(self.data_path.glob("*.gdb"))
            if gdb_files:
                self._gdb_path = gdb_files[0]
            else:
                raise DataNotFoundError("Geodatabase not found in NETL REE directory")
        path: Path = self._gdb_path  # type: ignore[assignment]
        return path

    def list_available(self) -> list[str]:
        """List available layers in the geodatabase."""
        if self._layers is not None:
            return list(self._layers)

        try:
            import fiona

            with fiona.open(self.gdb_path) as src:
                self._layers = list(src)
            return list(self._layers)
        except ImportError:
            # Without fiona, return expected layers
            return [
                "REE_Coal_Samples",
                "REE_Coal_Basins",
                "Coal_Resources",
            ]
        except (OSError, ValueError):
            return []

    def load(self, **kwargs: Any) -> pd.DataFrame:
        """
        Load data from the geodatabase.

        Args:
            **kwargs: Loader-specific parameters.
                layer: Layer name to load. If None, loads first available layer.

        Returns:
            DataFrame with layer data (without geometry)
        """
        layer: str | None = kwargs.get("layer")
        try:
            import geopandas as gpd  # noqa: F401

            gdf = self.load_with_geometry(layer)
            # Drop geometry for regular DataFrame
            return pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        except ImportError:
            raise ConfigurationError(
                "geopandas required for geodatabase loading. "
                "Install with: pip install cmm-data[geo]"
            )

    def load_with_geometry(self, layer: str | None = None) -> Any:
        """
        Load layer with geometry as GeoDataFrame.

        Args:
            layer: Layer name to load

        Returns:
            GeoDataFrame with geometry
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ConfigurationError(
                "geopandas required for geodatabase loading. "
                "Install with: pip install cmm-data[geo]"
            )

        cache_key = self._cache_key("gdf", layer)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._validate_path(self.data_path, "NETL REE directory")

        available = self.list_available()
        if layer is None:
            if available:
                layer = available[0]
            else:
                raise DataNotFoundError("No layers available in geodatabase")
        elif layer not in available:
            raise DataNotFoundError(f"Layer '{layer}' not found. Available: {available}")

        gdf = gpd.read_file(self.gdb_path, layer=layer)

        self._set_cached(cache_key, gdf)
        return gdf

    def get_ree_samples(self) -> pd.DataFrame:
        """
        Get REE sample data.

        Returns:
            DataFrame with REE concentration data
        """
        # Try common layer names
        for layer_name in ["REE_Coal_Samples", "REE_Samples", "Samples"]:
            try:
                return self.load(layer=layer_name)
            except DataNotFoundError:
                continue

        # Load first available
        return self.load()

    def get_coal_basins(self) -> Any:
        """
        Get coal basin boundaries with geometry.

        Returns:
            GeoDataFrame with basin polygons
        """
        for layer_name in ["REE_Coal_Basins", "Coal_Basins", "Basins"]:
            try:
                return self.load_with_geometry(layer_name)
            except DataNotFoundError:
                continue

        raise DataNotFoundError("Coal basins layer not found")

    def get_ree_statistics(self) -> dict:
        """
        Get statistics for REE concentrations.

        Returns:
            Dictionary with REE statistics by element
        """
        df = self.get_ree_samples()

        # Find REE columns (typically La, Ce, Pr, Nd, etc.)
        ree_elements = [
            "La",
            "Ce",
            "Pr",
            "Nd",
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
        ]

        stats = {}
        for elem in ree_elements:
            # Look for column containing element name
            elem_cols = [c for c in df.columns if elem in c and "ppm" in c.lower()]
            if not elem_cols:
                elem_cols = [c for c in df.columns if c == elem or c.startswith(f"{elem}_")]

            if elem_cols:
                col = elem_cols[0]
                values = pd.to_numeric(df[col], errors="coerce")
                valid = values.dropna()
                if len(valid) > 0:
                    stats[elem] = {
                        "column": col,
                        "count": len(valid),
                        "mean": valid.mean(),
                        "median": valid.median(),
                        "min": valid.min(),
                        "max": valid.max(),
                    }

        return stats

    def query_by_basin(self, basin_name: str) -> pd.DataFrame:
        """
        Query samples by coal basin.

        Args:
            basin_name: Basin name to filter

        Returns:
            Filtered DataFrame
        """
        df = self.get_ree_samples()

        basin_cols = [c for c in df.columns if "basin" in c.lower()]
        for col in basin_cols:
            mask = df[col].str.contains(basin_name, case=False, na=False)
            if mask.any():
                return df[mask]

        return pd.DataFrame()

    def query_by_state(self, state: str) -> pd.DataFrame:
        """
        Query samples by state.

        Args:
            state: State abbreviation or name

        Returns:
            Filtered DataFrame
        """
        df = self.get_ree_samples()

        state_cols = [c for c in df.columns if "state" in c.lower()]
        for col in state_cols:
            mask = df[col].str.contains(state, case=False, na=False)
            if mask.any():
                return df[mask]

        return pd.DataFrame()

    def describe(self) -> dict:
        """Describe the NETL REE dataset."""
        base = super().describe()
        base["layers"] = self.list_available()
        base["gdb_path"] = str(self.gdb_path) if self.data_path.exists() else None

        try:
            stats = self.get_ree_statistics()
            base["ree_elements_available"] = list(stats.keys())
        except (OSError, ValueError):
            pass

        return base
