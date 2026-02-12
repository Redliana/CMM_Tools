"""Geoscience Australia Chronostratigraphic Model loader."""

from __future__ import annotations

import zipfile
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigurationError, DataNotFoundError
from .base import BaseLoader

# Surface names in the GA 3D model
CHRONOSTRAT_SURFACES = [
    "Paleozoic_Top",
    "Neoproterozoic_Top",
    "Mesoproterozoic_Top",
    "Paleoproterozoic_Top",
    "Neoarchean_Top",
    "Mesoarchean_Top",
    "Paleoarchean_Top",
    "Eoarchean_Top",
    "Basement",
]


class GAChronostratigraphicLoader(BaseLoader):
    """
    Loader for Geoscience Australia 3D Chronostratigraphic Model.

    Provides access to 3D modelling surfaces and isochores for the
    preliminary chronostratigraphic model of Australia.

    Formats available:
    - GeoTIFF: Raster surfaces for GIS
    - XYZ: ASCII point data
    - ZMAP: Grid format for geological modeling software
    - PNG: Visualization images
    """

    dataset_name = "ga_chronostrat"

    # File patterns by format
    FORMAT_PATTERNS = {
        "geotiff": "149923_3D_Surfaces_GEOTIFF*.zip",
        "xyz": "149923_3D_Surfaces_XYZ*.zip",
        "zmap": "149923_3D_ZMAP*.zip",
        "png": "149923_3D_Surfaces_PNG*.zip",
        "confidence": "149923_3D_Confidence_GEOTIFF*.zip",
    }

    def list_available(self) -> list[str]:
        """List available data formats."""
        if not self.data_path.exists():
            return []

        available = []
        for format_name, pattern in self.FORMAT_PATTERNS.items():
            if list(self.data_path.glob(pattern)):
                available.append(format_name)

        return available

    def load(self, surface: str = "Paleozoic_Top", format: str = "xyz") -> pd.DataFrame:
        """
        Load a surface from the chronostratigraphic model.

        Args:
            surface: Surface name (e.g., 'Paleozoic_Top', 'Basement')
            format: Data format ('xyz', 'geotiff' requires rasterio)

        Returns:
            DataFrame with surface data (for XYZ format)
            For GeoTIFF, returns rasterio dataset if available
        """
        if format == "xyz":
            return self._load_xyz_surface(surface)
        elif format == "geotiff":
            return self._load_geotiff_surface(surface)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'xyz' or 'geotiff'")

    def _load_xyz_surface(self, surface: str) -> pd.DataFrame:
        """Load surface from XYZ format."""
        cache_key = self._cache_key("xyz", surface)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._validate_path(self.data_path, "GA Chronostratigraphic directory")

        # Find XYZ zip file
        xyz_files = list(self.data_path.glob(self.FORMAT_PATTERNS["xyz"]))
        if not xyz_files:
            raise DataNotFoundError("XYZ data not found. Download from GA eCat (record 149923)")

        zip_path = xyz_files[0]

        # Search for surface file in zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            # list files and find matching surface
            surface_file = None
            for name in zf.namelist():
                if surface.lower() in name.lower() and name.endswith(".xyz"):
                    surface_file = name
                    break

            if not surface_file:
                available_surfaces = [n for n in zf.namelist() if n.endswith(".xyz")]
                raise DataNotFoundError(
                    f"Surface '{surface}' not found. Available: {available_surfaces}"
                )

            # Read XYZ file
            with zf.open(surface_file) as f:
                df = pd.read_csv(f, sep=r"\s+", names=["x", "y", "z"], header=None, comment="#")

        df["surface"] = surface

        self._set_cached(cache_key, df)
        return df

    def _load_geotiff_surface(self, surface: str) -> Any:
        """Load surface from GeoTIFF format."""
        try:
            import rasterio
        except ImportError:
            raise ConfigurationError(
                "rasterio required for GeoTIFF loading. Install with: pip install cmm-data[geo]"
            )

        self._validate_path(self.data_path, "GA Chronostratigraphic directory")

        # Find GeoTIFF zip file
        tiff_files = list(self.data_path.glob(self.FORMAT_PATTERNS["geotiff"]))
        if not tiff_files:
            raise DataNotFoundError("GeoTIFF data not found. Download from GA eCat (record 149923)")

        zip_path = tiff_files[0]

        # Extract and open GeoTIFF
        with zipfile.ZipFile(zip_path, "r") as zf:
            surface_file = None
            for name in zf.namelist():
                if surface.lower() in name.lower() and name.endswith(".tif"):
                    surface_file = name
                    break

            if not surface_file:
                raise DataNotFoundError(f"Surface '{surface}' not found in GeoTIFF archive")

            # Extract to temp location and open
            extracted = zf.extract(surface_file, self.data_path / ".temp")
            return rasterio.open(extracted)

    def list_surfaces(self) -> list[str]:
        """List available surface names."""
        return CHRONOSTRAT_SURFACES.copy()

    def get_surface_extent(self, surface: str) -> dict[str, float]:
        """
        Get spatial extent of a surface.

        Args:
            surface: Surface name

        Returns:
            Dictionary with xmin, xmax, ymin, ymax, zmin, zmax
        """
        df = self._load_xyz_surface(surface)

        return {
            "xmin": df["x"].min(),
            "xmax": df["x"].max(),
            "ymin": df["y"].min(),
            "ymax": df["y"].max(),
            "zmin": df["z"].min(),
            "zmax": df["z"].max(),
            "point_count": len(df),
        }

    def get_depth_at_point(self, x: float, y: float, surface: str = "Basement") -> float | None:
        """
        Get depth to a surface at a specific location.

        Args:
            x: X coordinate (Albers Equal Area)
            y: Y coordinate (Albers Equal Area)
            surface: Surface name

        Returns:
            Depth value (z) or None if outside data extent
        """
        df = self._load_xyz_surface(surface)

        # Find nearest point (simple approach)
        distances = np.sqrt((df["x"] - x) ** 2 + (df["y"] - y) ** 2)
        min_idx = distances.idxmin()

        # Return None if too far from any data point
        if distances[min_idx] > 10000:  # 10km threshold
            return None

        return df.loc[min_idx, "z"]

    def get_model_info(self) -> dict:
        """Get information about the 3D model."""
        return {
            "title": "Preliminary 3D Chronostratigraphic Model of Australia",
            "record_id": "149923",
            "source": "Geoscience Australia",
            "spatial_reference": "EPSG:3577 (GDA94 / Australian Albers)",
            "surfaces": CHRONOSTRAT_SURFACES,
            "formats": self.list_available(),
            "url": "https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/149923",
        }

    def describe(self) -> dict:
        """Describe the GA chronostratigraphic dataset."""
        base = super().describe()
        base.update(self.get_model_info())
        return base
