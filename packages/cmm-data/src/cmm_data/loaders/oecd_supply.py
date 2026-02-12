"""OECD Supply Chain data loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..exceptions import DataNotFoundError
from .base import BaseLoader


class OECDSupplyChainLoader(BaseLoader):
    """
    Loader for OECD supply chain data.

    Provides access to:
    - Export Restrictions on Industrial Raw Materials
    - IEA Critical Minerals Outlook reports
    - ICIO documentation (Input-Output tables)
    - BTIGE documentation (Bilateral Trade data)
    """

    dataset_name = "oecd"

    # Subdirectories
    SUBDIRS = {
        "export_restrictions": "Export_Restrictions",
        "iea_minerals": "IEA_Critical_Minerals",
        "icio": "ICIO",
        "btige": "BTIGE",
    }

    def list_available(self) -> list[str]:
        """List available data categories."""
        if not self.data_path.exists():
            return []

        available = []
        for name, subdir in self.SUBDIRS.items():
            path = self.data_path / subdir
            if path.exists() and any(path.iterdir()):
                available.append(name)

        return available

    def load(self, dataset: str = "export_restrictions") -> pd.DataFrame:
        """
        Load OECD dataset.

        Note: Most OECD data is in PDF format. This returns metadata
        about available files. Use get_pdf_paths() for file locations.

        Args:
            dataset: Dataset name ('export_restrictions', 'iea_minerals', etc.)

        Returns:
            DataFrame with file metadata
        """
        if dataset not in self.SUBDIRS:
            raise DataNotFoundError(
                f"Unknown dataset: {dataset}. Available: {list(self.SUBDIRS.keys())}"
            )

        subdir = self.SUBDIRS[dataset]
        path = self.data_path / subdir

        if not path.exists():
            raise DataNotFoundError(f"Dataset '{dataset}' not found at {path}")

        # Build file inventory
        files = []
        for f in path.rglob("*"):
            if f.is_file():
                files.append(
                    {
                        "filename": f.name,
                        "path": str(f),
                        "extension": f.suffix.lower(),
                        "size_mb": f.stat().st_size / (1024 * 1024),
                        "category": dataset,
                    }
                )

        return pd.DataFrame(files)

    def get_pdf_paths(self, dataset: str) -> list[Path]:
        """
        Get paths to PDF files in a dataset.

        Args:
            dataset: Dataset name

        Returns:
            list of Path objects to PDF files
        """
        df = self.load(dataset)
        pdf_df = df[df["extension"] == ".pdf"]
        return [Path(p) for p in pdf_df["path"]]

    def get_export_restrictions_reports(self) -> list[Path]:
        """Get paths to Export Restrictions PDF reports."""
        return self.get_pdf_paths("export_restrictions")

    def get_iea_minerals_reports(self) -> list[Path]:
        """Get paths to IEA Critical Minerals Outlook PDFs."""
        return self.get_pdf_paths("iea_minerals")

    def get_icio_documentation(self) -> list[Path]:
        """Get paths to ICIO documentation files."""
        df = self.load("icio")
        return [Path(p) for p in df["path"]]

    def load_icio_tables(self, year: int | None = None) -> pd.DataFrame:
        """
        Load ICIO tables if available (requires manual download).

        Args:
            year: Specific year to load, or None for all years

        Returns:
            DataFrame with ICIO data

        Note:
            ICIO CSV files must be manually downloaded from OECD website
            due to Cloudflare protection.
        """
        icio_path = self.data_path / self.SUBDIRS["icio"]

        # Look for CSV files
        csv_files = list(icio_path.glob("*.csv"))
        if not csv_files:
            raise DataNotFoundError(
                "No ICIO CSV files found. These must be manually downloaded from:\n"
                "https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html\n\n"
                "See README.md for detailed instructions."
            )

        if year:
            # Filter to specific year
            year_files = [f for f in csv_files if str(year) in f.name]
            if not year_files:
                available_years = [f.name for f in csv_files]
                raise DataNotFoundError(
                    f"ICIO data for year {year} not found. Available: {available_years}"
                )
            csv_files = year_files

        # Load and concatenate
        dfs = []
        for f in csv_files:
            df = self._read_csv(f)
            df["_source_file"] = f.name
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    def get_minerals_coverage(self) -> dict:
        """
        Get information about minerals covered in OECD data.

        Returns:
            Dictionary with mineral coverage information
        """
        return {
            "export_restrictions": {
                "description": "Export restrictions on industrial raw materials",
                "commodities": 65,
                "countries": 82,
                "years": "2009-2023",
                "key_minerals": [
                    "potash",
                    "molybdenum",
                    "tungsten",
                    "zirconium",
                    "germanium",
                    "rare earths",
                    "lithium",
                    "cobalt",
                    "nickel",
                    "graphite",
                ],
            },
            "iea_critical_minerals": {
                "description": "IEA Critical Minerals Outlook",
                "minerals_count": 35,
                "key_minerals": [
                    "lithium",
                    "nickel",
                    "cobalt",
                    "graphite",
                    "copper",
                    "rare earth elements",
                    "manganese",
                    "silicon",
                    "chromium",
                ],
                "scenarios": ["STEPS", "APS", "NZE"],
            },
            "icio": {
                "description": "Inter-Country Input-Output tables",
                "years": "1995-2022",
                "economies": 81,
                "industries": 45,
                "notes": "2025 edition includes iron/steel vs non-ferrous metals split",
            },
        }

    def get_download_urls(self) -> dict[str, str]:
        """
        Get URLs for manual download of OECD data.

        Returns:
            Dictionary mapping dataset names to download URLs
        """
        return {
            "icio": "https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html",
            "btige": "https://www.oecd.org/en/data/datasets/bilateral-trade-in-goods-by-industry-and-end-use-category.html",
            "stan": "https://www.oecd.org/en/data/datasets/structural-analysis-database.html",
            "export_restrictions": "https://www.oecd.org/trade/topics/export-restrictions-on-industrial-raw-materials/",
            "iea_critical_minerals": "https://www.iea.org/data-and-statistics/data-tools/critical-minerals-data-explorer",
        }

    def describe(self) -> dict:
        """Describe the OECD supply chain dataset."""
        base = super().describe()
        base["subdirectories"] = self.SUBDIRS
        base["minerals_coverage"] = self.get_minerals_coverage()
        base["download_urls"] = self.get_download_urls()

        # Count files by type
        file_counts = {}
        for dataset in self.list_available():
            try:
                df = self.load(dataset)
                file_counts[dataset] = df["extension"].value_counts().to_dict()
            except (OSError, ValueError):
                pass
        base["file_counts"] = file_counts

        return base
