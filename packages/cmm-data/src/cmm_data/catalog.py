"""Data catalog and inventory functions."""

from __future__ import annotations

import pandas as pd

from .config import get_config
from .loaders.usgs_commodity import COMMODITY_NAMES, CRITICAL_MINERALS


def get_data_catalog() -> pd.DataFrame:
    """
    Get a catalog of all available CMM datasets.

    Returns:
        DataFrame with dataset information and availability status
    """
    config = get_config()

    datasets = [
        {
            "dataset": "usgs_commodity",
            "name": "USGS Mineral Commodity Summaries",
            "description": "World production and salient statistics for 80+ commodities",
            "loader": "USGSCommodityLoader",
            "source": "USGS",
            "format": "CSV",
        },
        {
            "dataset": "usgs_ore",
            "name": "USGS Ore Deposits Database",
            "description": "Geochemical analyses from ore deposits (356 fields)",
            "loader": "USGSOreDepositsLoader",
            "source": "USGS",
            "format": "CSV",
        },
        {
            "dataset": "osti",
            "name": "OSTI Technical Documents",
            "description": "Technical reports on critical minerals and materials",
            "loader": "OSTIDocumentsLoader",
            "source": "OSTI/DOE",
            "format": "JSON/TXT",
        },
        {
            "dataset": "preprocessed",
            "name": "Preprocessed Document Corpus",
            "description": "Unified corpus for LLM training (3,298 documents)",
            "loader": "PreprocessedCorpusLoader",
            "source": "CMM",
            "format": "JSONL",
        },
        {
            "dataset": "ga_chronostrat",
            "name": "GA 3D Chronostratigraphic Model",
            "description": "3D modelling surfaces for Australia (9 surfaces)",
            "loader": "GAChronostratigraphicLoader",
            "source": "Geoscience Australia",
            "format": "GeoTIFF/XYZ/ZMAP",
        },
        {
            "dataset": "netl_ree",
            "name": "NETL REE and Coal Database",
            "description": "REE data from coal and coal-related resources",
            "loader": "NETLREECoalLoader",
            "source": "NETL/DOE",
            "format": "Geodatabase",
        },
        {
            "dataset": "oecd",
            "name": "OECD Supply Chain Data",
            "description": "Export restrictions, IEA minerals, ICIO documentation",
            "loader": "OECDSupplyChainLoader",
            "source": "OECD/IEA",
            "format": "PDF/CSV",
        },
    ]

    df = pd.DataFrame(datasets)

    # Check availability
    status = config.validate()
    df["available"] = df["dataset"].map(lambda x: status.get(x, False))

    # Get paths
    def get_path_safe(ds):
        try:
            return str(config.get_path(ds))
        except (ValueError, KeyError):
            return None

    df["path"] = df["dataset"].map(get_path_safe)

    return df


def list_commodities() -> list[str]:
    """
    list all available USGS commodity codes.

    Returns:
        list of commodity codes (e.g., ['abras', 'alumi', ...])
    """
    return sorted(COMMODITY_NAMES.keys())


def list_critical_minerals() -> list[str]:
    """
    list DOE critical minerals commodity codes.

    Returns:
        list of critical mineral codes
    """
    return CRITICAL_MINERALS.copy()


def get_commodity_info(code: str) -> dict:
    """
    Get information about a commodity.

    Args:
        code: Commodity code (e.g., 'lithi')

    Returns:
        Dictionary with commodity information
    """
    name = COMMODITY_NAMES.get(code, code.title())
    is_critical = code in CRITICAL_MINERALS

    return {
        "code": code,
        "name": name,
        "is_critical_mineral": is_critical,
        "data_types": ["world", "salient"],
    }


def search_all_datasets(query: str, datasets: list[str] | None = None) -> pd.DataFrame:
    """
    Search across multiple datasets.

    Args:
        query: Search query string
        datasets: list of datasets to search (default: all)

    Returns:
        DataFrame with search results from all datasets
    """
    from .loaders.osti_docs import OSTIDocumentsLoader
    from .loaders.preprocessed import PreprocessedCorpusLoader
    from .loaders.usgs_commodity import USGSCommodityLoader

    results = []

    if datasets is None:
        datasets = ["usgs_commodity", "osti", "preprocessed"]

    if "usgs_commodity" in datasets:
        try:
            loader = USGSCommodityLoader()
            # Search commodity names
            for code, name in COMMODITY_NAMES.items():
                if query.lower() in name.lower() or query.lower() in code.lower():
                    results.append(
                        {
                            "dataset": "usgs_commodity",
                            "type": "commodity",
                            "id": code,
                            "name": name,
                            "match_field": "commodity_name",
                        }
                    )
        except (OSError, ValueError):
            pass

    if "osti" in datasets:
        try:
            loader = OSTIDocumentsLoader()
            docs = loader.search_documents(query, limit=20)
            for _, row in docs.iterrows():
                results.append(
                    {
                        "dataset": "osti",
                        "type": "document",
                        "id": row.get("osti_id", ""),
                        "name": row.get("title", ""),
                        "match_field": "title/abstract",
                    }
                )
        except (OSError, ValueError):
            pass

    if "preprocessed" in datasets:
        try:
            loader = PreprocessedCorpusLoader()
            docs = loader.search(query, limit=20)
            for _, row in docs.iterrows():
                results.append(
                    {
                        "dataset": "preprocessed",
                        "type": "document",
                        "id": row.get("id", ""),
                        "name": row.get("title", str(row.get("text", ""))[:50]),
                        "match_field": "text",
                    }
                )
        except (OSError, ValueError):
            pass

    return pd.DataFrame(results)


def get_dataset_summary() -> dict:
    """
    Get a summary of all available data.

    Returns:
        Dictionary with summary statistics
    """
    catalog = get_data_catalog()

    summary = {
        "total_datasets": len(catalog),
        "available_datasets": catalog["available"].sum(),
        "sources": catalog["source"].unique().tolist(),
        "by_source": catalog.groupby("source")["available"].sum().to_dict(),
    }

    # Add commodity counts
    summary["total_commodities"] = len(COMMODITY_NAMES)
    summary["critical_minerals"] = len(CRITICAL_MINERALS)

    return summary
