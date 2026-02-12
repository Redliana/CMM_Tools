"""
CMM Data - Critical Minerals Modeling Data Access Library

This package provides unified access to datasets for critical minerals
supply chain modeling, including USGS commodity data, ore deposits,
OECD trade data, and geospatial resources.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .catalog import get_data_catalog, list_commodities, list_critical_minerals
from .config import CMMDataConfig, configure, get_config
from .exceptions import CMMDataError, ConfigurationError, DataNotFoundError


# Lazy imports for loaders to avoid import overhead
def __getattr__(name):
    if name == "USGSCommodityLoader":
        from .loaders.usgs_commodity import USGSCommodityLoader

        return USGSCommodityLoader
    elif name == "USGSOreDepositsLoader":
        from .loaders.usgs_ore import USGSOreDepositsLoader

        return USGSOreDepositsLoader
    elif name == "OSTIDocumentsLoader":
        from .loaders.osti_docs import OSTIDocumentsLoader

        return OSTIDocumentsLoader
    elif name == "PreprocessedCorpusLoader":
        from .loaders.preprocessed import PreprocessedCorpusLoader

        return PreprocessedCorpusLoader
    elif name == "GAChronostratigraphicLoader":
        from .loaders.ga_chronostrat import GAChronostratigraphicLoader

        return GAChronostratigraphicLoader
    elif name == "NETLREECoalLoader":
        from .loaders.netl_ree import NETLREECoalLoader

        return NETLREECoalLoader
    elif name == "OECDSupplyChainLoader":
        from .loaders.oecd_supply import OECDSupplyChainLoader

        return OECDSupplyChainLoader
    elif name == "MindatLoader":
        from .loaders.mindat import MindatLoader

        return MindatLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Convenience functions
def load_usgs_commodity(commodity: str, data_type: str = "world"):
    """
    Load USGS commodity data.

    Args:
        commodity: Commodity code (e.g., 'lithi', 'cobal', 'raree')
        data_type: 'world' for world production or 'salient' for salient statistics

    Returns:
        pandas.DataFrame with commodity data
    """
    from .loaders.usgs_commodity import USGSCommodityLoader

    loader = USGSCommodityLoader()
    if data_type == "world":
        return loader.load_world_production(commodity)
    elif data_type == "salient":
        return loader.load_salient_statistics(commodity)
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'world' or 'salient'")


def load_ore_deposits(table: str = "all"):
    """
    Load USGS ore deposits data.

    Args:
        table: Table name or 'all' for complete dataset

    Returns:
        pandas.DataFrame with ore deposit data
    """
    from .loaders.usgs_ore import USGSOreDepositsLoader

    loader = USGSOreDepositsLoader()
    return loader.load(table)


def search_documents(query: str, **kwargs):
    """
    Search OSTI documents.

    Args:
        query: Search query string
        **kwargs: Additional search parameters

    Returns:
        pandas.DataFrame with search results
    """
    from .loaders.osti_docs import OSTIDocumentsLoader

    loader = OSTIDocumentsLoader()
    return loader.search_documents(query, **kwargs)


def iter_corpus_documents(**kwargs):
    """
    Iterate over preprocessed corpus documents.

    Yields:
        dict: Document metadata and content
    """
    from .loaders.preprocessed import PreprocessedCorpusLoader

    loader = PreprocessedCorpusLoader()
    yield from loader.iter_documents(**kwargs)


__all__ = [
    # Configuration
    "CMMDataConfig",
    # Exceptions
    "CMMDataError",
    "ConfigurationError",
    "DataNotFoundError",
    "GAChronostratigraphicLoader",
    "MindatLoader",
    "NETLREECoalLoader",
    "OECDSupplyChainLoader",
    "OSTIDocumentsLoader",
    "PreprocessedCorpusLoader",
    # Loaders
    "USGSCommodityLoader",
    "USGSOreDepositsLoader",
    # Version
    "__version__",
    "configure",
    "get_config",
    # Catalog
    "get_data_catalog",
    "iter_corpus_documents",
    "list_commodities",
    "list_critical_minerals",
    "load_ore_deposits",
    # Convenience functions
    "load_usgs_commodity",
    "search_documents",
]
