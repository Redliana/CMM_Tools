"""Configuration management for CMM Data package."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ConfigurationError


def _find_data_root() -> Path:
    """
    Find the root directory containing CMM data.

    Searches in order:
    1. CMM_DATA_PATH environment variable
    2. Parent directories looking for Globus_Sharing
    3. Common installation paths
    """
    # Check environment variable first
    env_path = os.environ.get("CMM_DATA_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Try to find Globus_Sharing relative to this package
    current = Path(__file__).resolve()
    for _ in range(10):  # Limit search depth
        current = current.parent
        if current.name == "Globus_Sharing":
            return current
        globus = current / "Globus_Sharing"
        if globus.exists():
            return globus

    # Common paths to check
    common_paths = [
        Path.home() / "Documents" / "Projects" / "Science_Projects" / "MPII_CMM" / "Globus_Sharing",
        Path.home() / "Globus_Sharing",
        Path("/data/cmm/Globus_Sharing"),
    ]

    for path in common_paths:
        if path.exists():
            return path

    # Return None - will need to be configured
    return None


@dataclass
class CMMDataConfig:
    """Configuration for CMM Data package."""

    # Root data directory
    data_root: Path | None = None

    # Individual dataset paths (relative to data_root)
    usgs_data_dir: str = "USGS_Data"
    usgs_ore_deposits_dir: str = "USGS_Ore_Deposits"
    osti_retrieval_dir: str = "OSTI_retrieval"
    preprocessed_dir: str = "Data/preprocessed"
    ga_chronostrat_dir: str = "GA_149923_Chronostratigraphic"
    netl_ree_dir: str = "NETL_REE_Coal"
    oecd_supply_dir: str = "OECD_Supply_Chain_Data"
    mindat_dir: str = "Mindat"

    # Caching settings
    cache_enabled: bool = True
    cache_dir: Path | None = None
    cache_ttl_seconds: int = 3600  # 1 hour

    def __post_init__(self):
        if self.data_root is None:
            self.data_root = _find_data_root()
        elif isinstance(self.data_root, str):
            self.data_root = Path(self.data_root)

        if self.cache_dir is None and self.data_root:
            self.cache_dir = self.data_root / ".cmm_cache"

    def get_path(self, dataset: str) -> Path:
        """Get the full path to a dataset directory."""
        if self.data_root is None:
            raise ConfigurationError(
                "Data root not configured. set CMM_DATA_PATH environment variable "
                "or call cmm_data.configure(data_root='/path/to/Globus_Sharing')"
            )

        path_map = {
            "usgs": self.usgs_data_dir,
            "usgs_commodity": self.usgs_data_dir,
            "usgs_ore": self.usgs_ore_deposits_dir,
            "osti": self.osti_retrieval_dir,
            "preprocessed": self.preprocessed_dir,
            "ga": self.ga_chronostrat_dir,
            "ga_chronostrat": self.ga_chronostrat_dir,
            "netl": self.netl_ree_dir,
            "netl_ree": self.netl_ree_dir,
            "oecd": self.oecd_supply_dir,
            "mindat": self.mindat_dir,
        }

        rel_path = path_map.get(dataset.lower())
        if rel_path is None:
            raise ConfigurationError(f"Unknown dataset: {dataset}")

        return self.data_root / rel_path

    def validate(self) -> dict:
        """
        Validate configuration and return availability status.

        Returns:
            dict mapping dataset names to availability status
        """
        datasets = [
            "usgs_commodity",
            "usgs_ore",
            "osti",
            "preprocessed",
            "ga_chronostrat",
            "netl_ree",
            "oecd",
            "mindat",
        ]

        status = {}
        for ds in datasets:
            try:
                path = self.get_path(ds)
                status[ds] = path.exists()
            except ConfigurationError:
                status[ds] = False

        return status


# Global configuration instance
_config: CMMDataConfig | None = None


def get_config() -> CMMDataConfig:
    """Get the current configuration, creating default if needed."""
    global _config
    if _config is None:
        _config = CMMDataConfig()
    return _config


def configure(
    data_root: str | None = None,
    cache_enabled: bool | None = None,
    cache_dir: str | None = None,
    cache_ttl_seconds: int | None = None,
    **kwargs,
) -> CMMDataConfig:
    """
    Configure the CMM Data package.

    Args:
        data_root: Root directory containing CMM data (Globus_Sharing)
        cache_enabled: Whether to enable caching
        cache_dir: Directory for cache files
        cache_ttl_seconds: Cache time-to-live in seconds
        **kwargs: Additional configuration options

    Returns:
        Updated CMMDataConfig instance
    """
    global _config

    config_kwargs = {}
    if data_root is not None:
        config_kwargs["data_root"] = Path(data_root)
    if cache_enabled is not None:
        config_kwargs["cache_enabled"] = cache_enabled
    if cache_dir is not None:
        config_kwargs["cache_dir"] = Path(cache_dir)
    if cache_ttl_seconds is not None:
        config_kwargs["cache_ttl_seconds"] = cache_ttl_seconds
    config_kwargs.update(kwargs)

    _config = CMMDataConfig(**config_kwargs)
    return _config
