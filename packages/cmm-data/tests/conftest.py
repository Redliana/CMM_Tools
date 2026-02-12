"""Shared fixtures for cmm-data tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cmm_data.config import CMMDataConfig


@pytest.fixture(autouse=True)
def _reset_global_config() -> None:
    """Reset the global config singleton before each test.

    This prevents state leakage between tests that call ``configure()``
    or ``get_config()``.
    """
    import cmm_data.config as cfg_module

    cfg_module._config = None
    yield
    cfg_module._config = None


@pytest.fixture()
def tmp_data_root(tmp_path: Path) -> Path:
    """Create a temporary data root with minimal directory structure.

    Returns:
        Path to the temporary data root directory.
    """
    # Create subdirectories that CMMDataConfig expects
    for subdir in [
        "USGS_Data",
        "USGS_Ore_Deposits",
        "OSTI_retrieval",
        "Data/preprocessed",
        "GA_149923_Chronostratigraphic",
        "NETL_REE_Coal",
        "OECD_Supply_Chain_Data",
        "Mindat",
    ]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def config_with_root(tmp_data_root: Path) -> CMMDataConfig:
    """Return a ``CMMDataConfig`` pointing at a temporary data root.

    Returns:
        Fully initialised CMMDataConfig with existing directories.
    """
    return CMMDataConfig(data_root=tmp_data_root)
