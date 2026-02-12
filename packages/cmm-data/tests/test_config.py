"""Tests for cmm_data.config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from cmm_data.config import CMMDataConfig, configure, get_config
from cmm_data.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# CMMDataConfig construction and defaults
# ---------------------------------------------------------------------------


class TestCMMDataConfigDefaults:
    """Verify default field values on ``CMMDataConfig``."""

    def test_default_data_root_is_none_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """data_root should be None when no env var is set and the auto-search fails."""
        monkeypatch.delenv("CMM_DATA_PATH", raising=False)
        # Force auto-search to fail by pointing to a non-existent path
        cfg = CMMDataConfig(data_root=Path("/nonexistent/path/that/will/not/exist"))
        assert cfg.data_root == Path("/nonexistent/path/that/will/not/exist")

    def test_default_dataset_dirs(self) -> None:
        """All default relative directory names should match the expected strings."""
        cfg = CMMDataConfig(data_root=Path("/fake"))
        assert cfg.usgs_data_dir == "USGS_Data"
        assert cfg.usgs_ore_deposits_dir == "USGS_Ore_Deposits"
        assert cfg.osti_retrieval_dir == "OSTI_retrieval"
        assert cfg.preprocessed_dir == "Data/preprocessed"
        assert cfg.ga_chronostrat_dir == "GA_149923_Chronostratigraphic"
        assert cfg.netl_ree_dir == "NETL_REE_Coal"
        assert cfg.oecd_supply_dir == "OECD_Supply_Chain_Data"
        assert cfg.mindat_dir == "Mindat"

    def test_default_cache_settings(self) -> None:
        """Cache should be enabled by default with a 1-hour TTL."""
        cfg = CMMDataConfig(data_root=Path("/fake"))
        assert cfg.cache_enabled is True
        assert cfg.cache_ttl_seconds == 3600

    def test_cache_dir_derived_from_data_root(self, tmp_path: Path) -> None:
        """cache_dir should default to data_root / '.cmm_cache'."""
        cfg = CMMDataConfig(data_root=tmp_path)
        assert cfg.cache_dir == tmp_path / ".cmm_cache"

    def test_string_data_root_converted_to_path(self, tmp_path: Path) -> None:
        """If data_root is passed as a string it should be coerced to Path."""
        cfg = CMMDataConfig(data_root=str(tmp_path))  # type: ignore[arg-type]
        assert isinstance(cfg.data_root, Path)
        assert cfg.data_root == tmp_path


# ---------------------------------------------------------------------------
# CMMDataConfig.get_path()
# ---------------------------------------------------------------------------


class TestGetPath:
    """Tests for ``CMMDataConfig.get_path``."""

    def test_known_dataset_returns_path(self, config_with_root: CMMDataConfig) -> None:
        """get_path should return a composed path for a known dataset."""
        path = config_with_root.get_path("usgs_commodity")
        assert path == config_with_root.data_root / "USGS_Data"

    def test_alias_datasets_resolve(self, config_with_root: CMMDataConfig) -> None:
        """Short aliases like 'usgs' and 'ga' should resolve correctly."""
        assert config_with_root.get_path("usgs") == config_with_root.get_path("usgs_commodity")
        assert config_with_root.get_path("ga") == config_with_root.get_path("ga_chronostrat")
        assert config_with_root.get_path("netl") == config_with_root.get_path("netl_ree")

    def test_unknown_dataset_raises(self, config_with_root: CMMDataConfig) -> None:
        """An unknown dataset name should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Unknown dataset"):
            config_with_root.get_path("nonexistent_dataset")

    def test_no_data_root_raises(self) -> None:
        """If data_root is None, get_path must raise ConfigurationError."""
        cfg = CMMDataConfig.__new__(CMMDataConfig)
        cfg.data_root = None
        with pytest.raises(ConfigurationError, match="Data root not configured"):
            cfg.get_path("usgs")

    def test_case_insensitive_lookup(self, config_with_root: CMMDataConfig) -> None:
        """Dataset names should be matched case-insensitively."""
        assert config_with_root.get_path("USGS_COMMODITY") == (
            config_with_root.data_root / "USGS_Data"
        )


# ---------------------------------------------------------------------------
# CMMDataConfig.validate()
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for ``CMMDataConfig.validate``."""

    def test_all_datasets_present(self, config_with_root: CMMDataConfig) -> None:
        """When all subdirectories exist, validate should report True for each."""
        status = config_with_root.validate()
        # The fixture creates all expected subdirectories
        assert isinstance(status, dict)
        for ds in ["usgs_commodity", "usgs_ore", "osti", "preprocessed"]:
            assert status[ds] is True

    def test_missing_directory_reported_false(self, tmp_path: Path) -> None:
        """A missing dataset directory should be reported as False."""
        cfg = CMMDataConfig(data_root=tmp_path)
        status = cfg.validate()
        # tmp_path has no subdirectories
        assert all(v is False for v in status.values())


# ---------------------------------------------------------------------------
# Module-level configure() and get_config()
# ---------------------------------------------------------------------------


class TestConfigureAndGetConfig:
    """Tests for the module-level ``configure`` and ``get_config`` helpers."""

    def test_get_config_returns_instance(self) -> None:
        """get_config should return a CMMDataConfig instance."""
        cfg = get_config()
        assert isinstance(cfg, CMMDataConfig)

    def test_get_config_is_singleton(self) -> None:
        """Successive calls to get_config should return the same object."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_configure_sets_data_root(self, tmp_path: Path) -> None:
        """configure(data_root=...) should update the global config."""
        cfg = configure(data_root=str(tmp_path))
        assert cfg.data_root == tmp_path
        assert get_config() is cfg

    def test_configure_sets_cache_options(self, tmp_path: Path) -> None:
        """configure should accept and apply cache-related kwargs."""
        cfg = configure(
            data_root=str(tmp_path),
            cache_enabled=False,
            cache_ttl_seconds=7200,
        )
        assert cfg.cache_enabled is False
        assert cfg.cache_ttl_seconds == 7200

    def test_configure_with_env_variable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Setting CMM_DATA_PATH env var should be picked up by default config."""
        monkeypatch.setenv("CMM_DATA_PATH", str(tmp_path))
        cfg = CMMDataConfig()
        assert cfg.data_root == tmp_path
