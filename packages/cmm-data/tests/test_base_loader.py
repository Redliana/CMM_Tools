"""Tests for cmm_data.loaders.base module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cmm_data.config import CMMDataConfig
from cmm_data.exceptions import DataNotFoundError
from cmm_data.loaders.base import BaseLoader

# ---------------------------------------------------------------------------
# Concrete subclass for testing (BaseLoader is abstract)
# ---------------------------------------------------------------------------


class StubLoader(BaseLoader):
    """Minimal concrete implementation of BaseLoader for test purposes."""

    dataset_name = "usgs_commodity"

    def load(self, **kwargs: Any) -> pd.DataFrame:
        """Return an empty DataFrame."""
        return pd.DataFrame()

    def list_available(self) -> list[str]:
        """Return a fixed list of items."""
        return ["item_a", "item_b"]


# ---------------------------------------------------------------------------
# _cache_key()
# ---------------------------------------------------------------------------


class TestCacheKey:
    """Tests for ``BaseLoader._cache_key``."""

    def test_deterministic(self, config_with_root: CMMDataConfig) -> None:
        """Calling _cache_key with identical arguments should produce the same hash."""
        loader = StubLoader(config=config_with_root)
        k1 = loader._cache_key("world", commodity="lithi")
        k2 = loader._cache_key("world", commodity="lithi")
        assert k1 == k2

    def test_different_args_yield_different_keys(self, config_with_root: CMMDataConfig) -> None:
        """Different arguments should produce different cache keys."""
        loader = StubLoader(config=config_with_root)
        k1 = loader._cache_key("world", commodity="lithi")
        k2 = loader._cache_key("world", commodity="cobal")
        assert k1 != k2

    def test_returns_hex_string(self, config_with_root: CMMDataConfig) -> None:
        """The cache key should be a hexadecimal string (md5 digest)."""
        loader = StubLoader(config=config_with_root)
        key = loader._cache_key("test")
        assert isinstance(key, str)
        assert all(c in "0123456789abcdef" for c in key)

    def test_kwarg_order_invariant(self, config_with_root: CMMDataConfig) -> None:
        """Kwargs order should not affect the hash (json.dumps with sort_keys=True)."""
        loader = StubLoader(config=config_with_root)
        k1 = loader._cache_key(a="1", b="2")
        k2 = loader._cache_key(b="2", a="1")
        assert k1 == k2


# ---------------------------------------------------------------------------
# _validate_path()
# ---------------------------------------------------------------------------


class TestValidatePath:
    """Tests for ``BaseLoader._validate_path``."""

    def test_raises_on_missing_path(self, config_with_root: CMMDataConfig) -> None:
        """_validate_path should raise DataNotFoundError when path is absent."""
        loader = StubLoader(config=config_with_root)
        fake_path = Path("/absolutely/does/not/exist")
        with pytest.raises(DataNotFoundError, match="not found"):
            loader._validate_path(fake_path, description="Test file")

    def test_no_error_on_existing_path(
        self, config_with_root: CMMDataConfig, tmp_path: Path
    ) -> None:
        """_validate_path should succeed silently for an existing path."""
        loader = StubLoader(config=config_with_root)
        existing_file = tmp_path / "exists.txt"
        existing_file.write_text("data")
        loader._validate_path(existing_file, description="Test file")  # should not raise

    def test_error_message_includes_description(self, config_with_root: CMMDataConfig) -> None:
        """The raised error should embed the caller-supplied description."""
        loader = StubLoader(config=config_with_root)
        with pytest.raises(DataNotFoundError, match="World production directory"):
            loader._validate_path(Path("/no/such"), description="World production directory")


# ---------------------------------------------------------------------------
# _find_file()
# ---------------------------------------------------------------------------


class TestFindFile:
    """Tests for ``BaseLoader._find_file``."""

    def test_finds_matching_file(self, config_with_root: CMMDataConfig, tmp_path: Path) -> None:
        """_find_file should return the path of a matching file."""
        loader = StubLoader(config=config_with_root)
        target = tmp_path / "mcs2023-lithi_world.csv"
        target.write_text("a,b\n1,2\n")
        result = loader._find_file("mcs*-lithi_world.csv", directory=tmp_path)
        assert result == target

    def test_raises_when_no_match(self, config_with_root: CMMDataConfig, tmp_path: Path) -> None:
        """_find_file should raise DataNotFoundError when nothing matches."""
        loader = StubLoader(config=config_with_root)
        with pytest.raises(DataNotFoundError, match="No file matching"):
            loader._find_file("no_match*.csv", directory=tmp_path)


# ---------------------------------------------------------------------------
# Caching round-trip
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    """Verify in-memory caching via _get_cached / _set_cached."""

    def test_set_and_get(self, config_with_root: CMMDataConfig) -> None:
        """Data stored via _set_cached should be retrievable via _get_cached."""
        loader = StubLoader(config=config_with_root)
        loader._set_cached("my_key", {"hello": "world"})
        assert loader._get_cached("my_key") == {"hello": "world"}

    def test_get_returns_none_for_unknown_key(self, config_with_root: CMMDataConfig) -> None:
        """_get_cached should return None when the key has never been set."""
        loader = StubLoader(config=config_with_root)
        assert loader._get_cached("unknown_key") is None

    def test_cache_disabled(self, tmp_path: Path) -> None:
        """When cache_enabled is False, _get_cached should always return None."""
        cfg = CMMDataConfig(data_root=tmp_path, cache_enabled=False)
        loader = StubLoader(config=cfg)
        loader._set_cached("key", "value")
        assert loader._get_cached("key") is None
