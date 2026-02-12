"""Base loader class for all CMM data loaders."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..config import get_config
from ..exceptions import DataNotFoundError

if TYPE_CHECKING:
    from pathlib import Path


class BaseLoader(ABC):
    """
    Abstract base class for all CMM data loaders.

    Provides common functionality for data loading, caching, and validation.
    """

    # Override in subclasses
    dataset_name: str = "base"

    def __init__(self, config=None):
        """
        Initialize the loader.

        Args:
            config: Optional CMMDataConfig instance. Uses global config if not provided.
        """
        self.config = config or get_config()
        self._cache = {}

    @property
    def data_path(self) -> Path:
        """Get the path to this loader's dataset directory."""
        return self.config.get_path(self.dataset_name)

    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """
        Load data from the dataset.

        Args:
            **kwargs: Loader-specific parameters

        Returns:
            pandas.DataFrame with the loaded data
        """
        pass

    @abstractmethod
    def list_available(self) -> list[str]:
        """
        list available data items in this dataset.

        Returns:
            list of available data item identifiers
        """
        pass

    def describe(self) -> dict[str, Any]:
        """
        Describe this dataset.

        Returns:
            Dictionary with dataset metadata
        """
        return {
            "name": self.dataset_name,
            "path": str(self.data_path),
            "available": self.data_path.exists(),
            "items": self.list_available() if self.data_path.exists() else [],
        }

    def query(self, **kwargs) -> pd.DataFrame:
        """
        Query the dataset with filters.

        Default implementation loads all data then filters.
        Override in subclasses for more efficient querying.

        Args:
            **kwargs: Query parameters (column=value filters)

        Returns:
            Filtered pandas.DataFrame
        """
        df = self.load()

        for col, value in kwargs.items():
            if col in df.columns:
                if isinstance(value, (list, tuple)):
                    df = df[df[col].isin(value)]
                else:
                    df = df[df[col] == value]

        return df

    def _cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": kwargs,
            "dataset": self.dataset_name,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached(self, key: str) -> Any | None:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self.config.cache_enabled:
            return None

        # Check memory cache first
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["time"] < self.config.cache_ttl_seconds:
                return entry["data"]
            else:
                del self._cache[key]

        # Check disk cache
        if self.config.cache_dir:
            cache_file = self.config.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    mtime = cache_file.stat().st_mtime
                    if time.time() - mtime < self.config.cache_ttl_seconds:
                        with open(cache_file, "rb") as f:
                            data = pickle.load(f)
                        self._cache[key] = {"data": data, "time": mtime}
                        return data
                    else:
                        cache_file.unlink()
                except (OSError, ValueError, pickle.UnpicklingError):
                    pass

        return None

    def _set_cached(self, key: str, data: Any) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            data: Data to cache
        """
        if not self.config.cache_enabled:
            return

        # Memory cache
        self._cache[key] = {"data": data, "time": time.time()}

        # Disk cache
        if self.config.cache_dir:
            try:
                self.config.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self.config.cache_dir / f"{key}.pkl"
                with open(cache_file, "wb") as f:
                    pickle.dump(data, f)
            except OSError:
                pass  # Silently fail disk caching

    def _validate_path(self, path: Path, description: str = "File") -> None:
        """
        Validate that a path exists.

        Args:
            path: Path to validate
            description: Description for error message

        Raises:
            DataNotFoundError: If path does not exist
        """
        if not path.exists():
            raise DataNotFoundError(f"{description} not found: {path}")

    def _find_file(self, pattern: str, directory: Path | None = None) -> Path:
        """
        Find a file matching a pattern.

        Args:
            pattern: Glob pattern
            directory: Directory to search (defaults to data_path)

        Returns:
            Path to matching file

        Raises:
            DataNotFoundError: If no matching file found
        """
        directory = directory or self.data_path
        matches = list(directory.glob(pattern))

        if not matches:
            raise DataNotFoundError(f"No file matching '{pattern}' found in {directory}")

        return matches[0]

    def _read_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        """
        Read a CSV file with common options.

        Args:
            path: Path to CSV file
            **kwargs: Additional pandas read_csv options

        Returns:
            pandas.DataFrame
        """
        default_kwargs = {
            "encoding": "utf-8",
            "encoding_errors": "replace",
        }
        default_kwargs.update(kwargs)

        try:
            return pd.read_csv(path, **default_kwargs)
        except UnicodeDecodeError:
            default_kwargs["encoding"] = "latin-1"
            return pd.read_csv(path, **default_kwargs)
