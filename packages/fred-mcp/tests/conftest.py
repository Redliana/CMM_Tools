"""Shared pytest fixtures for fred-mcp."""

from __future__ import annotations

import pytest

from fred_mcp.client import FredClient


@pytest.fixture
def client() -> FredClient:
    """Return a FredClient instance for tests."""
    return FredClient()
