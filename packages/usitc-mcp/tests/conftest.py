"""Shared pytest fixtures for usitc-mcp."""

from __future__ import annotations

import pytest

from usitc_mcp.client import USITCClient


@pytest.fixture
def client() -> USITCClient:
    """Return a USITCClient instance for tests."""
    return USITCClient()
