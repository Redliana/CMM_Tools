"""Shared pytest fixtures for census-aes-mcp."""

from __future__ import annotations

import pytest

from census_aes_mcp.client import CensusClient


@pytest.fixture
def client() -> CensusClient:
    """Return a CensusClient instance for tests."""
    return CensusClient()
