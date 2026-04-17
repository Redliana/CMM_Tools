"""Shared pytest fixtures for worldbank-mcp."""

from __future__ import annotations

import pytest

from worldbank_mcp.client import WorldBankClient


@pytest.fixture
def client() -> WorldBankClient:
    """Return a WorldBankClient instance for tests."""
    return WorldBankClient()
