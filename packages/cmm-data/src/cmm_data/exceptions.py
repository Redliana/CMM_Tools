"""Custom exceptions for the CMM Data package."""

from __future__ import annotations


class CMMDataError(Exception):
    """Base exception for CMM Data package."""

    pass


class DataNotFoundError(CMMDataError):
    """Raised when requested data cannot be found."""

    pass


class ConfigurationError(CMMDataError):
    """Raised when there is a configuration problem."""

    pass


class ParseError(CMMDataError):
    """Raised when data cannot be parsed."""

    pass


class ValidationError(CMMDataError):
    """Raised when data validation fails."""

    pass
