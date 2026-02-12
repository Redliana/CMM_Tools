"""Tests for cmm_data.exceptions module."""

from __future__ import annotations

import pytest

from cmm_data.exceptions import (
    CMMDataError,
    ConfigurationError,
    DataNotFoundError,
    ParseError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Verify the custom exception class hierarchy."""

    def test_cmm_data_error_is_exception(self) -> None:
        """CMMDataError should be a subclass of Exception."""
        assert issubclass(CMMDataError, Exception)

    def test_data_not_found_is_cmm_data_error(self) -> None:
        """DataNotFoundError should be a subclass of CMMDataError."""
        assert issubclass(DataNotFoundError, CMMDataError)

    def test_configuration_error_is_cmm_data_error(self) -> None:
        """ConfigurationError should be a subclass of CMMDataError."""
        assert issubclass(ConfigurationError, CMMDataError)

    def test_parse_error_is_cmm_data_error(self) -> None:
        """ParseError should be a subclass of CMMDataError."""
        assert issubclass(ParseError, CMMDataError)

    def test_validation_error_is_cmm_data_error(self) -> None:
        """ValidationError should be a subclass of CMMDataError."""
        assert issubclass(ValidationError, CMMDataError)


class TestExceptionInstantiation:
    """Verify that each exception can be raised and caught correctly."""

    @pytest.mark.parametrize(
        "exc_cls",
        [CMMDataError, DataNotFoundError, ConfigurationError, ParseError, ValidationError],
    )
    def test_exception_carries_message(self, exc_cls: type[CMMDataError]) -> None:
        """Each exception should preserve the message passed at construction."""
        msg = f"Test message for {exc_cls.__name__}"
        exc = exc_cls(msg)
        assert str(exc) == msg

    def test_catch_subclass_via_base(self) -> None:
        """Raising a subclass should be catchable via the base class."""
        with pytest.raises(CMMDataError):
            raise DataNotFoundError("missing")

    def test_data_not_found_not_caught_as_configuration_error(self) -> None:
        """DataNotFoundError should *not* be caught by a ConfigurationError handler."""
        with pytest.raises(DataNotFoundError):
            try:
                raise DataNotFoundError("nope")
            except ConfigurationError:
                pytest.fail("DataNotFoundError was incorrectly caught as ConfigurationError")
