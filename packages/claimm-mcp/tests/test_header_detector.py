"""Tests for claimm_mcp.header_detector module."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from claimm_mcp.config import Settings

# ---------------------------------------------------------------------------
# We test the internal helper methods of HeaderDetector by instantiating it
# with a patched get_settings() so no real EDX key is required.
# ---------------------------------------------------------------------------


def _make_detector(settings: Settings) -> Any:
    """Create a HeaderDetector using a pre-built Settings object.

    Args:
        settings: A Settings instance (from the fixture).

    Returns:
        A HeaderDetector instance.
    """
    with patch("claimm_mcp.header_detector.get_settings", return_value=settings):
        from claimm_mcp.header_detector import HeaderDetector

        return HeaderDetector()


class TestDetectDelimiter:
    """Tests for HeaderDetector._detect_delimiter()."""

    def test_comma_delimited(self, settings: Settings) -> None:
        """Comma-separated lines should be detected as comma delimiter."""
        detector = _make_detector(settings)
        assert detector._detect_delimiter("a,b,c,d") == ","

    def test_tab_delimited(self, settings: Settings) -> None:
        """Tab-separated lines should be detected as tab delimiter."""
        detector = _make_detector(settings)
        assert detector._detect_delimiter("a\tb\tc\td") == "\t"

    def test_semicolon_delimited(self, settings: Settings) -> None:
        """Semicolon-separated lines should be detected as semicolon delimiter."""
        detector = _make_detector(settings)
        assert detector._detect_delimiter("a;b;c;d") == ";"

    def test_pipe_delimited(self, settings: Settings) -> None:
        """Pipe-separated lines should be detected as pipe delimiter."""
        detector = _make_detector(settings)
        assert detector._detect_delimiter("a|b|c|d") == "|"

    def test_no_delimiter_defaults_to_comma(self, settings: Settings) -> None:
        """When no delimiters are found, should default to comma."""
        detector = _make_detector(settings)
        assert detector._detect_delimiter("abcdefg") == ","

    def test_mixed_delimiters_picks_most_frequent(self, settings: Settings) -> None:
        """When multiple delimiters appear, the most frequent one wins."""
        detector = _make_detector(settings)
        # 3 commas vs 1 tab
        result = detector._detect_delimiter("a,b,c,d\te")
        assert result == ","

    def test_empty_line_defaults_to_comma(self, settings: Settings) -> None:
        """An empty line should yield the comma default."""
        detector = _make_detector(settings)
        assert detector._detect_delimiter("") == ","


class TestInferType:
    """Tests for HeaderDetector._infer_type()."""

    def test_empty_values_returns_unknown(self, settings: Settings) -> None:
        """Empty value list should return type 'unknown'."""
        detector = _make_detector(settings)
        result = detector._infer_type([])
        assert result["type"] == "unknown"

    def test_integer_values(self, settings: Settings) -> None:
        """All-integer values should be detected as 'integer'."""
        detector = _make_detector(settings)
        result = detector._infer_type(["1", "2", "3", "42"])
        assert result["type"] == "integer"

    def test_float_values(self, settings: Settings) -> None:
        """Values with decimal points should be detected as 'float'."""
        detector = _make_detector(settings)
        result = detector._infer_type(["1.5", "2.7", "3.14"])
        assert result["type"] == "float"
        assert result.get("metadata", {}).get("precision") == "double"

    def test_mixed_int_and_float(self, settings: Settings) -> None:
        """Mix of int and float should be detected as 'float'."""
        detector = _make_detector(settings)
        result = detector._infer_type(["1", "2.5", "3"])
        assert result["type"] == "float"

    def test_scientific_notation_is_float(self, settings: Settings) -> None:
        """Scientific notation values should be detected as 'float'."""
        detector = _make_detector(settings)
        result = detector._infer_type(["1.2e5", "3.4E-2"])
        assert result["type"] == "float"

    def test_thousands_separator(self, settings: Settings) -> None:
        """Numeric values with thousands separators should still be numeric."""
        detector = _make_detector(settings)
        result = detector._infer_type(["1,000", "2,500", "10,000"])
        assert result["type"] == "integer"

    def test_date_values(self, settings: Settings) -> None:
        """Date-like values with dashes or slashes should be type 'date'."""
        detector = _make_detector(settings)
        result = detector._infer_type(["2024-01-15", "2024-02-20", "2024-03-10"])
        assert result["type"] == "date"

    def test_datetime_values(self, settings: Settings) -> None:
        """Datetime-like values with colons should be type 'datetime'."""
        detector = _make_detector(settings)
        result = detector._infer_type(
            [
                "2024-01-15 10:30:00",
                "2024-02-20 14:45:00",
            ]
        )
        assert result["type"] == "datetime"

    def test_boolean_values(self, settings: Settings) -> None:
        """Boolean-like values should be detected as 'boolean'."""
        detector = _make_detector(settings)
        result = detector._infer_type(["true", "false", "True", "False"])
        assert result["type"] == "boolean"

    def test_boolean_yes_no(self, settings: Settings) -> None:
        """Yes/no values should be detected as 'boolean'."""
        detector = _make_detector(settings)
        result = detector._infer_type(["yes", "no", "Yes", "No"])
        assert result["type"] == "boolean"

    def test_string_values(self, settings: Settings) -> None:
        """Non-numeric, non-date, non-boolean values should be 'string'."""
        detector = _make_detector(settings)
        result = detector._infer_type(["lithium", "cobalt", "nickel"])
        assert result["type"] == "string"
        assert "max_length" in result.get("metadata", {})

    def test_string_max_length(self, settings: Settings) -> None:
        """String type metadata should include the max observed length."""
        detector = _make_detector(settings)
        result = detector._infer_type(["ab", "abcdef", "abc"])
        assert result["type"] == "string"
        assert result["metadata"]["max_length"] == 6


class TestDetectColumnTypes:
    """Tests for HeaderDetector._detect_column_types()."""

    def test_basic_column_types(self, settings: Settings) -> None:
        """Should correctly type a mix of columns."""
        detector = _make_detector(settings)
        headers = ["id", "value", "name"]
        sample_rows = [
            ["1", "10.5", "alpha"],
            ["2", "20.3", "beta"],
            ["3", "30.1", "gamma"],
        ]
        result = detector._detect_column_types(headers, sample_rows)

        assert len(result) == 3
        assert result[0]["name"] == "id"
        assert result[0]["type"] == "integer"
        assert result[1]["name"] == "value"
        assert result[1]["type"] == "float"
        assert result[2]["name"] == "name"
        assert result[2]["type"] == "string"

    def test_empty_sample_rows(self, settings: Settings) -> None:
        """With no sample rows, all columns should be 'unknown'."""
        detector = _make_detector(settings)
        headers = ["a", "b", "c"]
        result = detector._detect_column_types(headers, [])
        assert all(col["type"] == "unknown" for col in result)

    def test_nullable_detection(self, settings: Settings) -> None:
        """Columns with empty values should be marked as nullable."""
        detector = _make_detector(settings)
        headers = ["id", "maybe_null"]
        sample_rows = [
            ["1", "hello"],
            ["2", ""],
            ["3", "world"],
        ]
        result = detector._detect_column_types(headers, sample_rows)
        assert result[0]["nullable"] is False
        assert result[1]["nullable"] is True

    def test_sample_values_included(self, settings: Settings) -> None:
        """Result should include up to 3 non-empty sample values."""
        detector = _make_detector(settings)
        headers = ["letter"]
        sample_rows = [["a"], ["b"], ["c"], ["d"], ["e"]]
        result = detector._detect_column_types(headers, sample_rows)
        assert len(result[0]["sample_values"]) <= 3

    def test_short_rows_handled(self, settings: Settings) -> None:
        """Rows shorter than the header count should not raise errors."""
        detector = _make_detector(settings)
        headers = ["a", "b", "c"]
        sample_rows = [
            ["1", "2"],  # missing column c
            ["3"],  # missing columns b and c
        ]
        result = detector._detect_column_types(headers, sample_rows)
        assert len(result) == 3
        # Column c should be nullable since it's always missing
        assert result[2]["nullable"] is True


class TestParseCsvContent:
    """Tests for HeaderDetector._parse_csv_content()."""

    def test_basic_csv_parsing(self, settings: Settings, sample_csv_content: str) -> None:
        """Should parse valid CSV content and return success."""
        detector = _make_detector(settings)
        result = detector._parse_csv_content(
            content=sample_csv_content,
            resource_id="test-id",
            delimiter=None,
            partial=False,
        )
        assert result["success"] is True
        assert result["resource_id"] == "test-id"
        assert result["column_count"] == 5
        assert result["headers"] == [
            "sample_id",
            "element",
            "concentration_ppm",
            "date_collected",
            "is_valid",
        ]
        assert result["rows_sampled"] == 5

    def test_tsv_parsing(self, settings: Settings, sample_tsv_content: str) -> None:
        """Should correctly parse tab-delimited content."""
        detector = _make_detector(settings)
        result = detector._parse_csv_content(
            content=sample_tsv_content,
            resource_id="test-tsv",
            delimiter=None,
            partial=False,
        )
        assert result["success"] is True
        assert result["delimiter"] == "\t"
        assert result["headers"] == ["name", "value", "category"]

    def test_explicit_delimiter(self, settings: Settings) -> None:
        """When delimiter is explicitly provided, should use it."""
        detector = _make_detector(settings)
        content = "a;b;c\n1;2;3\n4;5;6"
        result = detector._parse_csv_content(
            content=content,
            resource_id="test-semi",
            delimiter=";",
            partial=False,
        )
        assert result["success"] is True
        assert result["delimiter"] == ";"
        assert result["headers"] == ["a", "b", "c"]

    def test_empty_content(self, settings: Settings) -> None:
        """Empty content should return success=False."""
        detector = _make_detector(settings)
        result = detector._parse_csv_content(
            content="",
            resource_id="empty-id",
            delimiter=None,
            partial=False,
        )
        assert result["success"] is False
        assert "error" in result

    def test_whitespace_only_content(self, settings: Settings) -> None:
        """Whitespace-only content should still produce a result dict."""
        detector = _make_detector(settings)
        result = detector._parse_csv_content(
            content="   \n  \n  ",
            resource_id="ws-id",
            delimiter=None,
            partial=False,
        )
        assert isinstance(result, dict)
        assert "resource_id" in result

    def test_header_only_csv(self, settings: Settings) -> None:
        """CSV with only headers and no data rows should still succeed."""
        detector = _make_detector(settings)
        result = detector._parse_csv_content(
            content="col_a,col_b,col_c",
            resource_id="header-only",
            delimiter=None,
            partial=False,
        )
        assert result["success"] is True
        assert result["column_count"] == 3
        assert result["rows_sampled"] == 0

    def test_partial_flag_passed_through(self, settings: Settings) -> None:
        """The partial_download field should reflect the input flag."""
        detector = _make_detector(settings)
        result = detector._parse_csv_content(
            content="a,b\n1,2",
            resource_id="partial-id",
            delimiter=None,
            partial=True,
        )
        assert result["partial_download"] is True

    def test_sample_rows_capped_at_five(self, settings: Settings) -> None:
        """Should return at most 5 sample rows."""
        detector = _make_detector(settings)
        lines = ["h1,h2"] + [f"{i},{i * 10}" for i in range(20)]
        content = "\n".join(lines)
        result = detector._parse_csv_content(
            content=content,
            resource_id="many-rows",
            delimiter=None,
            partial=False,
        )
        assert result["success"] is True
        assert result["rows_sampled"] == 5

    def test_bytes_fetched_calculated(self, settings: Settings, sample_csv_content: str) -> None:
        """bytes_fetched should match the UTF-8 byte length of the content."""
        detector = _make_detector(settings)
        result = detector._parse_csv_content(
            content=sample_csv_content,
            resource_id="bytes-id",
            delimiter=None,
            partial=False,
        )
        assert result["bytes_fetched"] == len(sample_csv_content.encode("utf-8"))
