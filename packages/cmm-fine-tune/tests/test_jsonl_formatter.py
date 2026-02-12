"""Tests for JSONL formatter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cmm_fine_tune.data.jsonl_formatter import (
    format_chat_example,
    qa_pair_to_chat_example,
    write_jsonl,
)
from cmm_fine_tune.models import QAPair


@pytest.fixture
def sample_pair():
    return QAPair(
        question="What is the total cobalt trade value?",
        answer="The total cobalt trade value was $1.5 million in 2023.",
        commodity="cobalt",
        complexity_level="L1",
        template_id="trade_total_value",
        source_data={"commodity": "cobalt", "year": 2023, "value_usd": 1500000},
    )


class TestChatConversion:
    def test_converts_to_chat(self, sample_pair):
        example = qa_pair_to_chat_example(sample_pair)
        assert len(example.messages) == 3
        assert example.messages[0].role == "system"
        assert example.messages[1].role == "user"
        assert example.messages[2].role == "assistant"
        assert example.messages[1].content == sample_pair.question
        assert example.messages[2].content == sample_pair.answer

    def test_format_json_line(self, sample_pair):
        example = qa_pair_to_chat_example(sample_pair)
        line = format_chat_example(example)
        data = json.loads(line)
        assert "messages" in data
        assert len(data["messages"]) == 3
        assert data["messages"][0]["role"] == "system"
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][2]["role"] == "assistant"

    def test_json_escaping(self):
        pair = QAPair(
            question='What about "special" characters & symbols?',
            answer="Values include $1,000 and \u00a31,000\nwith newlines.",
            commodity="test",
            complexity_level="L1",
            template_id="test",
        )
        example = qa_pair_to_chat_example(pair)
        line = format_chat_example(example)
        # Should be valid JSON
        data = json.loads(line)
        assert '"special"' in data["messages"][1]["content"]


class TestWriteJsonl:
    def test_write_and_read(self, sample_pair):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            count = write_jsonl([sample_pair, sample_pair], path)
            assert count == 2
            assert path.exists()

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2
            for line in lines:
                data = json.loads(line)
                assert "messages" in data
