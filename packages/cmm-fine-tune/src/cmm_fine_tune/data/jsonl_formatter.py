"""Convert QAPairs to mlx-lm chat JSONL format."""

from __future__ import annotations

import json
from pathlib import Path

from cmm_fine_tune.constants import CMM_SYSTEM_PROMPT
from cmm_fine_tune.models import ChatExample, ChatMessage, QAPair


def qa_pair_to_chat_example(pair: QAPair, system_prompt: str = CMM_SYSTEM_PROMPT) -> ChatExample:
    """Convert a single QAPair to a ChatExample with system/user/assistant messages."""
    return ChatExample(
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=pair.question),
            ChatMessage(role="assistant", content=pair.answer),
        ]
    )


def format_chat_example(example: ChatExample) -> str:
    """Format a ChatExample as a single JSON line for mlx-lm.

    Output: {"messages": [{"role": "system", "content": "..."}, ...]}
    """
    return json.dumps(
        {"messages": [m.model_dump() for m in example.messages]},
        ensure_ascii=False,
    )


def write_jsonl(
    pairs: list[QAPair],
    output_path: Path,
    system_prompt: str = CMM_SYSTEM_PROMPT,
) -> int:
    """Convert QAPairs to chat JSONL and write to file.

    Returns the number of examples written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            example = qa_pair_to_chat_example(pair, system_prompt)
            f.write(format_chat_example(example) + "\n")
            count += 1
    return count
