"""Batch inference using mlx_lm for evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

from cmm_fine_tune.constants import CMM_SYSTEM_PROMPT
from cmm_fine_tune.models import GoldQAPair

logger = logging.getLogger(__name__)


def run_inference(
    model_id: str,
    questions: list[GoldQAPair],
    adapter_path: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> list[str]:
    """Run inference on a list of gold QA questions.

    Returns a list of generated answers (same order as input).
    """
    from mlx_lm import generate, load

    logger.info("Loading model %s...", model_id)
    load_kwargs: dict = {}
    if adapter_path and Path(adapter_path).exists():
        load_kwargs["adapter_path"] = adapter_path
        logger.info("Loading adapter from %s", adapter_path)
    model, tokenizer = load(model_id, **load_kwargs)

    answers: list[str] = []
    for i, qa in enumerate(questions):
        prompt = _build_prompt(qa.question, tokenizer)
        logger.debug("Generating answer %d/%d...", i + 1, len(questions))
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
        )
        answers.append(response.strip())

    logger.info("Generated %d answers", len(answers))
    return answers


def _build_prompt(question: str, tokenizer) -> str:
    """Build a chat prompt using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": CMM_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback for tokenizers without chat template
    return f"System: {CMM_SYSTEM_PROMPT}\n\nUser: {question}\n\nAssistant:"
