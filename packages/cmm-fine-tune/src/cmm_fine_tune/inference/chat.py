"""Interactive REPL chat with rich formatting.

Usage: cmm-chat --model mlx-community/phi-4-bf16 --adapter adapters/phi4_lora/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cmm_fine_tune.constants import CMM_SYSTEM_PROMPT, DEFAULT_MODEL

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive CMM chat")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="MLX model ID")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
    except ImportError:
        print("Install 'rich' for formatted output: pip install rich")
        sys.exit(1)

    console = Console()

    console.print(
        Panel(
            f"[bold]CMM Fine-Tuned Chat[/bold]\n"
            f"Model: {args.model}\n"
            f"Adapter: {args.adapter or 'None'}\n"
            f"Type 'quit' or 'exit' to leave. 'clear' to reset context.",
            title="CMM Chat",
        )
    )

    # Load model
    console.print("[dim]Loading model...[/dim]")
    from mlx_lm import generate, load

    load_kwargs: dict = {}
    if args.adapter and Path(args.adapter).exists():
        load_kwargs["adapter_path"] = args.adapter
    model, tokenizer = load(args.model, **load_kwargs)
    console.print("[green]Model loaded![/green]\n")

    conversation: list[dict[str, str]] = [{"role": "system", "content": CMM_SYSTEM_PROMPT}]

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break
        if user_input.lower() == "clear":
            conversation = [{"role": "system", "content": CMM_SYSTEM_PROMPT}]
            console.print("[dim]Context cleared.[/dim]\n")
            continue

        conversation.append({"role": "user", "content": user_input})

        # Build prompt
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
            prompt += "\nassistant:"

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temp=args.temperature,
        )

        response = response.strip()
        conversation.append({"role": "assistant", "content": response})

        console.print()
        console.print("[bold green]Assistant:[/bold green]")
        console.print(Markdown(response))
        console.print()


if __name__ == "__main__":
    main()
