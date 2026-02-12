.PHONY: fmt lint type-check test check clean

fmt:
	uv run ruff format packages/

lint:
	uv run ruff check packages/ --fix

type-check:
	uv run mypy packages/

test:
	uv run pytest

check: fmt lint type-check test
	@echo "All checks passed."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
