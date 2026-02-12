# CMM_Tools

Critical Minerals & Materials (CMM) tools monorepo — MCP servers, data libraries,
fine-tuning pipelines, and embedding utilities for the DOE Critical Minerals Modeling
initiative.

## Packages

| Package | Description | Type |
|---------|-------------|------|
| `bgs-mcp` | BGS World Mineral Statistics MCP server | MCP server |
| `claimm-mcp` | NETL EDX CLaiMM critical minerals MCP server | MCP server |
| `uncomtrade-mcp` | UN Comtrade international trade data MCP server | MCP server |
| `osti-mcp` | OSTI DOE technical reports MCP server | MCP server |
| `cmm-docs-mcp` | CMM document server (1,100+ PDFs, 500+ CSVs) | MCP server |
| `arxiv-mcp` | ArXiv paper search MCP server | MCP server |
| `scholar-mcp` | Google Scholar MCP server | MCP server |
| `scholar-api` | Google Scholar core search library | Library |
| `cmm-data` | Critical minerals data access library | Library |
| `cmm-fine-tune` | Phi-4 LoRA fine-tuning pipeline | Pipeline |
| `cmm-embedding` | Domain-specific embedding training & evaluation | Pipeline |

## Quick Start

```bash
# Install all packages in development mode
uv sync

# Run linting and formatting checks
make check

# Run a specific MCP server
uv run python -m bgs_mcp.server
```

## Development

```bash
# Format code
make fmt

# Lint
make lint

# Type-check
make type-check

# Run tests
make test

# Run all checks
make check
```

## Project Structure

```
CMM_Tools/
├── pyproject.toml          # uv workspace root + shared dev config
├── packages/               # All sub-packages
│   ├── bgs-mcp/
│   ├── claimm-mcp/
│   ├── uncomtrade-mcp/
│   ├── osti-mcp/
│   ├── cmm-docs-mcp/
│   ├── cmm-data/
│   ├── cmm-fine-tune/
│   ├── arxiv-mcp/
│   ├── scholar-api/
│   ├── scholar-mcp/
│   └── cmm-embedding/
├── schemas/                # Canonical data schemas
├── scripts/                # Utility scripts
└── tests/                  # Cross-package integration tests
```

## License

MIT
