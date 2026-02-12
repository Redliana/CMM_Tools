# claimm-mcp

MCP server for searching and managing NETL EDX CLaiMM (Critical Minerals and
Materials) datasets. Supports AI-powered search, schema detection, dataset
CRUD operations, and file uploads.

## Tools

### Read Tools

| Tool | Description |
|------|-------------|
| `search_claimm_data` | AI-powered natural language search across CLAIMM datasets |
| `list_claimm_datasets` | List available datasets, optionally filtered by category |
| `get_dataset_details` | Get full metadata and resource list for a dataset |
| `get_resource_details` | Get metadata and download URL for a specific file |
| `get_download_url` | Get the direct download URL for a resource |
| `detect_file_schema` | Detect column headers and types from a CSV/Excel file |
| `detect_dataset_schemas` | Detect schemas for all tabular files in a dataset |
| `ask_about_data` | Ask natural language questions about CLAIMM data |

### Write Tools

| Tool | Description |
|------|-------------|
| `create_dataset` | Create a new dataset in EDX |
| `update_dataset` | Update an existing dataset's metadata |
| `upload_file` | Upload a file to an existing dataset |
| `update_file` | Update file metadata or replace file content |
| `delete_file` | Delete a file from EDX |
| `delete_dataset` | Delete a dataset and all its files |

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `EDX_API_KEY` | Required | NETL EDX API key for data access |
| `OPENAI_API_KEY` | Optional | OpenAI key for LLM-powered search |
| `ANTHROPIC_API_KEY` | Optional | Anthropic key for LLM-powered search |
| `GOOGLE_API_KEY` | Optional | Google AI key for LLM-powered search |
| `XAI_API_KEY` | Optional | xAI (Grok) key for LLM-powered search |
| `DEFAULT_LLM_PROVIDER` | Optional | LLM provider: openai, anthropic, google, xai |

At least one LLM key is recommended for AI-powered search and Q&A tools.

## Usage

```sh
uv run claimm-mcp
```

### Claude Desktop

```json
{
  "mcpServers": {
    "claimm-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/CMM_Tools", "run", "claimm-mcp"],
      "env": {
        "EDX_API_KEY": "your-edx-key"
      }
    }
  }
}
```
