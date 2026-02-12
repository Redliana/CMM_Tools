# osti-mcp

MCP server for OSTI (Office of Scientific and Technical Information) document
metadata. Provides access to DOE technical reports and journal articles on
critical minerals including rare earths, lithium, cobalt, nickel, copper,
graphite, gallium, and germanium.

## Tools

| Tool | Description |
|------|-------------|
| `get_osti_overview` | Get collection statistics (counts by commodity and type) |
| `list_commodities` | List available commodity category codes and names |
| `search_osti_documents` | Search documents with text query and filters |
| `get_osti_document` | Get full metadata for a specific document by OSTI ID |
| `get_documents_by_commodity` | Browse documents for a specific commodity category |
| `get_recent_documents` | Get the most recently published documents |

## Configuration

No environment variables are required. The server uses a bundled document
catalog.

## Usage

```sh
uv run osti-mcp
```

### Claude Desktop

```json
{
  "mcpServers": {
    "osti-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/CMM_Tools", "run", "osti-mcp"],
      "env": {}
    }
  }
}
```
