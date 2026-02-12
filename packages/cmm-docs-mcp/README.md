# cmm-docs-mcp

MCP server for the CMM document collection: 1,137+ DOE technical reports and
journal articles (PDFs) plus 579+ CSV datasets on critical mineral commodities.
Provides full-text search, OCR extraction, batch processing, and CSV querying.

## Tools

### Document Tools

| Tool | Description |
|------|-------------|
| `list_documents` | List documents, optionally filtered by commodity code |
| `get_document_metadata` | Get full metadata for a document by OSTI ID |
| `read_document` | Extract text content from a PDF document |
| `export_citation` | Export a document citation in BibTeX format |
| `search_by_commodity` | Find all resources related to a specific commodity |

### Search Tools

| Tool | Description |
|------|-------------|
| `search_documents` | Full-text search across all indexed PDFs |
| `find_similar` | Find documents similar to a given document (TF-IDF) |
| `build_index` | Build or rebuild the full-text search index |
| `get_index_status` | Get search index statistics |

### OCR Tools

| Tool | Description |
|------|-------------|
| `ocr_document` | Extract text from a PDF using Mistral OCR |
| `get_ocr_status` | Check if Mistral OCR is configured |
| `triage_documents` | Identify documents that would benefit from OCR |
| `analyze_document_for_ocr` | Analyze a single document for OCR suitability |
| `extract_document_full` | Full extraction with images, tables, and structure |
| `analyze_chart` | Analyze a chart image using Pixtral Large |
| `extract_and_analyze_document` | Full extraction with automatic chart analysis |

### Batch Processing Tools

| Tool | Description |
|------|-------------|
| `estimate_batch_cost` | Estimate cost before running batch OCR |
| `process_documents_batch` | Batch process documents for LLM fine-tuning |
| `get_batch_status` | Get current batch processing status |
| `process_single_for_finetune` | Process one document for fine-tuning output |

### Data Tools

| Tool | Description |
|------|-------------|
| `list_datasets` | List available CSV datasets by category |
| `get_schema` | Get column schema for a specific CSV dataset |
| `query_csv` | Query a CSV file with filters |
| `read_csv_sample` | Read first N rows from a CSV dataset |

### Utility Tools

| Tool | Description |
|------|-------------|
| `get_statistics` | Get overall collection statistics |
| `get_commodities` | Get commodity codes and descriptions |

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `CMM_DOCS_DATA_ROOT` | Required | Root directory containing OSTI PDFs and CSV data |
| `MISTRAL_API_KEY` | Optional | Mistral API key for OCR and chart analysis |

## Usage

```sh
uv run cmm-docs-mcp
```

### Claude Desktop

```json
{
  "mcpServers": {
    "cmm-docs-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/CMM_Tools", "run", "cmm-docs-mcp"],
      "env": {
        "CMM_DOCS_DATA_ROOT": "/path/to/data",
        "MISTRAL_API_KEY": "your-mistral-key"
      }
    }
  }
}
```
