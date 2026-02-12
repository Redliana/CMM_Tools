# CMM MCP Server — Critical Minerals and Materials Document Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that gives Claude (and other MCP-compatible LLM clients) programmatic access to a curated collection of DOE technical reports, USGS mineral-commodity datasets, and Mistral-powered OCR extraction — all focused on Critical Minerals and Materials (CMM).

> **Note:** For international trade data queries (UN Comtrade), use the dedicated **[UNComtrade_MCP](../../Data_Needs/UNComtrade_MCP/)** server, which provides a more comprehensive set of trade analysis tools.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Sources](#data-sources)
4. [Capabilities & Tools](#capabilities--tools)
   - [Document Tools](#document-tools)
   - [Search Tools](#search-tools)
   - [OCR Tools](#ocr-tools)
   - [Batch Processing Tools](#batch-processing-tools)
   - [Data Tools](#data-tools)
   - [Utility Tools](#utility-tools)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [Commodity Codes](#commodity-codes)
11. [Technical Details](#technical-details)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The CMM MCP Server is designed to support research and analysis of critical minerals and materials supply chains. It exposes a rich set of tools over the MCP stdio transport so that Claude Code (or any MCP client) can:

- Browse, read, and search **1,137 DOE/OSTI technical reports and journal articles** (PDFs organized by commodity)
- Query **579+ CSV datasets** from USGS, LISA Model, and NETL covering mineral commodities, ore deposits, and industry trends
- Perform **full-text search** (SQLite FTS5 with BM25 ranking) and **similarity search** (TF-IDF cosine similarity) across the entire document collection
- Run **Mistral OCR** on scanned or image-heavy PDFs for high-quality text extraction
- Analyze **scientific charts and figures** with Pixtral Large vision model to extract numerical data
- **Batch-process** documents through an OCR + chart-analysis pipeline to generate JSONL fine-tuning data for LLMs
- Export **BibTeX citations** for any document in the collection

The server is built with [FastMCP](https://github.com/jlowin/fastmcp) and communicates over **stdio transport**, making it straightforward to register with Claude Code.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Code / MCP Client                │
└────────────────────────────┬────────────────────────────────┘
                             │  stdio (JSON-RPC)
┌────────────────────────────▼────────────────────────────────┐
│                    server.py  (FastMCP)                      │
│                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ document_tools│ │  search.py   │ │      ocr.py          │ │
│  │ .py           │ │  FTS5+TF-IDF │ │  Mistral OCR +       │ │
│  │ PDF read,     │ │              │ │  Pixtral Large +     │ │
│  │ metadata,     │ │              │ │  PDF Triager         │ │
│  │ citations     │ │              │ │                      │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
│  ┌──────────────┐ ┌──────────────────────────────────────┐ │
│  │ data_tools.py│ │ batch_processor.py                   │ │
│  │ CSV query,   │ │ Batch OCR pipeline → JSONL output    │ │
│  │ schemas      │ │                                      │ │
│  └──────────────┘ └──────────────────────────────────────┘ │
│                                                             │
│                      config.py                              │
│              (paths, constants, env vars)                    │
└─────────────────────────────────────────────────────────────┘
         │                │
   ┌─────▼──────┐  ┌─────▼──────┐
   │ OSTI PDFs  │  │ CSV Data   │
   │ (1,137 docs│  │ (579+ files│
   │  by commod.)│  │  USGS/LISA/│
   │            │  │  NETL)     │
   └────────────┘  └────────────┘
```

---

## Data Sources

| Source | Description | Volume |
|--------|-------------|--------|
| **OSTI (DOE)** | Technical reports and journal articles retrieved from the DOE Office of Scientific and Technical Information | ~1,137 PDFs organized by commodity category |
| **USGS Mineral Commodities** | Salient statistics and world production data for mineral commodities | Multiple CSVs |
| **USGS Ore Deposits** | Geochemical and geospatial ore deposit data | Multiple CSVs |
| **LISA Model** | LISA (Lithium-Ion Supply Analysis) extracted model data | Multiple CSVs |
| **NETL REE Coal** | National Energy Technology Laboratory rare earth element data from coal sources (geodatabase exports) | Multiple CSVs |
| **UN Comtrade** | International trade data available via the separate **[UNComtrade_MCP](../../Data_Needs/UNComtrade_MCP/)** server | See UNComtrade_MCP |

---

## Capabilities & Tools

The server registers **21 MCP tools** across six functional groups.

### Document Tools

| Tool | Description |
|------|-------------|
| `list_documents(commodity?, limit?)` | List documents from the OSTI collection, optionally filtered by commodity code or subdomain |
| `get_document_metadata(osti_id)` | Retrieve full metadata (title, authors, abstract, DOI, subjects, etc.) for a document |
| `read_document(osti_id, max_chars?)` | Extract text content from a PDF using PyMuPDF (default limit: 50,000 chars) |
| `export_citation(osti_id)` | Export a BibTeX-formatted citation for the document |
| `search_by_commodity(commodity)` | Find all documents related to a specific commodity code or name |

### Search Tools

| Tool | Description |
|------|-------------|
| `search_documents(query, limit?)` | Full-text search across all indexed PDFs using SQLite FTS5 with BM25 ranking |
| `find_similar(osti_id, limit?)` | Find related documents using TF-IDF cosine similarity |
| `build_index()` | Build or rebuild the full-text search index (extracts text from all ~1,137 PDFs; ~30 min) |
| `get_index_status()` | Check the status of the search index (document count, FTS/TF-IDF readiness) |

### OCR Tools

| Tool | Description |
|------|-------------|
| `ocr_document(osti_id, commodity?)` | Extract text from a PDF using Mistral OCR (ideal for scanned documents) |
| `get_ocr_status()` | Check whether Mistral OCR is configured and available |
| `triage_documents(limit?, commodity?)` | Analyze PDFs across the collection to identify candidates that would benefit from OCR (image-heavy, low text extraction, encoding issues) |
| `analyze_document_for_ocr(osti_id)` | Detailed per-page analysis of a single document's OCR candidacy |
| `extract_document_full(osti_id, save_images?)` | Full extraction with images, tables, and structured content via Mistral OCR |
| `analyze_chart(image_path, custom_prompt?)` | Analyze a chart/plot image with Pixtral Large to extract axis labels, data points, and trends |
| `extract_and_analyze_document(osti_id, analyze_charts?)` | Combined OCR extraction + automatic chart/figure analysis in one pass |

### Batch Processing Tools

| Tool | Description |
|------|-------------|
| `estimate_batch_cost(osti_ids?)` | Estimate Mistral API cost before running batch processing |
| `process_documents_batch(osti_ids?, analyze_charts?, resume?)` | Batch-process documents through OCR + chart analysis; output as JSONL for LLM fine-tuning |
| `get_batch_status()` | Check current batch processing status and output locations |
| `process_single_for_finetune(osti_id, analyze_charts?)` | Process a single document and format for fine-tuning |

The batch pipeline produces:
- **Per-document JSON** in `index/batch_output/`
- **Consolidated JSONL** in `index/finetune_data/documents.jsonl`
- **Extraction summary** in `index/finetune_data/extraction_summary.md`
- **Resumable state** in `index/batch_output/processing_state.json`

### Data Tools

| Tool | Description |
|------|-------------|
| `list_datasets(category?)` | List available CSV datasets, optionally filtered by category (USGS, LISA, NETL) |
| `get_schema(dataset)` | Get column names, types, row counts, and sample values for a dataset |
| `query_csv(dataset, filters?, columns?, limit?)` | Query a CSV with filters (`>`, `<` for numeric; `~` for substring match) |
| `read_csv_sample(dataset, n_rows?)` | Preview the first N rows of a dataset |

### Utility Tools

| Tool | Description |
|------|-------------|
| `get_statistics()` | Get overall collection statistics (documents, datasets, search index, commodities) |
| `get_commodities()` | Get the list of commodity codes and subdomain codes with descriptions |

---

## Prerequisites

- **Python 3.10+**
- **Claude Code** (or any MCP-compatible client)
- **Mistral AI API key** (optional; required for OCR and chart analysis features)
- The underlying data collections (OSTI PDFs, CSV datasets, document catalog) at the paths configured in `config.py`

---

## Installation

1. **Clone the repository** (or copy the `cmm_mcp_server/` directory to your preferred location).

2. **Create and activate a virtual environment:**

   ```bash
   cd cmm_mcp_server
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   # .venv\Scripts\activate    # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   The key dependencies are:

   | Package | Purpose |
   |---------|---------|
   | `fastmcp` | MCP server framework (stdio transport) |
   | `pymupdf` (fitz) | PDF text extraction and page analysis |
   | `pandas` | CSV data loading and querying |
   | `scikit-learn` | TF-IDF vectorization and cosine similarity |
   | `numpy` | Numerical operations for search/similarity |
   | `mistralai` | Mistral OCR and Pixtral Large vision APIs |
   | `python-dotenv` | Environment variable loading from `.env` |

4. **Configure environment variables** (see [Configuration](#configuration) below).

---

## Configuration

### Environment Variables (`.env`)

Create a `.env` file in the `cmm_mcp_server/` directory with the following keys:

```env
# Required for OCR and chart analysis features
MISTRAL_API_KEY=your_mistral_api_key_here
```

This key is **optional** — the server will start without it, but OCR and chart-analysis tools will return informative error messages if invoked without it.

### Data Paths (`config.py`)

The server expects data at the paths defined in `config.py`. The key paths are:

| Constant | Description | Default |
|----------|-------------|---------|
| `DATA_ROOT` | Base directory for all data | `~/Documents/Projects/.../LLM_Fine_Tuning/Claude` |
| `OSTI_DIR` | OSTI retrieval output directory | `DATA_ROOT/OSTI_retrieval` |
| `OSTI_PDFS_DIR` | PDFs organized by commodity subdirectories | `OSTI_DIR/pdfs` |
| `OSTI_CATALOG` | Document catalog JSON file | `OSTI_DIR/document_catalog.json` |
| `SCHEMAS_JSON` | Pre-computed CSV schema definitions | `DATA_ROOT/schemas/all_schemas.json` |
| `USGS_DATA_DIR` | USGS mineral commodity data | `DATA_ROOT/USGS_Data` |
| `USGS_ORE_DIR` | USGS ore deposit data | `DATA_ROOT/USGS_Ore_Deposits` |
| `LISA_MODEL_DIR` | LISA model extracted data | `DATA_ROOT/LISA_Model/extracted_data` |
| `NETL_REE_DIR` | NETL REE coal geodatabase CSVs | `DATA_ROOT/NETL_REE_Coal/geodatabase_csv` |
| `INDEX_DIR` | Search index and batch output | `DATA_ROOT/cmm_mcp_server/index` |

**If you deploy this on a different machine**, update `DATA_ROOT` in `config.py` to point to where your data resides.

### Configurable Limits (`config.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_CSV_ROWS` | 100 | Maximum rows returned from CSV queries |
| `MAX_PDF_CHARS` | 50,000 | Maximum characters extracted from a single PDF |
| `MAX_SEARCH_RESULTS` | 20 | Maximum search results returned |

---

## Usage

### Register with Claude Code

```bash
claude mcp add --transport stdio cmm-docs -- python /path/to/cmm_mcp_server/server.py
```

Or, if using a virtual environment:

```bash
claude mcp add --transport stdio cmm-docs -- /path/to/cmm_mcp_server/.venv/bin/python /path/to/cmm_mcp_server/server.py
```

### Run Standalone (for testing)

```bash
cd cmm_mcp_server
python server.py
```

The server will start on **stdio** and wait for MCP JSON-RPC messages.

### Build the Search Index

Before using `search_documents` or `find_similar`, you need to build the search index. This is a one-time operation (approximately 30 minutes for the full collection):

```
> Use the build_index tool to create the search index.
```

The index is persisted to `index/search_index.db` (FTS5) and `index/tfidf_vectors.pkl` (TF-IDF) and does not need to be rebuilt unless the document collection changes.

### Example Interactions

**Search for documents about solvent extraction:**
```
> Search for documents about "solvent extraction rare earth separation"
```

**Query USGS commodity data:**
```
> List the available USGS datasets, then show me the schema for the lithium salient statistics file
```

**Process a document for fine-tuning:**
```
> Process document 3004920 through OCR and chart analysis for fine-tuning
```

---

## Project Structure

```
cmm_mcp_server/
├── server.py              # Main MCP server — registers all 21 tools with FastMCP
├── config.py              # Configuration: paths, API keys, constants, commodity/subdomain codes
├── document_tools.py      # DocumentManager: PDF reading, metadata, citations, commodity browsing
├── search.py              # SearchIndex: SQLite FTS5 full-text search + TF-IDF similarity
├── ocr.py                 # MistralOCR: OCR extraction, chart analysis (Pixtral Large), PDF triage
├── batch_processor.py     # BatchProcessor: batch OCR pipeline → JSONL fine-tuning data
├── data_tools.py          # DataManager: CSV listing, schema inspection, filtered queries
├── __init__.py            # Package marker
├── requirements.txt       # Python dependencies
├── .env                   # API keys (gitignored)
├── .gitignore             # Ignores .env, index/, __pycache__/
└── index/                 # Generated at runtime (gitignored)
    ├── search_index.db        # SQLite FTS5 database
    ├── tfidf_vectors.pkl      # Pickled TF-IDF matrix + vectorizer
    ├── extracted_images/      # Images extracted from PDFs via OCR
    ├── batch_output/          # Per-document JSON from batch processing
    │   └── processing_state.json
    └── finetune_data/         # Consolidated fine-tuning output
        ├── documents.jsonl
        └── extraction_summary.md
```

---

## Commodity Codes

The document collection is organized by the following commodity categories:

| Code | Description |
|------|-------------|
| `HREE` | Heavy Rare Earth Elements (Dy, Tb) |
| `LREE` | Light Rare Earth Elements (Nd, Pr) |
| `CO` | Cobalt |
| `LI` | Lithium |
| `GA` | Gallium |
| `GR` | Graphite |
| `NI` | Nickel |
| `CU` | Copper |
| `GE` | Germanium |
| `OTH` | Other (Mn, Ti, PGM, W) |

### Research Subdomains

| Code | Description |
|------|-------------|
| `T-EC` | Extraction Chemistry |
| `T-PM` | Processing Metallurgy |
| `T-GO` | Geological Occurrence |
| `S-ST` | Supply Chain Topology |
| `G-PR` | Policy/Regulatory |

---

## Technical Details

### Search Engine

The search system uses a **dual-index** approach:

1. **SQLite FTS5** — Full-text search with BM25 ranking. The FTS5 virtual table indexes OSTI ID, title, description, authors, subjects, and extracted PDF content. Queries support the standard FTS5 match syntax (boolean operators, phrase search, column filters).

2. **TF-IDF + Cosine Similarity** — A scikit-learn `TfidfVectorizer` (10,000 features, unigrams + bigrams, English stop words removed) is fitted on the concatenation of title + description + PDF text for each document. The resulting sparse matrix is saved to disk and used for `find_similar` queries via cosine similarity.

### OCR Pipeline

The OCR subsystem has three components:

- **MistralOCR** (`ocr.py`) — Wraps the Mistral OCR API (`mistral-ocr-latest` model) for high-quality text extraction from PDFs. Supports basic text extraction, full extraction with images/tables, and combined extraction + chart analysis.

- **Pixtral Large** (`ocr.py`) — Uses the `pixtral-large-latest` vision model to analyze extracted chart/plot images. Extracts chart type, axis labels, data series, data points, and trends in structured markdown format.

- **PDFTriager** (`ocr.py`) — Analyzes PDFs using PyMuPDF to identify documents that would benefit from OCR re-processing. Evaluates each page for image coverage, text extraction quality, character density, and encoding artifacts. Assigns a priority score to rank candidates.

### Batch Processing

The `BatchProcessor` (`batch_processor.py`) orchestrates large-scale document conversion:

1. Identifies OCR candidates (via the triager or an explicit list)
2. Runs Mistral OCR on each document with rate limiting (20 requests/minute)
3. Optionally runs Pixtral Large chart analysis on extracted images
4. Saves per-document JSON records and a consolidated JSONL file
5. Supports **resumable processing** via a state file — interrupted jobs can be continued from where they left off

The JSONL output format is structured for LLM instruction fine-tuning, with metadata, full text, table contents, and figure descriptions combined into a single text field per document.

### Singleton Pattern

All major components (`DocumentManager`, `DataManager`, `SearchIndex`, `MistralOCR`, `PDFTriager`, `BatchProcessor`) use a **singleton pattern** via module-level `get_*()` factory functions. This ensures that catalogs, indexes, and API clients are loaded once and reused across tool invocations within a session.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Search index not built` error | Run the `build_index` tool to create the FTS5 + TF-IDF indexes |
| `Mistral OCR not configured` | Add a valid `MISTRAL_API_KEY` to the `.env` file |
| `PDF not found for OSTI ID` | Verify that the PDF exists under `OSTI_PDFS_DIR/{commodity}/{osti_id}_*.pdf` |
| `Dataset not found` | Check that `SCHEMAS_JSON` points to a valid `all_schemas.json` and the CSV paths within it are correct |
| Slow search index build | The initial build processes ~1,137 PDFs and takes ~30 minutes. Subsequent queries use the cached index. |
| `DATA_ROOT` path errors | Update `DATA_ROOT` in `config.py` to match your local data directory layout |
