#!/usr/bin/env python3
"""
CMM MCP Server - Critical Minerals and Materials Document Server

Provides Claude with access to:
- 1,137 DOE technical reports and journal articles (PDFs)
- 579+ CSV datasets on mineral commodities
- Full-text search across all documents
- Similarity search to find related papers
- BibTeX citation export

Usage:
    python server.py

Then configure in Claude Code:
    claude mcp add --transport stdio cmm-docs -- python /path/to/server.py
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

from cmm_docs.batch_processor import get_batch_processor
from cmm_docs.config import COMMODITIES, SUBDOMAINS
from cmm_docs.data_tools import get_data_manager
from cmm_docs.document_tools import get_document_manager
from cmm_docs.ocr import get_mistral_ocr, get_pdf_triager
from cmm_docs.search import get_search_index

# Create MCP server
mcp = FastMCP(
    name="CMM Document Server",
    version="1.0.0",
)

# =============================================================================
# Document Tools
# =============================================================================


@mcp.tool()
def list_documents(commodity: str | None = None, limit: int = 50) -> list:
    """
    list available documents from the OSTI collection.

    Args:
        commodity: Filter by commodity code (HREE, LREE, CO, LI, GA, GR, NI, CU, GE, OTH)
                   or subdomain (subdomain_T-EC, subdomain_T-PM, etc.)
        limit: Maximum number of documents to return (default 50)

    Returns:
        list of document summaries with OSTI ID, title, authors, and category
    """
    dm = get_document_manager()
    return dm.list_documents(commodity=commodity, limit=limit)


@mcp.tool()
def get_document_metadata(osti_id: str) -> dict:
    """
    Get full metadata for a document by OSTI ID.

    Args:
        osti_id: The OSTI document identifier (e.g., "3004920")

    Returns:
        Full document metadata including title, authors, abstract, DOI, etc.
    """
    dm = get_document_manager()
    result = dm.get_metadata(osti_id)
    if result is None:
        return {"error": f"Document not found: {osti_id}"}
    return result


@mcp.tool()
def read_document(osti_id: str, max_chars: int = 50000) -> str:
    """
    Extract and return text content from a PDF document.

    Args:
        osti_id: The OSTI document identifier
        max_chars: Maximum characters to extract (default 50000)

    Returns:
        Extracted text content from the PDF
    """
    dm = get_document_manager()
    return dm.read_document(osti_id, max_chars=max_chars)


@mcp.tool()
def export_citation(osti_id: str) -> str:
    """
    Export document citation in BibTeX format.

    Args:
        osti_id: The OSTI document identifier

    Returns:
        BibTeX formatted citation string
    """
    dm = get_document_manager()
    return dm.export_citation(osti_id)


@mcp.tool()
def search_by_commodity(commodity: str) -> dict:
    """
    Find all resources related to a specific commodity.

    Args:
        commodity: Commodity code (HREE, LREE, CO, LI, GA, GR, NI, CU, GE, OTH)
                   or full name (e.g., "lithium", "cobalt")

    Returns:
        Commodity information and list of related documents
    """
    dm = get_document_manager()
    return dm.search_by_commodity(commodity)


# =============================================================================
# Search Tools
# =============================================================================


@mcp.tool()
def search_documents(query: str, limit: int = 20) -> list:
    """
    Full-text search across all indexed PDF documents.

    Args:
        query: Search query (e.g., "solvent extraction rare earth")
        limit: Maximum results to return (default 20)

    Returns:
        list of matching documents with relevance scores
    """
    idx = get_search_index()
    return idx.search(query, limit=limit)


@mcp.tool()
def find_similar(osti_id: str, limit: int = 5) -> list:
    """
    Find documents similar to a given document using TF-IDF similarity.

    Args:
        osti_id: Reference document OSTI ID
        limit: Number of similar documents to return (default 5)

    Returns:
        list of similar documents with similarity scores
    """
    idx = get_search_index()
    return idx.find_similar(osti_id, limit=limit)


@mcp.tool()
def build_index() -> dict:
    """
    Build or rebuild the full-text search index.

    This extracts text from all PDFs and creates a searchable index.
    Takes approximately 30 minutes for 1,137 documents.

    Returns:
        Indexing statistics
    """
    idx = get_search_index()

    def progress(current, total, message):
        print(f"[{current}/{total}] {message}", file=sys.stderr)

    return idx.build_index(progress_callback=progress)


@mcp.tool()
def get_index_status() -> dict:
    """
    Get the status of the search index.

    Returns:
        Index statistics including document count and indexing status
    """
    idx = get_search_index()
    return idx.get_index_stats()


# =============================================================================
# OCR Tools
# =============================================================================


@mcp.tool()
def ocr_document(osti_id: str, commodity: str | None = None) -> dict:
    """
    Extract text from a PDF document using Mistral OCR.

    This provides high-quality text extraction, especially for scanned documents
    or PDFs with complex layouts, tables, and equations.

    Args:
        osti_id: The OSTI document identifier
        commodity: Optional commodity category to narrow PDF search

    Returns:
        Extracted text content with page-by-page breakdown
    """
    ocr = get_mistral_ocr()
    return ocr.extract_text_by_osti_id(osti_id, commodity)


@mcp.tool()
def get_ocr_status() -> dict:
    """
    Check if Mistral OCR is configured and available.

    Returns:
        OCR configuration status
    """
    ocr = get_mistral_ocr()
    return {
        "available": ocr.is_available(),
        "model": ocr.model if ocr.is_available() else None,
        "message": "Mistral OCR is ready"
        if ocr.is_available()
        else "Set MISTRAL_API_KEY in .env file to enable OCR",
    }


@mcp.tool()
def triage_documents(limit: int | None = None, commodity: str | None = None) -> dict:
    """
    Analyze PDFs to identify candidates that would benefit from Mistral OCR.

    Examines documents for:
    - Image-heavy pages (scanned documents)
    - Low text extraction (< 200 chars/page)
    - Text quality issues (encoding problems, gibberish)
    - Pages with images but little text

    Args:
        limit: Maximum documents to analyze (None for all ~1100 docs)
        commodity: Filter by commodity category (HREE, LREE, CO, LI, etc.)

    Returns:
        Prioritized list of OCR candidates with reasons and statistics
    """
    triager = get_pdf_triager()

    def progress(current, total, message):
        print(f"[{current}/{total}] {message}", file=sys.stderr)

    return triager.triage_documents(limit=limit, commodity=commodity, progress_callback=progress)


@mcp.tool()
def analyze_document_for_ocr(osti_id: str) -> dict:
    """
    Analyze a single document to determine if it would benefit from OCR.

    Args:
        osti_id: The OSTI document identifier

    Returns:
        Detailed analysis including page-by-page breakdown and OCR recommendation
    """

    triager = get_pdf_triager()

    # Find the document in catalog
    doc = triager.catalog.get(osti_id)
    if not doc:
        return {"error": f"Document {osti_id} not found in catalog"}

    commodity = doc.get("commodity_category", "")
    pdf_path = triager._find_pdf(osti_id, commodity)

    if not pdf_path:
        return {"error": f"PDF not found for {osti_id}"}

    analysis = triager.analyze_pdf(pdf_path)
    analysis["osti_id"] = osti_id
    analysis["title"] = doc.get("title", "Unknown")
    analysis["commodity"] = commodity

    return analysis


@mcp.tool()
def extract_document_full(osti_id: str, save_images: bool = True) -> dict:
    """
    Full document extraction with images, tables, and structured content.

    Uses Mistral OCR to extract:
    - Text with formatting preserved
    - Embedded images (saved to disk if save_images=True)
    - Tables in markdown format
    - Document structure

    Args:
        osti_id: The OSTI document identifier
        save_images: Whether to save extracted images to disk

    Returns:
        Full extraction results including text, images, tables, and statistics
    """
    from cmm_docs.ocr import EXTRACTED_IMAGES_DIR

    ocr = get_mistral_ocr()
    triager = get_pdf_triager()

    # Find the document
    doc = triager.catalog.get(osti_id)
    if not doc:
        return {"error": f"Document {osti_id} not found in catalog"}

    commodity = doc.get("commodity_category", "")
    pdf_path = triager._find_pdf(osti_id, commodity)

    if not pdf_path:
        return {"error": f"PDF not found for {osti_id}"}

    # set up output directory for images
    output_dir = None
    if save_images:
        output_dir = EXTRACTED_IMAGES_DIR / osti_id
        output_dir.mkdir(parents=True, exist_ok=True)

    result = ocr.extract_full(pdf_path, output_dir=output_dir)

    if result.get("success"):
        result["osti_id"] = osti_id
        result["title"] = doc.get("title", "Unknown")
        result["commodity"] = commodity

    return result


@mcp.tool()
def analyze_chart(image_path: str, custom_prompt: str | None = None) -> dict:
    """
    Analyze a chart/plot image using Pixtral Large to extract numerical data.

    Pixtral Large can:
    - Identify chart types (line, bar, scatter, etc.)
    - Extract axis labels and units
    - Extract data points and values
    - Identify trends and patterns

    Args:
        image_path: Path to an image file (PNG, JPG)
        custom_prompt: Optional custom analysis prompt

    Returns:
        Chart analysis with extracted data points
    """
    ocr = get_mistral_ocr()
    return ocr.analyze_chart(image_path, custom_prompt)


@mcp.tool()
def extract_and_analyze_document(osti_id: str, analyze_charts: bool = True) -> dict:
    """
    Full document extraction with automatic chart analysis.

    Combines OCR extraction with Pixtral Large vision analysis to:
    1. Extract all text, images, and tables from the PDF
    2. Identify and analyze charts/plots in the document
    3. Extract numerical data from scientific visualizations

    Args:
        osti_id: The OSTI document identifier
        analyze_charts: Whether to analyze extracted images as charts

    Returns:
        Full extraction results plus chart analyses with extracted data
    """
    from cmm_docs.ocr import EXTRACTED_IMAGES_DIR

    ocr = get_mistral_ocr()
    triager = get_pdf_triager()

    # Find the document
    doc = triager.catalog.get(osti_id)
    if not doc:
        return {"error": f"Document {osti_id} not found in catalog"}

    commodity = doc.get("commodity_category", "")
    pdf_path = triager._find_pdf(osti_id, commodity)

    if not pdf_path:
        return {"error": f"PDF not found for {osti_id}"}

    # set up output directory
    output_dir = EXTRACTED_IMAGES_DIR / osti_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if analyze_charts:
        result = ocr.extract_and_analyze_charts(pdf_path, output_dir=output_dir)
    else:
        result = ocr.extract_full(pdf_path, output_dir=output_dir)

    if result.get("success"):
        result["osti_id"] = osti_id
        result["title"] = doc.get("title", "Unknown")
        result["commodity"] = commodity

    return result


# =============================================================================
# Batch Processing Tools
# =============================================================================


@mcp.tool()
def estimate_batch_cost(osti_ids: list[str] | None = None) -> dict:
    """
    Estimate cost before running batch processing.

    Args:
        osti_ids: list of document IDs (None = all 81 OCR candidates)

    Returns:
        Cost estimate with page counts and pricing breakdown
    """
    processor = get_batch_processor()
    return processor.estimate_cost(osti_ids)


@mcp.tool()
def process_documents_batch(
    osti_ids: list[str] | None = None, analyze_charts: bool = True, resume: bool = True
) -> dict:
    """
    Batch process documents for LLM fine-tuning.

    Pipeline:
    1. Mistral OCR extracts text and images from PDFs
    2. Pixtral Large analyzes charts/figures → text descriptions
    3. Output saved as JSONL for fine-tuning

    Args:
        osti_ids: list of document IDs (None = all OCR candidates)
        analyze_charts: Whether to analyze images with Pixtral
        resume: Whether to resume from previous state

    Returns:
        Processing results with output file locations
    """
    processor = get_batch_processor()

    def progress(current, total, message):
        print(f"[{current}/{total}] {message}", file=sys.stderr)

    return processor.process_batch(
        osti_ids=osti_ids, analyze_charts=analyze_charts, resume=resume, progress_callback=progress
    )


@mcp.tool()
def get_batch_status() -> dict:
    """
    Get current batch processing status.

    Returns:
        Processing status including counts and output locations
    """
    processor = get_batch_processor()
    return processor.get_processing_status()


@mcp.tool()
def process_single_for_finetune(osti_id: str, analyze_charts: bool = True) -> dict:
    """
    Process a single document and format for fine-tuning.

    Args:
        osti_id: Document OSTI ID
        analyze_charts: Whether to analyze charts with Pixtral

    Returns:
        Processed document ready for fine-tuning
    """
    processor = get_batch_processor()
    return processor.process_document(osti_id, analyze_charts=analyze_charts)


# =============================================================================
# Data Tools
# =============================================================================


@mcp.tool()
def list_datasets(category: str | None = None) -> list:
    """
    list available CSV datasets.

    Args:
        category: Optional category filter (e.g., "USGS", "LISA", "NETL")

    Returns:
        list of available datasets with file counts and row counts
    """
    dm = get_data_manager()
    return dm.list_datasets(category=category)


@mcp.tool()
def get_schema(dataset: str) -> dict:
    """
    Get schema/column information for a specific dataset.

    Args:
        dataset: Dataset filename (e.g., "ChemData1.csv", "mcs2023-lithi_salient.csv")

    Returns:
        Schema with column names, types, and sample values
    """
    dm = get_data_manager()
    return dm.get_schema(dataset)


@mcp.tool()
def query_csv(
    dataset: str,
    filters: dict | None = None,
    columns: list[str] | None = None,
    limit: int = 100,
) -> dict:
    """
    Query a CSV file with optional filters.

    Args:
        dataset: Dataset filename
        filters: Column filters (e.g., {"Year": 2023, "Element": "~lithium"})
                 Prefix with > or < for numeric comparisons
                 Prefix with ~ for contains/substring search
        columns: list of columns to return (None for all)
        limit: Maximum rows to return (default 100)

    Returns:
        Query results with matching data
    """
    dm = get_data_manager()
    return dm.query_csv(dataset, filters=filters, columns=columns, limit=limit)


@mcp.tool()
def read_csv_sample(dataset: str, n_rows: int = 10) -> dict:
    """
    Read first N rows from a CSV dataset.

    Args:
        dataset: Dataset filename
        n_rows: Number of rows to read (default 10)

    Returns:
        Sample data from the dataset
    """
    dm = get_data_manager()
    return dm.read_csv_sample(dataset, n_rows=n_rows)


# =============================================================================
# Utility Tools
# =============================================================================


@mcp.tool()
def get_statistics() -> dict:
    """
    Get overall collection statistics.

    Returns:
        Statistics about documents, datasets, commodities, etc.
    """
    dm = get_document_manager()
    data_mgr = get_data_manager()
    idx = get_search_index()

    return {
        "documents": dm.get_statistics(),
        "datasets": data_mgr.get_statistics(),
        "search_index": idx.get_index_stats(),
        "commodities": COMMODITIES,
        "subdomains": SUBDOMAINS,
    }


@mcp.tool()
def get_commodities() -> dict:
    """
    Get list of commodity codes and descriptions.

    Returns:
        Dictionary of commodity codes and their descriptions
    """
    return {
        "commodities": COMMODITIES,
        "subdomains": SUBDOMAINS,
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the CMM Document Server."""
    print("Starting CMM Document Server...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
