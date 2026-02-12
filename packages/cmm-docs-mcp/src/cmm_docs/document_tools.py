"""
Document tools for CMM MCP Server
Handles PDF reading, metadata, and citations
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import fitz  # PyMuPDF

from cmm_docs.config import COMMODITIES, MAX_PDF_CHARS, OSTI_CATALOG, OSTI_PDFS_DIR

if TYPE_CHECKING:
    from pathlib import Path


class DocumentManager:
    """Manages OSTI document catalog and PDF extraction"""

    def __init__(self):
        self.catalog = self._load_catalog()
        self._build_index()

    def _load_catalog(self) -> list:
        """Load document catalog from JSON"""
        if not OSTI_CATALOG.exists():
            return []
        with open(OSTI_CATALOG) as f:
            return json.load(f)

    def _build_index(self):
        """Build lookup indices for fast access"""
        self.by_id = {doc.get("osti_id"): doc for doc in self.catalog}
        self.by_commodity = {}
        for doc in self.catalog:
            cat = doc.get("commodity_category", "unknown")
            if cat not in self.by_commodity:
                self.by_commodity[cat] = []
            self.by_commodity[cat].append(doc)

    def list_documents(self, commodity: str | None = None, limit: int = 50) -> list:
        """
        list documents, optionally filtered by commodity.

        Args:
            commodity: Filter by commodity code (e.g., "LI", "HREE")
            limit: Maximum number of documents to return

        Returns:
            list of document summaries
        """
        if commodity:
            commodity = commodity.upper()
            # Handle subdomain prefix
            if commodity.startswith("SUBDOMAIN_"):
                commodity = commodity
            docs = self.by_commodity.get(commodity, [])
        else:
            docs = self.catalog

        results = []
        for doc in docs[:limit]:
            results.append(
                {
                    "osti_id": doc.get("osti_id"),
                    "title": doc.get("title"),
                    "authors": doc.get("authors", [])[:3],  # First 3 authors
                    "publication_date": doc.get("publication_date"),
                    "commodity_category": doc.get("commodity_category"),
                    "product_type": doc.get("product_type"),
                }
            )

        return results

    def get_metadata(self, osti_id: str) -> dict | None:
        """
        Get full metadata for a document by OSTI ID.

        Args:
            osti_id: The OSTI document identifier

        Returns:
            Full document metadata or None if not found
        """
        return self.by_id.get(osti_id)

    def get_pdf_path(self, osti_id: str) -> Path | None:
        """Find PDF file path for a given OSTI ID"""
        doc = self.by_id.get(osti_id)
        if not doc:
            return None

        commodity = doc.get("commodity_category", "")
        commodity_dir = OSTI_PDFS_DIR / commodity

        if not commodity_dir.exists():
            return None

        # Find PDF file (format: {osti_id}_{title}.pdf)
        for pdf in commodity_dir.glob(f"{osti_id}_*.pdf"):
            return pdf

        return None

    def read_document(self, osti_id: str, max_chars: int = MAX_PDF_CHARS) -> str:
        """
        Extract text content from a PDF document.

        Args:
            osti_id: The OSTI document identifier
            max_chars: Maximum characters to extract

        Returns:
            Extracted text content
        """
        pdf_path = self.get_pdf_path(osti_id)
        if not pdf_path:
            return f"Error: PDF not found for OSTI ID {osti_id}"

        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            total_chars = 0

            for page in doc:
                page_text = page.get_text()
                if total_chars + len(page_text) > max_chars:
                    # Truncate to limit
                    remaining = max_chars - total_chars
                    text_parts.append(page_text[:remaining])
                    text_parts.append(f"\n\n[Truncated at {max_chars} characters]")
                    break
                text_parts.append(page_text)
                total_chars += len(page_text)

            doc.close()
            return "\n".join(text_parts)

        except (OSError, ValueError) as e:
            return f"Error reading PDF: {e!s}"

    def export_citation(self, osti_id: str, format: str = "bibtex") -> str:
        """
        Export document citation in BibTeX format.

        Args:
            osti_id: The OSTI document identifier
            format: Citation format (currently only "bibtex" supported)

        Returns:
            Formatted citation string
        """
        doc = self.by_id.get(osti_id)
        if not doc:
            return f"Error: Document not found for OSTI ID {osti_id}"

        # Extract fields
        title = doc.get("title", "Unknown Title")
        authors = doc.get("authors", [])
        year = doc.get("publication_date", "")[:4] if doc.get("publication_date") else "n.d."
        doi = doc.get("doi", "")
        product_type = doc.get("product_type", "misc")

        # Format authors for BibTeX
        author_str = " and ".join(authors) if authors else "Unknown"

        # Determine entry type
        if "Journal" in product_type:
            entry_type = "article"
            journal = doc.get("journal_name", "")
            volume = doc.get("journal_volume", "")
            journal_field = f"  journal = {{{journal}}}," if journal else ""
            volume_field = f"\n  volume = {{{volume}}}," if volume else ""
        else:
            entry_type = "techreport"
            journal_field = ""
            volume_field = ""
            institution = doc.get("research_orgs", [""])[0] if doc.get("research_orgs") else ""
            if institution:
                journal_field = f"  institution = {{{institution}}},"

        # Build BibTeX entry
        bibtex = f"""@{entry_type}{{osti_{osti_id},
  title = {{{title}}},
  author = {{{author_str}}},
  year = {{{year}}},
{journal_field}{volume_field}
  doi = {{{doi}}},
  url = {{https://www.osti.gov/biblio/{osti_id}}}
}}"""

        return bibtex

    def get_statistics(self) -> dict:
        """Get collection statistics"""
        stats = {
            "total_documents": len(self.catalog),
            "by_commodity": {},
            "by_product_type": {},
        }

        for commodity, docs in self.by_commodity.items():
            stats["by_commodity"][commodity] = len(docs)

        for doc in self.catalog:
            ptype = doc.get("product_type", "Unknown")
            stats["by_product_type"][ptype] = stats["by_product_type"].get(ptype, 0) + 1

        return stats

    def search_by_commodity(self, commodity: str) -> dict:
        """
        Find all resources related to a specific commodity.

        Args:
            commodity: Commodity code or name

        Returns:
            Dictionary with commodity info and related documents
        """
        commodity = commodity.upper()

        # Get commodity description
        description = COMMODITIES.get(commodity, f"Unknown commodity: {commodity}")

        # Get documents
        docs = self.list_documents(commodity=commodity, limit=100)

        return {
            "commodity_code": commodity,
            "description": description,
            "document_count": len(docs),
            "documents": docs[:20],  # Return first 20
        }


# Singleton instance
_document_manager = None


def get_document_manager() -> DocumentManager:
    """Get or create DocumentManager singleton"""
    global _document_manager
    if _document_manager is None:
        _document_manager = DocumentManager()
    return _document_manager
