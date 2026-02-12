"""
Batch processing for OCR extraction and chart analysis.
Converts scanned PDFs and scientific figures to text for LLM fine-tuning.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from cmm_docs.config import INDEX_DIR
from cmm_docs.ocr import EXTRACTED_IMAGES_DIR, get_mistral_ocr, get_pdf_triager

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Output directories
BATCH_OUTPUT_DIR = INDEX_DIR / "batch_output"
FINETUNE_DATA_DIR = INDEX_DIR / "finetune_data"


class BatchProcessor:
    """
    Batch processor for converting PDFs to fine-tuning data.

    Pipeline:
    1. Mistral OCR extracts text and images from PDFs
    2. Pixtral Large analyzes charts/figures and converts to text
    3. Output saved in formats suitable for LLM fine-tuning
    """

    def __init__(self):
        self.ocr = get_mistral_ocr()
        self.triager = get_pdf_triager()

        # Ensure output directories exist
        BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FINETUNE_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # State file for resumable processing
        self.state_file = BATCH_OUTPUT_DIR / "processing_state.json"

        # Rate limiting
        self.requests_per_minute = 20
        self.last_request_time = 0

    def _rate_limit(self):
        """Simple rate limiting to avoid API throttling"""
        min_interval = 60.0 / self.requests_per_minute
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def estimate_cost(self, osti_ids: list[str] | None = None) -> dict:
        """
        Estimate processing cost before running batch.

        Args:
            osti_ids: list of document IDs to process (None = all OCR candidates)

        Returns:
            Cost estimate with page counts and pricing
        """
        if osti_ids is None:
            # Get OCR candidates from triage
            triage_result = self.triager.triage_documents()
            candidates = triage_result.get("candidates", [])
            osti_ids = [c["osti_id"] for c in candidates]

        total_pages = 0
        total_images_est = 0
        documents = []

        for osti_id in osti_ids:
            doc = self.triager.catalog.get(osti_id)
            if not doc:
                continue

            commodity = doc.get("commodity_category", "")
            pdf_path = self.triager._find_pdf(osti_id, commodity)

            if pdf_path and pdf_path.exists():
                # Get page count using PyMuPDF
                import fitz

                try:
                    pdf_doc = fitz.open(pdf_path)
                    page_count = len(pdf_doc)
                    # Estimate ~1-2 images per page for scientific docs
                    image_est = page_count * 1.5
                    pdf_doc.close()

                    total_pages += page_count
                    total_images_est += image_est
                    documents.append(
                        {
                            "osti_id": osti_id,
                            "pages": page_count,
                            "title": doc.get("title", "Unknown")[:60],
                        }
                    )
                except (OSError, ValueError):
                    pass

        # Pricing estimates (as of 2025)
        ocr_cost_per_1000_pages = 1.00  # $1 per 1000 pages
        pixtral_cost_per_1000_images = 2.00  # Estimate for vision analysis

        ocr_cost = (total_pages / 1000) * ocr_cost_per_1000_pages
        pixtral_cost = (total_images_est / 1000) * pixtral_cost_per_1000_images

        return {
            "document_count": len(documents),
            "total_pages": total_pages,
            "estimated_images": int(total_images_est),
            "estimated_cost": {
                "ocr": f"${ocr_cost:.2f}",
                "pixtral": f"${pixtral_cost:.2f}",
                "total": f"${ocr_cost + pixtral_cost:.2f}",
            },
            "documents": documents[:20],  # Show first 20
            "note": "Pixtral cost is estimated. Actual cost depends on image count.",
        }

    def load_state(self) -> dict:
        """Load processing state for resumption"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"processed": [], "failed": [], "last_updated": None}

    def save_state(self, state: dict):
        """Save processing state"""
        state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def process_document(
        self, osti_id: str, analyze_charts: bool = True, save_images: bool = True
    ) -> dict:
        """
        Process a single document through the full pipeline.

        Args:
            osti_id: Document OSTI ID
            analyze_charts: Whether to analyze images with Pixtral
            save_images: Whether to save extracted images to disk

        Returns:
            Processed document data ready for fine-tuning
        """
        doc = self.triager.catalog.get(osti_id)
        if not doc:
            return {"error": f"Document {osti_id} not found in catalog"}

        commodity = doc.get("commodity_category", "")
        pdf_path = self.triager._find_pdf(osti_id, commodity)

        if not pdf_path:
            return {"error": f"PDF not found for {osti_id}"}

        # set up output directory for images
        output_dir = None
        if save_images:
            output_dir = EXTRACTED_IMAGES_DIR / osti_id
            output_dir.mkdir(parents=True, exist_ok=True)

        # Rate limit API calls
        self._rate_limit()

        # Extract with OCR
        if analyze_charts:
            result = self.ocr.extract_and_analyze_charts(pdf_path, output_dir=output_dir)
        else:
            result = self.ocr.extract_full(pdf_path, output_dir=output_dir)

        if not result.get("success"):
            return {"error": result.get("error"), "osti_id": osti_id}

        # Build fine-tuning record
        finetune_record = {
            "osti_id": osti_id,
            "title": doc.get("title", ""),
            "authors": doc.get("authors", []),
            "abstract": doc.get("description", ""),
            "commodity": commodity,
            "doi": doc.get("doi", ""),
            "publication_date": doc.get("publication_date", ""),
            "subjects": doc.get("subjects", []),
            # Extracted content
            "text": result.get("text", ""),
            "page_count": result.get("page_count", 0),
            "char_count": result.get("statistics", {}).get("total_chars", 0),
            # Tables as text
            "tables": [t.get("content", "") for t in result.get("tables", [])],
            # Chart analyses as text
            "chart_analyses": [],
            # Metadata
            "extraction_model": result.get("model", ""),
            "processed_at": datetime.now().isoformat(),
        }

        # Add chart analyses
        for chart in result.get("chart_analyses", []):
            finetune_record["chart_analyses"].append(
                {
                    "page": chart.get("page"),
                    "description": chart.get("analysis", ""),
                }
            )

        return finetune_record

    def process_batch(
        self,
        osti_ids: list[str] | None = None,
        analyze_charts: bool = True,
        resume: bool = True,
        progress_callback: Callable | None = None,
    ) -> dict:
        """
        Process multiple documents in batch.

        Args:
            osti_ids: list of document IDs (None = all OCR candidates)
            analyze_charts: Whether to analyze charts with Pixtral
            resume: Whether to resume from previous state
            progress_callback: Optional callback(current, total, message)

        Returns:
            Batch processing results
        """
        # Get document list
        if osti_ids is None:
            triage_result = self.triager.triage_documents()
            candidates = triage_result.get("candidates", [])
            osti_ids = [c["osti_id"] for c in candidates]

        # Load state for resumption
        state = self.load_state() if resume else {"processed": [], "failed": []}

        # Filter out already processed documents
        if resume:
            remaining = [oid for oid in osti_ids if oid not in state["processed"]]
        else:
            remaining = osti_ids
            state = {"processed": [], "failed": []}

        total = len(osti_ids)
        processed_count = len(state["processed"])

        results = {
            "total": total,
            "already_processed": processed_count,
            "to_process": len(remaining),
            "successful": [],
            "failed": [],
        }

        # Process each document
        for i, osti_id in enumerate(remaining):
            current = processed_count + i + 1

            if progress_callback:
                progress_callback(current, total, f"Processing {osti_id}")

            try:
                record = self.process_document(osti_id, analyze_charts=analyze_charts)

                if record.get("error"):
                    state["failed"].append(osti_id)
                    results["failed"].append({"osti_id": osti_id, "error": record["error"]})
                else:
                    # Save individual record
                    self._save_record(record)
                    state["processed"].append(osti_id)
                    results["successful"].append(osti_id)

                # Save state periodically
                if (i + 1) % 5 == 0:
                    self.save_state(state)

            except (OSError, ValueError, KeyError) as e:
                logger.error(f"Error processing {osti_id}: {e}")
                state["failed"].append(osti_id)
                results["failed"].append({"osti_id": osti_id, "error": str(e)})

        # Final state save
        self.save_state(state)

        # Generate consolidated output files
        self._generate_finetune_files()

        results["output_files"] = {
            "jsonl": str(FINETUNE_DATA_DIR / "documents.jsonl"),
            "state": str(self.state_file),
        }

        return results

    def _save_record(self, record: dict):
        """Save a single processed record"""
        osti_id = record.get("osti_id")
        output_file = BATCH_OUTPUT_DIR / f"{osti_id}.json"

        with open(output_file, "w") as f:
            json.dump(record, f, indent=2)

    def _generate_finetune_files(self):
        """Generate consolidated fine-tuning data files"""
        # Collect all processed records
        records = []
        for json_file in BATCH_OUTPUT_DIR.glob("*.json"):
            if json_file.name == "processing_state.json":
                continue
            try:
                with open(json_file) as f:
                    records.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                pass

        if not records:
            return

        # Generate JSONL for fine-tuning
        jsonl_path = FINETUNE_DATA_DIR / "documents.jsonl"
        with open(jsonl_path, "w") as f:
            for record in records:
                # Create fine-tuning format
                finetune_entry = self._format_for_finetuning(record)
                f.write(json.dumps(finetune_entry) + "\n")

        # Generate summary markdown
        summary_path = FINETUNE_DATA_DIR / "extraction_summary.md"
        with open(summary_path, "w") as f:
            f.write("# Document Extraction Summary\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Total documents: {len(records)}\n\n")

            total_chars = sum(r.get("char_count", 0) for r in records)
            total_charts = sum(len(r.get("chart_analyses", [])) for r in records)

            f.write(f"Total characters: {total_chars:,}\n")
            f.write(f"Total charts analyzed: {total_charts}\n\n")

            f.write("## Documents\n\n")
            for r in records:
                f.write(f"### {r.get('osti_id')} - {r.get('title', 'Unknown')[:60]}\n")
                f.write(f"- Commodity: {r.get('commodity')}\n")
                f.write(f"- Pages: {r.get('page_count')}\n")
                f.write(f"- Characters: {r.get('char_count'):,}\n")
                f.write(f"- Charts: {len(r.get('chart_analyses', []))}\n\n")

    def _format_for_finetuning(self, record: dict) -> dict:
        """
        Format a record for LLM fine-tuning.

        Creates a structure suitable for instruction fine-tuning with
        document content and chart descriptions as context.
        """
        # Build the full text content including chart descriptions
        content_parts = []

        # Title and metadata
        content_parts.append(f"Title: {record.get('title', 'Unknown')}")
        if record.get("authors"):
            authors = ", ".join(record["authors"][:5])
            content_parts.append(f"Authors: {authors}")
        content_parts.append(f"Commodity: {record.get('commodity', 'Unknown')}")
        content_parts.append("")

        # Abstract
        if record.get("abstract"):
            content_parts.append("Abstract:")
            content_parts.append(record["abstract"])
            content_parts.append("")

        # Main text
        content_parts.append("Content:")
        content_parts.append(record.get("text", ""))

        # Tables
        if record.get("tables"):
            content_parts.append("")
            content_parts.append("Tables:")
            for i, table in enumerate(record["tables"], 1):
                content_parts.append(f"\nTable {i}:")
                content_parts.append(table)

        # Chart analyses
        if record.get("chart_analyses"):
            content_parts.append("")
            content_parts.append("Figure Descriptions:")
            for i, chart in enumerate(record["chart_analyses"], 1):
                content_parts.append(f"\nFigure {i} (Page {chart.get('page', '?')}):")
                content_parts.append(chart.get("description", ""))

        full_text = "\n".join(content_parts)

        return {
            "osti_id": record.get("osti_id"),
            "title": record.get("title"),
            "commodity": record.get("commodity"),
            "subjects": record.get("subjects", []),
            "text": full_text,
            "metadata": {
                "doi": record.get("doi"),
                "publication_date": record.get("publication_date"),
                "page_count": record.get("page_count"),
                "char_count": len(full_text),
                "chart_count": len(record.get("chart_analyses", [])),
                "table_count": len(record.get("tables", [])),
            },
        }

    def get_processing_status(self) -> dict:
        """Get current processing status"""
        state = self.load_state()

        # Count output files
        output_files = list(BATCH_OUTPUT_DIR.glob("*.json"))
        output_count = len([f for f in output_files if f.name != "processing_state.json"])

        return {
            "processed_count": len(state.get("processed", [])),
            "failed_count": len(state.get("failed", [])),
            "output_files": output_count,
            "last_updated": state.get("last_updated"),
            "output_directory": str(BATCH_OUTPUT_DIR),
            "finetune_directory": str(FINETUNE_DATA_DIR),
        }


# Singleton instance
_batch_processor = None


def get_batch_processor() -> BatchProcessor:
    """Get or create BatchProcessor singleton"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor
