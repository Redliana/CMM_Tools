"""
Mistral OCR integration for CMM MCP Server
Provides high-quality text extraction from PDFs using Mistral's OCR API
Includes triage functionality to identify PDFs that would benefit from OCR
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # PyMuPDF

from cmm_docs.config import INDEX_DIR, MISTRAL_API_KEY, OSTI_CATALOG, OSTI_PDFS_DIR

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Default output directory for extracted images
EXTRACTED_IMAGES_DIR = INDEX_DIR / "extracted_images"


class MistralOCR:
    """Wrapper for Mistral OCR API with advanced extraction capabilities"""

    def __init__(self):
        self.api_key = MISTRAL_API_KEY
        self.client = None
        self.model = "mistral-ocr-latest"
        self.vision_model = "pixtral-large-latest"

        if self.api_key and self.api_key != "your_api_key_here":
            from mistralai import Mistral

            self.client = Mistral(api_key=self.api_key)

    def is_available(self) -> bool:
        """Check if Mistral OCR is configured and available"""
        return self.client is not None

    def extract_text_from_pdf(
        self, pdf_path: Path, include_images: bool = False, table_format: str = "markdown"
    ) -> dict:
        """
        Extract text from a PDF file using Mistral OCR.

        Args:
            pdf_path: Path to the PDF file
            include_images: Whether to include image descriptions
            table_format: Format for tables ("markdown", "html", or None)

        Returns:
            Dictionary with extracted content and metadata
        """
        if not self.is_available():
            return {"error": "Mistral OCR not configured. set MISTRAL_API_KEY in .env file."}

        if not pdf_path.exists():
            return {"error": f"PDF file not found: {pdf_path}"}

        try:
            # Read and encode the PDF
            with open(pdf_path, "rb") as f:
                pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

            # Call Mistral OCR API
            response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_data}",
                },
                include_image_base64=include_images,
            )

            # Extract text from all pages
            full_text = []
            page_contents = []

            for page in response.pages:
                page_text = page.markdown if hasattr(page, "markdown") else str(page)
                full_text.append(page_text)
                page_contents.append(
                    {
                        "page_number": page.index + 1
                        if hasattr(page, "index")
                        else len(page_contents) + 1,
                        "content": page_text,
                    }
                )

            return {
                "success": True,
                "text": "\n\n".join(full_text),
                "pages": page_contents,
                "page_count": len(page_contents),
                "model": self.model,
            }

        except (OSError, ValueError, KeyError) as e:
            logger.error(f"Mistral OCR failed for {pdf_path}: {e}")
            return {"error": str(e)}

    def extract_text_by_osti_id(self, osti_id: str, commodity: str | None = None) -> dict:
        """
        Extract text from a PDF by OSTI ID.

        Args:
            osti_id: The OSTI document identifier
            commodity: Optional commodity category to narrow search

        Returns:
            Dictionary with extracted content
        """
        # Find the PDF file
        pdf_path = self._find_pdf(osti_id, commodity)

        if not pdf_path:
            return {"error": f"PDF not found for OSTI ID: {osti_id}"}

        result = self.extract_text_from_pdf(pdf_path)
        if result.get("success"):
            result["osti_id"] = osti_id
            result["pdf_path"] = str(pdf_path)

        return result

    def _find_pdf(self, osti_id: str, commodity: str | None = None) -> Path | None:
        """Find PDF file by OSTI ID"""
        if commodity:
            # Search in specific commodity directory
            commodity_dir = OSTI_PDFS_DIR / commodity
            if commodity_dir.exists():
                for pdf in commodity_dir.glob(f"{osti_id}_*.pdf"):
                    return pdf

        # Search all commodity directories
        for commodity_dir in OSTI_PDFS_DIR.iterdir():
            if commodity_dir.is_dir():
                for pdf in commodity_dir.glob(f"{osti_id}_*.pdf"):
                    return pdf

        return None

    def extract_full(
        self, pdf_path: Path, table_format: str = "markdown", output_dir: Path | None = None
    ) -> dict:
        """
        Full extraction with images, tables, and structured content.

        Args:
            pdf_path: Path to the PDF file
            table_format: Format for tables ("markdown", "html", or None)
            output_dir: Directory to save extracted images (None to skip saving)

        Returns:
            Dictionary with:
            - text: Full extracted text
            - pages: Per-page content with markdown
            - images: list of extracted images with metadata
            - tables: list of extracted tables
            - statistics: Extraction statistics
        """
        if not self.is_available():
            return {"error": "Mistral OCR not configured. set MISTRAL_API_KEY in .env file."}

        if not pdf_path.exists():
            return {"error": f"PDF file not found: {pdf_path}"}

        try:
            # Read and encode the PDF
            with open(pdf_path, "rb") as f:
                pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

            # Call Mistral OCR API with full options
            response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_data}",
                },
                include_image_base64=True,
            )

            # Process response
            full_text = []
            pages = []
            all_images = []
            all_tables = []

            for page in response.pages:
                page_num = page.index + 1 if hasattr(page, "index") else len(pages) + 1
                page_text = page.markdown if hasattr(page, "markdown") else str(page)
                full_text.append(page_text)

                # Extract images from this page
                page_images = []
                if hasattr(page, "images") and page.images:
                    for img in page.images:
                        img_data = {
                            "page": page_num,
                            "id": img.id if hasattr(img, "id") else f"img-{len(all_images)}",
                            "image_base64": img.image_base64
                            if hasattr(img, "image_base64")
                            else None,
                        }
                        page_images.append(img_data)
                        all_images.append(img_data)

                # Extract tables from this page
                page_tables = []
                if hasattr(page, "tables") and page.tables:
                    for tbl in page.tables:
                        tbl_data = {
                            "page": page_num,
                            "content": tbl.markdown if hasattr(tbl, "markdown") else str(tbl),
                        }
                        page_tables.append(tbl_data)
                        all_tables.append(tbl_data)

                pages.append(
                    {
                        "page_number": page_num,
                        "content": page_text,
                        "image_count": len(page_images),
                        "table_count": len(page_tables),
                    }
                )

            # Save images if output directory specified
            saved_images = []
            if output_dir and all_images:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                saved_images = self._save_images(all_images, output_dir)

            return {
                "success": True,
                "text": "\n\n".join(full_text),
                "pages": pages,
                "page_count": len(pages),
                "images": all_images,
                "image_count": len(all_images),
                "tables": all_tables,
                "table_count": len(all_tables),
                "saved_images": saved_images,
                "model": self.model,
                "statistics": {
                    "total_chars": len("\n\n".join(full_text)),
                    "chars_per_page": len("\n\n".join(full_text)) / max(1, len(pages)),
                    "pages_with_images": sum(1 for p in pages if p["image_count"] > 0),
                    "pages_with_tables": sum(1 for p in pages if p["table_count"] > 0),
                },
            }

        except (OSError, ValueError, KeyError) as e:
            logger.error(f"Mistral OCR full extraction failed for {pdf_path}: {e}")
            return {"error": str(e)}

    def _save_images(self, images: list[dict], output_dir: Path) -> list[str]:
        """Save extracted images to disk"""
        saved = []
        for img in images:
            if img.get("image_base64"):
                try:
                    # Determine image format from base64 header or default to png
                    img_data = img["image_base64"]
                    if img_data.startswith("data:"):
                        # Extract format and data
                        header, data = img_data.split(",", 1)
                        if "jpeg" in header or "jpg" in header:
                            ext = "jpg"
                        elif "png" in header:
                            ext = "png"
                        else:
                            ext = "png"
                    else:
                        data = img_data
                        ext = "png"

                    filename = f"page{img['page']}_{img['id']}.{ext}"
                    filepath = output_dir / filename

                    with open(filepath, "wb") as f:
                        f.write(base64.b64decode(data))

                    saved.append(str(filepath))
                except (OSError, ValueError) as e:
                    logger.warning(f"Failed to save image {img['id']}: {e}")

        return saved

    def analyze_chart(self, image_source: str, prompt: str | None = None) -> dict:
        """
        Analyze a chart/plot image using Pixtral Large to extract data.

        Args:
            image_source: Either a file path or base64 encoded image
            prompt: Custom prompt for analysis (default extracts numerical data)

        Returns:
            Analysis results including extracted data points
        """
        if not self.is_available():
            return {"error": "Mistral API not configured"}

        default_prompt = """Analyze this scientific chart/plot and extract:
1. Chart type (line, bar, scatter, etc.)
2. X-axis label and units
3. Y-axis label and units
4. All data series names/labels
5. Key data points as a table with columns: Series, X-value, Y-value
6. Any trends or notable observations

Format the data points as a markdown table for easy parsing."""

        prompt = prompt or default_prompt

        try:
            # Handle file path vs base64
            if Path(image_source).exists():
                with open(image_source, "rb") as f:
                    img_data = base64.standard_b64encode(f.read()).decode("utf-8")
                # Detect format
                if image_source.lower().endswith((".jpg", ".jpeg")):
                    mime = "image/jpeg"
                else:
                    mime = "image/png"
                image_url = f"data:{mime};base64,{img_data}"
            # Assume it's already base64
            elif not image_source.startswith("data:"):
                image_url = f"data:image/png;base64,{image_source}"
            else:
                image_url = image_source

            # Call Pixtral Large for chart analysis
            response = self.client.chat.complete(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            return {
                "success": True,
                "analysis": response.choices[0].message.content,
                "model": self.vision_model,
            }

        except (OSError, ValueError, KeyError) as e:
            logger.error(f"Chart analysis failed: {e}")
            return {"error": str(e)}

    def extract_and_analyze_charts(
        self, pdf_path: Path, output_dir: Path | None = None, chart_prompt: str | None = None
    ) -> dict:
        """
        Full extraction with automatic chart analysis.

        Extracts content from PDF, identifies likely charts/plots,
        and analyzes them using Pixtral Large.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            chart_prompt: Custom prompt for chart analysis

        Returns:
            Full extraction results plus chart analyses
        """
        # First do full extraction
        result = self.extract_full(pdf_path, output_dir=output_dir)

        if not result.get("success"):
            return result

        # Analyze each extracted image as a potential chart
        chart_analyses = []
        for img in result.get("images", []):
            if img.get("image_base64"):
                analysis = self.analyze_chart(img["image_base64"], chart_prompt)
                if analysis.get("success"):
                    chart_analyses.append(
                        {
                            "page": img["page"],
                            "image_id": img["id"],
                            "analysis": analysis["analysis"],
                        }
                    )

        result["chart_analyses"] = chart_analyses
        result["charts_analyzed"] = len(chart_analyses)

        return result


class PDFTriager:
    """Analyzes PDFs to identify candidates for OCR enhancement"""

    # Thresholds for triage
    MIN_CHARS_PER_PAGE = 200  # Below this suggests scanned/image content
    MIN_TEXT_RATIO = 0.1  # Text area vs page area
    GIBBERISH_THRESHOLD = 0.3  # Ratio of non-printable/unusual chars

    def __init__(self):
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> dict:
        """Load document catalog"""
        if OSTI_CATALOG.exists():
            with open(OSTI_CATALOG) as f:
                catalog = json.load(f)
                return {doc["osti_id"]: doc for doc in catalog}
        return {}

    def analyze_pdf(self, pdf_path: Path) -> dict:
        """
        Analyze a single PDF for OCR candidacy.

        Returns analysis with:
        - page_count: Number of pages
        - total_chars: Total extracted characters
        - chars_per_page: Average characters per page
        - image_pages: Pages that are primarily images
        - text_quality: Quality score (0-1)
        - ocr_recommended: Whether OCR would help
        - reasons: list of reasons for recommendation
        """
        result = {
            "pdf_path": str(pdf_path),
            "file_size": pdf_path.stat().st_size if pdf_path.exists() else 0,
            "page_count": 0,
            "total_chars": 0,
            "chars_per_page": 0,
            "image_pages": [],
            "low_text_pages": [],
            "text_quality": 1.0,
            "ocr_recommended": False,
            "reasons": [],
            "priority": 0,  # Higher = more likely to benefit from OCR
        }

        if not pdf_path.exists():
            result["reasons"].append("File not found")
            return result

        if result["file_size"] == 0:
            result["reasons"].append("Empty file")
            return result

        try:
            doc = fitz.open(pdf_path)
            result["page_count"] = len(doc)

            page_analyses = []
            total_chars = 0

            for page_num, page in enumerate(doc):
                page_analysis = self._analyze_page(page, page_num)
                page_analyses.append(page_analysis)
                total_chars += page_analysis["char_count"]

                if page_analysis["is_image_page"]:
                    result["image_pages"].append(page_num + 1)
                if page_analysis["is_low_text"]:
                    result["low_text_pages"].append(page_num + 1)

            doc.close()

            result["total_chars"] = total_chars
            result["chars_per_page"] = total_chars / max(1, result["page_count"])

            # Calculate text quality
            result["text_quality"] = self._calculate_text_quality(page_analyses)

            # Determine if OCR is recommended
            self._evaluate_ocr_recommendation(result)

        except (OSError, ValueError) as e:
            result["reasons"].append(f"Error analyzing PDF: {e!s}")
            result["ocr_recommended"] = True
            result["priority"] = 5

        return result

    def _analyze_page(self, page, page_num: int) -> dict:
        """Analyze a single page"""
        text = page.get_text()
        char_count = len(text)

        # Get page dimensions
        rect = page.rect
        page_area = rect.width * rect.height

        # Count images
        image_list = page.get_images()
        image_count = len(image_list)

        # Calculate image coverage (approximate)
        image_area = 0
        for img in image_list:
            try:
                xref = img[0]
                img_rect = page.get_image_bbox(xref)
                if img_rect:
                    image_area += img_rect.width * img_rect.height
            except (OSError, ValueError):
                pass

        image_coverage = image_area / max(1, page_area)

        # Check for gibberish/encoding issues
        gibberish_ratio = self._detect_gibberish(text)

        # Determine if this is primarily an image page
        is_image_page = (image_count > 0 and image_coverage > 0.5 and char_count < 500) or (
            image_count > 0 and char_count < 100
        )

        is_low_text = char_count < self.MIN_CHARS_PER_PAGE and not is_image_page

        return {
            "page_num": page_num + 1,
            "char_count": char_count,
            "image_count": image_count,
            "image_coverage": image_coverage,
            "gibberish_ratio": gibberish_ratio,
            "is_image_page": is_image_page,
            "is_low_text": is_low_text,
        }

    def _detect_gibberish(self, text: str) -> float:
        """Detect ratio of gibberish/encoding issues in text"""
        if not text:
            return 0.0

        # Count problematic characters
        # - Replacement characters
        # - Excessive special characters
        # - Non-printable characters (except newlines/tabs)

        problematic = 0
        total = len(text)

        for char in text:
            code = ord(char)
            if (
                char == "\ufffd"  # Replacement character
                or (code < 32 and char not in "\n\r\t")  # Non-printable
                or (code >= 0xFFF0 and code <= 0xFFFF)  # Specials block
                or char in "□■●○◊◘◙"  # Common PDF extraction artifacts
            ):
                problematic += 1

        return problematic / max(1, total)

    def _calculate_text_quality(self, page_analyses: list[dict]) -> float:
        """Calculate overall text quality score (0-1)"""
        if not page_analyses:
            return 0.0

        total_gibberish = sum(p["gibberish_ratio"] for p in page_analyses)
        avg_gibberish = total_gibberish / len(page_analyses)

        # Quality is inverse of gibberish ratio
        return max(0, 1 - avg_gibberish * 3)  # Scale up gibberish impact

    def _evaluate_ocr_recommendation(self, result: dict):
        """Determine if OCR is recommended and calculate priority"""
        reasons = []
        priority = 0

        # Check for image-heavy document
        if result["image_pages"]:
            image_ratio = len(result["image_pages"]) / max(1, result["page_count"])
            if image_ratio > 0.3:
                reasons.append(
                    f"{len(result['image_pages'])} image-heavy pages ({image_ratio:.0%})"
                )
                priority += 3
            elif image_ratio > 0.1:
                reasons.append(f"{len(result['image_pages'])} image-heavy pages")
                priority += 2

        # Check for low text extraction
        if result["chars_per_page"] < self.MIN_CHARS_PER_PAGE:
            reasons.append(f"Low text: {result['chars_per_page']:.0f} chars/page")
            priority += 3
        elif result["chars_per_page"] < self.MIN_CHARS_PER_PAGE * 2:
            reasons.append(f"Below average text: {result['chars_per_page']:.0f} chars/page")
            priority += 1

        # Check for text quality issues
        if result["text_quality"] < 0.7:
            reasons.append(f"Text quality issues (score: {result['text_quality']:.2f})")
            priority += 2

        # Check for many low-text pages
        if result["low_text_pages"]:
            low_ratio = len(result["low_text_pages"]) / max(1, result["page_count"])
            if low_ratio > 0.2:
                reasons.append(f"{len(result['low_text_pages'])} low-text pages")
                priority += 1

        result["reasons"] = reasons
        result["priority"] = priority
        result["ocr_recommended"] = priority >= 2

    def triage_documents(
        self,
        limit: int | None = None,
        commodity: str | None = None,
        progress_callback: Callable | None = None,
    ) -> dict:
        """
        Triage all documents to identify OCR candidates.

        Args:
            limit: Maximum documents to analyze (None for all)
            commodity: Filter by commodity category
            progress_callback: Optional callback(current, total, message)

        Returns:
            Triage results with candidates list and statistics
        """
        # Get list of documents to analyze
        docs_to_analyze = []

        for osti_id, doc in self.catalog.items():
            doc_commodity = doc.get("commodity_category", "")

            if commodity and doc_commodity != commodity:
                continue

            pdf_path = self._find_pdf(osti_id, doc_commodity)
            if pdf_path:
                docs_to_analyze.append(
                    {
                        "osti_id": osti_id,
                        "title": doc.get("title", "Unknown"),
                        "commodity": doc_commodity,
                        "pdf_path": pdf_path,
                    }
                )

        if limit:
            docs_to_analyze = docs_to_analyze[:limit]

        total = len(docs_to_analyze)
        candidates = []
        stats = {
            "total_analyzed": 0,
            "ocr_recommended": 0,
            "empty_files": 0,
            "errors": 0,
            "by_priority": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        }

        for i, doc in enumerate(docs_to_analyze):
            if progress_callback and i % 50 == 0:
                progress_callback(i, total, f"Analyzing {doc['osti_id']}")

            analysis = self.analyze_pdf(doc["pdf_path"])
            analysis["osti_id"] = doc["osti_id"]
            analysis["title"] = doc["title"]
            analysis["commodity"] = doc["commodity"]

            stats["total_analyzed"] += 1

            if analysis.get("file_size", 0) == 0:
                stats["empty_files"] += 1
            elif "Error" in str(analysis.get("reasons", [])):
                stats["errors"] += 1
            elif analysis["ocr_recommended"]:
                stats["ocr_recommended"] += 1
                candidates.append(analysis)

                # Track by priority
                priority = min(5, max(1, analysis["priority"]))
                stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1

        # Sort candidates by priority (highest first)
        candidates.sort(key=lambda x: (-x["priority"], x["chars_per_page"]))

        return {
            "candidates": candidates,
            "stats": stats,
            "summary": {
                "total_documents": total,
                "ocr_candidates": len(candidates),
                "recommendation_rate": len(candidates) / max(1, total),
            },
        }

    def _find_pdf(self, osti_id: str, commodity: str | None = None) -> Path | None:
        """Find PDF file by OSTI ID"""
        if commodity:
            commodity_dir = OSTI_PDFS_DIR / commodity
            if commodity_dir.exists():
                for pdf in commodity_dir.glob(f"{osti_id}_*.pdf"):
                    return pdf

        for commodity_dir in OSTI_PDFS_DIR.iterdir():
            if commodity_dir.is_dir():
                for pdf in commodity_dir.glob(f"{osti_id}_*.pdf"):
                    return pdf

        return None


# Singleton instances
_mistral_ocr = None
_pdf_triager = None


def get_mistral_ocr() -> MistralOCR:
    """Get or create MistralOCR singleton"""
    global _mistral_ocr
    if _mistral_ocr is None:
        _mistral_ocr = MistralOCR()
    return _mistral_ocr


def get_pdf_triager() -> PDFTriager:
    """Get or create PDFTriager singleton"""
    global _pdf_triager
    if _pdf_triager is None:
        _pdf_triager = PDFTriager()
    return _pdf_triager
