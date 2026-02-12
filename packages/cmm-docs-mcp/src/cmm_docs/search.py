"""
Search functionality for CMM MCP Server
Implements full-text search with SQLite FTS5 and similarity search with TF-IDF
"""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3

import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cmm_docs.config import (
    INDEX_DIR,
    MAX_SEARCH_RESULTS,
    OSTI_CATALOG,
    OSTI_PDFS_DIR,
    SEARCH_DB,
    TFIDF_VECTORS,
)
from cmm_docs.ocr import get_mistral_ocr

logger = logging.getLogger(__name__)


class SearchIndex:
    """Full-text search and similarity search index"""

    def __init__(self):
        self.db_path = SEARCH_DB
        self.vectors_path = TFIDF_VECTORS
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = []

        # Ensure index directory exists
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        # Load existing index if available
        self._load_index()

    def _load_index(self):
        """Load existing search index and TF-IDF vectors"""
        if self.vectors_path.exists():
            try:
                with open(self.vectors_path, "rb") as f:
                    data = pickle.load(f)
                    self.vectorizer = data["vectorizer"]
                    self.tfidf_matrix = data["matrix"]
                    self.doc_ids = data["doc_ids"]
                logger.info(f"Loaded TF-IDF index with {len(self.doc_ids)} documents")
            except (OSError, ValueError, pickle.UnpicklingError) as e:
                logger.warning(f"Failed to load TF-IDF index: {e}")

    def is_indexed(self) -> bool:
        """Check if search index exists"""
        return self.db_path.exists() and self.tfidf_matrix is not None

    def get_index_stats(self) -> dict:
        """Get index statistics"""
        stats = {
            "indexed": self.is_indexed(),
            "document_count": 0,
            "fts_indexed": self.db_path.exists(),
            "tfidf_indexed": self.tfidf_matrix is not None,
        }

        if self.db_path.exists():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            stats["document_count"] = cursor.fetchone()[0]
            conn.close()

        return stats

    def build_index(self, progress_callback=None) -> dict:
        """
        Build full-text search index from all PDFs.
        This extracts text from all PDFs and indexes it.

        Args:
            progress_callback: Optional callback function(current, total, message)

        Returns:
            Statistics about the indexing process
        """
        # Load catalog
        if not OSTI_CATALOG.exists():
            return {"error": "Document catalog not found"}

        with open(OSTI_CATALOG) as f:
            catalog = json.load(f)

        total_docs = len(catalog)
        indexed = 0
        errors = 0
        texts = []
        doc_ids = []

        # Create/reset FTS database
        conn = sqlite3.connect(self.db_path)
        conn.execute("DROP TABLE IF EXISTS documents")
        conn.execute("DROP TABLE IF EXISTS documents_fts")

        conn.execute("""
            CREATE TABLE documents (
                osti_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                authors TEXT,
                subjects TEXT,
                commodity TEXT,
                content TEXT
            )
        """)

        conn.execute("""
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                osti_id,
                title,
                description,
                authors,
                subjects,
                content,
                content='documents',
                content_rowid='rowid'
            )
        """)

        # Process each document
        for i, doc in enumerate(catalog):
            osti_id = doc.get("osti_id")
            if not osti_id:
                continue

            if progress_callback and i % 50 == 0:
                progress_callback(i, total_docs, f"Processing {osti_id}")

            # Get metadata
            title = doc.get("title", "")
            description = doc.get("description", "")
            authors = ", ".join(doc.get("authors", []))
            subjects = ", ".join(doc.get("subjects", []))
            commodity = doc.get("commodity_category", "")

            # Extract PDF text
            content = self._extract_pdf_text(osti_id, commodity)

            if content:
                indexed += 1
            else:
                errors += 1
                content = ""  # Use empty string for missing PDFs

            # Insert into database
            conn.execute(
                """
                INSERT INTO documents (osti_id, title, description, authors, subjects, commodity, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (osti_id, title, description, authors, subjects, commodity, content),
            )

            # Store for TF-IDF
            full_text = f"{title} {description} {content}"
            texts.append(full_text)
            doc_ids.append(osti_id)

        # Build FTS index
        conn.execute("""
            INSERT INTO documents_fts(documents_fts) VALUES('rebuild')
        """)
        conn.commit()
        conn.close()

        # Build TF-IDF index
        if texts:
            self.vectorizer = TfidfVectorizer(
                max_features=10000, stop_words="english", ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.doc_ids = doc_ids

            # Save TF-IDF data
            with open(self.vectors_path, "wb") as f:
                pickle.dump(
                    {
                        "vectorizer": self.vectorizer,
                        "matrix": self.tfidf_matrix,
                        "doc_ids": self.doc_ids,
                    },
                    f,
                )

        return {
            "total_documents": total_docs,
            "indexed": indexed,
            "errors": errors,
            "fts_indexed": True,
            "tfidf_indexed": True,
        }

    def _extract_pdf_text(self, osti_id: str, commodity: str, use_ocr_fallback: bool = True) -> str:
        """Extract text from a PDF file, with optional Mistral OCR fallback"""
        commodity_dir = OSTI_PDFS_DIR / commodity
        if not commodity_dir.exists():
            return ""

        # Find PDF file
        pdf_path = None
        for pdf in commodity_dir.glob(f"{osti_id}_*.pdf"):
            pdf_path = pdf
            break

        if not pdf_path:
            return ""

        # Try PyMuPDF first
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            text = "\n".join(text_parts)

            # If we got meaningful text, return it
            if text.strip():
                return text

        except (OSError, ValueError) as e:
            logger.warning(f"PyMuPDF failed for {pdf_path}: {e}")

        # Fallback to Mistral OCR if enabled and available
        if use_ocr_fallback:
            ocr = get_mistral_ocr()
            if ocr.is_available():
                logger.info(f"Using Mistral OCR fallback for {osti_id}")
                result = ocr.extract_text_from_pdf(pdf_path)
                if result.get("success"):
                    return result.get("text", "")
                else:
                    logger.warning(f"Mistral OCR failed for {pdf_path}: {result.get('error')}")

        return ""

    def search(self, query: str, limit: int = MAX_SEARCH_RESULTS) -> list[dict]:
        """
        Full-text search across indexed documents.

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            list of matching documents with relevance scores
        """
        if not self.db_path.exists():
            return [{"error": "Search index not built. Use build_index first."}]

        conn = sqlite3.connect(self.db_path)

        # Use FTS5 search with BM25 ranking
        cursor = conn.execute(
            """
            SELECT
                d.osti_id,
                d.title,
                d.description,
                d.commodity,
                bm25(documents_fts) as score
            FROM documents_fts
            JOIN documents d ON documents_fts.rowid = d.rowid
            WHERE documents_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """,
            (query, limit),
        )

        results = []
        for row in cursor:
            results.append(
                {
                    "osti_id": row[0],
                    "title": row[1],
                    "description": row[2][:300] + "..." if len(row[2]) > 300 else row[2],
                    "commodity": row[3],
                    "relevance_score": abs(row[4]),  # BM25 returns negative scores
                }
            )

        conn.close()
        return results

    def find_similar(self, osti_id: str, limit: int = 5) -> list[dict]:
        """
        Find documents similar to a given document using TF-IDF cosine similarity.

        Args:
            osti_id: Reference document OSTI ID
            limit: Number of similar documents to return

        Returns:
            list of similar documents with similarity scores
        """
        if self.tfidf_matrix is None:
            return [{"error": "Similarity index not built. Use build_index first."}]

        if osti_id not in self.doc_ids:
            return [{"error": f"Document {osti_id} not found in index"}]

        # Get index of reference document
        ref_idx = self.doc_ids.index(osti_id)

        # Calculate cosine similarity
        ref_vector = self.tfidf_matrix[ref_idx]
        similarities = cosine_similarity(ref_vector, self.tfidf_matrix).flatten()

        # Get top similar documents (excluding self)
        similar_indices = similarities.argsort()[::-1]

        results = []
        for idx in similar_indices:
            if len(results) >= limit:
                break
            if idx == ref_idx:
                continue  # Skip self

            doc_id = self.doc_ids[idx]
            results.append(
                {
                    "osti_id": doc_id,
                    "similarity_score": float(similarities[idx]),
                }
            )

        # Enrich with metadata
        if OSTI_CATALOG.exists():
            with open(OSTI_CATALOG) as f:
                catalog = json.load(f)
                catalog_dict = {d["osti_id"]: d for d in catalog}

            for result in results:
                doc = catalog_dict.get(result["osti_id"], {})
                result["title"] = doc.get("title", "Unknown")
                result["commodity"] = doc.get("commodity_category", "Unknown")

        return results


# Singleton instance
_search_index = None


def get_search_index() -> SearchIndex:
    """Get or create SearchIndex singleton"""
    global _search_index
    if _search_index is None:
        _search_index = SearchIndex()
    return _search_index
