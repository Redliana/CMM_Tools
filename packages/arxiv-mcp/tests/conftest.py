"""Shared fixtures for arxiv-mcp tests."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import pytest

# -- Sample ArXiv Atom XML data for testing ----------------------------------

SAMPLE_ARXIV_ENTRY_XML = """\
<entry xmlns="http://www.w3.org/2005/Atom"
       xmlns:arxiv="http://arxiv.org/schemas/atom">
  <id>http://arxiv.org/abs/2301.07041v1</id>
  <title>  Attention Is All You Need\n  </title>
  <summary>  We propose a new simple network architecture, the Transformer,\n based solely on attention mechanisms.  </summary>
  <published>2023-01-17T18:00:00Z</published>
  <author><name>Ashish Vaswani</name></author>
  <author><name>Noam Shazeer</name></author>
  <author><name>Niki Parmar</name></author>
  <author><name>Jakob Uszkoreit</name></author>
  <category term="cs.CL"/>
  <category term="cs.AI"/>
  <link title="pdf" href="http://arxiv.org/pdf/2301.07041v1" rel="related" type="application/pdf"/>
</entry>
"""

SAMPLE_ARXIV_ENTRY_NO_PDF_XML = """\
<entry xmlns="http://www.w3.org/2005/Atom"
       xmlns:arxiv="http://arxiv.org/schemas/atom">
  <id>http://arxiv.org/abs/9901.00001v1</id>
  <title>Minimal Entry</title>
  <summary>Short abstract.</summary>
  <published>1999-01-01T00:00:00Z</published>
  <author><name>Jane Doe</name></author>
  <category term="math.AG"/>
</entry>
"""

SAMPLE_ARXIV_ENTRY_EMPTY_XML = """\
<entry xmlns="http://www.w3.org/2005/Atom"
       xmlns:arxiv="http://arxiv.org/schemas/atom">
</entry>
"""

SAMPLE_ARXIV_FEED_XML = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query: all:transformer</title>
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">2</opensearch:totalResults>
  {SAMPLE_ARXIV_ENTRY_XML}
  {SAMPLE_ARXIV_ENTRY_NO_PDF_XML}
</feed>
"""

SAMPLE_ARXIV_EMPTY_FEED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query: all:nonexistent</title>
</feed>
"""


@pytest.fixture()
def arxiv_namespace() -> dict[str, str]:
    """Return the ArXiv Atom XML namespace mapping."""
    return {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }


@pytest.fixture()
def sample_entry_element() -> ET.Element:
    """Return a parsed XML Element for a typical ArXiv entry.

    This entry has four authors, two categories, and a PDF link.
    """
    return ET.fromstring(SAMPLE_ARXIV_ENTRY_XML)


@pytest.fixture()
def sample_entry_no_pdf_element() -> ET.Element:
    """Return a parsed XML Element for an entry without an explicit PDF link."""
    return ET.fromstring(SAMPLE_ARXIV_ENTRY_NO_PDF_XML)


@pytest.fixture()
def sample_entry_empty_element() -> ET.Element:
    """Return a parsed XML Element for an entry with no child elements."""
    return ET.fromstring(SAMPLE_ARXIV_ENTRY_EMPTY_XML)


@pytest.fixture()
def sample_paper_dict() -> dict[str, Any]:
    """Return a sample paper metadata dictionary matching parse_arxiv_entry output."""
    return {
        "id": "2301.07041v1",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        "summary": (
            "We propose a new simple network architecture, the Transformer,"
            " based solely on attention mechanisms."
        ),
        "published": "2023-01-17T18:00:00Z",
        "categories": ["cs.CL", "cs.AI"],
        "pdf_url": "http://arxiv.org/pdf/2301.07041v1",
    }


@pytest.fixture()
def sample_paper_dict_few_authors() -> dict[str, Any]:
    """Return a paper dict with only two authors (below the truncation threshold)."""
    return {
        "id": "9901.00001v1",
        "title": "Minimal Entry",
        "authors": ["Jane Doe"],
        "summary": "Short abstract.",
        "published": "1999-01-01T00:00:00Z",
        "categories": ["math.AG"],
        "pdf_url": "http://arxiv.org/pdf/9901.00001v1",
    }


@pytest.fixture()
def sample_feed_xml() -> str:
    """Return a complete ArXiv Atom feed XML string with two entries."""
    return SAMPLE_ARXIV_FEED_XML


@pytest.fixture()
def sample_empty_feed_xml() -> str:
    """Return an ArXiv Atom feed XML string with zero entries."""
    return SAMPLE_ARXIV_EMPTY_FEED_XML
