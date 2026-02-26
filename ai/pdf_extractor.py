# ai/pdf_extractor.py
"""
PDF Text and Table Extraction for LiFi-PV Papers.

Uses pdfplumber to extract text and structured tables from PDF files.
Falls back to basic PyPDF2 text extraction if pdfplumber is unavailable.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


MAX_CHARS = 100_000  # Cap for Gemini context


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF, page by page.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        Concatenated text from all pages, capped at MAX_CHARS.

    Raises:
        ImportError: If neither pdfplumber nor PyPDF2 is installed.
        FileNotFoundError: If the PDF doesn't exist.
    """
    if PDFPLUMBER_AVAILABLE:
        return _extract_with_pdfplumber(pdf_path)
    elif PYPDF2_AVAILABLE:
        return _extract_with_pypdf2(pdf_path)
    else:
        raise ImportError(
            "No PDF library available. Install one:\n"
            "  pip install pdfplumber   (recommended)\n"
            "  pip install PyPDF2       (fallback)")


def extract_tables(pdf_path: str) -> list:
    """
    Extract tables from a PDF as list of dicts.

    Each table is represented as:
        {'page': int, 'headers': list[str], 'rows': list[list[str]]}

    Args:
        pdf_path: Path to PDF file.

    Returns:
        List of table dicts. Empty list if no tables found or pdfplumber unavailable.
    """
    if not PDFPLUMBER_AVAILABLE:
        logger.warning("pdfplumber not available — cannot extract tables")
        return []

    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for tbl in page_tables:
                    if not tbl or len(tbl) < 2:
                        continue
                    headers = [str(c or '').strip() for c in tbl[0]]
                    rows = []
                    for row in tbl[1:]:
                        rows.append([str(c or '').strip() for c in row])
                    tables.append({
                        'page': i + 1,
                        'headers': headers,
                        'rows': rows,
                    })
    except Exception as e:
        logger.error("Table extraction failed: %s", e)

    logger.info("Extracted %d tables from %s", len(tables), pdf_path)
    return tables


def format_tables_as_text(tables: list) -> str:
    """
    Format extracted tables into readable text for the LLM prompt.

    Args:
        tables: List of table dicts from extract_tables().

    Returns:
        Formatted string representation of all tables.
    """
    if not tables:
        return ""

    parts = []
    for i, tbl in enumerate(tables):
        parts.append(f"\n--- Table {i+1} (page {tbl['page']}) ---")
        parts.append(" | ".join(tbl['headers']))
        parts.append("-" * 60)
        for row in tbl['rows']:
            parts.append(" | ".join(row))

    return "\n".join(parts)


def _extract_with_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber."""
    text_parts = []
    total_len = 0
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ''
            text_parts.append(f"\n--- Page {i+1} ---\n{page_text}")
            total_len += len(page_text)
            if total_len > MAX_CHARS:
                logger.warning("Text capped at %d chars (page %d)", MAX_CHARS, i+1)
                break
    result = "\n".join(text_parts)
    logger.info("Extracted %d chars from %s (pdfplumber)", len(result), pdf_path)
    return result[:MAX_CHARS]


def _extract_with_pypdf2(pdf_path: str) -> str:
    """Extract text using PyPDF2 (fallback)."""
    reader = PdfReader(pdf_path)
    text_parts = []
    total_len = 0
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ''
        text_parts.append(f"\n--- Page {i+1} ---\n{page_text}")
        total_len += len(page_text)
        if total_len > MAX_CHARS:
            logger.warning("Text capped at %d chars (page %d)", MAX_CHARS, i+1)
            break
    result = "\n".join(text_parts)
    logger.info("Extracted %d chars from %s (PyPDF2)", len(result), pdf_path)
    return result[:MAX_CHARS]
