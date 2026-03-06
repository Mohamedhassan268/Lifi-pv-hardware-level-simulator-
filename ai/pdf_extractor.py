# ai/pdf_extractor.py
"""
PDF Text and Table Extraction for LiFi-PV Papers.

Uses pdfplumber to extract text and structured tables from PDF files.
Falls back to basic PyPDF2 text extraction if pdfplumber is unavailable.
Includes intelligent text pre-filtering to find parameter-dense sections.
"""

import re
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

# Keywords that indicate a paragraph contains hardware parameter info
_PARAM_KEYWORDS = re.compile(
    r'\b('
    r'[0-9]+\.?[0-9]*\s*'
    r'(?:kHz|MHz|GHz|Hz|mW|mA|uA|µA|nF|pF|µF|uF|kOhm|k\u03a9|kohm|\u03a9|Ohm|'
    r'cm2|cm\u00b2|mm2|mm\u00b2|meter|m\b|dB|dBm|A/W|W/A|V|mV|kbps|Mbps|bps|'
    r'lux|cd|sr|deg|°)'
    r'|KXOB|INA\d|TLV\d|ADA\d|LXM\d|BSD\d|BPW\d|OPT\d|SFH\d|SM\d|'
    r'solar\s+cell|photodiode|photovoltaic|responsivity|capacitance|'
    r'shunt\s+resistance|junction|half[\s-]angle|modulation|'
    r'bias\s+current|data\s+rate|bit\s+rate|BER|bit\s+error|'
    r'bandpass|band[\s-]pass|high[\s-]pass|low[\s-]pass|cutoff|'
    r'DC[\s-]DC|boost\s+converter|switching\s+freq|inductor|'
    r'amplifier|gain|comparator|sense\s+resistor|transimpedance|'
    r'harvested\s+power|optical\s+power|radiated\s+power|'
    r'LED|receiver|transmitter|channel\s+gain|distance|'
    r'active\s+area|photocurrent|MPP|V_?mpp|I_?mpp|P_?mpp'
    r')\b',
    re.IGNORECASE
)

# Section headings that likely contain parameter descriptions
_SECTION_KEYWORDS = re.compile(
    r'\b(system\s+model|circuit\s+design|experimental|setup|'
    r'parameter|component|specification|configuration|'
    r'receiver|transmitter|channel|measurement|result|'
    r'table|fig\w*\s+\d)\b',
    re.IGNORECASE
)


def extract_text(pdf_path: str, progress_callback=None) -> tuple:
    """
    Extract all text from a PDF, page by page.

    Tries text-based extraction first (pdfplumber/PyPDF2).
    If very little text is found (< 100 chars), falls back to OCR
    using EasyOCR if available.

    Args:
        pdf_path: Path to PDF file.
        progress_callback: Optional callback for OCR progress updates.

    Returns:
        Tuple of (text: str, ocr_used: bool).

    Raises:
        ImportError: If no PDF library and no OCR available.
        FileNotFoundError: If the PDF doesn't exist.
    """
    text = ''
    ocr_used = False

    # Try text-based extraction first
    if PDFPLUMBER_AVAILABLE:
        text = _extract_with_pdfplumber(pdf_path)
    elif PYPDF2_AVAILABLE:
        text = _extract_with_pypdf2(pdf_path)

    # If text extraction yielded enough content, return it
    if len(text.strip()) >= 100:
        return text, False

    # Text-based extraction failed or yielded little — try OCR
    logger.warning("Text-based extraction yielded only %d chars, attempting OCR...",
                   len(text.strip()))

    from ai.ocr_engine import is_ocr_available, ocr_pdf

    if is_ocr_available():
        try:
            text = ocr_pdf(pdf_path, progress_callback=progress_callback)
            ocr_used = True
            logger.info("OCR fallback succeeded: %d chars extracted", len(text))
            return text, True
        except Exception as e:
            logger.error("OCR fallback failed: %s", e)
            raise RuntimeError(
                f"Text-based extraction found very little text and OCR failed: {e}\n"
                f"The PDF may be corrupted or in an unsupported format."
            ) from e

    # No OCR available — give helpful error
    if not PDFPLUMBER_AVAILABLE and not PYPDF2_AVAILABLE:
        raise ImportError(
            "No PDF library available. Install one:\n"
            "  pip install pdfplumber   (recommended)\n"
            "  pip install PyPDF2       (fallback)\n"
            "For scanned PDFs, also install:\n"
            "  pip install easyocr pdf2image")
    else:
        raise RuntimeError(
            "Very little text extracted from PDF. "
            "The file may be image-based (scanned).\n"
            "Install OCR support with:\n"
            "  pip install easyocr pdf2image\n"
            "Windows also needs Poppler:\n"
            "  https://github.com/oschwartz10612/poppler-windows/releases")


def extract_tables(pdf_path: str) -> list:
    """
    Extract tables from a PDF as list of dicts.

    Each table is represented as:
        {'page': int, 'headers': list[str], 'rows': list[list[str]]}

    Filters out empty tables (all cells blank).

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

                    # Filter out empty tables (all cells blank)
                    all_content = ''.join(headers) + ''.join(
                        c for row in rows for c in row)
                    if not all_content.strip():
                        continue

                    tables.append({
                        'page': i + 1,
                        'headers': headers,
                        'rows': rows,
                    })
    except Exception as e:
        logger.error("Table extraction failed: %s", e)

    logger.info("Extracted %d non-empty tables from %s", len(tables), pdf_path)
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
        return "(No tables extracted from PDF)"

    parts = []
    for i, tbl in enumerate(tables):
        parts.append(f"\n--- Table {i+1} (page {tbl['page']}) ---")
        parts.append(" | ".join(tbl['headers']))
        parts.append("-" * 60)
        for row in tbl['rows']:
            parts.append(" | ".join(row))

    return "\n".join(parts)


def extract_key_text(full_text: str, max_chars: int = 15000) -> str:
    """
    Extract parameter-relevant sections from the full paper text.

    Uses keyword matching to identify paragraphs likely containing
    hardware parameter values. This reduces noise for small LLMs.

    Args:
        full_text: Full extracted text from the PDF.
        max_chars: Maximum output length.

    Returns:
        Filtered text containing only parameter-relevant paragraphs.
    """
    # Split into paragraphs (lines separated by blank lines or page markers)
    paragraphs = re.split(r'\n\s*\n|--- Page \d+ ---', full_text)

    scored_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < 20:
            continue

        # Score by number of parameter keyword matches
        matches = _PARAM_KEYWORDS.findall(para)
        section_matches = _SECTION_KEYWORDS.findall(para)
        score = len(matches) * 2 + len(section_matches)

        # Bonus for lines with "=" or ":" followed by numbers (likely parameter definitions)
        param_defs = re.findall(r'[=:]\s*[\d.]+', para)
        score += len(param_defs) * 3

        if score > 0:
            scored_paragraphs.append((score, para))

    # Sort by score descending, take the most relevant paragraphs
    scored_paragraphs.sort(key=lambda x: -x[0])

    result_parts = []
    total_len = 0
    for score, para in scored_paragraphs:
        if total_len + len(para) > max_chars:
            break
        result_parts.append(para)
        total_len += len(para)

    result = '\n\n'.join(result_parts)
    logger.info("Key text extraction: %d/%d chars (%d relevant paragraphs)",
                len(result), len(full_text), len(result_parts))
    return result


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
