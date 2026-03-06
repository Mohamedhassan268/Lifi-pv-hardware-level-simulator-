# ai/ocr_engine.py
"""
OCR Engine for scanned PDF papers.

Uses EasyOCR (PyTorch-based) with GPU acceleration to extract text
from image-based/scanned PDFs. Falls back to CPU if no GPU available.

PDF-to-image conversion uses PyMuPDF (fitz) — pure Python, no Poppler needed.
Falls back to pdf2image + Poppler if PyMuPDF is unavailable.

Dependencies:
    pip install easyocr pymupdf
    (or: pip install easyocr pdf2image + Poppler)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

MAX_CHARS = 100_000

# Check availability at import time
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


def is_ocr_available() -> bool:
    """Check if OCR dependencies (easyocr + a PDF-to-image backend) are installed."""
    has_pdf_backend = PYMUPDF_AVAILABLE or PDF2IMAGE_AVAILABLE
    return EASYOCR_AVAILABLE and has_pdf_backend


def _pdf_to_images_pymupdf(pdf_path: str, dpi: int = 300) -> list:
    """Convert PDF pages to numpy arrays using PyMuPDF (no Poppler needed)."""
    from PIL import Image
    import io

    doc = fitz.open(pdf_path)
    images = []
    zoom = dpi / 72  # fitz uses 72 DPI by default
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(np.array(img))

    doc.close()
    return images


def _pdf_to_images_pdf2image(pdf_path: str, dpi: int = 300) -> list:
    """Convert PDF pages to numpy arrays using pdf2image + Poppler."""
    pil_images = convert_from_path(pdf_path, dpi=dpi)
    return [np.array(img) for img in pil_images]


def ocr_pdf(pdf_path: str, languages: list = None,
            progress_callback=None) -> str:
    """
    Extract text from a scanned/image-based PDF using EasyOCR.

    Converts each PDF page to an image, then runs OCR on it.
    Uses GPU if available (EasyOCR auto-detects CUDA).

    Args:
        pdf_path: Path to the PDF file.
        languages: List of language codes (default: ['en']).
        progress_callback: Optional callback(page, total, text_so_far).

    Returns:
        Concatenated OCR text from all pages, formatted with
        page markers matching pdfplumber output format.

    Raises:
        ImportError: If easyocr or PDF-to-image backend is not installed.
        RuntimeError: If PDF conversion or OCR fails.
    """
    if not EASYOCR_AVAILABLE:
        raise ImportError(
            "easyocr is not installed. Install it with:\n"
            "  pip install easyocr")
    if not PYMUPDF_AVAILABLE and not PDF2IMAGE_AVAILABLE:
        raise ImportError(
            "No PDF-to-image backend available. Install one:\n"
            "  pip install pymupdf          (recommended, no Poppler needed)\n"
            "  pip install pdf2image        (needs Poppler installed)")

    if languages is None:
        languages = ['en']

    # Initialize EasyOCR reader (GPU auto-detected)
    logger.info("Initializing EasyOCR reader (languages=%s)...", languages)
    try:
        reader = easyocr.Reader(languages, gpu=True)
        logger.info("EasyOCR initialized with GPU support")
    except Exception:
        logger.warning("GPU initialization failed, falling back to CPU")
        reader = easyocr.Reader(languages, gpu=False)

    # Convert PDF pages to images
    logger.info("Converting PDF pages to images: %s", pdf_path)
    try:
        if PYMUPDF_AVAILABLE:
            logger.info("Using PyMuPDF for PDF-to-image conversion")
            images = _pdf_to_images_pymupdf(pdf_path)
        else:
            logger.info("Using pdf2image + Poppler for PDF-to-image conversion")
            images = _pdf_to_images_pdf2image(pdf_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert PDF to images: {e}\n"
            f"Try: pip install pymupdf"
        ) from e

    total_pages = len(images)
    logger.info("PDF has %d pages, starting OCR...", total_pages)

    text_parts = []
    total_len = 0

    for i, img_array in enumerate(images):
        if progress_callback:
            progress_callback(i + 1, total_pages, f"OCR page {i+1}/{total_pages}...")

        logger.info("OCR processing page %d/%d...", i + 1, total_pages)

        # Run OCR on the page image
        results = reader.readtext(img_array, detail=0, paragraph=True)

        # Join detected text blocks
        page_text = '\n'.join(results)
        text_parts.append(f"\n--- Page {i+1} ---\n{page_text}")

        total_len += len(page_text)
        if total_len > MAX_CHARS:
            logger.warning("OCR text capped at %d chars (page %d/%d)",
                           MAX_CHARS, i + 1, total_pages)
            break

    result = "\n".join(text_parts)
    logger.info("OCR complete: %d chars from %d pages", len(result), total_pages)
    return result[:MAX_CHARS]
