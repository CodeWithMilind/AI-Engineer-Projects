import fitz
import logging
from pathlib import Path
from typing import Dict, List, Tuple


def setup_logging() -> None:
    """
    Configure logging to display timestamps and messages.
    
    Why this function exists:
    - Helps track the execution flow of the program
    - Makes debugging easier by showing what steps have completed
    - Provides visibility into when errors occur
    
    How it works:
    - Sets the logging level to INFO
    - Formats messages with timestamp, logger name, and message
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_pdf(pdf_path: Path) -> Tuple[fitz.Document, int, str]:
    """
    Load a PDF file and return the document object, page count, and filename.
    
    Why this function exists:
    - Encapsulates PDF loading logic to keep app.py clean
    - Adds error handling for missing files
    - Provides basic metadata about the PDF
    
    How PyMuPDF works internally:
    - fitz.open() creates a Document object that represents the PDF
    - It reads the PDF structure but doesn't load all content into memory at once
    - Pages are accessed lazily when needed
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (Document object, page count, filename)
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading PDF: {pdf_path}")
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    return doc, len(doc), pdf_path.name


def extract_page_layouts(doc: fitz.Document) -> Dict[str, List]:
    """
    Extract multiple layout formats from each page of the PDF.
    
    Why this function exists:
    - Different layout formats serve different purposes
    - "blocks" are good for high-level section detection
    - "dict" gives fine-grained control over lines and spans
    - "text" is the raw text for quick verification
    
    What PyMuPDF's get_text() methods return:
    - get_text(): Plain text of the page
    - get_text("blocks"): List of tuples (x0, y0, x1, y1, text, block_no, block_type)
    - get_text("dict"): Nested dictionary with blocks, lines, spans, and their properties
    
    Args:
        doc: PyMuPDF Document object
        
    Returns:
        Dictionary with keys "plain_text", "blocks", "layout_dict" containing
        the extracted data for all pages
    """
    logger = logging.getLogger(__name__)
    logger.info("Extracting page layouts...")
    
    plain_text_pages = []
    blocks_pages = []
    layout_dict_pages = []
    
    for page_num in range(len(doc)):
        logger.info(f"Processing page {page_num + 1}/{len(doc)}")
        page = doc.load_page(page_num)
        
        # Extract plain text
        plain_text = page.get_text()
        plain_text_pages.append(plain_text)
        
        # Extract blocks
        blocks = page.get_text("blocks")
        blocks_pages.append(blocks)
        
        # Extract dictionary structure
        layout_dict = page.get_text("dict")
        layout_dict_pages.append(layout_dict)
    
    return {
        "plain_text": plain_text_pages,
        "blocks": blocks_pages,
        "layout_dict": layout_dict_pages
    }
