from pathlib import Path

import fitz


class PDFExtractionError(Exception):
    """Exception raised when PDF text extraction fails."""
    pass


def extract_pdf_text(pdf_path: Path) -> dict:
    """Extract text from every page in a PDF and return page metadata."""
    try:
        document = fitz.open(str(pdf_path))
    except Exception as exc:
        raise PDFExtractionError("Unable to open PDF file. The file may be corrupted.") from exc

    try:
        total_pages = document.page_count
        if total_pages == 0:
            raise PDFExtractionError("PDF is empty and contains no pages.")

        pages = []
        for page_index in range(total_pages):
            page = document.load_page(page_index)
            text = page.get_text().strip()
            pages.append({
                "page_number": page_index + 1,
                "text": text,
            })

        if all(not page_data["text"] for page_data in pages):
            raise PDFExtractionError("PDF contains no extractable text.")

        return {
            "total_pages": total_pages,
            "pages": pages,
        }
    finally:
        document.close()
