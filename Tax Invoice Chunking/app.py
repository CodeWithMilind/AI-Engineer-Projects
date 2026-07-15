import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from all pages of a PDF.

    Args:
        pdf_path (Path): Path to the PDF file

    Returns:
        str: Combined text from all pages
    """

    # Check if file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Open PDF
    document = fitz.open(pdf_path)

    print("=" * 60)
    print(f"File Name : {pdf_path.name}")
    print(f"Total Pages : {len(document)}")
    print("=" * 60)

    full_text = ""

    # Read page by page
    for page_number in range(len(document)):
        page = document.load_page(page_number)

        page_text = page.get_text()

        print(f"\n-------- PAGE {page_number + 1} --------\n")
        print(page_text)

        full_text += page_text + "\n"

    document.close()

    return full_text


def main():

    pdf_path = Path("data/invoices/invoice.pdf")

    try:
        extracted_text = extract_text_from_pdf(pdf_path)

        print("\n" + "=" * 60)
        print("PDF TEXT EXTRACTION COMPLETED")
        print("=" * 60)

        print(f"\nTotal Characters Extracted : {len(extracted_text)}")

    except Exception as e:
        print(f"\nError : {e}")


if __name__ == "__main__":
    main()