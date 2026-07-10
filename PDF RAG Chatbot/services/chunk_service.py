from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pdf_pages(
    pages: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """Convert extracted PDF pages into overlapping text chunks.

    Parameters:
    - pages: list of extracted page dictionaries. Each page must include:
        - page_number: int
        - text: str
    - chunk_size: maximum number of characters in each chunk.
    - chunk_overlap: number of characters shared between adjacent chunks.

    Returns:
    - A list of chunks, each containing:
        - chunk_id: sequential chunk identifier
        - page_number: source page number
        - chunk_text: text content of the chunk
    """
    if not isinstance(pages, list):
        raise ValueError("Pages must be a list of extracted page dictionaries.")
    
    
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


    chunk_list: List[Dict] = []
    chunk_id = 1

    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("Each page entry must be a dictionary.")

        page_number = page.get("page_number")
        text = page.get("text")

        if page_number is None or text is None:
            raise ValueError("Each page dictionary must include 'page_number' and 'text'.")
        if not isinstance(text, str):
            raise ValueError("Page text must be a string.")

        raw_chunks = splitter.split_text(text)
        for raw_chunk in raw_chunks:
            chunk_list.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "chunk_text": raw_chunk.strip(),
                }
            )
            chunk_id += 1

    return chunk_list
