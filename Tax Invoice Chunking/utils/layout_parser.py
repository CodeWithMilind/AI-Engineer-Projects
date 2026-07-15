import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


def parse_blocks(blocks: List[Tuple]) -> List[Dict[str, Any]]:
    """
    Parse PyMuPDF blocks into structured dictionaries.
    
    Why this function exists:
    - Converts raw block tuples into readable, structured data
    - Calculates width and height from coordinates
    - Makes it easy to work with block properties
    
    What blocks represent:
    - A block is a rectangular region of text on the page
    - Blocks are the top-level containers in PyMuPDF's layout model
    - Each block has coordinates, text, and type (text, image, etc.)
    
    What coordinates represent:
    - x0: Left x-coordinate of the block
    - y0: Top y-coordinate of the block
    - x1: Right x-coordinate of the block
    - y1: Bottom y-coordinate of the block
    - Coordinates are in points (1 point = 1/72 inch)
    - (0, 0) is the top-left corner of the page
    
    Args:
        blocks: List of block tuples from page.get_text("blocks")
        
    Returns:
        List of structured block dictionaries
    """
    logger = logging.getLogger(__name__)
    logger.info("Parsing blocks...")
    
    parsed_blocks = []
    
    for i, block in enumerate(blocks):
        x0, y0, x1, y1, text, block_no, block_type = block
        
        parsed_block = {
            "block_number": i,
            "block_type": block_type,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "width": x1 - x0,
            "height": y1 - y0,
            "text": text.strip()
        }
        
        parsed_blocks.append(parsed_block)
    
    return parsed_blocks


def print_blocks(parsed_blocks: List[Dict[str, Any]]) -> None:
    """
    Print parsed blocks in a readable format.
    
    Why this function exists:
    - Helps visualize the block structure of the document
    - Makes it easy to verify that blocks are being parsed correctly
    - Useful for debugging and understanding the document layout
    
    Args:
        parsed_blocks: List of structured block dictionaries
    """
    logger = logging.getLogger(__name__)
    logger.info("Printing blocks...")
    
    for block in parsed_blocks:
        if not block["text"]:
            continue
            
        print("\n" + "=" * 60)
        print(f"BLOCK {block['block_number']}")
        print("\nCoordinates")
        print(f"  x0 : {block['x0']:.2f}")
        print(f"  y0 : {block['y0']:.2f}")
        print(f"  x1 : {block['x1']:.2f}")
        print(f"  y1 : {block['y1']:.2f}")
        print(f"\nWidth : {block['width']:.2f}")
        print(f"Height : {block['height']:.2f}")
        print("\nText")
        print(f"\n{block['text']}")
        print("=" * 60)


def parse_layout_dict(layout_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Traverse and parse the full dictionary structure from page.get_text("dict").
    
    Why this function exists:
    - Provides access to the most detailed layout information
    - Breaks down blocks into lines and spans
    - Allows analysis at the character/span level (font size, color, etc.)
    
    Hierarchy in PyMuPDF's dict:
    - Page (top level)
      ↓
    - Blocks (containers of text)
      ↓
    - Lines (horizontal lines of text)
      ↓
    - Spans (contiguous text with same properties - font, size, color)
      ↓
    - Text (actual characters)
    
    Why spans are important:
    - Spans represent text with consistent formatting
    - Useful for detecting headings (different font size)
    - Helps identify emphasized text (bold, italic)
    
    Args:
        layout_dict: Dictionary from page.get_text("dict")
        
    Returns:
        List of structured block dictionaries with lines and spans
    """
    logger = logging.getLogger(__name__)
    logger.info("Parsing dictionary structure...")
    
    parsed_layout = []
    
    for block in layout_dict.get("blocks", []):
        if block.get("type") != 0:  # Skip non-text blocks
            continue
            
        parsed_block = {
            "block_number": block.get("number"),
            "x0": block.get("bbox")[0],
            "y0": block.get("bbox")[1],
            "x1": block.get("bbox")[2],
            "y1": block.get("bbox")[3],
            "lines": []
        }
        
        for line in block.get("lines", []):
            parsed_line = {
                "x0": line.get("bbox")[0],
                "y0": line.get("bbox")[1],
                "x1": line.get("bbox")[2],
                "y1": line.get("bbox")[3],
                "spans": []
            }
            
            for span in line.get("spans", []):
                parsed_span = {
                    "x0": span.get("bbox")[0],
                    "y0": span.get("bbox")[1],
                    "x1": span.get("bbox")[2],
                    "y1": span.get("bbox")[3],
                    "text": span.get("text"),
                    "font": span.get("font"),
                    "size": span.get("size"),
                    "color": span.get("color")
                }
                parsed_line["spans"].append(parsed_span)
            
            parsed_block["lines"].append(parsed_line)
        
        parsed_layout.append(parsed_block)
    
    return parsed_layout


def print_layout_dict(parsed_layout: List[Dict[str, Any]]) -> None:
    """
    Print the full layout hierarchy (Block → Line → Span → Text).
    
    Why this function exists:
    - Visualizes the complete document structure
    - Shows how text is organized into lines and spans
    - Helps understand formatting details (font, size)
    
    Args:
        parsed_layout: List of structured block dictionaries with lines and spans
    """
    logger = logging.getLogger(__name__)
    logger.info("Printing layout dictionary...")
    
    for block in parsed_layout:
        print("\n" + "=" * 60)
        print("BLOCK")
        print(f"  x0: {block['x0']:.2f}, y0: {block['y0']:.2f}, x1: {block['x1']:.2f}, y1: {block['y1']:.2f}")
        print("  ↓")
        
        for line in block["lines"]:
            print("  LINE")
            print(f"    x0: {line['x0']:.2f}, y0: {line['y0']:.2f}, x1: {line['x1']:.2f}, y1: {line['y1']:.2f}")
            print("    ↓")
            
            for span in line["spans"]:
                print("    SPAN")
                print(f"      x0: {span['x0']:.2f}, y0: {span['y0']:.2f}, x1: {span['x1']:.2f}, y1: {span['y1']:.2f}")
                print("      ↓")
                print(f"      TEXT: {span['text']}")
                print(f"      Font: {span['font']}, Size: {span['size']:.2f}")
            print()
        print("=" * 60)


def save_layout_data(
    plain_text_pages: List[str],
    blocks_pages: List[List[Dict[str, Any]]],
    layout_dict_pages: List[List[Dict[str, Any]]],
    output_dir: Path
) -> None:
    """
    Save all layout data to files.
    
    Why this function exists:
    - Persists the extracted layout information for later use
    - Allows other stages of the pipeline to use this data without reprocessing
    - Provides a way to inspect the data offline
    
    Why layout understanding is required before chunking:
    - Chunking needs to know about document structure to split intelligently
    - Without layout, you might split in the middle of a table or section
    - Layout helps keep related information together in the same chunk
    - Enables section-aware chunking (e.g., keep vendor info in one chunk)
    
    Args:
        plain_text_pages: List of plain text for each page
        blocks_pages: List of parsed blocks for each page
        layout_dict_pages: List of parsed layout dicts for each page
        output_dir: Directory to save the files
    """
    logger = logging.getLogger(__name__)
    logger.info("Saving layout data...")
    
    output_dir.mkdir(exist_ok=True)
    
    # Save plain text
    plain_text_path = output_dir / "plain_text.txt"
    with open(plain_text_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(plain_text_pages))
    logger.info(f"Saved plain text to {plain_text_path}")
    
    # Save blocks
    blocks_path = output_dir / "blocks.json"
    with open(blocks_path, "w", encoding="utf-8") as f:
        json.dump(blocks_pages, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved blocks to {blocks_path}")
    
    # Save layout dict
    layout_path = output_dir / "layout.json"
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(layout_dict_pages, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved layout to {layout_path}")
