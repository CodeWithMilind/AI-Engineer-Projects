import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ColumnBoundary:
    """Data class to represent a column boundary with x coordinates and header text."""
    x0: float
    x1: float
    header_text: str


@dataclass
class DetectedTable:
    """Data class to represent a detected table with all necessary info."""
    blocks: List[Dict[str, Any]]
    header_block: Dict[str, Any]
    column_boundaries: List[ColumnBoundary]
    y0: float
    y1: float


def detect_column_boundaries(header_block: Dict[str, Any]) -> List[ColumnBoundary]:
    """
    Detect column boundaries from a header block using the x coordinates of its lines/spans.
    
    How it works:
        1. Collect all x0 and x1 coordinates from each line's bbox in the header block
        2. Each line represents a column header, so we use its coordinates as the column boundary
        3. We join consecutive spans in the same line to get the full header text
    
    Why this is important:
        - The header block's lines are perfectly aligned with the columns of the table
        - Using these coordinates ensures we correctly group data into columns later
    
    Args:
        header_block: A block from page.get_text("dict") that is the table header
        
    Returns:
        List of ColumnBoundary objects, one per column
    """
    logger.info("Detecting column boundaries...")
    
    column_boundaries = []
    
    for line in header_block.get("lines", []):
        # Join all spans in the line to get the full header text
        header_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
        
        # Get the x coordinates from the line's bbox
        line_bbox = line.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x0 = line_bbox[0]
        x1 = line_bbox[2]
        
        if header_text:
            column_boundaries.append(ColumnBoundary(
                x0=x0,
                x1=x1,
                header_text=header_text
            ))
            logger.debug(f"  Detected column: '{header_text}' at x0={x0:.2f}, x1={x1:.2f}")
    
    logger.info(f"Detected {len(column_boundaries)} columns")
    return column_boundaries


def is_table_block_candidate(block: Dict[str, Any], header_block: Dict[str, Any]) -> bool:
    """
    Check if a block is a candidate for being part of the table (same x range as header, below header).
    
    Args:
        block: The block to check
        header_block: The header block of the table
        
    Returns:
        True if the block is a table candidate, False otherwise
    """
    header_bbox = header_block.get("bbox", [0.0,0.0,0.0,0.0])
    block_bbox = block.get("bbox", [0.0,0.0,0.0,0.0])
    
    # Check if block is below the header
    if block_bbox[1] < header_bbox[3]:
        return False
    
    # Check if block's x range overlaps significantly with the header's x range
    header_x0, header_x1 = header_bbox[0], header_bbox[2]
    block_x0, block_x1 = block_bbox[0], block_bbox[2]
    
    # Calculate overlap
    overlap_x0 = max(header_x0, block_x0)
    overlap_x1 = min(header_x1, block_x1)
    if overlap_x1 <= overlap_x0:
        return False
    
    overlap_ratio = (overlap_x1 - overlap_x0) / (header_x1 - header_x0)
    return overlap_ratio > 0.5  # At least 50% overlap


def detect_tables(layout_pages: List[Dict[str, Any]]) -> List[DetectedTable]:
    """
    Detect tables in the parsed layout (from page.get_text("dict")).
    
    Detection algorithm:
        1. Iterate over all blocks in all pages
        2. For each block, check if it looks like a table header (multiple lines, bold text, etc.)
        3. Once a header is found, collect all blocks below it that are in the same x range
        4. Detect column boundaries from the header block's lines
        5. Return a list of DetectedTable objects
    
    How enterprise systems do it (simplified):
        - They also use ML models, but we're using rule-based which is great for structured invoices
        - Rule-based systems are faster, more transparent, and easier to debug for consistent layouts
    
    Args:
        layout_pages: List of pages, each page is a dict from page.get_text("dict") (has a "blocks" key)
        
    Returns:
        List of DetectedTable objects
    """
    logger.info("Detecting tables...")
    
    detected_tables = []
    
    for page_num, page_dict in enumerate(layout_pages):
        page_blocks = page_dict.get("blocks", [])  # Get blocks from the page dict
        logger.debug(f"Processing page {page_num + 1} for tables...")
        
        for i, block in enumerate(page_blocks):
            # Skip non-text blocks
            if block.get("type", 0) != 0:
                continue
                
            # Heuristic for table header:
            # 1. Has multiple lines
            # 2. At least one bold span
            # 3. All lines are approximately on the same horizontal line (y0 within 2 points)
            lines = block.get("lines", [])
            has_bold = any(
                "bold" in span.get("font", "").lower()
                for line in lines
                for span in line.get("spans", [])
            )
            
            # Check if all lines are horizontally aligned
            all_aligned = False
            if len(lines) >= 2:
                first_line_y0 = lines[0].get("bbox", [0,0,0,0])[1]
                all_aligned = all(
                    abs(line.get("bbox", [0,0,0,0])[1] - first_line_y0) < 2.0
                    for line in lines
                )
            
            if len(lines) >= 2 and has_bold and all_aligned:
                # Let's check if this is a table header by seeing if there are blocks below it in the same x range
                # Collect candidate blocks (below this block, overlapping x range)
                candidate_blocks = [block]  # Include the header block itself
                header_block = block
                header_y1 = header_block.get("bbox", [0,0,0,0])[3]  # bbox is (x0, y0, x1, y1)
                min_y0 = header_y1
                max_y1 = header_block.get("bbox", [0,0,0,0])[1]
                
                for j in range(i + 1, len(page_blocks)):
                    candidate = page_blocks[j]
                    if candidate.get("type", 0) != 0:
                        continue
                    if is_table_block_candidate(candidate, header_block):
                        candidate_blocks.append(candidate)
                        min_y0 = min(min_y0, candidate.get("bbox", [float("inf"),0,0,0])[1])
                        max_y1 = max(max_y1, candidate.get("bbox", [0,0,0,0])[3])
                
                # If we have at least 2 blocks (header + 1 data row), it's a table
                if len(candidate_blocks) >= 2:
                    logger.info(f"Found table candidate with {len(candidate_blocks)} blocks on page {page_num + 1}")
                    column_boundaries = detect_column_boundaries(header_block)
                    detected_table = DetectedTable(
                        blocks=candidate_blocks,
                        header_block=header_block,
                        column_boundaries=column_boundaries,
                        y0=min_y0,
                        y1=max_y1
                    )
                    detected_tables.append(detected_table)
                    # Skip the blocks we already added to this table
                    i = j  # Not perfect but works for our case
    
    logger.info(f"Detected {len(detected_tables)} table(s)")
    return detected_tables
