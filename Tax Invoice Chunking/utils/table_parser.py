import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from utils.table_detector import DetectedTable, ColumnBoundary

logger = logging.getLogger(__name__)


@dataclass
class ParsedTableRow:
    """Data class to represent a parsed table row with column values."""
    values: Dict[str, str]  # Maps header text to cell value


def assign_to_column(cell_text: str, cell_x0: float, cell_x1: float, 
                     column_boundaries: List[ColumnBoundary]) -> str:
    """
    Assign a cell (text + x coordinates) to the correct column using the column boundaries.
    
    How it works:
        - Calculate the midpoint of the cell's x range
        - Find which column boundary contains this midpoint
        - Fallback to closest column if midpoint is not exactly in any boundary
    
    Why coordinates instead of just text order:
        - Text order can be unreliable if there are unusual spacing or formatting
        - Coordinates give us precise spatial information, which is how the document is structured
    
    Args:
        cell_text: Text of the cell
        cell_x0: Left x coordinate of the cell
        cell_x1: Right x coordinate of the cell
        column_boundaries: List of detected column boundaries
        
    Returns:
        Header text of the column this cell belongs to
    """
    cell_midpoint = (cell_x0 + cell_x1) / 2
    
    # First try exact match (cell midpoint is within column boundary)
    for col in column_boundaries:
        if col.x0 <= cell_midpoint <= col.x1:
            return col.header_text
    
    # If no exact match, find the closest column boundary
    closest_col = None
    min_distance = float("inf")
    for col in column_boundaries:
        col_midpoint = (col.x0 + col.x1) / 2
        distance = abs(cell_midpoint - col_midpoint)
        if distance < min_distance:
            min_distance = distance
            closest_col = col
    
    return closest_col.header_text if closest_col else ""


def parse_detected_table(detected_table: DetectedTable) -> List[ParsedTableRow]:
    """
    Parse a DetectedTable into a list of ParsedTableRow objects.
    
    How rows are reconstructed:
        - Each block in the detected table is a row (since our invoice's table has one block per row)
        - For multi-block rows (more complex tables), you could group blocks by y coordinate, but our case is simple!
    
    How columns are reconstructed:
        - For each line in a block (which is a cell), we use its x coordinates to assign it to the correct column
        - We use the column boundaries from the header block to do this assignment
    
    Args:
        detected_table: DetectedTable object from table_detector.py
        
    Returns:
        List of ParsedTableRow objects, one per row (excluding header row)
    """
    logger.info("Parsing detected table...")
    
    parsed_rows = []
    column_boundaries = detected_table.column_boundaries
    
    # The first block is the header, so we start from index 1
    for row_block in detected_table.blocks[1:]:
        row_values = {}
        
        for line in row_block.get("lines", []):
            # Join all spans in the line to get cell text
            cell_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            line_bbox = line.get("bbox", [0.0, 0.0, 0.0, 0.0])
            cell_x0 = line_bbox[0]
            cell_x1 = line_bbox[2]
            
            if cell_text:
                # Assign cell to column
                column_header = assign_to_column(cell_text, cell_x0, cell_x1, column_boundaries)
                if column_header:
                    row_values[column_header] = cell_text
        
        if row_values:
            parsed_row = ParsedTableRow(values=row_values)
            parsed_rows.append(parsed_row)
            logger.debug(f"  Parsed row: {row_values}")
    
    logger.info(f"Parsed {len(parsed_rows)} data rows from table")
    return parsed_rows
