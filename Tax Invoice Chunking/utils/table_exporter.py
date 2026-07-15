import logging
import json
from typing import List, Dict, Any
from pathlib import Path
from utils.table_detector import DetectedTable, ColumnBoundary
from utils.table_parser import ParsedTableRow

logger = logging.getLogger(__name__)


def table_to_dict(table_name: str, column_boundaries: List[ColumnBoundary], 
                  parsed_rows: List[ParsedTableRow]) -> Dict[str, Any]:
    """
    Convert a parsed table into a dictionary (for JSON export).
    
    Args:
        table_name: Name of the table (e.g., "Invoice Items")
        column_boundaries: List of ColumnBoundary objects (for column headers)
        parsed_rows: List of ParsedTableRow objects
        
    Returns:
        Dictionary in the format:
        {
            "table_name": "Invoice Items",
            "columns": ["Item", "Qty", ...],
            "rows": [{"Item": "Laptop", ...}, ...]
        }
    """
    logger.info("Converting table to dictionary...")
    
    columns = [col.header_text for col in column_boundaries]
    rows = [row.values for row in parsed_rows]
    
    return {
        "table_name": table_name,
        "columns": columns,
        "rows": rows
    }


def print_table_pretty(table_name: str, column_boundaries: List[ColumnBoundary], 
                       parsed_rows: List[ParsedTableRow]) -> None:
    """
    Print the table in a pretty, human-readable format to the terminal.
    
    Args:
        table_name: Name of the table
        column_boundaries: List of ColumnBoundary objects
        parsed_rows: List of ParsedTableRow objects
    """
    logger.info("Printing table...")
    columns = [col.header_text for col in column_boundaries]
    
    # Calculate column widths (minimum 10, or max length of header or any cell in column)
    column_widths = {}
    for col in columns:
        max_len = len(col)
        for row in parsed_rows:
            cell_len = len(row.values.get(col, ""))
            if cell_len > max_len:
                max_len = cell_len
        column_widths[col] = max(max_len, 10)
    
    # Print table header
    print("\n" + "=" * 80)
    print(f"TABLE DETECTED: {table_name}")
    print("=" * 80)
    
    # Print column headers
    header_line = "  ".join(col.ljust(column_widths[col]) for col in columns)
    print(header_line)
    print("-" * len(header_line))
    
    # Print each row
    for row in parsed_rows:
        row_line = "  ".join(
            row.values.get(col, "").ljust(column_widths[col])
            for col in columns
        )
        print(row_line)
    
    print("=" * 80 + "\n")


def save_tables_to_json(tables_data: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Save all tables to a single JSON file (tables.json).
    
    Args:
        tables_data: List of table dictionaries (from table_to_dict)
        output_dir: Directory to save the file in
    """
    logger.info("Saving tables to JSON...")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "tables.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tables_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved tables to {output_path}")


def save_invoice_structured(sections: Dict[str, List[str]], tables_data: List[Dict[str, Any]], 
                            output_dir: Path) -> None:
    """
    Save all structured invoice data (sections + tables) to invoice_structured.json.
    
    This is useful for downstream tasks like semantic chunking or RAG, as it contains all
    the information in a machine-readable format!
    
    Args:
        sections: Dictionary of sections (from section_detector.py)
        tables_data: List of table dictionaries (from table_to_dict)
        output_dir: Directory to save the file in
    """
    logger.info("Saving structured invoice data...")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "invoice_structured.json"
    
    structured_data = {
        "sections": sections,
        "tables": tables_data
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved structured invoice data to {output_path}")
