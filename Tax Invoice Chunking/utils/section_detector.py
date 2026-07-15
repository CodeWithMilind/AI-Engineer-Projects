import logging
import re
from typing import Dict, List, Any


def detect_sections(parsed_blocks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Detect invoice sections using rule-based logic with vertical position ordering.
    
    Why this function exists:
    - Organizes unstructured blocks into meaningful sections
    - Provides a foundation for intelligent chunking later
    - Demonstrates how layout information (y-coordinates) can be used
    
    How rule-based detection works now:
    - Uses section order based on typical invoice layout (top to bottom)
    - First find all section headers and record their y-positions
    - Assign each block to the section whose header is the closest above it
    - Fallback to keyword matching if needed
    
    Sections in typical order (top to bottom):
    1. HEADER: Invoice number, date, etc. (top of page)
    2. VENDOR: Seller information
    3. CUSTOMER: Buyer information
    4. ITEMS: List of products/services
    5. TOTALS: Subtotal, tax, grand total
    6. BANK: Payment details
    
    Args:
        parsed_blocks: List of structured block dictionaries (must have y0)
        
    Returns:
        Dictionary mapping section names to lists of text content
    """
    logger = logging.getLogger(__name__)
    logger.info("Detecting sections...")
    
    sections = {
        "HEADER": [],
        "VENDOR": [],
        "CUSTOMER": [],
        "ITEMS": [],
        "TOTALS": [],
        "BANK": []
    }
    
    # Define section order (top to bottom) and their keywords
    section_order = ["HEADER", "VENDOR", "CUSTOMER", "ITEMS", "TOTALS", "BANK"]
    section_keywords = {
        "HEADER": ["invoice", "date"],
        "VENDOR": ["vendor", "seller", "from", "issued by"],
        "CUSTOMER": ["customer", "buyer", "bill to"],  # Removed "to" because too common
        "ITEMS": ["item", "product", "description", "quantity", "rate"],
        "TOTALS": ["total", "subtotal", "gst", "grand total"],  # Removed "tax" because it matches "TAX INVOICE"
        "BANK": ["bank", "account", "ifsc", "payment"]
    }
    
    # Step 1: Find all section header blocks and their y0 positions
    section_headers = []
    for block in parsed_blocks:
        # Only check the FIRST LINE of the block for section keywords
        first_line = block["text"].split("\n")[0].strip().lower()
        
        # First check if it's a HEADER (prioritize this)
        assigned_section = None
        for keyword in section_keywords["HEADER"]:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, first_line):
                assigned_section = "HEADER"
                break
        
        # If not HEADER, check other sections in reverse order
        if not assigned_section:
            for section in reversed(section_order):
                if section == "HEADER":
                    continue  # Already checked
                for keyword in section_keywords[section]:
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, first_line):
                        assigned_section = section
                        break
                if assigned_section:
                    break
        
        if assigned_section:
            section_headers.append({
                "section": assigned_section,
                "y0": block["y0"],
                "block": block
            })
    
    # Step 2: Sort section headers by y0 (top to bottom)
    section_headers.sort(key=lambda x: x["y0"])
    
    # Step 3: Assign each block to the correct section
    for block in parsed_blocks:
        if not block["text"].strip():
            continue
            
        block_y0 = block["y0"]
        
        # Find the last section header that is above or at this block
        assigned_section = "HEADER"  # Default to header
        for header in reversed(section_headers):
            if block_y0 >= header["y0"]:
                assigned_section = header["section"]
                break
        
        sections[assigned_section].append(block["text"])
    
    logger.info("Section detection completed.")
    return sections


def print_sections(sections: Dict[str, List[str]]) -> None:
    """
    Print detected sections in a readable format.
    
    Why this function exists:
    - Makes it easy to verify that sections are being detected correctly
    - Provides a clear overview of the document structure
    
    Args:
        sections: Dictionary mapping section names to lists of text content
    """
    logger = logging.getLogger(__name__)
    logger.info("Printing sections...")
    
    for section, content in sections.items():
        print("\n" + "=" * 20 + f" {section} " + "=" * 20)
        if content:
            for item in content:
                print(f"\n{item}")
        else:
            print("\n(No content detected)")
