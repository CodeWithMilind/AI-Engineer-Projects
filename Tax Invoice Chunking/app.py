from pathlib import Path
from utils import (
    setup_logging,
    load_pdf,
    extract_page_layouts,
    parse_blocks,
    parse_layout_dict,
    save_layout_data,
    detect_sections,
    print_sections,
    detect_tables,
    parse_detected_table,
    table_to_dict,
    print_table_pretty,
    save_tables_to_json,
    save_invoice_structured
)


def main():
    """
    Main function to run the Document Layout Understanding and Table Parsing stages.
    
    Updated pipeline:
        PDF → Load → Extract Layouts → Parse Blocks/Layout → Save Layout Data
            → Detect Sections → Detect Tables → Parse Tables → Print/Save Structured Data
    """
    setup_logging()
    
    pdf_path = Path("data/invoices/invoice.pdf")
    output_dir = Path("extracted")
    
    try:
        # Step 1: Load PDF
        doc, page_count, filename = load_pdf(pdf_path)
        print("=" * 60)
        print(f"File Name : {filename}")
        print(f"Total Pages : {page_count}")
        print("=" * 60)
        
        # Step 2: Extract layouts
        layouts = extract_page_layouts(doc)
        doc.close()
        
        # Step 3: Parse blocks and layout dict (don't print to keep output clean)
        all_parsed_blocks = []
        for blocks in layouts["blocks"]:
            parsed = parse_blocks(blocks)
            all_parsed_blocks.append(parsed)
        
        all_parsed_layout = []
        for layout_dict in layouts["layout_dict"]:
            parsed = parse_layout_dict(layout_dict)
            all_parsed_layout.append(parsed)
        
        # Step 4: Save layout data
        save_layout_data(
            layouts["plain_text"],
            all_parsed_blocks,
            all_parsed_layout,
            output_dir
        )
        
        # Step 5: Detect sections (using first page as example)
        sections = {}
        if all_parsed_blocks:
            sections = detect_sections(all_parsed_blocks[0])
            print_sections(sections)
        
        # Step 6: Detect and parse tables
        tables_data = []
        detected_tables = detect_tables(layouts["layout_dict"])
        for i, detected_table in enumerate(detected_tables):
            parsed_rows = parse_detected_table(detected_table)
            table_name = f"Invoice Items {i+1}" if i > 0 else "Invoice Items"
            table_dict = table_to_dict(table_name, detected_table.column_boundaries, parsed_rows)
            tables_data.append(table_dict)
            # Print the table nicely
            print_table_pretty(table_name, detected_table.column_boundaries, parsed_rows)
        
        # Step 7: Save tables and structured invoice data
        save_tables_to_json(tables_data, output_dir)
        save_invoice_structured(sections, tables_data, output_dir)
        
        print("\n" + "=" * 60)
        print("DOCUMENT LAYOUT UNDERSTANDING AND TABLE PARSING COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()