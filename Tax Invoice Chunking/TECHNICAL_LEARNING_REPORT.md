# Invoice Document Understanding System

---

## Project Overview

### What is This Project?
This is an **Enterprise Invoice Document Understanding System** built entirely from scratch—no shortcuts, no pre-built libraries for layout analysis (no LangChain, no LlamaParse, no Unstructured). The goal is to understand how production-grade Document AI systems work under the hood.

### Why Document Understanding Matters?
Modern businesses process millions of documents daily—invoices, receipts, purchase orders, contracts, etc. Manually extracting data from these documents is:
- Time-consuming
- Error-prone
- Expensive

Document Understanding systems automate this process, turning unstructured/semi-structured PDFs into structured, machine-readable data that can be used for:
- Automated bookkeeping
- Expense tracking
- Tax calculation
- Audit trails
- Business intelligence

### Why Invoices Are Hard for AI?
Invoices are deceptively complex:
- **Varied Layouts**: No two vendors use the exact same invoice format
- **Tables**: Contain line items with critical quantity/price information
- **Key-Value Pairs**: Vendor name, invoice number, date, total—all in different places
- **Fonts/Formatting**: Bold, italic, different sizes to indicate importance
- **Coordinates Matter**: Text placement defines meaning (e.g., "Bill To" vs "Ship To")
- **Scanned vs Digital**: Some invoices are scanned images (not covered in this stage)

---

## Problem Statement

### Why Plain PDF Extraction Isn’t Enough?
If you only extract plain text from a PDF invoice (like using `page.get_text()` in PyMuPDF), you lose all **spatial context**.

#### Example: Plain Text vs Layout-Aware Extraction
Suppose we have this part of an invoice:

```
Item           Qty      Unit Price     Total
Laptop          2        50,000       100,000
Mouse           3           500         1,500
```

Plain text extraction gives you:
```
Item
Qty
Unit Price
Total
Laptop
2
50,000
100,000
Mouse
3
500
1,500
```

The problem? You have no idea which numbers belong to which items! This is where **layout understanding** becomes critical—we need to preserve the spatial relationships between text elements.

#### Key Problems with Plain Extraction:
1. **Lost Layout**: Tables become lists of random words
2. **Broken Tables**: Relationships between columns/rows disappear
3. **Mixed Text**: Vendor info, customer info, line items all jumbled together
4. **Missing Hierarchy**: No way to tell header from body from footer

---

## Goals
- **Learn by Building**: Implement everything manually to understand internal workings
- **Production-Grade Code**: Use modularity, type hints, logging, error handling
- **Preserve Spatial Context**: Extract coordinates, blocks, lines, spans
- **Rule-Based First**: No ML (yet)—understand rule-based systems before adding complexity
- **Structured Output**: Convert unstructured PDFs into machine-readable JSON
- **Maintainable Pipeline**: Keep each stage separate for future improvements

---

## Current Architecture

### Pipeline Diagram (ASCII)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Invoice Document Understanding                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────────┐  │
│  │  Invoice    │────▶│   PDF Reader     │────▶│  Layout Extraction      │  │
│  │    PDF      │     │  (PyMuPDF)       │     │  (Blocks/Lines/Spans)   │  │
│  └─────────────┘     └──────────────────┘     └─────────────────────────┘  │
│                                                     │                        │
│                                                     ▼                        │
│  ┌─────────────────────────┐     ┌───────────────────────────────────────┐ │
│  │ Structured JSON Export  │◀────│   Table Detection & Reconstruction    │ │
│  └─────────────────────────┘     └───────────────────────────────────────┘ │
│                                                     │                        │
│                                                     ▼                        │
│                              ┌─────────────────────────┐                   │
│                              │   Section Detection     │                   │
│                              └─────────────────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure
Let’s look at the project directory tree and explain what each part does.

```
invoice-rag/
├── app.py                          # Main entry point (orchestrates the pipeline)
├── requirements.txt                # Dependencies (only PyMuPDF currently)
├── TECHNICAL_LEARNING_REPORT.md    # This file
├── data/
│   └── invoices/
│       └── invoice.pdf            # Sample input invoice
├── extracted/                     # All generated output files
│   ├── plain_text.txt
│   ├── blocks.json
│   ├── layout.json
│   ├── tables.json
│   └── invoice_structured.json
├── chunks/                        # Reserved for future stage
├── embeddings/                    # Reserved for future stage
├── vector_db/                     # Reserved for future stage
├── models/                        # Reserved for future stage
└── utils/
    ├── __init__.py                # Exports all utility functions
    ├── pdf_reader.py              # Handles PDF loading and basic layout extraction
    ├── layout_parser.py           # Parses blocks, lines, spans, and saves to JSON
    ├── section_detector.py        # Rule-based section detection (Header/Vendor/Customer/etc.)
    ├── table_detector.py          # Detects tables using spatial coordinates
    ├── table_parser.py            # Reconstructs tables into structured rows/columns
    └── table_exporter.py          # Exports tables to JSON and prints neatly
```

### Detailed Module Explanations

#### `app.py`
- **Purpose**: Orchestrates the entire pipeline
- **Why it exists**: Keeps business logic out of utility modules—only responsible for calling functions in order
- **Key actions**:
  - Sets up logging
  - Loads the PDF
  - Extracts and parses layout
  - Saves layout data
  - Detects sections
  - Detects, parses, and saves tables

#### `requirements.txt`
- **Purpose**: Lists project dependencies
- **Current content**: `pymupdf`

#### `utils/__init__.py`
- **Purpose**: Exports all public utility functions/classes so `app.py` can import them easily
- **Key exports**:
  - `setup_logging()`, `load_pdf()`, `extract_page_layouts()`
  - `parse_blocks()`, `parse_layout_dict()`, `save_layout_data()`
  - `detect_sections()`, `print_sections()`
  - `detect_tables()`, `parse_detected_table()`
  - `table_to_dict()`, `print_table_pretty()`, `save_tables_to_json()`, `save_invoice_structured()`

#### `utils/pdf_reader.py`
- **Purpose**: Handles initial PDF loading and raw layout extraction
- **Why it exists**: Centralizes all PDF reading logic—separates concerns
- **Key functions**:
  1. `setup_logging()`: Configures Python logging to show timestamps and module names
  2. `load_pdf()`: Opens a PDF file, returns the PyMuPDF doc, page count, and filename
  3. `extract_page_layouts()`: For each page, extracts plain text, blocks, and the full layout dict from PyMuPDF

#### `utils/layout_parser.py`
- **Purpose**: Takes raw layout data from PyMuPDF and converts it into structured, usable formats
- **Why it exists**: Raw PyMuPDF output is low-level—this makes it easy to work with
- **Key functions**:
  1. `parse_blocks()`: Converts PyMuPDF block tuples into structured dictionaries with coordinates and text
  2. `parse_layout_dict()`: Traverses PyMuPDF's full layout dictionary (blocks → lines → spans)
  3. `save_layout_data()`: Writes plain text, blocks, and layout dict to files in `extracted/`

#### `utils/section_detector.py`
- **Purpose**: Uses rule-based logic to group blocks into logical invoice sections
- **Why it exists**: Turns a list of blocks into meaningful sections like "Vendor" or "Items"
- **Key functions**:
  1. `detect_sections()`: Detects sections by looking for keywords in the first line of blocks (and checks vertical ordering)
  2. `print_sections()`: Prints detected sections in a readable format

#### `utils/table_detector.py`
- **Purpose**: Detects tables using spatial information
- **Why it exists**: Identifies where tables are located in the document
- **Key components**:
  - `ColumnBoundary` (dataclass): Stores column x0/x1 and header text
  - `DetectedTable` (dataclass): Stores table blocks, header, columns, and coordinates
  - `detect_column_boundaries()`: Finds columns from the header block's lines
  - `is_table_block_candidate()`: Checks if a block should be part of the table
  - `detect_tables()`: Main detection function using heuristics (bold text, multiple lines, horizontally aligned lines)

#### `utils/table_parser.py`
- **Purpose**: Takes a detected table and reconstructs its rows and columns
- **Why it exists**: Converts a list of blocks into a structured table
- **Key components**:
  - `ParsedTableRow` (dataclass): Stores a dictionary of column header → cell value
  - `assign_to_column()`: Uses cell coordinates to assign it to the correct column
  - `parse_detected_table()`: Reconstructs all rows in the table

#### `utils/table_exporter.py`
- **Purpose**: Exports structured tables to JSON and prints them nicely
- **Why it exists**: Makes table output human-readable and machine-readable
- **Key functions**:
  1. `table_to_dict()`: Converts a parsed table into a dictionary for JSON export
  2. `print_table_pretty()`: Prints the table in a readable ASCII format in the terminal
  3. `save_tables_to_json()`: Writes all tables to `extracted/tables.json`
  4. `save_invoice_structured()`: Writes sections and tables together to `extracted/invoice_structured.json`

---

## Data Flow

Let’s trace an invoice step by step through the pipeline.

### Step 1: PDF → PyMuPDF
We start with `invoice.pdf` in `data/invoices/`.
- Call `load_pdf(pdf_path)` in `pdf_reader.py`
- PyMuPDF opens the file, creates a `Document` object
- We get: filename, page count, and the doc object

### Step 2: PyMuPDF → Raw Layout Data
Next, call `extract_page_layouts(doc)`:
- For each page in the document:
  - `page.get_text()` → plain text string
  - `page.get_text("blocks")` → list of block tuples (x0, y0, x1, y1, text, block_no, block_type)
  - `page.get_text("dict")` → full hierarchical dictionary (page → blocks → lines → spans)
- Store all this in a dictionary: `{"plain_text": [...], "blocks": [...], "layout_dict": [...]}`

### Step 3: Raw Layout Data → Parsed Data
Call `parse_blocks()` and `parse_layout_dict()` on the extracted data:
- `parse_blocks()`: Converts block tuples into structured dictionaries with `block_number`, `x0`, `y0`, `x1`, `y1`, `width`, `height`, `text`
- `parse_layout_dict()`: Traverses the layout dict and creates a structured version with blocks, lines, spans, and their coordinates

### Step 4: Parsed Data → Sections
Call `detect_sections()` on the parsed blocks:
- Check first line of each block for keywords (e.g., "Vendor", "Customer")
- Use vertical ordering to assign blocks to sections (blocks after the "Vendor" keyword go in Vendor section, etc.)
- Use word boundaries for keyword matching (to avoid "TAX INVOICE" matching "Total")

### Step 5: Raw Layout → Tables
Call `detect_tables()` using `layouts["layout_dict"]`:
- Find blocks that are table headers:
  - At least 2 lines
  - At least one bold span
  - All lines are horizontally aligned (same y0 ± 2 points)
- Collect all blocks below the header that are in the same x range
- Detect columns from the header's lines
- Call `parse_detected_table()` to reconstruct rows
- Call `save_tables_to_json()` and `save_invoice_structured()`

---

## Stage 1: PDF Reading

### Why PyMuPDF?
PyMuPDF (also called `fitz`) was chosen because:
1. **Fast**: Written in C, much faster than pure Python libraries
2. **Layout Preservation**: Gives access to low-level layout info (blocks, lines, spans, coordinates)
3. **No External Dependencies**: Just one package to install
4. **Mature**: Has been around for years, stable and reliable

### How Pages Are Read
In `pdf_reader.py`:
- `doc = fitz.open(pdf_path)`: Opens the PDF
- `page = doc.load_page(page_num)`: Loads a single page
- `doc.close()`: Closes the document (important to avoid memory leaks!)

### Output Generated
At this stage, we generate the raw data that later stages will parse:
- Plain text per page
- Blocks per page (tuples)
- Layout dict per page (hierarchical)

---

## Stage 2: Layout Understanding

### Core Concepts
To understand how this stage works, we need to know PyMuPDF’s layout hierarchy:

```
Page
  └── Blocks
        └── Lines
              └── Spans
                    └── Text
```

#### What Are Blocks?
- **Definition**: A rectangular region of text or image on the page
- **Why they matter**: Groups related text (e.g., vendor info is one block, line item is another block)
- **Properties**: `x0`, `y0` (top-left corner), `x1`, `y1` (bottom-right corner), `block_type` (0 for text, 1 for image), `text`

#### What Are Lines?
- **Definition**: A horizontal line of text inside a block
- **Why they matter**: Represents a single row of text (e.g., "Invoice No: INV-2026-001" is a line)
- **Properties**: `x0`, `y0`, `x1`, `y1`, list of spans

#### What Are Spans?
- **Definition**: A contiguous sequence of text with the same font and size
- **Why they matter**: Tells us about formatting (e.g., bold for headers)
- **Properties**: `x0`, `y0`, `x1`, `y1`, `text`, `font`, `size`, `color`

#### What Are Coordinates?
- **Units**: Points (1 point = 1/72 inch)
- **Origin**: Top-left corner of the page (0,0)
- **Why they matter**: The single most important piece of information for document understanding! Spatial context defines meaning.

### Dictionary Structure Example (Simplified)
Here’s a tiny snippet of what `page.get_text("dict")` returns:
```json
{
  "blocks": [
    {
      "type": 0,
      "bbox": [240.13, 76.74, 355.15, 101.53],
      "lines": [
        {
          "bbox": [240.13, 76.74, 355.15, 101.53],
          "spans": [
            {
              "text": "TAX INVOICE",
              "font": "Helvetica-Bold",
              "size": 18.0,
              "color": 0
            }
          ]
        }
      ]
    }
  ]
}
```

---

## Stage 3: Section Detection

### Sections We Detect
We currently detect 6 sections:
1. **Header**: Invoice number, date
2. **Vendor**: Seller name, address, GSTIN
3. **Customer**: Buyer name, address, GSTIN
4. **Items**: Line items (table)
5. **Totals**: Subtotal, tax, grand total
6. **Bank Details**: Account name, bank, account number, IFSC

### How Detection Works
We use a **rule-based approach** with 3 key ideas:
1. **Keyword Matching**: Look for section headers in the first line of a block (e.g., "Vendor", "Customer")
2. **Vertical Ordering**: Use block y-coordinates to assign blocks to sections (blocks after a section header go in that section)
3. **Word Boundaries**: Use regex word boundaries (`\b`) to avoid partial matches (so "tax" doesn't match "TAX INVOICE")

### Heuristics Used
In `utils/section_detector.py`:
- Check the first line of each block (section headers are usually at the top of a block)
- Prioritize HEADER section first so "TAX INVOICE" doesn't get assigned to TOTALS
- Check for sections in reverse order when looking at non-header sections to avoid "to" matching "Customer" when it's part of another word

---

## Stage 4: Table Understanding

### How Rows Are Detected
For our sample invoice, each line item is a separate block! So:
- The first block after the header is the first data row
- The second block after the header is the second data row
- And so on!

For more complex tables where a single row spans multiple blocks, we could group blocks by y-coordinate—but our invoice is simple!

### How Columns Are Detected
Columns are detected from the **table header block**:
- Each line in the header block represents one column
- We use the line's `bbox[0]` (x0) and `bbox[2]` (x1) as the column boundaries

### How Coordinates Are Used
1. **Column Assignment**: For each cell (line in a data block), calculate its midpoint: `(x0 + x1)/2`
2. **Find Column**: Check which column boundary contains this midpoint; if none, use the closest column
3. **Why This Works**: Cells in the same column are vertically aligned—their x-coordinates will fall within the column boundaries from the header!

### Why This Is Better Than Plain Text
- **Preserves Relationships**: "Laptop" stays with "Qty: 2" and "Total: 100,000"
- **Machine-Readable**: Can easily import into a database or spreadsheet
- **Enables Analytics**: You can calculate total quantity, average price, etc.

---

## Output Files

### `extracted/plain_text.txt`
- **What it is**: Concatenated plain text from all pages
- **Why we save it**: Useful for quick verification, and for future stages that might use plain text (but we’ll mostly use layout data!)

### `extracted/blocks.json`
- **What it is**: List of parsed blocks per page (structured dictionaries, not tuples)
- **Structure per block**:
  ```json
  {
    "block_number": 0,
    "block_type": 0,
    "x0": 240.1278076171875,
    "y0": 76.73999786376953,
    "x1": 355.1477966308594,
    "y1": 101.5260009765625,
    "width": 115.01998901367188,
    "height": 24.78600311279297,
    "text": "TAX INVOICE"
  }
  ```

### `extracted/layout.json`
- **What it is**: Full parsed layout (blocks → lines → spans) per page
- **Why we save it**: Contains all low-level layout information for debugging or future improvements

### `extracted/tables.json`
- **What it is**: Structured data for all detected tables
- **Structure**:
  ```json
  [
    {
      "table_name": "Invoice Items",
      "columns": ["Item", "Qty", "Unit Price (I)", "Total (I)"],
      "rows": [
        {"Item": "Laptop", "Qty": "2", "Unit Price (I)": "50,000", "Total (I)": "100,000"},
        ...
      ]
    }
  ]
  ```

### `extracted/invoice_structured.json`
- **What it is**: The most important output file! Combines both sections and tables into one structured file.
- **Structure**:
  ```json
  {
    "sections": { ... },
    "tables": [ ... ]
  }
  ```

---

## Sample Data Flow

Let’s take one line item from our invoice and see how it flows through the system.

### Input (PDF):
```
Laptop          2        50,000        100,000
```

### Step 1: Raw PyMuPDF Block
```python
(x0=91.24, y0=263.25, x1=481.31, y1=276.99, text="Laptop\n2\n50,000\n100,000", block_no=5, block_type=0)
```

### Step 2: Parsed Layout Block
```json
{
  "block_number": 5,
  "x0": 91.24,
  "y0": 263.25,
  "x1": 481.31,
  "y1": 276.99,
  "lines": [
    {"spans": [{"text": "Laptop"}], "bbox": [91.24, 263.25, 121.82, 276.99]},
    {"spans": [{"text": "2"}], "bbox": [287.66, 263.25, 293.22, 276.99]},
    {"spans": [{"text": "50,000"}], "bbox": [350.75, 263.25, 381.33, 276.99]},
    {"spans": [{"text": "100,000"}], "bbox": [445.17, 263.25, 481.31, 276.99]}
  ]
}
```

### Step 3: Assigned to Columns
- "Laptop" → Column 0 ("Item")
- "2" → Column 1 ("Qty")
- "50,000" → Column 2 ("Unit Price (I)")
- "100,000" → Column 3 ("Total (I)")

### Step 4: Final Structured Output
```json
{"Item": "Laptop", "Qty": "2", "Unit Price (I)": "50,000", "Total (I)": "100,000"}
```

---

## Design Decisions

### 1. PyMuPDF (Instead of pdfplumber, PyPDF2, etc.)
- **Why?** PyMuPDF preserves layout information better and is faster than most other Python PDF libraries.
- **Alternative considered**: `pdfplumber`—also great for layout, but PyMuPDF was chosen for speed and simplicity.

### 2. JSON for Storage
- **Why?** JSON is:
  - Human-readable
  - Machine-readable
  - Supported by every programming language
  - Easy to debug
- **Alternative considered**: CSV—good for tables, but not good for hierarchical data like layout dict.

### 3. Coordinates Are Critical
- **Why we use them**: Spatial context defines meaning in documents. Without coordinates, tables become random text.
- **Why not just text order?** Text order in PDFs isn’t always top-to-bottom/left-to-right—coordinates are more reliable.

### 4. Logging Everywhere
- **Why?** Logging helps with debugging and understanding what the system is doing (especially important when things go wrong!).
- **What we log**:
  - PDF loading
  - Page processing
  - Block/line/span parsing
  - Section/table detection
  - Saving files

### 5. Modular Architecture
- **Why?** Each module does one thing and does it well (Single Responsibility Principle).
- **Benefits**:
  - Easy to test each module independently
  - Easy to update one stage without breaking others
  - Easy to add new features later (like semantic chunking)

---

## Challenges Faced

### 1. Too Many Tables Detected
- **Problem**: Vendor, Customer, and other blocks were being detected as tables because they had multiple lines and bold text!
- **Solution**: Added a heuristic that all lines in a table header must be **horizontally aligned** (same y0 ± 2 points). This fixed it immediately!

### 2. Partial Keyword Matches
- **Problem**: "TAX INVOICE" was being assigned to the TOTALS section because it contained the substring "tax"!
- **Solution**: Use regex word boundaries (`\b`) when matching keywords. Now "tax" only matches "tax" as a whole word, not part of "TAX INVOICE".

### 3. Common Words as Keywords
- **Problem**: The CUSTOMER section had "to" as a keyword, which matched all kinds of things!
- **Solution**: Removed "to" from the keywords list—only use more specific keywords like "Customer" or "Bill To".

### 4. PyMuPDF’s Data Structure
- **Problem**: Initially forgot that `page.get_text("dict")` returns a dict with a `blocks` key, not a list of blocks directly!
- **Solution**: Fixed `detect_tables()` to get `page_blocks = page_dict.get("blocks", [])` instead of treating `page_dict` as the blocks list.
- **Lesson**: Always check the library’s documentation carefully!

### 5. "I" in Currency Symbols
- **Problem**: PyMuPDF was extracting the Indian Rupee symbol as "I" (because of the font ZapfDingbats)!
- **Solution**: This is a known issue with PyMuPDF and certain fonts. For now, we just keep it as "I"—future stages could handle this with a lookup table, but it’s not critical for our current goals.

---

## Current Limitations

### 1. Only Rule-Based
- No ML—can’t handle weird or unseen layouts (only works for invoices similar to our sample)
- Solution for later: Add ML models like LayoutLM or TableTransformer for better generalization

### 2. Only Digital PDFs
- Can’t handle scanned PDFs (images)!
- Solution for later: Add OCR (using Tesseract or easyOCR)

### 3. Simple Tables Only
- Our table detection assumes one block per row—can’t handle:
  - Merged cells
  - Multi-line cells
  - Tables spanning multiple pages

### 4. Hardcoded Keywords
- Section detection relies on specific keywords (e.g., "Vendor")—if an invoice says "Seller" instead, it won’t detect the section
- Solution for later: Make keywords configurable, or use NER (Named Entity Recognition)

### 5. Single Invoice Tested
- Only tested on one sample invoice—we don’t know how it performs on other layouts!

---

## Future Roadmap
(Only brief mentions—future stages!)
- **Semantic Chunking**: Split structured data into meaningful chunks for RAG
- **Embeddings**: Convert chunks into vector embeddings using models like sentence-transformers
- **FAISS**: Store embeddings in a FAISS vector database for fast similarity search
- **Retriever**: Build a retriever to find relevant chunks for a query
- **LLM**: Connect to an LLM (like GPT-4o, Claude 3) to answer questions about invoices
- **Chat Interface**: Build a simple web UI to ask questions about invoices

---

## Interview Questions
Here are 30+ interview questions based on the current implementation, with answers and follow-ups!

---

### Questions About PDF Reading
1. **Question**: Why did you choose PyMuPDF over other PDF libraries like PyPDF2 or pdfplumber?
   - **Answer**: PyMuPDF is fast (written in C), preserves detailed layout information (blocks/lines/spans/coordinates), and has a simple API. PyPDF2 is good for basic tasks but doesn’t preserve layout well. pdfplumber is also good, but PyMuPDF was chosen for speed and simplicity for this project.
   - **Follow-up**: If you had to handle scanned PDFs, what would you add?
   - **Follow-up Answer**: OCR! Use libraries like Tesseract, easyOCR, or cloud services (AWS Textract, Google Vision) to extract text from images.

2. **Question**: What’s the difference between `page.get_text()`, `page.get_text("blocks")`, and `page.get_text("dict")` in PyMuPDF?
   - **Answer**: 
     - `page.get_text()`: Returns plain text as a single string (loses layout).
     - `page.get_text("blocks")`: Returns a list of tuples where each tuple is a block (x0,y0,x1,y1,text,block_no,block_type).
     - `page.get_text("dict")`: Returns a hierarchical dictionary with full layout info (page → blocks → lines → spans, each with coordinates, font info, etc.).
   - **Follow-up**: Which one do you use for table detection and why?
   - **Follow-up Answer**: `page.get_text("dict")` because we need line-level coordinates to detect columns and assign cells!

---

### Questions About Layout Understanding
3. **Question**: What is a "span" in PyMuPDF?
   - **Answer**: A span is a contiguous sequence of text with the same font, size, and color!
   - **Follow-up**: Why are spans important?
   - **Follow-up Answer**: They tell you about formatting—e.g., bold text is often a header or important label!

4. **Question**: Explain the layout hierarchy in PyMuPDF.
   - **Answer**: Page → Blocks → Lines → Spans. Pages contain blocks, blocks contain lines, lines contain spans, spans contain text!
   - **Follow-up**: What’s a block?
   - **Follow-up Answer**: A rectangular region of text or image—groups related content together!

5. **Question**: What are coordinates in PyMuPDF measured in, and where is the origin?
   - **Answer**: Coordinates are in points (1 point = 1/72 inch). The origin (0,0) is the **top-left corner** of the page!
   - **Follow-up**: Why is that different from typical image coordinates (where origin is often bottom-left)?
   - **Follow-up Answer**: PDFs are designed for printing, and the top-left origin is standard in desktop publishing!

---

### Questions About Section Detection
6. **Question**: How did you detect sections in the invoice?
   - **Answer**: Rule-based! We look for keywords in the first line of each block (e.g., "Vendor", "Customer") and use vertical ordering (block y-coordinates) to assign subsequent blocks to the section! We also use regex word boundaries to avoid partial matches!
   - **Follow-up**: What are some limitations of rule-based section detection?
   - **Follow-up Answer**: It only works for invoices with the exact keywords we specified! If an invoice uses "Seller" instead of "Vendor", it won’t detect the section!

7. **Question**: Why did you use regex word boundaries (`\b`) for keyword matching?
   - **Answer**: To avoid partial matches! For example, we don’t want "TAX INVOICE" to match the "tax" keyword in our TOTALS section! Word boundaries ensure we only match whole words!
   - **Follow-up**: What’s a regex word boundary?
   - **Follow-up Answer**: `\b` matches the position between a word character (a-z, A-Z, 0-9, _) and a non-word character!

8. **Question**: How did you decide which section a block belongs to?
   - **Answer**: We find the last section header block that is above the current block (using y-coordinates), then assign the block to that section!
   - **Follow-up**: What if a block is before any section header?
   - **Follow-up Answer**: It goes to the HEADER section by default!

---

### Questions About Table Understanding
9. **Question**: How did you detect tables in the invoice?
   - **Answer**: Rule-based using 3 heuristics:
     1. Block has at least 2 lines
     2. Block has at least one bold span
     3. All lines in the block are horizontally aligned (same y0 ± 2 points)
   - **Follow-up**: Why the horizontal alignment check?
   - **Follow-up Answer**: Because table headers have all their column labels on the same line! Sections like Vendor or Customer have lines stacked vertically, so this check filters them out!

10. **Question**: How did you detect column boundaries?
    - **Answer**: From the table header block! Each line in the header is a column—we use the line’s `bbox[0]` and `bbox[2]` as the column’s x0 and x1!
    - **Follow-up**: What if a column has multiple lines in the header?
    - **Follow-up Answer**: For our invoice, it doesn’t—but if it did, we could take the minimum x0 and maximum x1 of all lines in the column!

11. **Question**: How did you assign a cell to the correct column?
    - **Answer**: Calculate the cell’s midpoint (`(x0 + x1)/2`), then find the column boundary that contains this midpoint! If none, use the closest column!
    - **Follow-up**: Why use midpoint instead of x0 or x1?
    - **Follow-up Answer**: Midpoint is more robust—avoids edge cases where a cell is slightly overlapping two columns!

12. **Question**: Why is table understanding better than just plain text?
    - **Answer**: Because it preserves the **relationships** between columns and rows! For example, you know that "Laptop" has Qty 2 and Total 100,000—with plain text, you just have a list of words!
    - **Follow-up**: What are some use cases for structured tables?
    - **Follow-up Answer**: Automated bookkeeping, expense tracking, tax calculation, business intelligence!

---

### Questions About Architecture
13. **Question**: Why did you choose a modular architecture?
    - **Answer**: Single Responsibility Principle—each module does one thing! This makes it easy to test, debug, and update individual stages without breaking others!
    - **Follow-up**: Give an example of how modularity helps.
    - **Follow-up Answer**: If we want to replace rule-based section detection with ML later, we just need to change `section_detector.py`—no other files need to be touched!

14. **Question**: What’s the purpose of `app.py`?
    - **Answer**: It’s the main entry point—it orchestrates the pipeline! It doesn’t contain business logic—just calls utility functions in order!
    - **Follow-up**: Why not put everything in `app.py`?
    - **Follow-up Answer**: It would become a huge, unmaintainable mess! Modularity keeps code organized!

15. **Question**: Why did you use dataclasses for `ColumnBoundary`, `DetectedTable`, etc.?
    - **Answer**: Dataclasses make code cleaner! They automatically generate `__init__`, `__repr__`, and other dunder methods—less boilerplate code!
    - **Follow-up**: What’s a dataclass in Python?
    - **Follow-up Answer**: A decorator (`@dataclass`) that adds generated methods to a class—great for storing data!

---

### Questions About Output
16. **Question**: What’s the difference between `blocks.json` and `layout.json`?
    - **Answer**: `blocks.json` has a simplified structure with just block-level info (x0,y0,x1,y1,text). `layout.json` has the full hierarchy (blocks → lines → spans) with all font and coordinate details!
    - **Follow-up**: When would you use one over the other?
    - **Follow-up Answer**: Use `blocks.json` for simple tasks (like section detection), use `layout.json` when you need line/span-level details (like table detection)!

17. **Question**: What’s in `invoice_structured.json` and why is it important?
    - **Answer**: It’s the most important output file—combines both sections and tables into one structured JSON file! This is what future stages (like semantic chunking) will use!
    - **Follow-up**: Why not save sections and tables separately?
    - **Follow-up Answer**: We do save them separately too! But having them in one file is convenient for downstream use!

18. **Question**: Why did you choose JSON for output files?
    - **Answer**: JSON is human-readable, machine-readable, supported by every language, and easy to debug!
    - **Follow-up**: What other formats could you use?
    - **Follow-up Answer**: CSV (for tables), XML (for hierarchical data), Parquet (for big data), etc.—but JSON is the best balance of simplicity and functionality for this project!

---

### Questions About Design Decisions
19. **Question**: Why did you use logging instead of print statements?
    - **Answer**: Logging is more flexible! You can set different log levels (DEBUG, INFO, WARNING, ERROR), write logs to files, add timestamps and module names, and disable logging in production!
    - **Follow-up**: What log level did you use for most messages?
    - **Follow-up Answer**: INFO! For debug details, we could use DEBUG level!

20. **Question**: If you had to handle multi-page invoices, what would you change?
    - **Answer**: Currently, we process pages individually—for multi-page tables, we’d need to:
      - Track table state across pages
      - Check if a block on page 2 is part of the table from page 1
      - Handle headers/footers that repeat on every page
    - **Follow-up**: How would you track table state across pages?
    - **Follow-up Answer**: Store the last detected table’s column boundaries and y-range—if a block on the next page is within the x-range and below the previous table’s end, it’s part of the same table!

---

### Questions About Challenges
21. **Question**: What was the biggest challenge you faced so far?
    - **Answer**: Too many false positives for table detection! Vendor, Customer, and other blocks were being detected as tables! Fixed by adding the horizontal alignment check for header lines!
    - **Follow-up**: What’s a false positive?
    - **Follow-up Answer**: When the system detects something that isn’t actually there (e.g., detecting a table where there isn’t one)!

22. **Question**: How did you debug issues with the pipeline?
    - **Answer**:
      1. Logging—added lots of INFO and DEBUG logs
      2. Saved intermediate outputs (blocks.json, layout.json) to inspect
      3. Printed blocks and layout to the terminal to see what’s happening
    - **Follow-up**: What’s your debugging process in general?
    - **Follow-up Answer**: Reproduce the issue, isolate the problem, form a hypothesis, test the hypothesis, fix, verify!

23. **Question**: How did you handle the "I" instead of Rupee symbol issue?
    - **Answer**: It’s a known issue with PyMuPDF and ZapfDingbats font—for now, we just keep it as "I"! It’s not critical for our current goals, but future stages could add a lookup table to replace it!
    - **Follow-up**: Why does PyMuPDF do that?
    - **Follow-up Answer**: Some fonts (like ZapfDingbats) use non-standard character encodings—PyMuPDF maps them to the closest ASCII character!

---

### Questions About Limitations
24. **Question**: What are some limitations of the current system?
    - **Answer**:
      1. Only rule-based (can’t handle unseen layouts)
      2. Only works on digital PDFs (not scanned)
      3. Simple tables only (no merged cells, multi-page tables)
      4. Hardcoded keywords
      5. Only tested on one invoice
    - **Follow-up**: How would you fix the hardcoded keywords problem?
    - **Follow-up Answer**: Make keywords configurable via a YAML/JSON file, or use NER (Named Entity Recognition) to find entities like vendor names, invoice numbers, etc.!

25. **Question**: Can this system handle scanned PDFs? Why or why not?
    - **Answer**: No! Because scanned PDFs are just images—there’s no text to extract! We’d need to add OCR (Optical Character Recognition) first!
    - **Follow-up**: What OCR libraries would you use?
    - **Follow-up Answer**: Tesseract (free, open-source), easyOCR (simpler API), or cloud services like AWS Textract, Google Vision, Azure Computer Vision!

---

### Questions About Future Work
26. **Question**: What’s semantic chunking and why would you use it?
    - **Answer**: Semantic chunking is splitting a document into meaningful chunks based on content, not just arbitrary size! For our invoice, chunks could be:
      - Vendor section
      - Customer section
      - Each line item (or all line items)
      - Totals section
      This helps RAG systems retrieve more relevant information!
    - **Follow-up**: How is it different from regular chunking?
    - **Follow-up Answer**: Regular chunking splits by character count (e.g., every 512 tokens)—semantic chunking splits by meaning!

27. **Question**: What are embeddings and why would you use them?
    - **Answer**: Embeddings are numerical representations of text—they capture the semantic meaning of the text! Similar texts have similar embeddings! We’d use them to store chunks in a vector database!
    - **Follow-up**: What embedding models would you use?
    - **Follow-up Answer**: sentence-transformers (all-mpnet-base-v2 is a good default), OpenAI text-embedding-3-small, etc.!

28. **Question**: What is FAISS and why would you use it?
    - **Answer**: FAISS (Facebook AI Similarity Search) is a library for efficient similarity search of dense vectors! We’d use it to store embeddings and quickly find the most similar chunks to a user’s query!
    - **Follow-up**: Why not just use a regular database?
    - **Follow-up Answer**: Because FAISS is optimized for vector similarity search—regular databases would be too slow for large numbers of embeddings!

---

### Bonus Questions
29. **Question**: What is Document AI?
    - **Answer**: Document AI is a field of AI that focuses on understanding, extracting, and processing information from documents (PDFs, images, etc.)!
    - **Follow-up**: What’s the difference between Document AI and NLP?
    - **Follow-up Answer**: NLP focuses on text—Document AI focuses on documents (which include layout, images, tables, etc.)!

30. **Question**: If you could start over, what would you do differently?
    - **Answer**:
      1. Write more tests from the beginning
      2. Use type hints everywhere (we already do most places, but could be more strict)
      3. Add more sample invoices to test on earlier
      4. Use a config file for keywords and thresholds instead of hardcoding
    - **Follow-up**: What tests would you write?
    - **Follow-up Answer**: Unit tests for each utility function, integration tests for the full pipeline, regression tests to make sure changes don’t break existing features!

31. **Question**: How would you improve the section detection?
    - **Answer**:
      1. Use a config file for keywords instead of hardcoding
      2. Add NER to find entities (like vendor names, invoice numbers) without relying on keywords
      3. Use more layout features (like bold text, indentation)
      4. Add more sample invoices to test and refine rules
    - **Follow-up**: What NER models would you use?
    - **Follow-up Answer**: spaCy (with a custom model fine-tuned on invoices), LayoutLM, or cloud services!

32. **Question**: How would you improve the table detection?
    - **Answer**:
      1. Add support for multi-block rows (group blocks by y-coordinate)
      2. Add support for multi-page tables
      3. Add support for merged cells
      4. Use a pre-trained model like TableTransformer for better generalization
    - **Follow-up**: What is TableTransformer?
    - **Follow-up Answer**: A pre-trained model from Hugging Face that detects tables and table structure!

---

## Key Learnings

### 1. Layout Is Everything
Coordinates aren’t just extra data—they’re the most important data! Without spatial context, you lose the meaning of the document.

### 2. Rule-Based Systems Are Great (When They Work)
Rule-based systems are:
- Fast
- Transparent (you know exactly why a decision was made)
- Easy to debug
Perfect for well-structured documents like invoices (when the layout is consistent)!

### 3. Modularity Is Non-Negotiable
Keeping each stage separate makes the codebase maintainable. You can test, debug, and update individual parts without breaking everything else.

### 4. Logging Saves Lives
Print statements are okay for small scripts, but logging is essential for production systems. It helps you understand what’s happening when things go wrong.

### 5. Dataclasses Are Awesome
They reduce boilerplate and make code cleaner. Use them for data structures like `ColumnBoundary` or `ParsedTableRow`!

### 6. Start Small
Don’t try to build the perfect system all at once! Start with a simple sample invoice, get that working, then iterate.

### 7. Documentation Is Important
Writing this report helped me understand the system better! Even if no one else reads it, documenting your work helps you solidify your knowledge.

---

## Conclusion

This project demonstrates core knowledge of enterprise Document AI systems:
- **Layout Understanding**: We extract and use spatial information (coordinates, blocks, lines, spans)
- **Modular Architecture**: Each stage is separate and maintainable
- **Structured Output**: We turn unstructured PDFs into machine-readable JSON
- **Rule-Based Systems**: We understand how to build and debug rule-based systems (a critical skill before adding ML complexity)

By building everything from scratch, we gained a deep understanding of how Document AI systems work under the hood—knowledge that will be invaluable when we add ML, embeddings, FAISS, and LLMs in future stages!
