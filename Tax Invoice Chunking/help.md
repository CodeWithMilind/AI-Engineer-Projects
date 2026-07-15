                Invoice PDF
                     │
                     ▼
         1. PDF Reader (PyMuPDF)
                     │
                     ▼
      2. OCR (Only if scanned PDF)
                     │
                     ▼
      3. Layout Understanding
         ├─ Blocks
         ├─ Lines
         ├─ Spans
         ├─ Coordinates
         └─ Fonts
                     │
                     ▼
      4. Section Detection
         ├─ Header
         ├─ Vendor
         ├─ Customer
         ├─ Items
         ├─ Totals
         └─ Bank Details
                     │
                     ▼
      5. Table Understanding
         ├─ Detect tables
         ├─ Detect rows
         ├─ Detect columns
         └─ Build structured JSON
                     │
                     ▼
      6. Semantic Chunking
         ├─ Header chunk
         ├─ Vendor chunk
         ├─ Customer chunk
         ├─ Items chunk
         ├─ Totals chunk
         └─ Bank chunk
                     │
                     ▼
      7. Embeddings
                     │
                     ▼
      8. Vector Database (FAISS)
                     │
                     ▼
      9. Retriever
                     │
                     ▼
      10. LLM (Ollama/OpenAI)
                     │
                     ▼
      11. Chat Interface