# PDF RAG Chatbot: Problems and Solutions

This file documents all the issues encountered during development and their corresponding fixes!

---

## 1. Chunk Size Too Small
### Problem
- Chunking used `chunk_size=100` characters and `chunk_overlap=20` characters by default!
- This created way too small chunks that weren't useful for semantic retrieval!
- Chunks were so tiny they didn't contain enough context for Ollama to answer questions!

### Root Cause
Default parameters in `chunk_pdf_pages()` function in [services/chunk_service.py](file:///c:/Users/codew/OneDrive/Desktop/AI%20Engineer%20Projects/PDF%20RAG%20Chatbot/services/chunk_service.py) were set too low!

### Solution
Modified [services/chunk_service.py](file:///c:/Users/codew/OneDrive/Desktop/AI%20Engineer%20Projects/PDF%20RAG%20Chatbot/services/chunk_service.py):
- Changed `chunk_size` default from `100` → `1000`
- Changed `chunk_overlap` default from `20` → `200`

---

## 2. Index/Chunks Overwritten with Demo Data
### Problem
- Whenever `services/faiss_pipeline.py`'s main() function was run, it would overwrite `vector_store/faiss.index` and `vector_store/chunks_map.json` with demo data!
- This caused queries to return "I couldn't find this information in the uploaded document" even after a valid PDF was uploaded!

### Root Cause
`faiss_pipeline.py` has a `main()` function that creates and saves demo index/chunks!

### Solution
Avoid running `python services/faiss_pipeline.py` unless you specifically want to reset the index to demo data!

---

## 3. UnboundLocalError in /query Endpoint
### Problem
- When uvicorn reloaded the server (after detecting file changes), it did NOT call the `@app.on_event("startup")` function again!
- This meant the global variables `ollama_client`, `embeddings`, `faiss_index`, `chunks` were not initialized in the new process!
- Resulted in: `UnboundLocalError: cannot access local variable 'faiss_index' where it is not associated with a value`!

### Root Cause
- Global variables were only initialized once on server startup!
- No safeguards to reinitialize them after server reloads!

### Solution
Modified [app.py](file:///c:/Users/codew/OneDrive/Desktop/AI%20Engineer%20Projects/PDF%20RAG%20Chatbot/app.py):
1. Added explicit initialization of global variables at module scope!
2. Created an `initialize_globals()` function to ensure all globals are always set!
3. Modified the `/query` endpoint to:
   - Declare all four globals at the start!
   - Reinitialize `ollama_client` if it's `None`!
   - Still reload `faiss_index`/`chunks` if they're missing!
