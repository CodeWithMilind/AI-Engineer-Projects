# PDF RAG Chatbot - Complete Technical Learning Report

## Table of Contents
1. [Project Overview](#project-overview)
2. [Complete File Breakdown](#complete-file-breakdown)
3. [Project Flow](#project-flow)
4. [Core Concepts Explained](#core-concepts-explained)
5. [Backend Architecture](#backend-architecture)
6. [Interview Preparation](#interview-preparation)
7. [Code Walkthrough](#code-walkthrough)
8. [System Design](#system-design)
9. [One-Day Revision Notes](#one-day-revision-notes)

---

## Project Overview
This project is a **Retrieval-Augmented Generation (RAG) chatbot** that lets users upload PDFs and ask questions about their content. It uses:
- **FastAPI** for backend API endpoints
- **PyMuPDF (fitz)** for PDF text extraction
- **langchain-text-splitters** for text chunking
- **sentence-transformers** (via HuggingFace) for generating embeddings
- **FAISS** for similarity search
- **Ollama** for local LLM inference
- **HTML/CSS/JS** for frontend UI

---

## Complete File Breakdown

### 1. app.py - FastAPI Backend Entrypoint
#### Why this file exists
This is the main backend server file that defines all API endpoints, initializes global state, and orchestrates the entire RAG pipeline.

#### Responsibilities
- Serve the root endpoint
- Accept PDF uploads and rebuild the knowledge base
- Answer user questions via /query endpoint
- Expose helper endpoints (/extract-pdf, /chunk-pdf) for testing
- Manage global variables (ollama_client, embeddings, faiss_index, chunks)
- Handle server startup and reloads

#### Files that call it
This file is the entrypoint; it is called by uvicorn to start the server.

#### Functions it calls
- `utils.file_utils`: `generate_unique_filename`, `save_upload_file`, `validate_pdf_file`
- `services.pdf_extractor`: `extract_pdf_text`
- `services.chunk_service`: `chunk_pdf_pages`
- `services.faiss_pipeline`: `load_embedding_model`, `embed_chunks`, `build_faiss_index`, `save_faiss`, `load_faiss`
- `services.ollama_service`: `OllamaClient`
- `services.rag_service`: `answer_question`

#### Data entering
- PDF files via /upload endpoint (multipart/form-data)
- User questions via /query endpoint (JSON)

#### Data leaving
- Upload metadata (filename, file size, pages, chunks) as JSON
- Query responses (answer + sources) as JSON

#### Important functions (line-by-line)
##### initialize_globals() (lines 58‑65)
```python
def initialize_globals():
    global ollama_client, embeddings, faiss_index, chunks
    ollama_client = None
    embeddings = None
    faiss_index = None
    chunks = None

initialize_globals()
```
- **Purpose**: Ensures all global state variables are initialized on file load, preventing "UnboundLocalError" when server reloads.
- Called once at module level.

##### _ensure_embeddings_loaded() (lines 74‑78)
```python
def _ensure_embeddings_loaded() -> None:
    global embeddings
    if embeddings is None:
        logger.info("Loading embedding model for indexing and retrieval.")
        embeddings = load_embedding_model(model_name=EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
```
- **Purpose**: Lazy-loads the HuggingFace embedding model only when needed (first upload or query).
- Uses global `embeddings` variable.

##### _build_index_from_pdf() (lines 81‑118)
```python
def _build_index_from_pdf(pdf_path: Path) -> dict:
    global embeddings, faiss_index, chunks

    logger.info("=== PDF Upload Received === path=%s", pdf_path)
    _ensure_embeddings_loaded()

    extraction_result = extract_pdf_text(pdf_path)
    pages = extraction_result.get("pages", [])
    full_text = "\n".join(page.get("text", "") for page in pages)
    logger.info("\n===== PDF TEXT =====\n%s", full_text[:2000])
    if pages:
        logger.info("\n===== FIRST PAGE =====\n%s", pages[0].get("text", "")[:1000])
    logger.info("=== PDF Extracted === total_pages=%d", extraction_result.get("total_pages", 0))

    chunk_list = chunk_pdf_pages(pages)
    logger.info("\n===== NUMBER OF CHUNKS ===== %d", len(chunk_list))
    if chunk_list:
        logger.info("\n===== FIRST CHUNK =====\n%s", chunk_list[0].get("chunk_text", "")[:500])

    chunk_texts = [chunk.get("chunk_text", "") for chunk in chunk_list]
    vectors = embed_chunks(embeddings, chunk_texts)
    logger.info("\n===== EMBEDDING DIMENSION ===== %d", vectors.shape[1])
    logger.info("=== Embedding Shape === %s", vectors.shape)

    index = build_faiss_index(vectors)
    logger.info("\n===== NUMBER OF VECTORS IN FAISS ===== %d", getattr(index, "ntotal", len(chunk_texts)))

    save_faiss(index, chunk_texts, VECTOR_INDEX_PATH, CHUNKS_MAPPING_PATH)
    logger.info("=== FAISS Saved === index_path=%s mapping_path=%s", VECTOR_INDEX_PATH, CHUNKS_MAPPING_PATH)

    faiss_index, chunks = load_faiss(VECTOR_INDEX_PATH, CHUNKS_MAPPING_PATH)
    logger.info("=== Reloaded FAISS === chunk_count=%d", len(chunks))

    return {
        "filename": pdf_path.name,
        "pages": extraction_result.get("total_pages", 0),
        "chunks": len(chunk_list),
    }
```
- **Purpose**: Orchestrates the entire upload-to-index pipeline.
- **Flow**:
  1. Load embedding model
  2. Extract text from PDF
  3. Chunk the text
  4. Generate embeddings
  5. Build FAISS index
  6. Save index and chunk map
  7. Reload index into global state
- **Returns**: Metadata about the uploaded PDF and processing.

##### @app.on_event("startup") startup_load_resources() (lines 121‑138)
```python
@app.on_event("startup")
def startup_load_resources() -> None:
    global ollama_client, embeddings, faiss_index, chunks

    logger.info("Loading RAG resources on startup.")
    ollama_client = OllamaClient(server_url=OLLAMA_SERVER_URL, model_name=OLLAMA_MODEL_NAME)
    try:
        embeddings = load_embedding_model(model_name=EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        embeddings = None

    try:
        faiss_index, chunks = load_faiss(VECTOR_INDEX_PATH, CHUNKS_MAPPING_PATH)
    except Exception as exc:
        logger.error("Failed to load FAISS index or chunk mapping: %s", exc)
        faiss_index = None
        chunks = None
```
- **Purpose**: Initializes resources when server first starts.
- **Does**:
  1. Creates OllamaClient
  2. Loads embedding model
  3. Loads existing FAISS index (if any)

##### @app.post("/upload") async upload_pdf() (lines 146‑172)
```python
@app.post("/upload")
async def upload_pdf(upload_file: UploadFile = File(...)):
    try:
        validate_pdf_file(upload_file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    unique_name = generate_unique_filename(upload_file.filename)
    saved_path = UPLOAD_FOLDER / unique_name

    try:
        save_upload_file(upload_file, saved_path)
        file_size = saved_path.stat().st_size
        build_result = _build_index_from_pdf(saved_path)
        return {
            "filename": unique_name,
            "file_path": str(saved_path),
            "file_size": file_size,
            "pages": build_result.get("pages", 0),
            "chunks": build_result.get("chunks", 0),
        }
    except PDFExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to rebuild the knowledge base from the uploaded PDF.")
        raise HTTPException(status_code=500, detail="Failed to process uploaded PDF.") from exc
```
- **Purpose**: Main upload endpoint.
- **Steps**:
  1. Validate file is a PDF
  2. Generate unique filename
  3. Save file to uploads directory
  4. Rebuild index via `_build_index_from_pdf()`
  5. Return metadata

##### @app.post("/query") query_document() (lines 225‑266)
```python
@app.post("/query")
def query_document(query: QueryRequest):
    global ollama_client, embeddings, faiss_index, chunks

    if ollama_client is None:
        logger.info("Reinitializing Ollama client.")
        ollama_client = OllamaClient(server_url=OLLAMA_SERVER_URL, model_name=OLLAMA_MODEL_NAME)

    if faiss_index is None or chunks is None or embeddings is None:
        try:
            _ensure_embeddings_loaded()
            if faiss_index is None or chunks is None:
                faiss_index, chunks = load_faiss(VECTOR_INDEX_PATH, CHUNKS_MAPPING_PATH)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="RAG resources are not loaded. Upload a PDF first and try again.",
            ) from exc

    if not query.question.strip():
        raise HTTPException(status_code=400, detail="A non-empty question is required.")

    try:
        result = answer_question(
            user_question=query.question,
            index=faiss_index,
            chunks=chunks,
            embeddings=embeddings,
            ollama_client=ollama_client,
            top_k=query.top_k or 3,
        )
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected error during query: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during query.") from exc
```
- **Purpose**: Main question-answering endpoint.
- **Steps**:
  1. Reinitialize ollama_client if None (fixes reload issues)
  2. Ensure all resources are loaded
  3. Validate question
  4. Call `answer_question()` from `rag_service`
  5. Return answer + sources

---

### 2. utils/file_utils.py
#### Why this file exists
Provides utility functions for handling file uploads.

#### Responsibilities
- Generate unique filenames
- Save uploaded files
- Validate PDFs

#### Files that call it
app.py calls all 3 functions.

#### Functions it calls
- No external functions beyond standard library (uuid, pathlib, fastapi.UploadFile)

#### Data entering
- FastAPI UploadFile
- Original filename string
- Destination Path object

#### Data leaving
- Unique filename string
- Saved Path object
- Raises ValueError on validation failure

#### Important functions (line-by-line)
##### generate_unique_filename(original_filename) (lines 6‑12)
```python
def generate_unique_filename(original_filename: str) -> str:
    extension = Path(original_filename).suffix.lower()
    if extension != ".pdf":
        extension = ".pdf"
    unique_id = uuid.uuid4().hex
    return f"{unique_id}{extension}"
```
- **Purpose**: Creates a unique filename to avoid collisions.
- Uses `uuid.uuid4().hex` for random ID.
- Forces .pdf extension regardless of original.

##### save_upload_file(upload_file, destination) (lines 15‑20)
```python
def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        buffer.write(upload_file.file.read())
    return destination
```
- **Purpose**: Saves uploaded file bytes to disk.
- Creates parent directory if it doesn't exist.
- Reads entire file into memory (simple but not ideal for huge PDFs).

##### validate_pdf_file(upload_file) (lines 23‑29)
```python
def validate_pdf_file(upload_file: UploadFile) -> None:
    if upload_file.content_type != "application/pdf":
        raise ValueError("Uploaded file must be a PDF.")

    if not upload_file.filename.lower().endswith(".pdf"):
        raise ValueError("Uploaded file must use a .pdf extension.")
```
- **Purpose**: Validates file is a PDF via both Content-Type header and filename extension.
- Raises ValueError if invalid.

---

### 3. services/pdf_extractor.py
#### Why this file exists
Handles extracting raw text from PDF files.

#### Responsibilities
- Open PDF files
- Extract text page by page
- Handle errors (corrupted files, empty PDFs, no extractable text)
- Return structured extraction result

#### Files that call it
app.py calls `extract_pdf_text()`.

#### Functions it calls
- `fitz.open()` from PyMuPDF

#### Data entering
- Path object of PDF file

#### Data leaving
- Dictionary with `total_pages` and `pages` (list of {page_number, text})
- Raises `PDFExtractionError` on failure

#### Important functions (line-by-line)
##### extract_pdf_text(pdf_path) (lines 11‑40)
```python
def extract_pdf_text(pdf_path: Path) -> dict:
    try:
        document = fitz.open(str(pdf_path))
    except Exception as exc:
        raise PDFExtractionError("Unable to open PDF file. The file may be corrupted.") from exc

    try:
        total_pages = document.page_count
        if total_pages == 0:
            raise PDFExtractionError("PDF is empty and contains no pages.")

        pages = []
        for page_index in range(total_pages):
            page = document.load_page(page_index)
            text = page.get_text().strip()
            pages.append({
                "page_number": page_index + 1,
                "text": text,
            })

        if all(not page_data["text"] for page_data in pages):
            raise PDFExtractionError("PDF contains no extractable text.")

        return {
            "total_pages": total_pages,
            "pages": pages,
        }
    finally:
        document.close()
```
- **Purpose**: Extracts text from each page of PDF.
- **Steps**:
  1. Open PDF
  2. Check if empty
  3. Iterate pages, extract text, store with page number
  4. Check if all pages are blank
  5. Return structured result
- Uses `finally` block to ensure document is closed.

---

### 4. services/chunk_service.py
#### Why this file exists
Converts extracted PDF text into smaller, semantically meaningful chunks that can be embedded.

#### Responsibilities
- Validate input pages
- Split text into chunks using RecursiveCharacterTextSplitter
- Attach metadata (chunk_id, page_number) to each chunk
- Return list of chunk dictionaries

#### Files that call it
app.py calls `chunk_pdf_pages()`.

#### Functions it calls
- `RecursiveCharacterTextSplitter` from langchain_text_splitters

#### Data entering
- List of page dictionaries (each with page_number and text)
- Optional chunk_size and chunk_overlap

#### Data leaving
- List of chunk dictionaries: {chunk_id, page_number, chunk_text}

#### Important functions (line-by-line)
##### chunk_pdf_pages(pages, chunk_size=1000, chunk_overlap=200) (lines 6‑70)
```python
def chunk_pdf_pages(
    pages: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
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
```
- **Purpose**: Chunks each page's text.
- **Splitter config**:
  - `chunk_size`: max characters per chunk (1000 default)
  - `chunk_overlap`: overlapping characters between consecutive chunks (200 default)
  - `separators`: tries splitting on paragraph breaks, then line breaks, then spaces, then anything
- Processes pages one at a time, assigns sequential chunk_id.

---

### 5. services/faiss_pipeline.py
#### Why this file exists
Handles all embedding, FAISS index building, saving/loading, and querying.

#### Responsibilities
- Load HuggingFace embedding model
- Embed list of chunks
- Build FAISS index (with cosine similarity)
- Save index and chunk map to disk
- Load index and chunk map from disk
- Query index with user question

#### Files that call it
- app.py calls all functions except main()
- services/rag_service.py calls `query_index()`

#### Functions it calls
- `HuggingFaceEmbeddings` from langchain_huggingface
- faiss library functions
- numpy
- json, os

#### Data entering
- Model name, device
- List of chunk texts
- User question string
- File paths

#### Data leaving
- Embedding model instance
- Numpy array of embeddings
- FAISS index object
- Tuple (loaded index, chunks list)
- List of (chunk_text, similarity_score) tuples

#### Important functions (line-by-line)
##### load_embedding_model() (lines 29‑47)
```python
def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> HuggingFaceEmbeddings:
    embed = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    return embed
```
- **Purpose**: Loads the sentence transformer embedding model via LangChain wrapper.
- Default: `all-MiniLM-L6-v2` (lightweight, 384-dimensional embeddings).

##### embed_chunks() (lines 50‑66)
```python
def embed_chunks(embeddings: HuggingFaceEmbeddings, chunks: List[str]) -> np.ndarray:
    vectors = embeddings.embed_documents(chunks)
    arr = np.array(vectors, dtype=np.float32)
    return arr
```
- **Purpose**: Convert list of text chunks into embedding matrix.
- `embed_documents()` returns list of lists; converts to float32 numpy array for FAISS.

##### build_faiss_index() (lines 69‑98)
```python
def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    n, dim = vectors.shape

    # Normalize vectors in-place to unit length (L2 norm = 1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    vectors_norm = vectors / norms

    # Create an IndexFlatIP (inner product) index. With normalized vectors, inner product = cosine similarity.
    index = faiss.IndexFlatIP(dim)

    index.add(vectors_norm)
    return index
```
- **Purpose**: Builds FAISS index for cosine similarity search.
- **Steps**:
  1. Normalize vectors to unit length
  2. Create IndexFlatIP (inner product) index
  3. Add normalized vectors
- Normalization + inner product = cosine similarity!

##### save_faiss() (lines 101‑119)
```python
def save_faiss(index: faiss.Index, chunks: List[str], index_path: str, mapping_path: str) -> None:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
```
- **Purpose**: Saves index and chunks to disk.
- Saves index as binary file, chunks as JSON array.

##### load_faiss() (lines 122‑140)
```python
def load_faiss(index_path: str, mapping_path: str) -> Tuple[faiss.Index, List[str]]:
    index = faiss.read_index(index_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks
```
- **Purpose**: Loads index and chunks from disk.
- Returns tuple (index, chunks list).

##### query_index() (lines 143‑175)
```python
def query_index(index: faiss.Index, chunks: List[str], embeddings: HuggingFaceEmbeddings, query: str, k: int = 3) -> List[Tuple[str, float]]:
    q_vec = embeddings.embed_query(query)
    q = np.array(q_vec, dtype=np.float32).reshape(1, -1)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-6)

    distances, indices = index.search(q_norm, k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append((chunks[idx], float(score)))
    return results
```
- **Purpose**: Query FAISS index for top‑k most similar chunks.
- **Steps**:
  1. Embed user query
  2. Normalize query vector
  3. Search index for top‑k matches
  4. Map indices back to chunk texts
  5. Return list of (chunk_text, score)

---

### 6. services/rag_service.py
#### Why this file exists
Orchestrates the question-answering part of RAG: retrieval → prompt building → LLM inference.

#### Responsibilities
- Retrieve top‑k chunks
- Format retrieved chunks
- Build RAG prompt
- Call Ollama client
- Format and return response

#### Files that call it
app.py calls `answer_question()`.

#### Functions it calls
- `services.faiss_pipeline.query_index()`
- `services.ollama_service.OllamaClient.generate()`

#### Data entering
- User question string
- FAISS index
- Chunks list
- Embedding model
- Ollama client
- top_k

#### Data leaving
- Dict {answer: str, sources: [{chunk, score}]}

#### Important functions (line-by-line)
##### retrieve_top_chunks() (lines 14‑56)
```python
def retrieve_top_chunks(
    index,
    chunks: List[str],
    embeddings: HuggingFaceEmbeddings,
    user_question: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    if not user_question or not user_question.strip():
        raise ValueError("User question must be a non-empty string.")
    if not chunks:
        raise ValueError("Chunk list must not be empty.")

    logger.info("Querying FAISS for top %d chunks.", top_k)
    start_time = time.perf_counter()
    results = query_index(index, chunks, embeddings, user_question, k=top_k)
    retrieval_time = time.perf_counter() - start_time
    logger.info("Retrieval completed in %.3f seconds; retrieved %d chunks.", retrieval_time, len(results))

    sources: List[Dict[str, Any]] = []
    for chunk_index, (chunk_text, score) in enumerate(results, start=1):
        logger.info("===== Retrieved Chunk %d =====", chunk_index)
        logger.info("%s", chunk_text)
        logger.info("Similarity Score: %.4f", float(score))
        sources.append(
            {
                "chunk": chunk_index,
                "score": round(float(score), 4),
                "text": chunk_text,
            }
        )
    return sources
```
- **Purpose**: Wraps `query_index()` for logging and formatting.
- Returns sources with chunk index, score, and text.

##### format_retrieved_chunks() (lines 59‑71)
```python
def format_retrieved_chunks(chunks: List[Dict[str, Any]]) -> str:
    formatted_chunks: List[str] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text")
        else:
            text = chunk

        if isinstance(text, str) and text.strip():
            formatted_chunks.append(text.strip())
    return "\n\n".join(formatted_chunks)
```
- **Purpose**: Joins retrieved chunks into a single string for prompt injection.
- Handles both dict and string chunk formats.

##### build_prompt_template() (lines 74‑93)
```python
def build_prompt_template(retrieved_chunks: List[Dict[str, Any]], user_question: str) -> str:
    if not user_question or not user_question.strip():
        raise ValueError("User question must be a non-empty string.")

    formatted_context = format_retrieved_chunks(retrieved_chunks)
    return (
        "You are a helpful AI assistant for a PDF knowledge base.\n\n"
        "Use ONLY the provided context to answer the question.\n"
        "Do not invent facts, do not speculate, and do not add information that is not present in the context.\n"
        "If the answer is not explicitly available in the context, respond exactly:\n"
        "\"I couldn't find this information in the uploaded document.\"\n\n"
        "Keep the answer concise, factual, and preserve technical terminology when relevant.\n"
        "If multiple relevant passages are provided, combine them naturally into a single answer.\n\n"
        "Context:\n"
        f"{formatted_context}\n\n"
        "Question:\n"
        f"{user_question}\n\n"
        "Answer:\n"
    )
```
- **Purpose**: Builds the RAG prompt with strict instructions to avoid hallucinations.
- Includes retrieved context, user question, and detailed rules for LLM.

##### answer_question() (lines 96‑151)
```python
def answer_question(
    user_question: str,
    index,
    chunks: List[str],
    embeddings: HuggingFaceEmbeddings,
    ollama_client: Optional[OllamaClient] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    if not user_question or not user_question.strip():
        raise ValueError("Question must not be empty.")
    if not chunks:
        raise ValueError("Chunk list must not be empty.")
    if embeddings is None:
        raise ValueError("Embeddings are not available.")

    if ollama_client is None:
        ollama_client = OllamaClient()

    request_start = time.perf_counter()
    retrieved_chunks = retrieve_top_chunks(index, chunks, embeddings, user_question, top_k=top_k)

    if not retrieved_chunks:
        logger.warning("No relevant chunks retrieved for the question.")
        return {
            "answer": "I couldn't find this information in the uploaded document.",
            "sources": [],
        }

    prompt = build_prompt_template(retrieved_chunks, user_question)
    logger.info("=== Prompt Sent To Ollama ===\n%s", prompt)

    try:
        logger.info("Sending the RAG prompt to Ollama.")
        ollama_start = time.perf_counter()
        answer = ollama_client.generate(prompt)
        ollama_time = time.perf_counter() - ollama_start
        logger.info("=== Ollama Response ===\n%s", answer)
        logger.info("Ollama responded in %.3f seconds.", ollama_time)
    except OllamaError as exc:
        logger.exception("Ollama failed while generating an answer.")
        raise RuntimeError(f"Ollama is unavailable: {exc}") from exc

    total_request_time = time.perf_counter() - request_start
    logger.info("Total request completed in %.3f seconds.", total_request_time)

    return {
        "answer": answer,
        "sources": [
            {
                "chunk": source.get("chunk", 0),
                "score": source.get("score", 0.0),
            }
            for source in retrieved_chunks
        ],
    }
```
- **Purpose**: Main RAG orchestration function.
- **Steps**:
  1. Validate inputs
  2. Retrieve chunks
  3. If no chunks, return fallback answer
  4. Build prompt
  5. Call Ollama
  6. Return answer and sources

---

### 7. services/ollama_service.py
#### Why this file exists
Provides a lightweight client for communicating with a local Ollama server.

#### Responsibilities
- Define custom `OllamaError` exception
- Define `OllamaClient` dataclass
- Send prompts to Ollama's `/v1/completions` endpoint
- Parse and validate responses

#### Files that call it
app.py, services/rag_service.py

#### Functions it calls
- Standard library (json, urllib)

#### Data entering
- Prompt string
- Optional model override

#### Data leaving
- Generated text string
- Raises OllamaError on failures

#### Important classes/functions (line-by-line)
##### @dataclass OllamaClient (lines 16‑88)
```python
@dataclass
class OllamaClient:
    server_url: str = "http://localhost:11434"
    model_name: str = "llama3.2:3b"
    timeout: int = 120
    max_tokens: int = 150
    temperature: float = 0.0

    def generate(self, prompt: str, model_name: Optional[str] = None) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        payload = {
            "model": model_name or self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")

        request = Request(
            url=f"{self.server_url.rstrip('/')}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                if response.status != 200:
                    raise OllamaError(
                        f"Ollama server returned HTTP {response.status}: {raw}"
                    )
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise OllamaError(
                f"Ollama server returned HTTP {exc.code}: {message}"
            ) from exc
        except URLError as exc:
            raise OllamaError(
                f"Unable to reach Ollama server at {self.server_url}: {exc.reason}"
            ) from exc
        except Exception as exc:
            raise OllamaError("Unexpected error while calling Ollama.") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON.") from exc

        return self._extract_text(decoded)
```
- **Purpose**: Sends requests to Ollama's compatible completions endpoint.
- Uses only standard library (no external HTTP dependencies).
- Handles timeouts, connection errors, invalid responses.

##### @staticmethod _extract_text() (lines 89‑107)
```python
@staticmethod
def _extract_text(response_data: dict) -> str:
    if not isinstance(response_data, dict):
        raise OllamaError("Ollama response payload is not a JSON object.")

    if "completion" in response_data and isinstance(response_data["completion"], str):
        return response_data["completion"].strip()

    if "output" in response_data and isinstance(response_data["output"], str):
        return response_data["output"].strip()

    choices = response_data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            text = first.get("text") or first.get("message", {}).get("content")
            if isinstance(text, str):
                return text.strip()

    raise OllamaError("Ollama response did not contain a text completion.")
```
- **Purpose**: Extracts generated text from Ollama's response.
- Supports multiple response formats for compatibility.

---

### 8. frontend/index.html, style.css, script.js
#### index.html
- **Purpose**: Defines the UI structure.
- Contains: Dropzone for upload, chat window, question composer.

#### style.css
- **Purpose**: Styles the frontend with modern dark theme.
- Uses CSS variables for colors, flexbox/grid for layout, animations for typing.

#### script.js
- **Purpose**: Handles all frontend interactivity and API calls.
- **Key functions**:
  - Drag/drop file selection
  - Upload file to /upload endpoint
  - Send question to /query endpoint
  - Render chat bubbles
  - Show loading state

---

### 9. requirements.txt
#### Why this file exists
Lists Python package dependencies.

#### Packages
1. `fastapi`: Web framework for APIs
2. `uvicorn[standard]`: ASGI server to run FastAPI
3. `python-dotenv`: Load environment variables from .env
4. `PyMuPDF==1.27.2.3`: PDF text extraction (import fitz)
5. `langchain==0.0.309`: LangChain (though we mostly use parts like langchain-huggingface, langchain-text-splitters)
6. Additional installed packages (not in requirements.txt but used):
   - langchain-huggingface
   - langchain-text-splitters
   - faiss-cpu
   - sentence-transformers
   - numpy

---

### 10. .env.example
#### Why this file exists
Template for environment variables.

#### Variables (from .env.example + app.py defaults)
- `HOST`: Server host (default 127.0.0.1)
- `PORT`: Server port (default 8000)
- `OLLAMA_SERVER_URL`: Ollama server URL (default http://localhost:11434)
- `OLLAMA_MODEL_NAME`: Model to use (default llama3.2:3b)
- `VECTOR_INDEX_PATH`: Path to faiss.index (default vector_store/faiss.index)
- `CHUNKS_MAPPING_PATH`: Path to chunks_map.json (default vector_store/chunks_map.json)
- `EMBEDDING_MODEL_NAME`: HuggingFace model (default sentence-transformers/all-MiniLM-L6-v2)
- `EMBEDDING_DEVICE`: cpu/cuda (default cpu)

---

## Project Flow

### Sequence Diagram
```
Browser
  ↓ (User selects/drops PDF)
Frontend (script.js)
  ↓ (POST /upload with FormData)
FastAPI (app.py)
  ↓ (validate, save file)
utils.file_utils
  ↓ (extract text)
services.pdf_extractor
  ↓ (chunk text)
services.chunk_service
  ↓ (embed chunks)
services.faiss_pipeline (HuggingFace)
  ↓ (build FAISS index, save to disk)
services.faiss_pipeline (FAISS)
  ↓ (reload index into memory)
app.py
  ↓ (return success JSON)
Frontend

---

Browser
  ↓ (User types question, clicks Ask)
Frontend (script.js)
  ↓ (POST /query with JSON)
FastAPI (app.py)
  ↓ (embed question)
services.faiss_pipeline (HuggingFace)
  ↓ (query FAISS index)
services.faiss_pipeline (FAISS)
  ↓ (retrieve top‑k chunks)
services.rag_service
  ↓ (build RAG prompt)
services.rag_service
  ↓ (send prompt to Ollama)
services.ollama_service
  ↓ (get LLM answer)
services.ollama_service
  ↓ (format answer + sources)
app.py
  ↓ (return JSON)
Frontend
  ↓ (render chat bubble)
Browser
```

### Detailed Flow - Upload PDF
1. **User interaction**: User drags/drops PDF or clicks to browse
2. **Frontend**: Handles file selection, validates type
3. **Frontend → Backend**: Sends POST request to `/upload` with FormData containing file
4. **Backend (app.py)**:
   - Validates file
   - Generates unique filename
   - Saves file to `uploads/`
5. **PDF Extraction**: `extract_pdf_text()` reads PDF, extracts text per page
6. **Text Chunking**: `chunk_pdf_pages()` splits text into chunks of ~1000 chars with 200‑char overlap
7. **Embedding**: `embed_chunks()` converts each chunk into 384‑dimensional vector
8. **Index Building**: `build_faiss_index()` creates FAISS index with normalized vectors
9. **Persistence**: `save_faiss()` writes index to `vector_store/faiss.index` and chunks to `vector_store/chunks_map.json`
10. **Reload**: `load_faiss()` reads index and chunks back into memory and stores in global variables
11. **Response**: Backend sends JSON response with upload metadata
12. **Frontend**: Shows success message and upload status

### Detailed Flow - Ask Question
1. **User interaction**: User types question, clicks "Ask"
2. **Frontend**:
   - Validates question and that a PDF was uploaded
   - Shows loading bubble
3. **Frontend → Backend**: Sends POST to `/query` with JSON {question: "...", top_k:3}
4. **Backend (app.py)**:
   - Checks/initializes global resources
   - Calls `answer_question()`
5. **Retrieval**:
   - `retrieve_top_chunks()` calls `query_index()`
   - Question is embedded and normalized
   - FAISS finds top‑k most similar chunks via cosine similarity
   - Returns list of (chunk_text, score)
6. **Prompt Building**: `build_prompt_template()` creates prompt with context + question + instructions
7. **LLM Inference**: `OllamaClient.generate()` sends prompt to Ollama's `/v1/completions`
8. **Response Handling**:
   - Ollama returns generated answer
   - `answer_question()` formats answer + sources
9. **Backend → Frontend**: Returns JSON {answer: "...", sources: [...]}
10. **Frontend**:
    - Removes loading bubble
    - Renders assistant answer + sources chat bubbles
    - Scrolls to bottom

---

## Core Concepts Explained

### What is RAG?
**Retrieval-Augmented Generation (RAG)** is an AI technique that combines:
1. **Retrieval**: Search a knowledge base (our PDF chunks) for relevant information
2. **Generation**: Use an LLM to generate an answer *grounded* in the retrieved information

Why RAG instead of just asking an LLM?
- LLMs have a cutoff date for training data
- LLMs don't know your private documents
- RAG reduces hallucinations by grounding answers in actual text
- RAG provides citations (sources) so you can verify answers

### Why use RAG instead of sending the whole PDF?
1. **Context window limits**: LLMs have maximum input size (e.g., 4k, 8k, 32k tokens). Long PDFs won't fit.
2. **Cost**: More tokens = higher cost (for paid APIs).
3. **Relevance**: The LLM might get distracted by irrelevant parts of a long PDF; RAG delivers only the most relevant chunks.
4. **Latency**: Processing huge inputs takes longer.

### What is chunking?
**Chunking** is splitting a long text into smaller, semantically coherent pieces.

Why chunk?
- Embedding models have maximum sequence lengths (e.g., all-MiniLM-L6-v2 handles ~256 tokens, ~200 words).
- Smaller chunks let you retrieve only the specific part of the PDF that answers the question, rather than the whole thing.

### Why chunk size matters?
- **Too small**: Chunks don't contain enough context to be meaningful; the LLM can't answer because it doesn't have the full picture.
- **Too large**: Chunks include irrelevant information; retrieval is less accurate; may exceed embedding model's max length.
- **Rule of thumb**: 500–1500 characters per chunk, with 10–20% overlap. Our project uses 1000 chars, 200 overlap.

### Why overlap exists?
Overlap helps prevent information from being lost at the chunk boundaries. If a key concept is split across two chunks, the overlap ensures both chunks contain part of it, so at least one is retrieved.

### What are embeddings?
**Embeddings** are numerical representations of text (high‑dimensional vectors) where similar texts have vectors that are close together (in terms of cosine similarity or L2 distance).

For example:
- "Cat sits on mat" and "Kitten rests on rug" → similar vectors
- "Cat sits on mat" and "Stock market rises" → dissimilar vectors

### How are embeddings generated?
Using pre-trained **sentence transformer** models from HuggingFace! These models are trained on millions of text pairs to learn to map sentences to meaningful vectors.

Our project uses `sentence-transformers/all-MiniLM-L6-v2`, which is small, fast, and produces 384‑dimensional vectors.

### Why similar sentences have similar vectors?
Because during training, the model is taught to minimize the distance between embeddings of semantically similar texts and maximize the distance between dissimilar ones.

### Why use FAISS?
**FAISS (Facebook AI Similarity Search)** is a library for efficient similarity search and clustering of dense vectors.

Why not just calculate similarity to every chunk every time?
- For small datasets, that's fine (brute‑force).
- For large datasets (100k+ chunks), brute‑force is too slow! FAISS uses data structures and algorithms (like inverted indexes, HNSW graphs, etc.) to make search blazing fast.
- Our project uses `IndexFlatIP` (exact search, good for small‑medium datasets).

### How similarity search works?
1. Embed the user query.
2. Normalize the query vector (to use cosine similarity).
3. Use FAISS to find vectors in the index with the highest inner product (cosine similarity).
4. Map the indices of those vectors back to the original text chunks.

### What is cosine similarity?
Cosine similarity measures the cosine of the angle between two vectors.
- Range: -1 to 1
- 1: vectors are identical
- 0: vectors are perpendicular (no similarity)
- -1: vectors are opposite

Formula:
`cos(θ) = (A • B) / (|A| * |B|)`

If vectors are normalized (|A|=|B|=1), this simplifies to the dot product (which is what we do!).

### What is L2 distance?
L2 distance (Euclidean distance) is the straight‑line distance between two vectors in space.
- Range: 0 to infinity
- 0: vectors are identical
- Larger values = more dissimilar

For normalized vectors, cosine similarity and L2 distance are related (lower L2 = higher cosine similarity), but cosine is more common for text embeddings because it's invariant to vector magnitude (we care about direction, not length).

### Why Top‑K retrieval?
We retrieve the top k most similar chunks because:
- The single most similar chunk might be correct, but maybe a combination of a few chunks gives a better answer.
- K is a hyperparameter (we use 3 by default). Too small: miss context; too large: include irrelevant info and exceed context window.

### What prompt is sent to Ollama?
The prompt includes:
1. Instructions: "You are a helpful AI assistant... Use ONLY the provided context... Do not invent facts... If answer not available, say exactly: ..."
2. Context: The top k retrieved chunks, joined by newlines
3. User question
4. "Answer:" prefix to prompt generation

Example prompt:
```
You are a helpful AI assistant for a PDF knowledge base.

Use ONLY the provided context to answer the question.
Do not invent facts, do not speculate, and do not add information that is not present in the context.
If the answer is not explicitly available in the context, respond exactly:
"I couldn't find this information in the uploaded document."

Keep the answer concise, factual, and preserve technical terminology when relevant.
If multiple relevant passages are provided, combine them naturally into a single answer.

Context:
Milind Chaudhari
+91 8530871762 | Pune | codewithmilind@gmail.com | LinkedIn | GitHub | Portfolio
PROFESSIONAL SUMMARY
AI & Machine Learning student passionate about building production-ready AI applications using Python...

Question:
What is the candidate's full name?

Answer:
```

### Why is prompt engineering required?
LLMs are powerful but need clear instructions! Good prompts:
- Tell the LLM exactly what to do (and what NOT to do)
- Give constraints (only use provided context)
- Provide an exact fallback answer for when info is missing
- This drastically reduces hallucinations!

### Why hallucinations happen?
Hallucinations (when the LLM makes up facts) happen because:
- The LLM is a statistical model predicting next tokens; it doesn't "know" what's true.
- If it doesn't have the information, it tries to fill in the gaps with something plausible.
- RAG helps by giving it explicit context to reference!

### How RAG reduces hallucinations?
- By grounding the answer in explicit retrieved text from your document.
- By including strong prompt instructions: "Use ONLY the provided context... Do not invent facts..."
- By providing sources so you can verify the answer.

### What are limitations of this project?
1. **Single PDF only**: Uploading a new PDF replaces the knowledge base entirely.
2. **No persistent user sessions**: All state is in global variables; if server restarts, you have to re‑upload.
3. **Basic chunking**: Uses simple RecursiveCharacterTextSplitter; doesn't consider document structure (headings, sections, tables, images).
4. **FAISS IndexFlatIP only**: Exact search; not optimized for huge datasets (100k+ chunks).
5. **No authentication/authorization**: Anyone can upload/query.
6. **Local Ollama only**: No support for OpenAI, Claude, etc., without modification.
7. **No memory**: Chatbot doesn't remember previous questions/answers; every query is independent.

---

## Backend Architecture

### Folder Structure
```
PDF RAG Chatbot/
├── app.py                          # Main FastAPI app
├── requirements.txt                # Dependencies
├── .env.example                    # Env vars template
├── PROBLEMS_AND_SOLUTIONS.md       # Issues we fixed
├── frontend/                       # Frontend files
│   ├── index.html
│   ├── script.js
│   └── style.css
├── uploads/                        # Stored PDFs
├── vector_store/                   # FAISS index + chunks map
│   ├── faiss.index
│   └── chunks_map.json
├── utils/                          # Utility functions
│   └── file_utils.py
├── services/                       # Core business logic
│   ├── pdf_extractor.py
│   ├── chunk_service.py
│   ├── faiss_pipeline.py
│   ├── ollama_service.py
│   └── rag_service.py
├── tests/                          # (Minimal tests)
├── Code Practice/                  # (Scratch files)
└── venv/                           # Virtual environment
```

### Purpose of every folder
- **frontend/**: UI code (static HTML/CSS/JS)
- **uploads/**: Directory where uploaded PDFs are saved
- **vector_store/**: Directory where FAISS index and chunks JSON are saved
- **utils/**: Utility functions (file handling)
- **services/**: All core RAG logic
- **tests/**: Test files
- **Code Practice/**: Scratch/experiment files
- **venv/**: Virtual environment dependencies

### Dependency Flow
```
frontend (JS)
  ↓ (HTTP calls)
app.py
  ↓
  ├→ utils/file_utils.py
  ├→ services/pdf_extractor.py
  ├→ services/chunk_service.py
  ├→ services/faiss_pipeline.py
  │   ├→ HuggingFaceEmbeddings
  │   └→ FAISS
  ├→ services/ollama_service.py
  │   └→ Ollama (external process)
  └→ services/rag_service.py
      ├→ services/faiss_pipeline.query_index()
      └→ services/ollama_service.OllamaClient.generate()
```

### Call Hierarchy
```
app.py startup_load_resources()
  ├→ OllamaClient()
  ├→ load_embedding_model()
  └→ load_faiss()

app.py /upload endpoint
  ├→ validate_pdf_file()
  ├→ generate_unique_filename()
  ├→ save_upload_file()
  ├→ _build_index_from_pdf()
      ├→ _ensure_embeddings_loaded()
      ├→ extract_pdf_text()
      ├→ chunk_pdf_pages()
      ├→ embed_chunks()
      ├→ build_faiss_index()
      ├→ save_faiss()
      └→ load_faiss()

app.py /query endpoint
  └→ answer_question()
      ├→ retrieve_top_chunks()
      │   └→ query_index()
      ├→ build_prompt_template()
      └→ OllamaClient.generate()
```

---

## Interview Preparation

### 50 Interview Questions + Ideal Answers + Common Mistakes

---

#### 1. Explain your project.
**Ideal Answer**:
This is a PDF RAG chatbot that lets users upload PDFs and ask questions about their content. The tech stack is FastAPI for the backend, PyMuPDF for PDF extraction, langchain-text-splitters for chunking, HuggingFace sentence-transformers for embeddings, FAISS for similarity search, Ollama for local LLM inference, and vanilla HTML/CSS/JS for the frontend. When a user uploads a PDF, we extract text, chunk it, embed each chunk, build a FAISS index, and save it to disk. When they ask a question, we embed the query, retrieve the top‑k most similar chunks, build a prompt with context + question, send it to Ollama, and return the answer with sources.

**Common Mistakes**:
- Forgetting to mention key components (FAISS, embeddings, chunking, RAG concept)
- Being too vague ("it's a chatbot that uses AI")
- Not explaining the flow end-to-end

---

#### 2. Why did you choose FastAPI?
**Ideal Answer**:
FastAPI is modern, fast (high performance, based on Starlette and Pydantic), has automatic interactive docs (Swagger UI/Redoc), async support, strong type hints, automatic request validation with Pydantic models, and is easy to set up for REST APIs—perfect for this project!

**Common Mistakes**:
- "Because it's popular"
- Comparing to Flask/Django without specific reasons relevant to this project

---

#### 3. Why did you choose Ollama?
**Ideal Answer**:
Ollama lets you run LLMs locally (no API keys needed, no data sent to third parties, free), has a simple CLI and compatible API, supports tons of models (like Llama 3, Mistral, Gemma), and is easy to integrate with—great for development and privacy-focused use cases!

**Common Mistakes**:
- Not mentioning benefits of local inference (privacy, cost, no rate limits)
- Confusing Ollama with the models it runs

---

#### 4. Why did you choose FAISS?
**Ideal Answer**:
FAISS is optimized for fast similarity search on dense vectors, is easy to use, supports multiple index types (exact/approximate), is fast enough for our use case (IndexFlatIP for exact search on small‑medium datasets), and integrates well with numpy!

**Common Mistakes**:
- Not explaining what FAISS actually does
- Not mentioning cosine similarity or index types

---

#### 5. Why did you choose HuggingFace sentence-transformers?
**Ideal Answer**:
Sentence-transformers are specifically trained to produce good embeddings for semantic similarity tasks; all-MiniLM-L6-v2 is lightweight (fast inference, small file size), works well for retrieval, is easy to use via LangChain's HuggingFaceEmbeddings wrapper, and doesn't require GPU (runs on CPU just fine)!

**Common Mistakes**:
- Not naming the specific model
- Not explaining why sentence transformers vs regular transformers

---

#### 6. Why not use OpenAI (GPT) instead of Ollama?
**Ideal Answer**:
OpenAI is great, but Ollama lets us run locally: no API costs, no data leaving our machine (better privacy), no rate limits, perfect for development and this project's scope. If we wanted to switch to OpenAI, we'd just need to modify ollama_service.py to call OpenAI's API instead!

**Common Mistakes**:
- Badmouthing OpenAI; focus on tradeoffs
- Not mentioning that the project could be adapted

---

#### 7. How are embeddings stored?
**Ideal Answer**:
The embedding vectors are stored in a FAISS index file (vector_store/faiss.index), which is a binary file optimized for fast similarity search. We also store the original chunk texts in vector_store/chunks_map.json, so we can map FAISS's integer indices back to human-readable text!

**Common Mistakes**:
- Confusing vector storage with chunk storage
- Not mentioning both files

---

#### 8. How is similarity calculated?
**Ideal Answer**:
We use cosine similarity! First, we normalize all vectors (chunk embeddings and query embedding) to unit length. Then, we use FAISS's IndexFlatIP (inner product), which for normalized vectors equals cosine similarity. Higher inner product = higher cosine similarity = more similar!

**Common Mistakes**:
- Forgetting about normalization
- Confusing L2 distance with cosine similarity
- Not mentioning FAISS's role

---

#### 9. How is the prompt built?
**Ideal Answer**:
The prompt has four parts:
1. A clear system prompt that tells the LLM to only use provided context, not invent facts, and gives an exact fallback answer if information isn't found.
2. The retrieved context (top‑k chunks joined by newlines).
3. The user's question.
4. An "Answer:" prefix to signal where the LLM should start generating.

**Common Mistakes**:
- Not mentioning the system prompt's role in reducing hallucinations
- Being vague about what's in the prompt

---

#### 10. How is the PDF processed?
**Ideal Answer**:
When a PDF is uploaded:
1. We validate it's actually a PDF (Content-Type and .pdf extension).
2. Save it with a unique filename to uploads/.
3. Use PyMuPDF (fitz) to extract text page by page.
4. Split each page's text into 1000‑character chunks with 200‑character overlap using RecursiveCharacterTextSplitter.
5. Embed each chunk with sentence-transformers/all-MiniLM-L6-v2 to get 384‑dimensional vectors.
6. Normalize vectors, build FAISS IndexFlatIP, add vectors to index.
7. Save index and chunks to disk, reload into global state.

**Common Mistakes**:
- Missing steps (e.g., validation, saving, reloading)
- Not mentioning chunk size/overlap

---

#### 11. How would you scale this project to handle many users and many PDFs?
**Ideal Answer**:
- **Multi-PDF support**: Store PDFs and chunks in a database (PostgreSQL with pgvector for vector search, or use a dedicated vector DB like Pinecone/Weaviate/Qdrant). Add user accounts (authentication/authorization) so each user has their own documents.
- **Remove global state**: Don't store FAISS index in global variables; load per user or use a vector DB service.
- **Async processing**: Use background tasks (FastAPI BackgroundTasks or Celery + Redis) for PDF processing, so upload endpoints return quickly.
- **Caching**: Cache frequent queries (Redis).
- **Deployment**: Use Docker to containerize, deploy to cloud (AWS/GCP/Azure), use a managed vector DB, put API behind a load balancer, use CDN for frontend static files.
- **Scaling Ollama**: If using local Ollama, run multiple instances behind a load balancer or switch to a hosted LLM API.

**Common Mistakes**:
- Being too vague
- Forgetting about multi-user/multi-PDF aspects

---

#### 12. How would you support multiple PDFs?
**Ideal Answer**:
- Assign each PDF an ID.
- Store embeddings with metadata (pdf_id, user_id, chunk_id, page_number, chunk_text) in a vector DB (pgvector, Pinecone, Weaviate, Qdrant) instead of a single FAISS index file.
- When querying, filter by user_id and/or specific pdf_id to only search relevant PDFs.

**Common Mistakes**:
- Not mentioning vector DBs with filtering/metadata support

---

#### 13. How would you deploy this project?
**Ideal Answer**:
1. Containerize the backend with Docker (Dockerfile that installs dependencies, runs uvicorn).
2. Frontend: Serve static files via Nginx or from FastAPI's StaticFiles.
3. Use a .env file for config (don't commit secrets!).
4. Deploy to cloud service (AWS EC2/ECS, GCP Cloud Run, Azure App Service, or render.com, fly.io for simpler PaaS).
5. For Ollama: Either run it on the same cloud VM or use a hosted LLM API.
6. For vector storage: Use a managed vector DB (e.g., Pinecone, Weaviate Cloud) instead of local FAISS index files for persistence and scaling.

**Common Mistakes**:
- Forgetting about Ollama in deployment
- Not mentioning Docker/containers

---

#### 14. How would you improve retrieval quality?
**Ideal Answer**:
- **Better chunking**: Use semantic chunking instead of just character-based; split at logical sections/headings; use larger chunk size with smaller overlap or vice versa based on testing; use LangChain's Markdown/HTML splitters if PDF has structure.
- **Better embeddings**: Use a more powerful model (e.g., all-mpnet-base-v2 instead of all-MiniLM-L6-v2).
- **Re-ranking**: After retrieving top-k chunks with FAISS, use a cross-encoder to re-rank them and pick the top n (improves relevance).
- **Hybrid search**: Combine semantic search with keyword search (BM25).
- **Metadata filtering**: If storing metadata, filter by date, source, etc., if applicable.
- **Hypothetical Document Embeddings (HyDE)**: Ask LLM to generate a hypothetical answer, embed that, retrieve chunks similar to the hypothetical answer (for better recall).

**Common Mistakes**:
- Only suggesting "use a better model" without other practical changes

---

#### 15. What are production limitations of your current implementation?
**Ideal Answer**:
- **Global state**: FAISS index and chunks are stored in global variables; if server restarts, you lose state unless you reload from disk; doesn't work with multiple workers/processes (each worker would have its own global state).
- **No persistent session/user data**: No database, no auth, no way to remember previous chats.
- **Single PDF only**: Uploading a new PDF overwrites index.
- **No error resilience**: If Ollama is down, query fails; no retries.
- **No monitoring/observability**: No structured logging, no metrics (e.g., Prometheus), no tracing.
- **No testing**: Minimal tests; no unit tests for services, no integration tests for endpoints.
- **Security issues**: No CORS restrictions (allow_origins=["*"] is bad for production), no auth, no rate limiting, uploaded PDFs are saved without virus scanning.

**Common Mistakes**:
- Only listing one or two limitations; not being thorough

---

#### 16. Explain chunking and why we need it.
**Ideal Answer**:
(See "Core Concepts" above)

**Common Mistakes**:
- Not mentioning embedding model max sequence length
- Not explaining overlap

---

#### 17. What's the difference between IndexFlatIP and IndexFlatL2 in FAISS?
**Ideal Answer**:
- **IndexFlatIP**: Uses inner product (dot product); if vectors are normalized, this equals cosine similarity (good for text embeddings).
- **IndexFlatL2**: Uses Euclidean (L2) distance; good for when you care about magnitude of vectors (less common for text).

**Common Mistakes**:
- Mixing up which one is for cosine
- Not mentioning normalization for IndexFlatIP

---

#### 18. How would you add chat history/memory?
**Ideal Answer**:
- Store chat history in a database (e.g., PostgreSQL) with user_id, session_id, role (user/assistant), content, timestamp.
- When sending a new query, include the last N messages in the prompt so the LLM has context.
- Optionally: Use LangChain's ConversationSummaryMemory or similar to condense long chat history so it doesn't take too much context space.

**Common Mistakes**:
- Not mentioning a database for storage
- Forgetting about context window limits

---

#### 19. What is hallucination, and how does RAG reduce it?
**Ideal Answer**:
(See "Core Concepts" above)

**Common Mistakes**:
- Not defining hallucination clearly
- Not explaining RAG's role clearly

---

#### 20. What are embeddings?
**Ideal Answer**:
(See "Core Concepts" above)

**Common Mistakes**:
- Not mentioning they're numerical vectors
- Not mentioning similarity

---

#### 21. Why did you use cosine similarity over L2 distance?
**Ideal Answer**:
Cosine similarity measures the angle between vectors, so it's invariant to vector magnitude (we only care about the direction/meaning of the text, not how "long" the vector is). For text embeddings, cosine similarity is more intuitive and performs better for semantic similarity tasks!

**Common Mistakes**:
- Not explaining the difference clearly

---

#### 22. What are the dimensions of your embeddings?
**Ideal Answer**:
384 dimensions! Because we use sentence-transformers/all-MiniLM-L6-v2.

**Common Mistakes**:
- Guessing or not knowing the model's output dimension

---

#### 23. Walk through what happens when a user uploads a PDF.
**Ideal Answer**:
(See "Detailed Flow - Upload PDF" above)

**Common Mistakes**:
- Skipping steps or being too vague

---

#### 24. Walk through what happens when a user asks a question.
**Ideal Answer**:
(See "Detailed Flow - Ask Question" above)

**Common Mistakes**:
- Skipping retrieval or prompt building steps

---

#### 25. Why do you normalize vectors for cosine similarity?
**Ideal Answer**:
Normalizing vectors sets their L2 norm (length) to 1, so the dot product (inner product) equals cosine similarity! If we didn't normalize, longer vectors would have larger dot products even if they're less similar semantically.

**Common Mistakes**:
- Forgetting the math: cos(θ) = (A•B)/(|A||B|); if |A|=|B|=1, cos(θ)=A•B

---

#### 26. What is RecursiveCharacterTextSplitter?
**Ideal Answer**:
It's a LangChain text splitter that tries splitting text in order of a list of separators (for us, ["\n\n", "\n", " ", ""]). It first tries splitting on paragraph breaks; if a chunk is still too big, it tries line breaks; then spaces; then individual characters. This helps keep semantically related text together!

**Common Mistakes**:
- Not explaining the recursive part (trying separators in order)

---

#### 27. How would you test this project?
**Ideal Answer**:
1. **Unit tests**: For each service function (e.g., test_extract_pdf_text(), test_chunk_pdf_pages(), test_query_index(), test_build_prompt_template()).
2. **Integration tests**: Test endpoints (upload a test PDF, ask a known question, verify answer is correct).
3. **End-to-end tests**: Use Cypress/Playwright to simulate user interactions (upload PDF, ask question, check UI).
4. **Manual testing**: Try a variety of PDFs (text-heavy, scanned, images only—though scanned PDFs won't work with current extractor; note that limitation!).

**Common Mistakes**:
- Not mentioning different test levels (unit/integration/e2e)

---

#### 28. What would you do if the uploaded PDF is an image/scanned document (no extractable text)?
**Ideal Answer**:
Add OCR (Optical Character Recognition) using a library like Tesseract (via pytesseract or easyocr)! So if extract_pdf_text() finds no text, run OCR on each page's image to extract text!

**Common Mistakes**:
- Forgetting OCR is needed for scanned PDFs

---

#### 29. What's the purpose of the chunks_map.json file?
**Ideal Answer**:
FAISS stores vectors as integer indices (0,1,2,...), not text! chunks_map.json maps those indices back to the original chunk texts, so when we retrieve index 5, we can look up chunks[5] and get the actual text!

**Common Mistakes**:
- Not connecting FAISS indices to chunk texts

---

#### 30. Explain the FAISS index saving and loading flow.
**Ideal Answer**:
When saving:
1. Use `faiss.write_index(index, index_path)` to save the index as binary.
2. Use json.dump() to save chunks list to mapping_path.
When loading:
1. Use `faiss.read_index(index_path)` to load index.
2. Use json.load() to load chunks list.

**Common Mistakes**:
- Forgetting that both index and chunks need to be saved/loaded

---

#### 31. What is the role of global variables in app.py, and what are the issues with them?
**Ideal Answer**:
Global variables (ollama_client, embeddings, faiss_index, chunks) store state in memory for quick access, so we don't have to reload the model/index every query.

Issues:
- Not persistent across server restarts.
- Don't work with multiple workers (each worker has its own copy).
- Caused an UnboundLocalError when server reloaded because globals weren't initialized (we fixed that with initialize_globals()).
- Thread-safety (though FastAPI runs endpoints in separate threads by default; in our case, reads are okay, writes only on upload, which is infrequent).

**Common Mistakes**:
- Not mentioning downsides of global variables

---

#### 32. How would you handle very large PDFs?
**Ideal Answer**:
- **Stream processing**: Don't read entire PDF into memory at once (though PyMuPDF already loads pages incrementally).
- **Background processing**: Use Celery + Redis or FastAPI BackgroundTasks to process large PDFs in the background so the upload endpoint doesn't block; return a task ID, let frontend poll for completion.
- **Larger chunk size + smaller overlap?** Or split into multiple parts, but keep retrieval per part.
- **Use a vector DB**: Don't keep entire index in memory; use a vector DB that can handle large datasets on disk.

**Common Mistakes**:
- Not mentioning background tasks or async processing

---

#### 33. What is Pydantic and how is it used in this project?
**Ideal Answer**:
Pydantic is a library for data validation using Python type hints! In this project, we use Pydantic's BaseModel to define QueryRequest:
```python
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
```
FastAPI automatically validates that incoming POST data to /query matches this model; if not, returns 400 error automatically!

**Common Mistakes**:
- Not mentioning validation or BaseModel

---

#### 34. What are CORS and why do we have CORSMiddleware in app.py?
**Ideal Answer**:
CORS (Cross-Origin Resource Sharing) is a browser security mechanism that prevents web pages from making requests to domains different from their own. Since our frontend runs on, say, http://localhost:3000 and backend on http://localhost:8000, we need CORSMiddleware to allow the frontend to talk to the backend!

**Common Mistakes**:
- Forgetting that allow_origins=["*"] is not secure for production; explain that in production you should set it to your frontend's actual domain!

---

#### 35. How would you add authentication/authorization?
**Ideal Answer**:
- Use OAuth2 with JWT (JSON Web Tokens): Let users sign up/login, get access token, send token in Authorization header with every request.
- Use FastAPI's OAuth2PasswordBearer and PyJWT library.
- Store users in PostgreSQL (hashed passwords only!).
- Add dependencies to endpoints to check for valid tokens.

**Common Mistakes**:
- Not mentioning JWT or OAuth2
- Forgetting to hash passwords

---

#### 36. What libraries would you use for vector databases if not FAISS?
**Ideal Answer**:
- **Pinecone**: Managed cloud vector DB (easy to use, scalable).
- **Weaviate**: Open-source, can be self-hosted or managed.
- **Qdrant**: Open-source, fast, easy API.
- **pgvector**: Extension for PostgreSQL (great if you already use PostgreSQL, lets you store vectors and relational data in same DB).
- **ChromaDB**: Open-source, simple, good for development.

**Common Mistakes**:
- Only listing one or two options

---

#### 37. Explain top_k and how you chose its value.
**Ideal Answer**:
top_k is the number of most similar chunks we retrieve! We chose 3 as a default—it's a balance: enough context to answer most questions, but not so many that we exceed the LLM's context window or include irrelevant information. We could tune it based on testing with our specific use case!

**Common Mistakes**:
- Not explaining why 3 is a reasonable default

---

#### 38. How would you handle images/tables in PDFs?
**Ideal Answer**:
- **Tables**: Use a library like Camelot or Tabula to extract tables as structured text/CSV/Markdown, then include that in chunks.
- **Images**: Use a vision-language model (like GPT-4V, Claude 3, or local Llama 3.2 Vision via Ollama) to extract descriptions of images, then add those descriptions to chunks!

**Common Mistakes**:
- Forgetting that current code only extracts raw text, not tables/images

---

#### 39. What is the difference between embed_documents and embed_query in LangChain's HuggingFaceEmbeddings?
**Ideal Answer**:
- embed_documents(): Takes a list of texts, returns list of embeddings (used for chunks).
- embed_query(): Takes one text, returns one embedding (used for user question).
- In many sentence-transformer models, they do exactly the same thing, but the separation is there for models that treat queries and documents differently!

**Common Mistakes**:
- Not knowing the difference (or thinking they're totally different)

---

#### 40. How does RecursiveCharacterTextSplitter work with the separators list?
**Ideal Answer**:
It tries splitting the text using the first separator in the list. For each resulting split, if it's longer than chunk_size, it splits that part using the next separator in the list, and so on recursively until all chunks are <= chunk_size!

**Common Mistakes**:
- Not explaining the order of separators

---

#### 41. What is the maximum sequence length for the embedding model you used?
**Ideal Answer**:
all-MiniLM-L6-v2 has a max sequence length of 256 tokens (roughly 150–200 words), so our 1000-character chunks are roughly at the limit (depends on text, but we stay under)!

**Common Mistakes**:
- Not knowing the model's limits

---

#### 42. How would you implement re-ranking?
**Ideal Answer**:
After retrieving top‑k chunks (say top 20) with FAISS, use a cross-encoder model (like cross-encoder/ms-marco-MiniLM-L-6-v2 from HuggingFace) to score each (query, chunk) pair directly, then take the top n (say top 3) of those! Cross-encoders are better at fine-grained relevance but slower than bi-encoders (our sentence-transformers), so we use bi-encoder for retrieval first, cross-encoder for re-ranking!

**Common Mistakes**:
- Confusing bi-encoders and cross-encoders

---

#### 43. What are bi-encoders and cross-encoders?
**Ideal Answer**:
- **Bi-encoder**: Encodes texts (query or chunk) independently into embeddings; then we compute similarity between embeddings. Fast, good for retrieval (our sentence-transformers).
- **Cross-encoder**: Takes both query and chunk as input at same time, outputs a similarity score directly. Better quality, slower, good for re-ranking.

**Common Mistakes**:
- Mixing up which is which

---

#### 44. What is temperature in LLM generation and why did you set it to 0.0?
**Ideal Answer**:
Temperature controls randomness! Higher temperature (0.7–1.0) = more creative/random answers; lower temperature (0.0–0.3) = more deterministic/factual answers. We set temperature=0.0 because we want the most factual answer grounded in the context—no creativity needed!

**Common Mistakes**:
- Not explaining the scale (0–1 or sometimes 0–2 depending on API)
- Not justifying 0.0 for this use case

---

#### 45. What is max_tokens and why did you set it to 150?
**Ideal Answer**:
max_tokens is the maximum number of tokens the LLM will generate! We set it to 150 because we want concise answers—no long tangents! For longer answers, we could increase it.

**Common Mistakes**:
- Confusing max_tokens with context window size

---

#### 46. Walk through the OllamaClient.generate() function.
**Ideal Answer**:
(See ollama_service.py line-by-line explanation above)

**Common Mistakes**:
- Forgetting error handling parts

---

#### 47. Why did you use PyMuPDF (fitz) instead of other PDF libraries like PyPDF2?
**Ideal Answer**:
PyMuPDF is fast, extracts text accurately, preserves layout better than many other libraries, is easy to use, and is actively maintained!

**Common Mistakes**:
- Badmouthing other libraries; focus on why fitz fits this project

---

#### 48. What is the purpose of the finally block in extract_pdf_text()?
**Ideal Answer**:
The finally block ensures that `document.close()` is called *no matter what* (even if an exception is raised), so we don't leave the PDF file open, which could cause resource leaks!

**Common Mistakes**:
- Not mentioning resource leaks

---

#### 49. What's the difference between async and sync endpoints in FastAPI and why did we use async for upload?
**Ideal Answer**:
In FastAPI, async def endpoints run in the event loop; synchronous def endpoints run in a thread pool. We used async for upload_pdf() because saving a file and processing a PDF can involve I/O, and async lets the server handle other requests while waiting! However, in our case, the actual processing (extract_pdf_text(), etc.) is synchronous, so to be fully async, we could run those in a thread pool with asyncio.to_thread()!

**Common Mistakes**:
- Not explaining the difference between async/sync endpoints in FastAPI

---

#### 50. What was the hardest bug you faced in this project, and how did you fix it?
**Ideal Answer**:
The UnboundLocalError when the server reloaded! Because uvicorn would reload the app on code changes, the global variables (faiss_index, etc.) weren't initialized in the new process, so when we tried to access them in /query, we got "cannot access local variable 'faiss_index' where it is not associated with a value". We fixed it by:
1. Adding an initialize_globals() function that sets all globals to None.
2. Calling initialize_globals() at module level, so it runs on every load.
3. Adding code in /query endpoint to reinitialize ollama_client if it's None.

**Common Mistakes**:
- Not having a specific bug story (we have one here!)

---

## Code Walkthrough
Imagine you're sharing your screen in an interview. Here's exactly what to do and say!

### Step 1: Start with project structure
Open your project folder in your file explorer/IDE and say:
"Okay, let me walk you through the project structure first! We have:
- `app.py`: Main FastAPI backend
- `services/`: All the core RAG logic
- `utils/`: File utilities
- `frontend/`: HTML/CSS/JS UI
- `uploads/`: Where we save PDFs
- `vector_store/`: FAISS index and chunks map"

### Step 2: Open app.py first
Open `app.py`, scroll to the top and explain imports, then go through endpoints:
1. **First, show global variables and initialize_globals()**: "Okay, here we have our global variables for the Ollama client, embeddings model, FAISS index, and chunks list—and we added initialize_globals() to fix the UnboundLocalError on server reload."
2. **Then show QueryRequest model**: "Here's our Pydantic model for /query requests."
3. **Then show _build_index_from_pdf()**: "This is the core function for processing a PDF upload: extract text → chunk → embed → build FAISS index → save → reload into globals."
4. **Then show /upload endpoint**: "Here's the /upload endpoint: validates the file, saves it, calls _build_index_from_pdf(), returns metadata."
5. **Then show /query endpoint**: "Here's the /query endpoint: initializes Ollama client if needed, loads resources, calls answer_question() from rag_service."
6. **Show startup_load_resources()**: "On startup, we initialize everything."

### Step 3: Open utils/file_utils.py
"Next, let's look at file utils! This handles validating PDFs are really PDFs, generating unique filenames so they don't collide, and saving the file to disk."

### Step 4: Open services/pdf_extractor.py
"Now PDF extraction! We use PyMuPDF (fitz) to open the PDF, iterate over each page, extract text, and return it structured with page numbers! The finally block ensures the document is always closed."

### Step 5: Open services/chunk_service.py
"Chunking! We use RecursiveCharacterTextSplitter with 1000-char chunks and 200-char overlap—this helps keep semantic context and prevent boundary issues! We process each page and assign chunk IDs."

### Step 6: Open services/faiss_pipeline.py
"This is where embeddings and FAISS live! We load the sentence transformer model, embed chunks, build and save the index, and query the index! Key point: we normalize vectors so inner product equals cosine similarity!"

### Step 7: Open services/rag_service.py
"This is the RAG orchestrator! retrieve_top_chunks() gets the relevant chunks, build_prompt_template() builds our carefully engineered prompt to prevent hallucinations, and answer_question() ties it all together and calls Ollama!"

### Step 8: Open services/ollama_service.py
"Ollama client! Super lightweight, uses only standard library urllib to hit Ollama's /v1/completions endpoint, parses responses, and handles errors with custom OllamaError!"

### Step 9: Open frontend/index.html and script.js
"Quick look at frontend: index.html is the layout, script.js handles drag/drop, upload API call, query API call, and chat bubble rendering!"

### Step 10: Summarize the flow end-to-end again
"Okay, putting it all together: user uploads PDF → backend extracts/chunks/embeds/indexes → user asks question → backend retrieves relevant chunks → builds prompt → calls Ollama → returns answer + sources → frontend displays it!"

### Common Interview Questions to Anticipate During Walkthrough
- "Wait, why do you reload the index after saving it?"
  "To make sure the global variables are using the newly saved index/chunks, not the old ones!"
- "Why did you choose those chunk size/overlap values?"
  "They're a reasonable default! 1000 chars is enough to capture context, 200 overlap prevents boundary splits—we could tune with testing!"

---

## System Design

### High-Level Architecture Diagram
```
┌─────────────────┐
│   Frontend UI   │ (HTML/CSS/JS)
└────────┬────────┘
         │
         ↓ HTTP
┌─────────────────┐
│   FastAPI API   │
└────────┬────────┘
         │
         ├───────────────┐
         ↓               ↓
┌─────────────────┐ ┌──────────────────┐
│  Upload Flow    │ │  Query Flow      │
└────────┬────────┘ └────────┬─────────┘
         │                   │
         ↓                   ↓
┌─────────────────┐ ┌──────────────────┐
│  PDF Extractor  │ │   Embed Query    │
│  (PyMuPDF)      │ │  (HuggingFace)   │
└────────┬────────┘ └────────┬─────────┘
         │                   │
         ↓                   ↓
┌─────────────────┐ ┌──────────────────┐
│  Text Chunker   │ │   FAISS Search   │
│  (LangChain)    │ │  (Top‑k Chunks)  │
└────────┬────────┘ └────────┬─────────┘
         │                   │
         ↓                   ↓
┌─────────────────┐ ┌──────────────────┐
│  Embed Chunks   │ │   Build Prompt   │
│  (HuggingFace)  │ │  (Context+Q)     │
└────────┬────────┘ └────────┬─────────┘
         │                   │
         ↓                   ↓
┌─────────────────┐ ┌──────────────────┐
│  Build FAISS    │ │      Ollama      │
│  Save to Disk   │ │  (Local LLM)     │
└─────────────────┘ └────────┬─────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │  Return Answer  │
                    │  + Sources      │
                    └────────┬────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │  Frontend UI    │
                    └─────────────────┘
```

### Low-Level Architecture Diagram (Backend Components)
```
                     ┌─────────────────────────────────────────────┐
                     │              FastAPI App                    │
                     │  ┌────────────────────────────────────────┐ │
                     │  │      API Endpoints                     │ │
                     │  │  - GET /                               │ │
                     │  │  - POST /upload                        │ │
                     │  │  - POST /query                         │ │
                     │  └───────────────────────┬────────────────┘ │
                     └──────────────────────────┼──────────────────┘
                                                │
                          ┌─────────────────────┼───────────────────┐
                          │                     │                   │
                          ↓                     ↓                   ↓
                  ┌────────────────┐   ┌────────────────┐  ┌───────────────┐
                  │  file_utils    │   │ pdf_extractor  │  │ chunk_service │
                  └────────────────┘   └────────────────┘  └───────────────┘
                          │                     │                   │
                          └─────────────────────┼───────────────────┘
                                                │
                          ┌─────────────────────┼───────────────────┐
                          │                     │                   │
                          ↓                     ↓                   ↓
                  ┌────────────────┐   ┌────────────────┐  ┌───────────────┐
                  │ faiss_pipeline │   │  rag_service   │  │ollama_service │
                  └────────────────┘   └────────────────┘  └───────────────┘
                          │                     │                   │
                          └─────────────────────┼───────────────────┘
                                                │
                          ┌─────────────────────┼───────────────────┐
                          │                     │                   │
                          ↓                     ↓                   ↓
                  ┌────────────────┐   ┌────────────────┐  ┌───────────────┐
                  │ HuggingFace    │   │    FAISS       │  │   Ollama      │
                  │ (sentence-     │   │  (Index)       │  │  (Local LLM)  │
                  │  transformers) │   └────────────────┘  └───────────────┘
                  └────────────────┘
```

### Request Lifecycle (Upload)
1. **User sends POST /upload**:
   - Request hits FastAPI's /upload endpoint (async def).
   - validate_pdf_file() runs; raises 400 if invalid.
   - generate_unique_filename() creates unique name.
   - save_upload_file() writes file to uploads/.
   - _build_index_from_pdf() is called.
   - Inside _build_index_from_pdf():
     - _ensure_embeddings_loaded() loads model if needed.
     - extract_pdf_text() reads PDF, returns pages.
     - chunk_pdf_pages() splits into chunks.
     - embed_chunks() creates vectors.
     - build_faiss_index() builds index.
     - save_faiss() saves index + chunks.
     - load_faiss() reloads into globals.
   - Returns JSON: {filename, file_path, file_size, pages, chunks} → 200 OK.

### Request Lifecycle (Query)
1. **User sends POST /query**:
   - Request hits /query endpoint (def).
   - Checks if ollama_client is None; reinitializes if needed.
   - Checks if faiss_index/chunks/embeddings are None; tries to load.
   - Validates question is not empty; raises 400 if it is.
   - Calls answer_question().
   - Inside answer_question():
     - retrieve_top_chunks() calls query_index().
     - If no chunks, returns fallback answer.
     - build_prompt_template() creates prompt.
     - ollama_client.generate() calls Ollama's API.
     - Formats answer + sources.
   - Returns JSON: {answer, sources} → 200 OK.
   - If any error: returns appropriate 400/500/502/503 JSON error.

### Data Lifecycle
1. **PDF**: User uploads → saved to `uploads/{unique_id}.pdf`.
2. **Chunks**: Extracted from PDF → stored temporarily in memory → saved to `vector_store/chunks_map.json` as JSON array.
3. **Embeddings**: Generated from chunks → stored in FAISS index → saved to `vector_store/faiss.index` as binary.
4. **Queries**: User question → embedded → used for FAISS search → not stored (unless we add chat history later).
5. **Answers**: Generated by Ollama → returned to frontend → not stored (unless we add chat history).

### Error Handling
- **File validation**: ValueError → 400 Bad Request.
- **PDF extraction**: PDFExtractionError → 400 Bad Request.
- **Missing resources (Ollama down, no index)**: 502 Bad Gateway/503 Service Unavailable.
- **Invalid query**: ValueError →400 Bad Request.
- **Unexpected errors**: Catch-all →500 Internal Server Error + log exception.

### Logging
- Uses Python's standard logging module, level=INFO.
- Logs:
  - Upload received, PDF text snippet, first page, number of chunks, first chunk, embedding dimension, vectors in FAISS, save success, reload success.
  - Query received, retrieval time, retrieved chunks and scores, prompt sent, Ollama response and time, total request time.
  - Errors with full traceback.
- Log format: `"%(asctime)s [%(levelname)s] %(message)s"`.

### Configuration
- Uses `python-dotenv` to load from `.env` file.
- All configs are module-level constants in `app.py`:
  - OLLAMA_SERVER_URL
  - OLLAMA_MODEL_NAME
  - VECTOR_INDEX_PATH
  - CHUNKS_MAPPING_PATH
  - EMBEDDING_MODEL_NAME
  - EMBEDDING_DEVICE

### Scalability
**Current limitations for scaling**:
- Global state: Can't use multiple workers/processes.
- Local FAISS file: Not persisted if container restarts; can't share across multiple instances.
- Synchronous processing: Blocks event loop during PDF processing (though we use async endpoint, processing is sync).

**Scalability improvements**:
- Replace global FAISS index with managed vector DB (Pinecone/Weaviate/Qdrant/pgvector).
- Add message broker (Redis/RabbitMQ) + task queue (Celery) for background processing of PDFs.
- Use Docker Compose to run FastAPI, Ollama, vector DB, Redis all together.
- Add caching (Redis) for frequent queries.

### Performance Bottlenecks
1. **Embedding model inference**: Loading the model once on startup is good; embedding many chunks could be slow on CPU (could use GPU if available).
2. **Ollama inference**: Local LLMs can be slow on CPU; use GPU or faster model (smaller model like llama3.2:1b).
3. **FAISS IndexFlatIP**: For 100k+ chunks, switch to an approximate index like IndexIVFFlat for faster search.

### Security Issues (Current)
1. **CORS**: `allow_origins=["*"]` (allows any frontend to talk to backend; in production, set to your specific domain).
2. **No authentication**: Anyone can upload/query.
3. **No rate limiting**: Vulnerable to abuse/DoS.
4. **No file size limits**: Users could upload huge PDFs and crash the server.
5. **No virus scanning**: Uploaded PDFs are saved directly to disk without scanning.
6. **No input sanitization**: Though we validate PDFs, we should be careful with any user input (question text is passed to Ollama, which should be safe, but still).

### How to Deploy
1. **Dockerize**:
   - Create `Dockerfile` for backend:
     ```dockerfile
     FROM python:3.12-slim
     WORKDIR /app
     COPY requirements.txt .
     RUN pip install --no-cache-dir -r requirements.txt
     COPY . .
     EXPOSE 8000
     CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
     ```
   - Create `docker-compose.yml` to run FastAPI + Ollama (optional):
     ```yaml
     version: '3.8'
     services:
       backend:
         build: .
         ports:
           - "8000:8000"
         volumes:
           - ./uploads:/app/uploads
           - ./vector_store:/app/vector_store
         env_file:
           - .env
       # Uncomment if you want Ollama in container too
       # ollama:
       #   image: ollama/ollama:latest
       #   ports:
       #     - "11434:11434"
       #   volumes:
       #     - ollama_data:/root/.ollama
     # volumes:
     #   ollama_data:
     ```

2. **Cloud Deployment**:
   - Use Render, Fly.io, AWS ECS/Fargate, GCP Cloud Run, Azure App Service.
   - For Ollama: Either run it on the same VM, or use a hosted LLM (OpenAI, Anthropic, Together.ai, etc.).
   - For vector storage: Use a managed vector DB, or use a persistent volume for `vector_store/`.
   - Frontend: Serve static files via Nginx, or upload to S3 + CloudFront.

3. **Production Config**:
   - Set `allow_origins` to your frontend domain, not "*".
   - Add authentication.
   - Add rate limiting (slowapi with Limiter + MemoryStorage).
   - Use structured logging (e.g., json-log-formatter).
   - Add monitoring (Prometheus + Grafana, or a service like New Relic/Datadog).
   - Add health checks (FastAPI's /health endpoint).

### Future Improvements
1. **Multi-PDF and user accounts**: Auth + vector DB with metadata filtering.
2. **Chat history/memory**: Database + conversational prompts.
3. **Better chunking**: Semantic chunking, document structure-aware splitting.
4. **Re-ranking**: Cross-encoder re-ranking.
5. **Hybrid search**: BM25 + semantic.
6. **OCR for scanned PDFs**: Pytesseract/EasyOCR.
7. **Table extraction**: Camelot/Tabula.
8. **Image understanding**: Vision-language models for image descriptions.
9. **Streaming responses**: Stream Ollama's output to frontend token-by-token for better UX.
10. **Testing**: Full unit/integration/e2e test suite.
11. **CI/CD**: GitHub Actions/GitLab CI for testing and deployment.

---

## One Day Revision Notes
Use this to cram everything before your interview!

### Quick Recap - Tech Stack
- **Backend**: FastAPI, Uvicorn
- **PDF Extraction**: PyMuPDF (fitz)
- **Chunking**: langchain-text-splitters RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384D) via langchain-huggingface
- **Vector Search**: FAISS IndexFlatIP (cosine similarity, normalized vectors)
- **LLM**: Ollama (llama3.2:3b, temp=0.0, max_tokens=150)
- **Frontend**: Vanilla HTML/CSS/JavaScript

### Core Concepts Recap
- **RAG**: Retrieval-Augmented Generation → retrieves relevant chunks, then generates answer using only those chunks (reduces hallucinations, adds citations).
- **Embeddings**: Numerical vectors where similar texts are close together.
- **Cosine Similarity**: Measures angle between vectors; we normalize vectors so dot product = cosine similarity.
- **Chunking**: Splitting text so it fits embedding models and retrieves relevant parts; overlap fixes boundary issues.
- **FAISS**: Fast similarity search library; IndexFlatIP for exact search, IndexIVFFlat for approximate scaling.

### File Structure Quick Reference
```
app.py:
- Main FastAPI app, endpoints, global variables, _build_index_from_pdf()

utils/file_utils.py:
- validate_pdf_file(), generate_unique_filename(), save_upload_file()

services/pdf_extractor.py:
- extract_pdf_text(), PDFExtractionError

services/chunk_service.py:
- chunk_pdf_pages(pages, chunk_size=1000, chunk_overlap=200)

services/faiss_pipeline.py:
- load_embedding_model(), embed_chunks(), build_faiss_index(), save_faiss(), load_faiss(), query_index()

services/ollama_service.py:
- OllamaClient dataclass, generate(prompt), _extract_text(response)

services/rag_service.py:
- retrieve_top_chunks(), format_retrieved_chunks(), build_prompt_template(), answer_question()

frontend/: index.html, script.js, style.css
```

### Endpoints Quick Reference
1. **GET /**: Health check/message.
2. **POST /upload**: Accepts FormData with `upload_file`, processes PDF, returns metadata.
3. **POST /extract-pdf**: Accepts FormData with `upload_file`, returns extracted pages (test endpoint).
4. **POST /chunk-pdf**: Accepts FormData with `upload_file`, returns chunks (test endpoint).
5. **POST /query**: Accepts JSON {question: str, top_k?: int}, returns {answer: str, sources: [{chunk: int, score: float}]}.

### Environment Variables Quick Reference
- OLLAMA_SERVER_URL=http://localhost:11434
- OLLAMA_MODEL_NAME=llama3.2:3b
- VECTOR_INDEX_PATH=vector_store/faiss.index
- CHUNKS_MAPPING_PATH=vector_store/chunks_map.json
- EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
- EMBEDDING_DEVICE=cpu

### Key Bugs We Fixed Recap
1. **Chunk size too small**: Changed from 100 to 1000 chars, overlap from 20 to 200.
2. **UnboundLocalError on server reload**: Added initialize_globals() that sets all globals to None and calls at module load; added ollama_client reinitialization in /query.
3. **Demo chunks overwriting index**: Don't run faiss_pipeline.py main() after uploading PDFs!

### Project Flow Quick Recap (Upload)
```
Upload PDF → Validate → Save file → Extract text → Chunk → Embed → Build FAISS index → Save index/chunks → Reload to globals → Return success
```

### Project Flow Quick Recap (Query)
```
Ask question → Embed question → Retrieve top-k chunks → Build prompt → Send to Ollama → Parse answer → Return answer + sources
```

### Interview Questions Quick Recap
1. Explain project (end-to-end flow)
2. Why FastAPI, Ollama, FAISS, HuggingFace?
3. What's RAG, why use it?
4. What's chunking, why overlap?
5. What's embeddings, cosine similarity?
6. How do you scale this?
7. How do you add multiple PDFs/users/auth?
8. How do you improve retrieval?
9. Production limitations?
10. Hardest bug and fix?

Remember: Be confident, explain clearly, use specific details from your project! Don't be afraid to say "I don't know, but here's how I'd find out"! Good luck!
```

Now let's write this to a file!Okay, let's create the complete technical learning report as a Markdown file!
<seed:tool_call><function name="TodoWrite"><parameter name="todos" string="false":[{"content": "Create comprehensive learning report", "id": "1", "priority": "high", "status": "completed"}]