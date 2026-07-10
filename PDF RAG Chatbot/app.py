from pathlib import Path
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.chunk_service import chunk_pdf_pages
from services.faiss_pipeline import (
    build_faiss_index,
    embed_chunks,
    load_embedding_model,
    load_faiss,
    save_faiss,
)
from services.ollama_service import OllamaClient, OllamaError
from services.pdf_extractor import PDFExtractionError, extract_pdf_text
from services.rag_service import answer_question
from utils.file_utils import (
    generate_unique_filename,
    save_upload_file,
    validate_pdf_file,
)

load_dotenv()

app = FastAPI(title="PDF RAG Chatbot Starter")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:3b")
VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "vector_store/faiss.index")
CHUNKS_MAPPING_PATH = os.getenv("CHUNKS_MAPPING_PATH", "vector_store/chunks_map.json")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ollama_client: Optional[OllamaClient] = None
embeddings = None
faiss_index = None
chunks = None


def initialize_globals():
    global ollama_client, embeddings, faiss_index, chunks
    ollama_client = None
    embeddings = None
    faiss_index = None
    chunks = None


initialize_globals()


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


def _ensure_embeddings_loaded() -> None:
    global embeddings
    if embeddings is None:
        logger.info("Loading embedding model for indexing and retrieval.")
        embeddings = load_embedding_model(model_name=EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)


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


@app.get("/")
def read_root():
    return JSONResponse({"message": "PDF RAG Chatbot backend is running."})


@app.post("/upload")
async def upload_pdf(upload_file: UploadFile = File(...)):
    """Save the uploaded PDF and rebuild the FAISS index from its content."""
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


@app.post("/extract-pdf")
async def extract_pdf(upload_file: UploadFile = File(...)):
    """Upload a PDF, save it, and return extracted text for every page."""
    try:
        validate_pdf_file(upload_file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    unique_name = generate_unique_filename(upload_file.filename)
    saved_path = UPLOAD_FOLDER / unique_name
    try:
        save_upload_file(upload_file, saved_path)
        extraction_result = extract_pdf_text(saved_path)
        return extraction_result
    except PDFExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected server error while processing PDF.") from exc


@app.post("/chunk-pdf")
async def chunk_pdf(upload_file: UploadFile = File(...)):
    """Accept a PDF upload, extract text, chunk it, and return chunk metadata."""
    try:
        validate_pdf_file(upload_file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    unique_name = generate_unique_filename(upload_file.filename)
    saved_path = UPLOAD_FOLDER / unique_name

    try:
        save_upload_file(upload_file, saved_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to save uploaded PDF file.") from exc

    try:
        extraction_result = extract_pdf_text(saved_path)
        pages = extraction_result.get("pages", [])
        chunk_list = chunk_pdf_pages(pages)
        return {
            "total_chunks": len(chunk_list),
            "chunks": chunk_list,
        }
    except PDFExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected server error while chunking PDF.") from exc


@app.post("/query")
def query_document(query: QueryRequest):
    """Answer a user question using FAISS retrieval and Ollama generation."""
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
