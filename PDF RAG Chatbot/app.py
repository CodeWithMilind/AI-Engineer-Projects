from pathlib import Path
import os
import logging
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.chunk_service import chunk_pdf_pages
from services.faiss_pipeline import load_embedding_model, load_faiss
from services.ollama_service import OllamaClient, OllamaError
from services.rag_service import answer_question
from services.pdf_extractor import PDFExtractionError, extract_pdf_text
from utils.file_utils import (
    generate_unique_filename,
    save_upload_file,
    validate_pdf_file,
)

load_dotenv()

app = FastAPI(title="PDF RAG Chatbot Starter")
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


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


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
    """Accept a single PDF file upload, save it with a unique name, and return file metadata."""
    try:
        validate_pdf_file(upload_file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    unique_name = generate_unique_filename(upload_file.filename)
    saved_path = UPLOAD_FOLDER / unique_name

    try:
        save_upload_file(upload_file, saved_path)
        file_size = saved_path.stat().st_size
        return {
            "filename": unique_name,
            "file_path": str(saved_path),
            "file_size": file_size,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to save uploaded PDF file.") from exc


@app.post("/extract-pdf")
async def extract_pdf(upload_file: UploadFile = File(...)):
    """Upload a PDF, save it, and return extracted text for every page."""
    try:
        validate_pdf_file(upload_file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    saved_path = UPLOAD_FOLDER / upload_file.filename
    try:
        save_upload_file(upload_file, saved_path)
        extraction_result = extract_pdf_text(saved_path)
        return extraction_result
    except PDFExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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
        chunks = chunk_pdf_pages(pages)
        total_chunks = len(chunks)

        return {
            "total_chunks": total_chunks,
            "chunks": chunks,
        }
    except PDFExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected server error while chunking PDF.") from exc


@app.post("/query")
def query_document(query: QueryRequest):
    """Answer a user question using FAISS retrieval and Ollama generation."""
    if faiss_index is None or chunks is None or embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="RAG resources are not loaded. Build the FAISS index and restart the server.",
        )

    if not query.question.strip():
        raise HTTPException(status_code=400, detail="A non-empty question is required.")

    try:
        answer = answer_question(
            user_question=query.question,
            index=faiss_index,
            chunks=chunks,
            embeddings=embeddings,
            ollama_client=ollama_client,
            top_k=query.top_k or 3,
        )
        return {"answer": answer}
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected error during query: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during query.") from exc
