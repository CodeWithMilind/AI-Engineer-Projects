from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from services.faiss_pipeline import HuggingFaceEmbeddings, query_index
from services.ollama_service import OllamaClient, OllamaError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def retrieve_top_chunks(
    index,
    chunks: List[str],
    embeddings: HuggingFaceEmbeddings,
    user_question: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Retrieve the top matching chunk texts from the FAISS index.

    Args:
        index: The loaded FAISS index.
        chunks: The list of original chunk texts.
        embeddings: The embedding model used for query encoding.
        user_question: The question to match against stored chunks.
        top_k: The number of top chunks to return.

    Returns:
        A list of the top matching chunk text strings.
    """
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


def format_retrieved_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a single text block for prompt injection."""
    formatted_chunks: List[str] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text")
        else:
            text = chunk

        if isinstance(text, str) and text.strip():
            formatted_chunks.append(text.strip())

    return "\n\n".join(formatted_chunks)


def build_prompt_template(retrieved_chunks: List[Dict[str, Any]], user_question: str) -> str:
    """Build a concise, production-ready RAG prompt with explicit grounding rules."""
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


def answer_question(
    user_question: str,
    index,
    chunks: List[str],
    embeddings: HuggingFaceEmbeddings,
    ollama_client: Optional[OllamaClient] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Answer a user question using FAISS retrieval and Ollama generation."""
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
