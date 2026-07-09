from __future__ import annotations

import logging
from typing import List, Optional

from services.faiss_pipeline import HuggingFaceEmbeddings, query_index
from services.ollama_service import OllamaClient, OllamaError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def format_retrieved_chunks(chunks: List[str]) -> str:
    """Format retrieved chunks into a single text block for prompt injection."""
    return "\n\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip())


def build_prompt_template(retrieved_chunks: List[str], user_question: str) -> str:
    """Build the RAG prompt template with context and the user question."""
    if not user_question or not user_question.strip():
        raise ValueError("User question must be a non-empty string.")

    formatted_context = format_retrieved_chunks(retrieved_chunks)
    prompt = (
        "You are a helpful AI assistant.\n\n"
        "Answer ONLY using the provided context.\n\n"
        "If the answer is not present in the context, respond exactly:\n"
        "\"I couldn't find this information in the uploaded document.\"\n\n"
        "Context:\n"
        f"{formatted_context}\n\n"
        "Question:\n"
        f"{user_question}\n\n"
        "Answer:\n"
    )
    return prompt


def answer_question(
    user_question: str,
    index,
    chunks: List[str],
    embeddings: HuggingFaceEmbeddings,
    ollama_client: Optional[OllamaClient] = None,
    top_k: int = 3,
) -> str:
    """Answer a user question using FAISS retrieval and Ollama generation."""
    if ollama_client is None:
        ollama_client = OllamaClient()

    if not chunks:
        raise ValueError("Chunk list must not be empty.")

    logger.info("Querying FAISS for top %d chunks.", top_k)
    results = query_index(index, chunks, embeddings, user_question, k=top_k)

    if not results:
        logger.warning("No relevant chunks were retrieved for the user question.")
        return "I couldn't find this information in the uploaded document."

    retrieved_chunks = [chunk for chunk, _score in results]
    prompt = build_prompt_template(retrieved_chunks, user_question)

    try:
        logger.info("Sending prompt to Ollama.")
        response = ollama_client.generate(prompt)
        logger.info("Received response from Ollama.")
        return response
    except OllamaError as exc:
        logger.error("Ollama generation failed: %s", exc)
        raise
