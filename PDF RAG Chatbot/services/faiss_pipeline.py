"""
FAISS Embedding Pipeline

This script demonstrates how to:
1. Load a HuggingFace embedding model via LangChain's `HuggingFaceEmbeddings`.
2. Convert text chunks into embeddings (numpy arrays).
3. Build a FAISS index from normalized embeddings (for cosine similarity).
4. Save and load the FAISS index and the mapping from vector IDs to text chunks.
5. Embed a user query and retrieve the top-k most similar chunks with scores.

Notes for learners:
- We use LangChain only for the `HuggingFaceEmbeddings` wrapper (it simplifies HuggingFace usage).
- We use `faiss` directly to build and persist the vector index so you can see the low-level steps.
"""

from typing import List, Tuple
import json
import os

import numpy as np

# LangChain provides a convenient wrapper around HuggingFace sentence-transformer models.
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# FAISS is the vector index library for fast similarity search.
import faiss


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> HuggingFaceEmbeddings:
    """Load a HuggingFace embedding model via LangChain.

    Args:
        model_name: The HF model name (sentence-transformers recommended).
        device: Where to run the model: 'cpu' or 'cuda'.

    Returns:
        An instance of `HuggingFaceEmbeddings` that can create embeddings.

    Why use LangChain here:
        - It abstracts tokenizer/model loading and gives consistent methods
          like `embed_documents` and `embed_query` which return Python lists.
    """

    # Create the embeddings object. LangChain will load the specified model.
    # `model_kwargs` instructs the transformer to run on the requested device.
    embed = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    return embed


def embed_chunks(embeddings: HuggingFaceEmbeddings, chunks: List[str]) -> np.ndarray:
    """Convert a list of text chunks into a numpy array of embeddings.

    Args:
        embeddings: HuggingFaceEmbeddings instance.
        chunks: List of text strings (the chunks you created earlier).

    Returns:
        A numpy array of shape (n_chunks, embedding_dim) with dtype float32.
    """

    # LangChain's `embed_documents` returns a list of embedding vectors (lists of floats).
    vectors = embeddings.embed_documents(chunks)

    # Convert to a NumPy array of type float32, as FAISS expects float32 vectors.
    arr = np.array(vectors, dtype=np.float32)
    return arr


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Build a FAISS index for cosine similarity search.

    We normalize vectors to unit length and use an inner-product index. The inner
    product between normalized vectors equals their cosine similarity.

    Args:
        vectors: NumPy array of shape (n, dim) dtype float32.

    Returns:
        A FAISS Index object with the vectors added.
    """

    # Get dimensionality from vectors
    n, dim = vectors.shape

    # Normalize vectors in-place to unit length (L2 norm = 1)
    # Add a tiny epsilon to avoid division by zero.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    vectors_norm = vectors / norms

    # Create an IndexFlatIP (inner product) index. With normalized vectors,
    # inner product = cosine similarity.
    index = faiss.IndexFlatIP(dim)

    # Add normalized vectors to the index.
    index.add(vectors_norm)

    return index


def save_faiss(index: faiss.Index, chunks: List[str], index_path: str, mapping_path: str) -> None:
    """Save the FAISS index binary and the chunk mapping (id -> text).

    Args:
        index: FAISS Index to save.
        chunks: List of original text chunks in the same order as vectors.
        index_path: Path to write the FAISS binary (e.g., 'vector_store/faiss.index').
        mapping_path: Path to write the chunk mapping JSON (e.g., 'vector_store/chunks_map.json').
    """

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Write the FAISS index to disk
    faiss.write_index(index, index_path)

    # Save the chunks mapping as JSON. The position in the list maps to the vector ID.
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_faiss(index_path: str, mapping_path: str) -> Tuple[faiss.Index, List[str]]:
    """Load a FAISS index and the chunk mapping from disk.

    Args:
        index_path: Path to the FAISS binary file.
        mapping_path: Path to the chunks JSON file.

    Returns:
        A tuple `(index, chunks)` where `index` is a FAISS Index and `chunks` is the list of texts.
    """

    # Read the FAISS index
    index = faiss.read_index(index_path)

    # Read the chunks mapping
    with open(mapping_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


def query_index(index: faiss.Index, chunks: List[str], embeddings: HuggingFaceEmbeddings, query: str, k: int = 3) -> List[Tuple[str, float]]:
    """Query the FAISS index and return the top-k chunks with similarity scores.

    Args:
        index: FAISS index containing normalized vectors.
        chunks: List of original chunk texts (order matches index IDs).
        embeddings: HuggingFaceEmbeddings instance to embed the query.
        query: The user query string.
        k: Number of top results to return.

    Returns:
        A list of tuples (chunk_text, similarity_score) ordered by descending score.
    """

    # Embed the query into a vector (list of floats)
    q_vec = embeddings.embed_query(query)

    # Convert to NumPy array and ensure dtype float32
    q = np.array(q_vec, dtype=np.float32).reshape(1, -1)

    # Normalize the query vector to unit length (so inner product = cosine)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-6)

    # Search the FAISS index. `scores` are inner-products (cosine similarities).
    distances, indices = index.search(q_norm, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append((chunks[idx], float(score)))

    return results


def main():
    """Example runner that ties all steps together.

    - If `vector_store/chunks.json` exists, it will use that as the chunk list.
    - Otherwise it creates a small example chunk list to demonstrate the flow.
    """

    # Paths for persistence
    index_path = "vector_store/faiss.index"
    mapping_path = "vector_store/chunks_map.json"

    # Try to load existing chunks from disk, otherwise use demo chunks.
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        # Demo chunks for users who want to run this file immediately.
        chunks = [
            "LangChain helps connect LLMs with data and tools.",
            "FAISS is a library for efficient similarity search over vectors.",
            "HuggingFace provides many transformer models including sentence transformers.",
        ]

    # 1) Load embedding model
    embeddings = load_embedding_model()

    # 2) Convert chunks to embeddings
    vectors = embed_chunks(embeddings, chunks)

    # Print number of chunks and embedding dimension and shape info
    n_chunks, emb_dim = vectors.shape
    print(f"Number of chunks: {n_chunks}")
    print(f"Embedding dimension: {emb_dim}")
    print(f"Embeddings array shape: {vectors.shape}")

    # 3) Build FAISS index
    index = build_faiss_index(vectors)

    # 4) Save the FAISS index and mapping
    save_faiss(index, chunks, index_path, mapping_path)
    print(f"Saved FAISS index to: {index_path}")

    # 5) Load the FAISS index back from disk to simulate a separate process
    index2, chunks2 = load_faiss(index_path, mapping_path)
    print(f"Loaded FAISS index. Number of stored chunks: {len(chunks2)}")

    # 6) Accept a user query (for demo, we use input())
    query = input("Enter a query to search the FAISS index: ")

    # 7) Embed the query and search
    results = query_index(index2, chunks2, embeddings, query, k=3)

    # 8) Print retrieved chunks and similarity scores
    print("\nTop retrieved chunks:")
    for text, score in results:
        print(f"Score: {score:.4f} | Chunk: {text}")


if __name__ == "__main__":
    main()
