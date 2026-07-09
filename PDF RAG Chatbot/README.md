# PDF RAG Chatbot Starter

This is the initial project setup for a Python FastAPI backend designed for a production-ready RAG (retrieval-augmented generation) chatbot application.

## Project structure

- `app.py` - FastAPI application entrypoint.
- `requirements.txt` - Python dependencies.
- `.env.example` - example environment variables for local configuration.
- `uploads/` - user-uploaded PDFs will be stored here.
- `data/` - raw or processed data files, such as metadata and extracted text.
- `vector_store/` - future vector database or serialized embeddings storage.
- `services/` - application services, like PDF ingestion and retrieval logic.
- `utils/` - helper utilities and shared functions.
- `frontend/` - frontend assets or UI code for the chatbot.

## How to run

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate it:
   - Windows: `venv\Scripts\activate`
   - macOS / Linux: `source venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   uvicorn app:app --reload
   ```

## What each folder is for

- `uploads/`: store PDFs that users upload. Later, the backend will read these files and convert them into text.
- `data/`: keep processed outputs, such as extracted text chunks or metadata. This helps separate temporary inputs from processed data.
- `vector_store/`: reserve space for the vector database or disk-backed embeddings store that will power retrieval.
- `services/`: future business logic will live here, including modules for parsing PDFs, building embeddings, and querying the vector store.
- `utils/`: shared utilities for file handling, environment loading, validation, and small helper functions.
- `frontend/`: place UI code or static assets here, such as a React app, HTML interface, or simple chat page.

## Phase 1 goals

- Understand how a FastAPI backend is structured.
- Keep the project modular and ready for later AI-specific functionality.
- Avoid implementing embeddings, vector databases, or LLMs yet.

## Before Phase 2

Make sure you understand:

1. Python package structure and why folders like `services/` and `utils/` help with separation of concerns.
2. How FastAPI handles routes and startup using `app.py`.
3. The role of environment variables for configuration.
4. The difference between raw `uploads/`, processed `data/`, and retrieval-oriented `vector_store/` storage.
5. Why a frontend folder is useful even if the UI is not built yet.
