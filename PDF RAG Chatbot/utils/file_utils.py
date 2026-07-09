import uuid
from pathlib import Path
from fastapi import UploadFile


def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename while preserving the original extension."""
    extension = Path(original_filename).suffix.lower()
    if extension != ".pdf":
        extension = ".pdf"
    unique_id = uuid.uuid4().hex
    return f"{unique_id}{extension}"


def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save an uploaded file to disk and return the saved file path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        buffer.write(upload_file.file.read())
    return destination


def validate_pdf_file(upload_file: UploadFile) -> None:
    """Validate that an uploaded file is a PDF by checking its content type and extension."""
    if upload_file.content_type != "application/pdf":
        raise ValueError("Uploaded file must be a PDF.")

    if not upload_file.filename.lower().endswith(".pdf"):
        raise ValueError("Uploaded file must use a .pdf extension.")
