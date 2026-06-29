"""
Ingest API routes.
"""

import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import settings
from app.db.database import Database, get_db
from app.db.repositories import DocRepository
from app.models.api import IngestRequest, IngestResponse
from app.models.entities import DocCreate
from app.supported_files import SUPPORTED_INPUT_EXTENSIONS_LABEL, is_supported_input

router = APIRouter(prefix="/ingest", tags=["ingest"])


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_doc_id(sha256: str) -> str:
    """Generate doc_id from SHA256 (first 16 chars for readability)."""
    return sha256[:16]


@router.post("", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    db: Database = Depends(get_db),
) -> IngestResponse:
    """
    Ingest a document from a local path.

    Creates a doc_id based on file content hash.
    If the same file was already ingested, returns the existing doc_id.
    """
    if not settings.enable_local_path_ingest:
        raise HTTPException(
            status_code=403,
            detail="Local path ingestion is disabled. Use file upload or enable SEMARK_ENABLE_LOCAL_PATH_INGEST for trusted deployments.",
        )

    if not request.path:
        raise HTTPException(status_code=400, detail="path is required")

    source_path = Path(request.path)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.path}")

    if source_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Directory ingestion not yet supported. Please provide a file path.",
        )

    if not is_supported_input(source_path.name):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported extensions: {SUPPORTED_INPUT_EXTENSIONS_LABEL}",
        )

    # Compute hash and doc_id
    file_sha256 = compute_sha256(source_path)
    doc_id = compute_doc_id(file_sha256)

    repo = DocRepository(db)

    # Check if already exists
    existing = await repo.get(doc_id)
    if existing:
        return IngestResponse(
            doc_id=existing.doc_id,
            source_path=existing.source_path,
            ext=existing.ext,
            size_bytes=existing.size_bytes,
            already_exists=True,
        )

    # Create doc directory structure
    doc_dir = settings.get_doc_path(doc_id)
    source_dir = doc_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    # Copy file to workspace
    ext = source_path.suffix.lstrip(".").lower()
    target_path = source_dir / f"original.{ext}"

    import shutil
    shutil.copy2(source_path, target_path)

    # Create DB record
    doc = await repo.create(
        doc_id=doc_id,
        data=DocCreate(
            source_path=str(source_path),
            sha256=file_sha256,
            ext=ext,
            size_bytes=source_path.stat().st_size,
        ),
    )

    return IngestResponse(
        doc_id=doc.doc_id,
        source_path=doc.source_path,
        ext=doc.ext,
        size_bytes=doc.size_bytes,
        already_exists=False,
    )


@router.post("/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Database = Depends(get_db),
) -> IngestResponse:
    """
    Upload and ingest a document.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not is_supported_input(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported extensions: {SUPPORTED_INPUT_EXTENSIONS_LABEL}",
        )

    # Read file content and compute hash
    content = await file.read()
    file_sha256 = hashlib.sha256(content).hexdigest()
    doc_id = compute_doc_id(file_sha256)

    repo = DocRepository(db)

    # Check if already exists
    existing = await repo.get(doc_id)
    if existing:
        return IngestResponse(
            doc_id=existing.doc_id,
            source_path=existing.source_path,
            ext=existing.ext,
            size_bytes=existing.size_bytes,
            already_exists=True,
        )

    # Create doc directory and save file
    doc_dir = settings.get_doc_path(doc_id)
    source_dir = doc_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix.lstrip(".").lower()
    target_path = source_dir / f"original.{ext}"

    with open(target_path, "wb") as f:
        f.write(content)

    # Create DB record
    doc = await repo.create(
        doc_id=doc_id,
        data=DocCreate(
            source_path=file.filename,
            sha256=file_sha256,
            ext=ext,
            size_bytes=len(content),
        ),
    )

    return IngestResponse(
        doc_id=doc.doc_id,
        source_path=doc.source_path,
        ext=doc.ext,
        size_bytes=doc.size_bytes,
        already_exists=False,
    )
