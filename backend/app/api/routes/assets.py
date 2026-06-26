"""
Assets API routes.

Provides endpoints for:
- Run assets (figures, forms, pages)
- Source files
- Debug files (layout.pdf, spans.pdf) from parse cache
"""


from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.db.database import Database, get_db
from app.db.repositories import CacheRepository, DocRepository, RunRepository

router = APIRouter(prefix="/assets", tags=["assets"])


@router.get("/{doc_id}/{run_id}/{path:path}")
async def get_asset(
    doc_id: str,
    run_id: str,
    path: str,
    db: Database = Depends(get_db),
) -> FileResponse:
    """
    Get an asset file (image, figure, form, etc.).
    """
    doc_repo = DocRepository(db)
    run_repo = RunRepository(db)

    # Verify document and run exist
    doc = await doc_repo.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    run = await run_repo.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    if run.doc_id != doc_id:
        raise HTTPException(status_code=400, detail="Run does not belong to this document")

    # Build asset path
    asset_path = settings.get_run_path(doc_id, run_id) / "assets" / path

    # Security: ensure path is within assets directory
    try:
        asset_path = asset_path.resolve()
        base_path = (settings.get_run_path(doc_id, run_id) / "assets").resolve()
        if not str(asset_path).startswith(str(base_path)):
            raise HTTPException(status_code=400, detail="Invalid path")
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not asset_path.exists():
        raise HTTPException(status_code=404, detail=f"Asset not found: {path}")

    # Determine media type
    suffix = asset_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
        ".json": "application/json",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(asset_path, media_type=media_type)


@router.get("/{doc_id}/source")
async def get_source_file(
    doc_id: str,
    db: Database = Depends(get_db),
) -> FileResponse:
    """
    Get the original source file.
    """
    doc_repo = DocRepository(db)

    doc = await doc_repo.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    source_dir = settings.get_doc_path(doc_id) / "source"
    if not source_dir.exists():
        raise HTTPException(status_code=404, detail="Source file not found")

    # Find the original file
    source_files = list(source_dir.glob("original.*"))
    if not source_files:
        raise HTTPException(status_code=404, detail="Source file not found")

    source_path = source_files[0]

    # Determine media type
    suffix = source_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".ppt": "application/vnd.ms-powerpoint",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".html": "text/html",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        source_path,
        media_type=media_type,
        filename=f"original{suffix}",
    )


# ==================== Debug/Doctor Assets ====================


@router.get("/{doc_id}/debug/layout.pdf")
async def get_layout_pdf(
    doc_id: str,
    db: Database = Depends(get_db),
) -> FileResponse:
    """
    Get the layout.pdf debug file from parse cache.

    This file shows the reading order and layout detection results
    from MinerU, useful for debugging parsing issues.
    """
    doc_repo = DocRepository(db)
    CacheRepository(db)

    doc = await doc_repo.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    # Find layout.pdf in parse cache
    cache_dir = settings.get_doc_path(doc_id) / "cache" / "parse"
    if not cache_dir.exists():
        raise HTTPException(status_code=404, detail="Parse cache not found")

    # Search for layout.pdf in any cache entry
    for cache_entry in cache_dir.iterdir():
        if cache_entry.is_dir():
            for layout_pdf in cache_entry.rglob("layout.pdf"):
                return FileResponse(
                    layout_pdf,
                    media_type="application/pdf",
                    filename=f"{doc_id}_layout.pdf",
                )

    raise HTTPException(status_code=404, detail="layout.pdf not found in parse cache")


@router.get("/{doc_id}/debug/spans.pdf")
async def get_spans_pdf(
    doc_id: str,
    db: Database = Depends(get_db),
) -> FileResponse:
    """
    Get the spans.pdf debug file from parse cache.

    This file shows span-level detection results from MinerU.
    """
    doc_repo = DocRepository(db)

    doc = await doc_repo.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    # Find spans.pdf in parse cache
    cache_dir = settings.get_doc_path(doc_id) / "cache" / "parse"
    if not cache_dir.exists():
        raise HTTPException(status_code=404, detail="Parse cache not found")

    # Search for spans.pdf in any cache entry
    for cache_entry in cache_dir.iterdir():
        if cache_entry.is_dir():
            for spans_pdf in cache_entry.rglob("spans.pdf"):
                return FileResponse(
                    spans_pdf,
                    media_type="application/pdf",
                    filename=f"{doc_id}_spans.pdf",
                )

    raise HTTPException(status_code=404, detail="spans.pdf not found in parse cache")


@router.get("/{doc_id}/debug/model.json")
async def get_model_json(
    doc_id: str,
    db: Database = Depends(get_db),
) -> FileResponse:
    """
    Get the model.json (YOLO detection results) from parse cache.

    This file contains the layout detection results including
    category_id, score, and bounding box for each detected element.
    """
    doc_repo = DocRepository(db)

    doc = await doc_repo.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    # Find model.json in parse cache
    cache_dir = settings.get_doc_path(doc_id) / "cache" / "parse"
    if not cache_dir.exists():
        raise HTTPException(status_code=404, detail="Parse cache not found")

    # Search for model.json in any cache entry
    for cache_entry in cache_dir.iterdir():
        if cache_entry.is_dir():
            for model_json in cache_entry.rglob("*_model.json"):
                return FileResponse(
                    model_json,
                    media_type="application/json",
                    filename=f"{doc_id}_model.json",
                )

    raise HTTPException(status_code=404, detail="model.json not found in parse cache")


@router.get("/{doc_id}/debug/middle.json")
async def get_middle_json(
    doc_id: str,
    db: Database = Depends(get_db),
) -> FileResponse:
    """
    Get the middle.json (span/line-level info) from parse cache.

    This file contains detailed span and line information for advanced use cases.
    """
    doc_repo = DocRepository(db)

    doc = await doc_repo.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    # Find middle.json in parse cache
    cache_dir = settings.get_doc_path(doc_id) / "cache" / "parse"
    if not cache_dir.exists():
        raise HTTPException(status_code=404, detail="Parse cache not found")

    # Search for middle.json in any cache entry
    for cache_entry in cache_dir.iterdir():
        if cache_entry.is_dir():
            for middle_json in cache_entry.rglob("*_middle.json"):
                return FileResponse(
                    middle_json,
                    media_type="application/json",
                    filename=f"{doc_id}_middle.json",
                )

    raise HTTPException(status_code=404, detail="middle.json not found in parse cache")


@router.get("/{doc_id}/debug")
async def list_debug_files(
    doc_id: str,
    db: Database = Depends(get_db),
) -> dict:
    """
    List available debug files for a document.

    Returns URLs to available debug assets from parse cache.
    """
    doc_repo = DocRepository(db)

    doc = await doc_repo.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    cache_dir = settings.get_doc_path(doc_id) / "cache" / "parse"
    if not cache_dir.exists():
        return {"doc_id": doc_id, "files": []}

    files = []

    # Search for debug files in cache entries
    for cache_entry in cache_dir.iterdir():
        if cache_entry.is_dir():
            # Check for each debug file type
            for layout_pdf in cache_entry.rglob("layout.pdf"):
                files.append({
                    "name": "layout.pdf",
                    "url": f"/api/assets/{doc_id}/debug/layout.pdf",
                    "description": "Reading order and layout detection visualization",
                })
                break

            for spans_pdf in cache_entry.rglob("spans.pdf"):
                files.append({
                    "name": "spans.pdf",
                    "url": f"/api/assets/{doc_id}/debug/spans.pdf",
                    "description": "Span-level detection visualization",
                })
                break

            for model_json in cache_entry.rglob("*_model.json"):
                files.append({
                    "name": "model.json",
                    "url": f"/api/assets/{doc_id}/debug/model.json",
                    "description": "YOLO detection results (category_id, score, bbox)",
                })
                break

            for middle_json in cache_entry.rglob("*_middle.json"):
                files.append({
                    "name": "middle.json",
                    "url": f"/api/assets/{doc_id}/debug/middle.json",
                    "description": "Span/line-level information for advanced use",
                })
                break

    return {"doc_id": doc_id, "files": files}
