"""
Documents API routes.
"""

import shutil
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import settings
from app.db.database import Database, get_db
from app.db.repositories import CacheRepository, DocRepository, EnrichRepository, RunRepository
from app.models.api import (
    CacheInvalidateRequest,
    CacheInvalidateResponse,
    DocListResponse,
    DocResponse,
)

router = APIRouter(prefix="/docs", tags=["docs"])


@router.get("", response_model=DocListResponse)
async def list_docs(
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: Database = Depends(get_db),
) -> DocListResponse:
    """
    List all documents.
    """
    doc_repo = DocRepository(db)
    run_repo = RunRepository(db)

    # 獲取總數和當前頁數據
    total = await doc_repo.count()
    docs = await doc_repo.list_all(limit=limit, offset=offset)

    result = []
    for doc in docs:
        runs = await run_repo.list_by_doc(doc.doc_id)
        result.append(
            DocResponse(
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                ext=doc.ext,
                size_bytes=doc.size_bytes,
                created_at=doc.created_at,
                run_count=len(runs),
            )
        )

    return DocListResponse(docs=result, total=total)


@router.get("/{doc_id}", response_model=DocResponse)
async def get_doc(
    doc_id: str,
    db: Database = Depends(get_db),
) -> DocResponse:
    """
    Get document information.
    """
    doc_repo = DocRepository(db)
    run_repo = RunRepository(db)

    doc = await doc_repo.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    runs = await run_repo.list_by_doc(doc_id)

    return DocResponse(
        doc_id=doc.doc_id,
        source_path=doc.source_path,
        ext=doc.ext,
        size_bytes=doc.size_bytes,
        created_at=doc.created_at,
        run_count=len(runs),
    )


@router.delete("/{doc_id}")
async def delete_doc(
    doc_id: str,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    """
    Delete a document and all its runs.
    """
    doc_repo = DocRepository(db)
    run_repo = RunRepository(db)
    cache_repo = CacheRepository(db)
    enrich_repo = EnrichRepository(db)

    doc = await doc_repo.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    # Delete all runs (this also deletes run_stages)
    runs = await run_repo.list_by_doc(doc_id)
    for run in runs:
        await run_repo.delete(run.run_id)

    # Delete enrich entries (foreign key constraint)
    await enrich_repo.invalidate_by_doc(doc_id)

    # Delete cache entries
    await cache_repo.invalidate_by_doc(doc_id)

    # Delete document from DB
    await doc_repo.delete(doc_id)

    # Delete from filesystem
    doc_path = settings.get_doc_path(doc_id)
    if doc_path.exists():
        shutil.rmtree(doc_path)

    return {"message": "Document deleted", "doc_id": doc_id}


@router.post("/{doc_id}/cache/invalidate", response_model=CacheInvalidateResponse)
async def invalidate_cache(
    doc_id: str,
    request: CacheInvalidateRequest,
    db: Database = Depends(get_db),
) -> CacheInvalidateResponse:
    """
    Invalidate cache entries for a document.
    """
    doc_repo = DocRepository(db)
    cache_repo = CacheRepository(db)

    doc = await doc_repo.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    total_invalidated = 0

    if request.stages:
        # Invalidate specific stages
        for stage in request.stages:
            count = await cache_repo.invalidate_by_doc_stage(doc_id, stage)
            total_invalidated += count

            # Also delete cache files
            cache_dir = settings.get_doc_path(doc_id) / "cache" / stage.value
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
    else:
        # Invalidate all cache for this doc
        total_invalidated = await cache_repo.invalidate_by_doc(doc_id)

        # Delete all cache directories
        cache_dir = settings.get_doc_path(doc_id) / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    return CacheInvalidateResponse(invalidated_count=total_invalidated)


@router.post("/batch-delete")
async def batch_delete_docs(
    doc_ids: list[str],
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Batch delete multiple documents and all their runs.
    """
    doc_repo = DocRepository(db)
    run_repo = RunRepository(db)
    cache_repo = CacheRepository(db)
    enrich_repo = EnrichRepository(db)

    deleted: list[str] = []
    errors: list[dict[str, str]] = []

    for doc_id in doc_ids:
        try:
            doc = await doc_repo.get(doc_id)
            if not doc:
                errors.append({"doc_id": doc_id, "error": "Document not found"})
                continue

            # Delete all runs (this also deletes run_stages)
            runs = await run_repo.list_by_doc(doc_id)
            for run in runs:
                await run_repo.delete(run.run_id)

            # Delete enrich entries (foreign key constraint)
            await enrich_repo.invalidate_by_doc(doc_id)

            # Delete cache entries
            await cache_repo.invalidate_by_doc(doc_id)

            # Delete document from DB
            await doc_repo.delete(doc_id)

            # Delete from filesystem
            doc_path = settings.get_doc_path(doc_id)
            if doc_path.exists():
                shutil.rmtree(doc_path)

            deleted.append(doc_id)
        except Exception as e:
            errors.append({"doc_id": doc_id, "error": str(e)})

    return {
        "message": f"Deleted {len(deleted)} documents",
        "deleted": deleted,
        "errors": errors,
    }


@router.get("/{doc_id}/runs")
async def list_doc_runs(
    doc_id: str,
    limit: int = Query(default=50, le=200),
    db: Database = Depends(get_db),
) -> dict:
    """
    List all runs for a document.
    """
    doc_repo = DocRepository(db)
    run_repo = RunRepository(db)

    doc = await doc_repo.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    runs = await run_repo.list_by_doc(doc_id, limit=limit)

    return {
        "doc_id": doc_id,
        "runs": [
            {
                "run_id": r.run_id,
                "profile": r.profile,
                "status": r.status,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
            for r in runs
        ],
    }
