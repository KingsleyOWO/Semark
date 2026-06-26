"""
Download API routes.

Provides batch download with format conversion and ZIP packaging.
"""

import io
import json
import zipfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.routes.converters import md_to_docx, md_to_txt
from app.config import settings
from app.db.database import Database, get_db
from app.db.repositories import DocRepository, RunRepository

router = APIRouter(prefix="/runs", tags=["download"])


class FileType(StrEnum):
    """Available file types for download."""

    SOURCE = "source"
    DOCUMENTS = "documents"
    QUALITY = "quality"
    ASSETS_INDEX = "assets_index"
    ENRICHMENTS = "enrichments"


class OutputFormat(StrEnum):
    """Output format for markdown files."""

    MD = "md"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"


class DownloadRequest(BaseModel):
    """Request model for batch download."""

    run_ids: list[str] = Field(..., min_length=1, max_length=500)
    file_types: list[FileType] = Field(
        default=[FileType.DOCUMENTS],
        description="File types to include in download",
    )
    format: OutputFormat = Field(
        default=OutputFormat.MD,
        description="Output format for markdown files",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional split document IDs to include when downloading documents.",
    )


class DownloadManifest(BaseModel):
    """Manifest included in download ZIP."""

    created_at: str
    run_count: int
    file_types: list[str]
    format: str
    files: list[dict]


# Mapping of file types to source files
SOURCE_FILES = {
    FileType.SOURCE: "source.md",
    FileType.QUALITY: "quality.json",
    FileType.ASSETS_INDEX: "assets_index.jsonl",
    FileType.ENRICHMENTS: "enrichments.jsonl",
}

FALLBACK_SOURCE_FILES = {
    FileType.SOURCE: "rag.md",
}


@router.post("/download")
async def download_runs(
    request: DownloadRequest,
    db: Database = Depends(get_db),
) -> StreamingResponse:
    """
    Batch download multiple runs with format conversion.

    Returns a ZIP file containing requested outputs from all specified runs.

    - **run_ids**: List of run IDs to download (max 100)
    - **file_types**: Which output files to include (source, documents, quality, etc.)
    - **format**: Output format for markdown files (md, docx, txt, json)

    JSON files (quality, assets_index, enrichments) are always returned as-is.
    """
    run_repo = RunRepository(db)
    doc_repo = DocRepository(db)

    # Validate all runs exist and get doc info
    runs_with_docs = []
    for run_id in request.run_ids:
        run = await run_repo.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        doc = await doc_repo.get(run.doc_id)
        # Extract original filename without extension
        source_name = Path(doc.source_path).stem if doc else run.run_id[:12]
        runs_with_docs.append((run, source_name))

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    manifest_files = []

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for run, source_name in runs_with_docs:
            run_path = settings.get_run_path(run.doc_id, run.run_id)
            outputs_path = run_path / "outputs"

            for file_type in request.file_types:
                try:
                    if file_type == FileType.DOCUMENTS:
                        archive_folder_name = (
                            f"{source_name}_{run.run_id[:12]}"
                            if len(runs_with_docs) > 1
                            else source_name
                        )
                        document_files = _get_document_files(
                            outputs_path,
                            source_name,
                            request.format,
                            document_ids=request.document_ids,
                            archive_folder_name=archive_folder_name,
                        )
                        if document_files:
                            for filename, content in document_files:
                                zf.writestr(filename, content)
                                manifest_files.append(
                                    {
                                        "run_id": run.run_id,
                                        "doc_id": run.doc_id,
                                        "source_name": source_name,
                                        "file": filename,
                                        "size": len(content),
                                    }
                                )
                        else:
                            manifest_files.append(
                                {
                                    "run_id": run.run_id,
                                    "doc_id": run.doc_id,
                                    "source_name": source_name,
                                    "file": "documents",
                                    "error": "No split documents found",
                                }
                            )
                        continue

                    result = _get_file_content(
                        outputs_path,
                        file_type,
                        request.format,
                        run.run_id,
                        source_name,
                    )

                    if result is not None:
                        content, filename = result
                        # Put files directly in root (filename already includes source_name)
                        zf.writestr(filename, content)
                        manifest_files.append(
                            {
                                "run_id": run.run_id,
                                "doc_id": run.doc_id,
                                "source_name": source_name,
                                "file": filename,
                                "size": len(content),
                            }
                        )
                    else:
                        manifest_files.append(
                            {
                                "run_id": run.run_id,
                                "doc_id": run.doc_id,
                                "source_name": source_name,
                                "file": SOURCE_FILES[file_type],
                                "error": "File not found",
                            }
                        )
                except Exception as e:
                    # Log error but continue with other files
                    manifest_files.append(
                        {
                            "run_id": run.run_id,
                            "doc_id": run.doc_id,
                            "source_name": source_name,
                            "file": file_type.value,
                            "error": str(e),
                        }
                    )

        # End-user document downloads should contain only ingestible markdown files.
        # Keep manifest.json for mixed/debug downloads requested through the API.
        if request.file_types != [FileType.DOCUMENTS]:
            manifest = DownloadManifest(
                created_at=datetime.now().isoformat(),
                run_count=len(runs_with_docs),
                file_types=[ft.value for ft in request.file_types],
                format=request.format.value,
                files=manifest_files,
            )
            zf.writestr("manifest.json", manifest.model_dump_json(indent=2))

    zip_buffer.seek(0)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"runs_download_{timestamp}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.get("/{run_id}/documents/{document_id}/download")
async def download_split_document(
    run_id: str,
    document_id: str,
    format: OutputFormat = Query(default=OutputFormat.MD),
    db: Database = Depends(get_db),
) -> StreamingResponse:
    """Download one split document directly without ZIP packaging."""
    run_repo = RunRepository(db)
    doc_repo = DocRepository(db)

    run = await run_repo.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    doc = await doc_repo.get(run.doc_id)
    source_name = Path(doc.source_path).stem if doc else run.run_id[:12]
    outputs_path = settings.get_run_path(run.doc_id, run.run_id) / "outputs"

    entry = _get_document_entry(outputs_path, document_id, source_name=source_name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    filename = entry.get("filename")
    if not filename:
        raise HTTPException(status_code=404, detail=f"Document filename missing: {document_id}")
    document_path = (outputs_path / "documents" / filename).resolve()
    documents_dir = (outputs_path / "documents").resolve()
    if not document_path.is_relative_to(documents_dir) or not document_path.exists():
        raise HTTPException(status_code=404, detail=f"Document file not found: {document_id}")

    title = str(entry.get("title") or Path(filename).stem)
    content, download_name, media_type = _convert_markdown_document(
        md_bytes=document_path.read_bytes(),
        base_name=str(entry.get("download_base_name") or f"{source_name}_{Path(filename).stem}"),
        title=title,
        format=format,
    )

    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": _content_disposition(download_name)},
    )


def _get_file_content(
    outputs_path: Path,
    file_type: FileType,
    format: OutputFormat,
    run_id: str,
    source_name: str,
) -> tuple[bytes, str] | None:
    """
    Get file content with optional format conversion.

    Returns (content_bytes, filename) or None if source file doesn't exist.
    Uses source_name (original document name) for output filename.
    """
    source_file = SOURCE_FILES[file_type]
    source_path = outputs_path / source_file
    if not source_path.exists() and file_type in FALLBACK_SOURCE_FILES:
        source_path = outputs_path / FALLBACK_SOURCE_FILES[file_type]

    if not source_path.exists():
        return None

    content = source_path.read_bytes()

    # Get the type suffix (dataset, rag, quality, etc.)
    type_suffix = file_type.value

    # JSON files don't get format conversion
    if file_type in (FileType.QUALITY, FileType.ASSETS_INDEX, FileType.ENRICHMENTS):
        # Use source_name for JSON files too
        suffix = "json" if file_type == FileType.QUALITY else "jsonl"
        return content, f"{source_name}_{type_suffix}.{suffix}"

    # Markdown files - use source_name as base
    if format == OutputFormat.MD:
        return content, f"{source_name}_{type_suffix}.md"

    md_text = content.decode("utf-8")

    if format == OutputFormat.TXT:
        txt_content = md_to_txt(md_text)
        return txt_content.encode("utf-8"), f"{source_name}_{type_suffix}.txt"

    if format == OutputFormat.DOCX:
        docx_content = md_to_docx(md_text, title=f"{source_name}_{type_suffix}")
        return docx_content, f"{source_name}_{type_suffix}.docx"

    if format == OutputFormat.JSON:
        # Convert MD to structured JSON
        json_content = {
            "run_id": run_id,
            "source_name": source_name,
            "type": type_suffix,
            "content": md_text,
            "line_count": len(md_text.split("\n")),
            "char_count": len(md_text),
        }
        return (
            json.dumps(json_content, ensure_ascii=False, indent=2).encode("utf-8"),
            f"{source_name}_{type_suffix}.json",
        )

    return content, f"{source_name}_{type_suffix}.md"


def _get_document_files(
    outputs_path: Path,
    source_name: str,
    format: OutputFormat = OutputFormat.MD,
    document_ids: list[str] | None = None,
    archive_folder_name: str | None = None,
) -> list[tuple[str, bytes]]:
    documents_dir = outputs_path / "documents"
    if not documents_dir.exists():
        return []

    archive_folder_name = archive_folder_name or source_name
    selected_ids = set(document_ids or [])
    entries = _get_document_entries(outputs_path, source_name=source_name)

    if entries:
        selected_entries = [
            entry for entry in entries
            if not selected_ids or entry.get("document_id") in selected_ids
        ]
    elif selected_ids:
        return []
    else:
        selected_entries = [
            {
                "filename": path.name,
                "title": path.stem,
                "download_base_name": f"{source_name}_{path.stem}",
            }
            for path in sorted(documents_dir.glob("*.md"))
        ]

    files: list[tuple[str, bytes]] = []
    documents_root = documents_dir.resolve()
    for entry in selected_entries:
        filename = entry.get("filename")
        if not filename:
            continue
        path = (documents_dir / str(filename)).resolve()
        if not path.is_relative_to(documents_root) or not path.exists() or path.suffix != ".md":
            continue
        title = str(entry.get("title") or path.stem)
        base_name = str(entry.get("download_base_name") or f"{source_name}_{path.stem}")
        content, filename, _media_type = _convert_markdown_document(
            md_bytes=path.read_bytes(),
            base_name=base_name,
            title=title,
            format=format,
        )
        files.append((f"{archive_folder_name}_documents/{filename}", content))
    return files


def _get_document_entry(
    outputs_path: Path,
    document_id: str,
    source_name: str | None = None,
) -> dict | None:
    entries = _get_document_entries(outputs_path, source_name=source_name)
    for entry in entries:
        if entry.get("document_id") == document_id:
            return entry
    return None


def _get_document_entries(outputs_path: Path, source_name: str | None = None) -> list[dict]:
    index_path = outputs_path / "documents_index.json"
    if not index_path.exists():
        return []

    data = json.loads(index_path.read_text(encoding="utf-8"))
    documents = data.get("documents", []) if isinstance(data, dict) else data
    counters: dict[str, int] = {}
    entries: list[dict] = []
    for raw_entry in documents:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        filename = entry.get("filename")
        if not filename and entry.get("file"):
            filename = Path(str(entry["file"])).name
        if not filename:
            continue
        entry["filename"] = filename
        if source_name:
            entry["download_base_name"] = _document_download_base_name(
                source_name=source_name,
                entry=entry,
                fallback_stem=Path(str(filename)).stem,
                counters=counters,
            )
        entries.append(entry)
    return entries


def _document_download_base_name(
    source_name: str,
    entry: dict,
    fallback_stem: str,
    counters: dict[str, int],
) -> str:
    document_id = str(entry.get("document_id") or "")
    kind = str(entry.get("kind") or "").strip().lower()

    if document_id == "main" or kind == "main" or fallback_stem == "main":
        suffix = "main"
    else:
        prefix = _document_kind_prefix(
            kind=kind,
            document_id=document_id,
            fallback_stem=fallback_stem,
        )
        counters[prefix] = counters.get(prefix, 0) + 1
        suffix = f"{prefix}{counters[prefix]:02d}"

    return f"{source_name}_{suffix}"


def _document_kind_prefix(kind: str, document_id: str, fallback_stem: str) -> str:
    if kind == "form" or document_id.startswith("form") or fallback_stem.startswith("form"):
        return "form"
    if "figure" in kind or document_id.startswith("figure") or fallback_stem.startswith("figure"):
        return "figure"
    if "table" in kind or document_id.startswith("table") or fallback_stem.startswith("table"):
        return "table"
    if kind:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in kind).strip("_")
        if cleaned:
            return cleaned
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in fallback_stem).strip("_")
    return cleaned or "document"



def _convert_markdown_document(
    md_bytes: bytes,
    base_name: str,
    title: str,
    format: OutputFormat,
) -> tuple[bytes, str, str]:
    md_text = md_bytes.decode("utf-8")

    if format == OutputFormat.MD:
        return md_bytes, f"{base_name}.md", "text/markdown; charset=utf-8"

    if format == OutputFormat.TXT:
        return (
            md_to_txt(md_text).encode("utf-8"),
            f"{base_name}.txt",
            "text/plain; charset=utf-8",
        )

    if format == OutputFormat.DOCX:
        return (
            md_to_docx(md_text, title=title),
            f"{base_name}.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    if format == OutputFormat.JSON:
        content = {
            "title": title,
            "content": md_text,
            "line_count": len(md_text.split("\n")),
            "char_count": len(md_text),
        }
        return (
            json.dumps(content, ensure_ascii=False, indent=2).encode("utf-8"),
            f"{base_name}.json",
            "application/json; charset=utf-8",
        )

    return md_bytes, f"{base_name}.md", "text/markdown; charset=utf-8"


def _content_disposition(filename: str) -> str:
    ascii_fallback = "".join(ch if ord(ch) < 128 else "_" for ch in filename)
    encoded = quote(filename)
    return f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{encoded}"
