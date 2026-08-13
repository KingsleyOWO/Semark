"""
Download API routes.

Provides batch download with format conversion and ZIP packaging.
"""

import io
import json
import re
import zipfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
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
    # The split main document — the same file the document viewer renders.
    MAIN_TEXT = "main_text"
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

    # One id per document, not per run: the client collapses re-processed runs
    # before sending (``dedupe_by_doc`` would too, but only AFTER validation, so
    # an over-long list is rejected before dedupe can help). 2000 leaves ~2x
    # headroom over a 1000-document corpus.
    run_ids: list[str] = Field(..., min_length=1, max_length=2000)
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
    dedupe_by_doc: bool = Field(
        default=False,
        description=(
            "Keep only the newest run per document, so re-processed documents "
            "are exported once instead of once per run."
        ),
    )
    flatten: bool = Field(
        default=False,
        description=(
            "Put every split document at the archive root instead of in a "
            "per-source folder. A 300-document export otherwise extracts to 300 "
            "folders that have to be opened one by one to reach the markdown."
        ),
    )


class DownloadManifest(BaseModel):
    """Manifest included in download ZIP."""

    created_at: str
    run_count: int
    file_types: list[str]
    format: str
    files: list[dict]


# Mapping of file types to source files. MAIN_TEXT resolves to the split main
# document first (see _get_main_document_file); the entry here is the legacy
# fallback for runs packaged when outputs/main_text.md was still written.
SOURCE_FILES = {
    FileType.SOURCE: "source.md",
    FileType.MAIN_TEXT: "main_text.md",
    FileType.QUALITY: "quality.json",
    FileType.ASSETS_INDEX: "assets_index.jsonl",
    FileType.ENRICHMENTS: "enrichments.jsonl",
}

FALLBACK_SOURCE_FILES = {
    FileType.SOURCE: "rag.md",
    # Runs processed before main_text.md existed fall back to the full text.
    FileType.MAIN_TEXT: "source.md",
}


@router.post("/download")
async def download_runs(
    request: DownloadRequest,
    db: Database = Depends(get_db),
) -> StreamingResponse:
    """
    Batch download multiple runs with format conversion.

    Returns a ZIP file containing requested outputs from all specified runs.

    - **run_ids**: List of run IDs to download (max 2000, one per document)
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
        source_name = _source_name_for(doc.source_path if doc else None, run.run_id)
        runs_with_docs.append((run, source_name))

    if request.dedupe_by_doc:
        runs_with_docs = _dedupe_runs_by_doc(runs_with_docs)

    zip_buffer = await run_in_threadpool(_build_download_zip, runs_with_docs, request)

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


def _build_download_zip(runs_with_docs: list, request: DownloadRequest) -> io.BytesIO:
    """
    Build the whole ZIP in memory and return it positioned at the start.

    Every step here is blocking — reading each artifact off disk, converting
    markdown to docx, deflating — and there is no ``await`` anywhere in it, so
    running it inline froze the entire event loop for as long as the archive
    took to assemble. Measured on a 167-document corpus: a docx download held
    the process for 14.5s, during which an unrelated ``/api/health`` request
    took 9.7s instead of its usual sub-millisecond. Callers must hand this to a
    worker thread so one big download stays one slow request instead of an
    outage for everyone else.
    """

    zip_buffer = io.BytesIO()
    manifest_files = []
    used_entry_names: set[str] = set()

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
                            flatten=request.flatten,
                        )
                        if document_files:
                            for filename, content in document_files:
                                # Only flat exports can collide: the folder form
                                # already isolates each source under its own
                                # run-suffixed directory.
                                if request.flatten:
                                    filename = _dedupe_zip_entry_name(
                                        filename, run.run_id, used_entry_names
                                    )
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
                        filename = _dedupe_zip_entry_name(filename, run.run_id, used_entry_names)
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
    return zip_buffer


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
    source_name = _source_name_for(doc.source_path if doc else None, run.run_id)
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
    if file_type == FileType.MAIN_TEXT:
        resolved = _get_main_document_file(outputs_path, format, source_name)
        if resolved is not None:
            return resolved

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


def _dedupe_runs_by_doc(runs_with_docs: list) -> list:
    """
    Keep only the newest run per document. Run IDs are ULIDs, so lexicographic
    order matches creation order.
    """
    newest_run_by_doc: dict[str, str] = {}
    for run, _source_name in runs_with_docs:
        current = newest_run_by_doc.get(run.doc_id)
        if current is None or run.run_id > current:
            newest_run_by_doc[run.doc_id] = run.run_id
    return [
        (run, source_name)
        for run, source_name in runs_with_docs
        if newest_run_by_doc[run.doc_id] == run.run_id
    ]


def _source_name_for(source_path: str | None, run_id: str) -> str:
    """Filename-safe display name; uploads can carry edge whitespace in the stem."""
    if source_path:
        stem = Path(source_path).stem.strip()
        if stem:
            return stem
    return run_id[:12]


def _dedupe_zip_entry_name(filename: str, run_id: str, used_names: set[str]) -> str:
    """
    Batch zips store non-document file types — and, when ``flatten`` is set,
    split documents too — flat at the archive root, so two runs of the same
    source would otherwise overwrite each other on extraction.
    """
    if filename not in used_names:
        used_names.add(filename)
        return filename

    stem, dot, suffix = filename.rpartition(".")
    if not dot:
        stem, suffix = filename, ""
    candidate = f"{stem}_{run_id[:12]}{dot}{suffix}"
    counter = 2
    while candidate in used_names:
        candidate = f"{stem}_{run_id[:12]}_{counter}{dot}{suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def _get_main_document_file(
    outputs_path: Path,
    format: OutputFormat,
    source_name: str,
) -> tuple[bytes, str] | None:
    """
    Resolve the 主文 download to the split main document.

    The document viewer renders documents/main.md, so the download must serve
    the same file; outputs/main_text.md is only a legacy fallback handled by
    the caller.
    """
    documents_dir = (outputs_path / "documents").resolve()
    entries = _get_document_entries(outputs_path, source_name=source_name)
    entry = next(
        (item for item in entries if item.get("document_id") == "main"),
        None,
    ) or next(
        (item for item in entries if str(item.get("kind") or "") == "main"),
        None,
    )

    if entry is not None:
        path = (outputs_path / "documents" / str(entry["filename"])).resolve()
        title = str(entry.get("title") or Path(str(entry["filename"])).stem)
        base_name = str(entry.get("download_base_name") or f"{source_name}_main")
    else:
        path = (outputs_path / "documents" / "main.md").resolve()
        title = source_name
        base_name = f"{source_name}_main"

    if not path.is_relative_to(documents_dir) or not path.exists():
        return None

    content, download_name, _media_type = _convert_markdown_document(
        md_bytes=path.read_bytes(),
        base_name=base_name,
        title=title,
        format=format,
    )
    return content, download_name


def _get_document_files(
    outputs_path: Path,
    source_name: str,
    format: OutputFormat = OutputFormat.MD,
    document_ids: list[str] | None = None,
    archive_folder_name: str | None = None,
    flatten: bool = False,
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
        # Flat exports drop the per-source folder. The file name already starts
        # with the source name (``報告_main.md``, ``報告_table01.md``), so the
        # folder only bought protection against two sources sharing a stem —
        # which the caller's dedupe pass handles instead.
        files.append((filename if flatten else f"{archive_folder_name}_documents/{filename}", content))
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


# Characters a ZIP entry must not carry: they are path separators or reserved
# on Windows. The full-width forms (：？！) are legal and common in the Chinese
# titles this reads from, so only the ASCII set is stripped.
_ILLEGAL_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

# Titles longer than this are descriptions, not names. Split-asset titles fall
# back to the VLM's semantic caption when the source carries no caption, which
# produces a whole sentence ("圖片顯示「作者」二字，以白色字體呈現於…") — true,
# useful for retrieval, useless as a file name.
_ASSET_TITLE_MAX_LEN = 40
_MAIN_TITLE_MAX_LEN = 60

# Openings that mark a VLM description even when it fits the length limit.
_DESCRIPTION_OPENERS = (
    "圖片顯示",
    "畫面顯示",
    "此圖",
    "本圖",
    "該圖",
    "這張圖",
    "圖中",
    "此表",
    "本表",
    "該表",
    "此為",
    "the image",
    "this figure",
    "this image",
    "this table",
    "the figure",
)

# "Figure 1" / "表 3" / "主文" — a title that only restates the kind prefix
# standing right in front of it. Repeating it makes the name longer and says
# nothing new.
_PLACEHOLDER_TITLE = re.compile(
    r"^(figure|fig|table|tbl|form|image|document|main|untitled"
    r"|圖|圖片|圖表|表|表格|表單|附件|文件|主文|本文|正文)\s*\d*$",
    re.IGNORECASE,
)


def _clean_filename_fragment(title: str, max_len: int) -> str:
    """Reduce a document title to something safe to paste into a file name."""

    cleaned = _ILLEGAL_FILENAME_CHARS.sub(" ", title)
    cleaned = " ".join(cleaned.split())
    # Trailing dots and spaces are silently dropped by Windows, which would turn
    # two distinct names into one on extraction.
    cleaned = cleaned[:max_len].strip().rstrip(". ")
    return cleaned


def _filename_title_fragment(title: str, is_main: bool, source_name: str = "") -> str:
    """
    The readable tail of a download file name, or "" when the title cannot serve.

    Measured over the 2026-08 corpus (141 documents, 301 split assets): every
    main title was a real article title — none empty, numeric, or duplicated.
    Asset titles split almost evenly between real captions (48%) and VLM
    descriptions (48%), with 5% bare "Figure N" placeholders. Appending all of
    them would name half the files after a sentence, so assets have to earn it;
    the ones that do not keep exactly the name they have today.
    """

    stripped = title.strip()
    if not stripped:
        return ""
    if _PLACEHOLDER_TITLE.match(stripped):
        return ""

    if is_main:
        fragment = _clean_filename_fragment(stripped, _MAIN_TITLE_MAX_LEN)
    else:
        if len(stripped) > _ASSET_TITLE_MAX_LEN:
            return ""
        lowered = stripped.lower()
        if any(lowered.startswith(opener) for opener in _DESCRIPTION_OPENERS):
            return ""
        fragment = _clean_filename_fragment(stripped, _ASSET_TITLE_MAX_LEN)

    # That corpus uploaded numeric filenames, so title and source name never
    # met. Elsewhere the upload is named after the article it holds, and
    # appending the title verbatim yields 使用說明_main_使用說明.md.
    if fragment and _folded(fragment) in _folded(source_name):
        return ""
    return fragment


def _folded(value: str) -> str:
    """Casefold and drop whitespace, so near-identical names compare equal."""
    return "".join(value.split()).casefold()


def _document_download_base_name(
    source_name: str,
    entry: dict,
    fallback_stem: str,
    counters: dict[str, int],
) -> str:
    document_id = str(entry.get("document_id") or "")
    kind = str(entry.get("kind") or "").strip().lower()

    is_main = document_id == "main" or kind == "main" or fallback_stem == "main"
    if is_main:
        suffix = "main"
    else:
        prefix = _document_kind_prefix(
            kind=kind,
            document_id=document_id,
            fallback_stem=fallback_stem,
        )
        counters[prefix] = counters.get(prefix, 0) + 1
        suffix = f"{prefix}{counters[prefix]:02d}"

    # The kind prefix stays in front of the title: it is what keeps the files
    # sorted, parseable, and collision-free. The title is an added tail, never a
    # replacement.
    fragment = _filename_title_fragment(
        str(entry.get("title") or ""), is_main=is_main, source_name=source_name
    )
    if fragment:
        return f"{source_name}_{suffix}_{fragment}"
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
