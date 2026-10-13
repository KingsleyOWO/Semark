"""
Parse stage - Document parsing using MinerU.
"""

import asyncio
import json
import os
import shutil
import signal
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageChops

from app.adapters.metafile import drawable_word_copy
from app.adapters.mineru import MinerUAdapter, MinerUResult
from app.adapters.spreadsheet import parse_spreadsheet
from app.adapters.word import WordParseResult, parse_word_document
from app.config import MinerUConfig, settings
from app.core.cache import CacheManager
from app.db.database import Database
from app.pipeline.delivery import atomic_write_json
from app.supported_files import (
    CONVERT_TO_PDF_EXTENSIONS,
    SPREADSHEET_NATIVE_EXTENSIONS,
    SUPPORTED_INPUT_EXTENSIONS,
    SUPPORTED_INPUT_EXTENSIONS_LABEL,
    WORD_NATIVE_EXTENSIONS,
)

WORD_PARSER_VERSION = "native_docx=python-docx;merged_cells=v2;image_types=v2;pdf_fallback=v1;content_controls=v1;table_images=v1;word_provenance=v1;caption_context=v2;table_headers=display_context_v2;auto_numbering=v1;text_boxes=v1;metafile_pictures=v1;tracked_changes=accepted_v1"
PARSE_REPORT_FILENAME = "parse_report.json"
OFFICE_CONVERSION_TIMEOUT_SECONDS = 300
OFFICE_CLEANUP_TIMEOUT_SECONDS = 5
# A drawn Word metafile is exported as LibreOffice draws it: 288 dpi keeps a small
# logo's letters, and an A4 page still stays under nine megapixels before trimming.
METAFILE_ZOOM = 4
METAFILE_MARGIN = 16


@dataclass
class ParseStageResult:
    """Result from parse stage."""

    success: bool
    cache_hit: bool = False
    cache_path: Path | None = None
    content_list_path: Path | None = None
    mineru_result: MinerUResult | None = None
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


class ParseStage:
    """
    Parse stage - converts documents to MinerU structured output.

    Input: Original document file
    Output: MinerU output including content_list.json
    """

    def __init__(
        self,
        db: Database,
        config: MinerUConfig | None = None,
    ):
        self.db = db
        self.config = config or MinerUConfig()
        self.adapter = MinerUAdapter(self.config)
        self.cache_manager = CacheManager(db)

    async def run(
        self,
        doc_id: str,
        run_id: str,
        use_cache: bool = True,
        force: bool = False,
    ) -> ParseStageResult:
        """
        Run parse stage for a document.

        Args:
            doc_id: Document ID
            run_id: Run ID
            use_cache: Whether to use cached results
            force: Force re-parse even if cache exists

        Returns:
            ParseStageResult with paths to output
        """
        # Get source file
        source_path = self._get_source_path(doc_id)
        if not source_path:
            return ParseStageResult(
                success=False,
                error=f"Source file not found for doc_id: {doc_id}",
            )

        # Get MinerU version for cache key
        mineru_version = await self.adapter.get_version()

        is_word = source_path.suffix.lower() in WORD_NATIVE_EXTENSIONS
        cache_version = f"{mineru_version};{WORD_PARSER_VERSION}" if is_word else mineru_version

        # Check cache
        if use_cache and not force:
            cache_path = await self.cache_manager.get_parse_cache(
                doc_id=doc_id,
                config=self.config,
                version=cache_version,
            )
            if cache_path:
                # Find content_list.json in cache
                content_list = self._find_content_list(cache_path)
                report_path = cache_path / PARSE_REPORT_FILENAME
                cached_stats = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
                if content_list is not None:
                    return ParseStageResult(
                        success=True,
                        cache_hit=True,
                        cache_path=cache_path,
                        content_list_path=content_list,
                        stats={
                            **cached_stats,
                            "cache_hit": True,
                            "mineru_version": mineru_version,
                        },
                    )

        # Prepare temp directories
        run_path = settings.get_run_path(doc_id, run_id)
        temp_output = run_path / "temp_parse"
        temp_output.mkdir(parents=True, exist_ok=True)

        if source_path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
            return ParseStageResult(
                success=False,
                error=(
                    "Unsupported file type. "
                    f"Supported extensions: {SUPPORTED_INPUT_EXTENSIONS_LABEL}"
                ),
            )

        if source_path.suffix.lower() in SPREADSHEET_NATIVE_EXTENSIONS:
            return await self._run_native_spreadsheet_parse(
                doc_id=doc_id,
                source_path=source_path,
                temp_output=temp_output,
                mineru_version=mineru_version,
            )

        fallback_stats: dict[str, Any] = {}
        if is_word:
            native_result = await self._run_native_word_parse(
                doc_id=doc_id,
                source_path=source_path,
                temp_output=temp_output,
                mineru_version=mineru_version,
            )
            if native_result.success:
                return native_result
            fallback_stats = {
                "parser": "mineru",
                "pdf_fallback": True,
                "native_word_error": native_result.error,
                "conversion_warnings": [],
            }
            # Failed native parsing may have left partial output. Start the
            # PDF attempt cleanly so cache discovery cannot select that output.
            shutil.rmtree(temp_output, ignore_errors=True)
            temp_output.mkdir(parents=True, exist_ok=True)

        def failure(message: str, mineru_result: MinerUResult | None = None) -> ParseStageResult:
            if fallback_stats:
                message = f"{fallback_stats['native_word_error']}; PDF fallback failed: {message}"
            return ParseStageResult(
                success=False, error=message, mineru_result=mineru_result,
                stats={
                    **({"duration_seconds": mineru_result.duration_seconds} if mineru_result else {}),
                    **fallback_stats,
                },
            )

        input_path = source_path
        converted_pdf = None
        convert_dir = run_path / "temp_convert"
        try:
            if fallback_stats or self._needs_conversion(source_path):
                warnings = fallback_stats.get("conversion_warnings")
                converted_pdf, error = await self._convert_office_to_pdf(
                    source_path, convert_dir, warnings=warnings,
                )
                if error or converted_pdf is None:
                    return failure(f"Office conversion failed: {error or 'PDF not generated'}")
                input_path = converted_pdf

            result = await self.adapter.parse(
                input_path=input_path,
                output_dir=temp_output,
                config_override=self.config,
            )
            if not result.success:
                return failure(result.error or "MinerU failed", result)

            if fallback_stats:
                content_list = result.content_list_path
                if content_list is None or not content_list.is_file():
                    return failure("MinerU produced no content list", result)
                try:
                    items = json.loads(content_list.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    return failure(f"Invalid PDF content list: {exc}", result)
                if not isinstance(items, list) or not items:
                    return failure("PDF parser produced no content", result)
                # Keep the actual converted PDF beside the parse output for
                # preview rendering and text supplementation, including retries.
                shutil.copy2(converted_pdf, content_list.parent / "original_origin.pdf")

            stats = {
                "cache_hit": False,
                "mineru_version": mineru_version,
                "duration_seconds": result.duration_seconds,
                **result.stats,
                **fallback_stats,
            }
            if fallback_stats:
                atomic_write_json(temp_output / PARSE_REPORT_FILENAME, stats, ensure_ascii=False, indent=2)
            cache_path = await self.cache_manager.set_parse_cache(
                doc_id=doc_id,
                config=self.config,
                source_path=temp_output,
                version=cache_version,
            )
            return ParseStageResult(
                success=True,
                cache_path=cache_path,
                content_list_path=self._find_content_list(cache_path),
                mineru_result=result,
                stats=stats,
            )
        finally:
            shutil.rmtree(convert_dir, ignore_errors=True)
            shutil.rmtree(temp_output, ignore_errors=True)


    async def _run_native_spreadsheet_parse(
        self,
        doc_id: str,
        source_path: Path,
        temp_output: Path,
        mineru_version: str,
    ) -> ParseStageResult:
        """Parse XLSX with native spreadsheet structure instead of PDF conversion."""
        result = parse_spreadsheet(source_path, temp_output)
        return await self._store_native_parse_result(
            doc_id=doc_id,
            temp_output=temp_output,
            mineru_version=mineru_version,
            parser_version="native_spreadsheet=v2",
            result=result,
        )

    async def _run_native_word_parse(
        self,
        doc_id: str,
        source_path: Path,
        temp_output: Path,
        mineru_version: str,
    ) -> ParseStageResult:
        """Parse DOCX with native Word structure instead of PDF conversion."""
        try:
            result = parse_word_document(source_path, temp_output)
            if result.success and result.content_list_path:
                result.stats["metafiles_rendered"] = await render_word_metafiles(
                    result.content_list_path, result.markdown_path)
        except Exception as exc:
            result = WordParseResult(
                success=False, output_dir=temp_output,
                error=f"Word parse failed: {exc}",
            )
        return await self._store_native_parse_result(
            doc_id=doc_id,
            temp_output=temp_output,
            mineru_version=mineru_version,
            parser_version=WORD_PARSER_VERSION,
            result=result,
        )

    async def _store_native_parse_result(
        self,
        doc_id: str,
        temp_output: Path,
        mineru_version: str,
        parser_version: str,
        result: Any,
    ) -> ParseStageResult:
        if not result.success:
            shutil.rmtree(temp_output, ignore_errors=True)
            return ParseStageResult(
                success=False,
                error=result.error,
                stats=result.stats,
            )

        version = f"{mineru_version};{parser_version}"
        if parser_version == WORD_PARSER_VERSION:
            atomic_write_json(temp_output / PARSE_REPORT_FILENAME, result.stats, ensure_ascii=False, indent=2)
        cache_path = await self.cache_manager.set_parse_cache(
            doc_id=doc_id,
            config=self.config,
            source_path=temp_output,
            version=version,
        )
        shutil.rmtree(temp_output, ignore_errors=True)
        cached_content_list = self._find_content_list(cache_path)

        return ParseStageResult(
            success=True,
            cache_hit=False,
            cache_path=cache_path,
            content_list_path=cached_content_list,
            stats={
                "cache_hit": False,
                "mineru_version": mineru_version,
                **result.stats,
            },
        )

    def _get_source_path(self, doc_id: str) -> Path | None:
        """Get the source file path for a document."""
        source_dir = settings.get_doc_path(doc_id) / "source"
        if not source_dir.exists():
            return None

        # Find original file
        for f in source_dir.glob("original.*"):
            return f

        return None

    def _find_content_list(self, cache_path: Path) -> Path | None:
        """Find content_list.json in cache directory."""
        # Search recursively
        for f in cache_path.rglob("*_content_list.json"):
            return f
        return None

    @staticmethod
    async def _convert_office_to_pdf(
        source_path: Path,
        output_dir: Path,
        *,
        warnings: list[str] | None = None,
    ) -> tuple[Path | None, str | None]:
        """
        Convert Office/HTML document to PDF using LibreOffice.

        Args:
            source_path: Path to Office/HTML document
            output_dir: Directory to save converted PDF

        Returns:
            Tuple of (pdf_path, error_message)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_dir = output_dir / "lo_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"{source_path.stem}.pdf"
        pdf_path.unlink(missing_ok=True)
        # LibreOffice draws an EMF+ only picture blank; it converts a copy holding the bitmap.
        converted = source_path
        if source_path.suffix.lower() == ".docx":
            try:
                converted = drawable_word_copy(source_path, output_dir / "drawable") or source_path
            except (zipfile.BadZipFile, OSError):
                converted = source_path

        try:
            # Use an isolated LibreOffice profile so parallel conversions do not
            # contend on the host user's profile lock.
            proc = await asyncio.create_subprocess_exec(
                "libreoffice",
                "--headless",
                "--norestore",
                "--nolockcheck",
                f"-env:UserInstallation={profile_dir.resolve().as_uri()}",
                "--convert-to", "pdf",
                "--outdir", str(output_dir),
                str(converted),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=OFFICE_CONVERSION_TIMEOUT_SECONDS,
                )
            except (TimeoutError, asyncio.CancelledError) as exc:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.communicate(), timeout=OFFICE_CLEANUP_TIMEOUT_SECONDS)
                except TimeoutError:
                    # A descendant or canceled pipe reader can keep communicate()
                    # waiting even after the process group has been killed.
                    # Close pipe transports as well; cleanup must remain bounded.
                    transport = getattr(proc, "_transport", None)
                    if transport is not None:
                        transport.close()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return None, f"LibreOffice conversion timed out after {OFFICE_CONVERSION_TIMEOUT_SECONDS} seconds"

            if proc.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                if not error_msg:
                    error_msg = stdout.decode("utf-8", errors="replace").strip()
                return None, f"LibreOffice conversion failed: {error_msg or 'no output'}"

            if pdf_path.is_file() and pdf_path.stat().st_size > 0:
                if warnings is not None:
                    warnings.extend(
                        line.strip() for line in stderr.decode("utf-8", errors="replace").splitlines()
                        if line.strip() and not line.startswith("Warning: failed to launch javaldx")
                    )
                return pdf_path, None
            return None, "PDF not generated after conversion"

        except FileNotFoundError:
            return None, "LibreOffice not installed. Install with: sudo apt install libreoffice"
        except Exception as e:
            return None, f"Conversion error: {str(e)}"

    def _needs_conversion(self, source_path: Path) -> bool:
        """Check if file needs conversion to PDF."""
        return source_path.suffix.lower() in CONVERT_TO_PDF_EXTENSIONS


async def _render_metafile(path: Path) -> Path | None:
    """PNG of what LibreOffice draws for a metafile, without the page around it; None if it draws nothing."""
    folder = path.parent / (path.stem + "-render")
    try:
        pdf, error = await ParseStage._convert_office_to_pdf(path, folder)
        if error or pdf is None:
            return None
        with fitz.open(pdf) as document:
            if len(document) != 1:
                return None
            pixmap = document[0].get_pixmap(matrix=fitz.Matrix(METAFILE_ZOOM, METAFILE_ZOOM))
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        # LibreOffice places the drawing on a white page; keep what differs from it.
        box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).point(
            lambda v: 255 if v > 8 else 0).getbbox()
        if box is None:
            return None
        target = path.with_suffix(".png")
        image.crop((max(0, box[0] - METAFILE_MARGIN), max(0, box[1] - METAFILE_MARGIN),
                    min(image.width, box[2] + METAFILE_MARGIN), min(image.height, box[3] + METAFILE_MARGIN))).save(target)
        return target
    except Exception:
        return None
    finally:
        shutil.rmtree(folder, ignore_errors=True)


async def render_word_metafiles(content_list_path: Path, markdown_path: Path | None) -> int:
    """Export each WMF or drawn EMF picture as LibreOffice draws it; the vision model cannot load a metafile.

    The picture keeps its Word identity (rel_id, image_sha256); only the exported file
    changes. A metafile LibreOffice draws nothing for is left as it was.
    """
    items = json.loads(content_list_path.read_text(encoding="utf-8"))
    rendered: dict[str, str | None] = {}
    for item in items:
        source = item.get("img_path") if item.get("type") == "image" else None
        if not source or Path(source).suffix.lower() not in {".wmf", ".emf"}:
            continue
        if source not in rendered:
            png = await _render_metafile(content_list_path.parent / source)
            rendered[source] = str(Path(source).with_suffix(".png")) if png else None
        if rendered[source]:
            item["img_path"] = rendered[source]
            item.setdefault("metadata", {})["rendered_from"] = Path(source).suffix.lower().lstrip(".")
    done = {source: png for source, png in rendered.items() if png}
    if done:
        content_list_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        if markdown_path and markdown_path.is_file():
            text = markdown_path.read_text(encoding="utf-8")
            for source, png in done.items():
                text = text.replace(f"]({source})", f"]({png})")
            markdown_path.write_text(text, encoding="utf-8")
    return len(done)
