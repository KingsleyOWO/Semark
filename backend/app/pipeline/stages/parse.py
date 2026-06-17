"""
Parse stage - Document parsing using MinerU.
"""

import asyncio
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.adapters.mineru import MinerUAdapter, MinerUResult
from app.adapters.spreadsheet import parse_spreadsheet
from app.adapters.word import parse_word_document
from app.config import MinerUConfig, settings
from app.core.cache import CacheManager
from app.db.database import Database
from app.supported_files import (
    CONVERT_TO_PDF_EXTENSIONS,
    SPREADSHEET_NATIVE_EXTENSIONS,
    SUPPORTED_INPUT_EXTENSIONS,
    SUPPORTED_INPUT_EXTENSIONS_LABEL,
    WORD_NATIVE_EXTENSIONS,
)


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

        # Check cache
        if use_cache and not force:
            cache_path = await self.cache_manager.get_parse_cache(
                doc_id=doc_id,
                config=self.config,
                version=mineru_version,
            )
            if cache_path:
                # Find content_list.json in cache
                content_list = self._find_content_list(cache_path)
                return ParseStageResult(
                    success=True,
                    cache_hit=True,
                    cache_path=cache_path,
                    content_list_path=content_list,
                    stats={
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

        if source_path.suffix.lower() in WORD_NATIVE_EXTENSIONS:
            return await self._run_native_word_parse(
                doc_id=doc_id,
                source_path=source_path,
                temp_output=temp_output,
                mineru_version=mineru_version,
            )

        # Convert Office/HTML files to PDF if needed
        input_path = source_path
        converted_pdf = None
        if self._needs_conversion(source_path):
            convert_dir = run_path / "temp_convert"
            converted_pdf, error = await self._convert_office_to_pdf(source_path, convert_dir)
            if error:
                return ParseStageResult(
                    success=False,
                    error=f"Office conversion failed: {error}",
                )
            input_path = converted_pdf

        # Run MinerU
        result = await self.adapter.parse(
            input_path=input_path,
            output_dir=temp_output,
            config_override=self.config,
        )

        # Clean up converted PDF
        if converted_pdf and converted_pdf.parent.exists():
            shutil.rmtree(converted_pdf.parent, ignore_errors=True)

        if not result.success:
            return ParseStageResult(
                success=False,
                mineru_result=result,
                error=result.error,
                stats={"duration_seconds": result.duration_seconds},
            )

        # Store in cache
        cache_path = await self.cache_manager.set_parse_cache(
            doc_id=doc_id,
            config=self.config,
            source_path=temp_output,
            version=mineru_version,
        )

        # Clean up temp
        shutil.rmtree(temp_output, ignore_errors=True)

        # Find content_list in cache (not temp which was deleted)
        cached_content_list = self._find_content_list(cache_path)

        return ParseStageResult(
            success=True,
            cache_hit=False,
            cache_path=cache_path,
            content_list_path=cached_content_list,
            mineru_result=result,
            stats={
                "cache_hit": False,
                "mineru_version": mineru_version,
                "duration_seconds": result.duration_seconds,
                **result.stats,
            },
        )


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
        result = parse_word_document(source_path, temp_output)
        return await self._store_native_parse_result(
            doc_id=doc_id,
            temp_output=temp_output,
            mineru_version=mineru_version,
            parser_version="native_docx=python-docx",
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

    async def _convert_office_to_pdf(
        self,
        source_path: Path,
        output_dir: Path,
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
                str(source_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                if not error_msg:
                    error_msg = stdout.decode("utf-8", errors="replace").strip()
                return None, f"LibreOffice conversion failed: {error_msg or 'no output'}"

            # Find the generated PDF
            pdf_name = source_path.stem + ".pdf"
            pdf_path = output_dir / pdf_name

            if not pdf_path.exists():
                # Try to find any PDF in output dir
                for f in output_dir.glob("*.pdf"):
                    pdf_path = f
                    break

            if pdf_path.exists():
                return pdf_path, None
            else:
                return None, "PDF not generated after conversion"

        except FileNotFoundError:
            return None, "LibreOffice not installed. Install with: sudo apt install libreoffice"
        except Exception as e:
            return None, f"Conversion error: {str(e)}"

    def _needs_conversion(self, source_path: Path) -> bool:
        """Check if file needs conversion to PDF."""
        return source_path.suffix.lower() in CONVERT_TO_PDF_EXTENSIONS
