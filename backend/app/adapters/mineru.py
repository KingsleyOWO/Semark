"""
MinerU CLI adapter for document parsing.

Wraps the `mineru` CLI tool and handles its output.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.config import MinerUConfig, settings


@dataclass
class MinerUResult:
    """Result from MinerU parsing."""

    success: bool
    output_dir: Path
    content_list_path: Path | None = None
    middle_json_path: Path | None = None  # Span/line-level info for advanced use
    model_json_path: Path | None = None  # YOLO detection results
    markdown_path: Path | None = None
    layout_pdf_path: Path | None = None
    spans_pdf_path: Path | None = None
    images_dir: Path | None = None
    error: str | None = None
    stderr: str | None = None
    duration_seconds: float = 0
    stats: dict[str, Any] = field(default_factory=dict)


class MinerUAdapter:
    """
    Adapter for MinerU document parsing CLI.

    MinerU outputs:
    - {filename}/auto/{filename}.md - Markdown output
    - {filename}/auto/{filename}_content_list.json - Structured content
    - {filename}/auto/images/ - Extracted images
    - layout.pdf - Debug: reading order visualization
    - spans.pdf - Debug: span visualization
    """

    def __init__(self, config: MinerUConfig | None = None):
        self.config = config or MinerUConfig()
        self.cli_path = settings.mineru_cli_path

    async def parse(
        self,
        input_path: Path,
        output_dir: Path,
        config_override: MinerUConfig | None = None,
    ) -> MinerUResult:
        """
        Parse a document using MinerU CLI.

        Args:
            input_path: Path to input file (PDF, image, etc.)
            output_dir: Directory to store MinerU output
            config_override: Optional config override for this run

        Returns:
            MinerUResult with paths to output files
        """
        config = config_override or self.config
        config, api_fallback_reason = await self._config_with_available_api_url(config)

        # Build CLI arguments
        args = self._build_args(input_path, output_dir, config)

        # Build environment variables
        env = self._build_env(config)

        # Execute MinerU
        import time
        start_time = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                self.cli_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **env},
            )

            stdout, stderr = await proc.communicate()
            duration = time.time() - start_time

            if proc.returncode != 0:
                return MinerUResult(
                    success=False,
                    output_dir=output_dir,
                    error=f"MinerU exited with code {proc.returncode}",
                    stderr=stderr.decode("utf-8", errors="replace"),
                    duration_seconds=duration,
                )

            # Find output files
            result = self._find_outputs(input_path, output_dir)
            result.duration_seconds = duration
            result.stderr = stderr.decode("utf-8", errors="replace") if stderr else None
            if api_fallback_reason:
                result.stats["api_url_fallback"] = api_fallback_reason

            return result

        except FileNotFoundError:
            return MinerUResult(
                success=False,
                output_dir=output_dir,
                error=f"MinerU CLI not found: {self.cli_path}",
            )
        except Exception as e:
            return MinerUResult(
                success=False,
                output_dir=output_dir,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def _config_with_available_api_url(
        self,
        config: MinerUConfig,
    ) -> tuple[MinerUConfig, str | None]:
        """Drop api_url when the configured MinerU API is unreachable.

        MinerU can start its own local API when --api-url is omitted. A stale
        localhost api_url should not make ordinary document parsing fail.
        """
        if not config.api_url:
            return config, None

        parsed = urlparse(config.api_url)
        host = parsed.hostname
        if not host:
            return config, f"ignored invalid api_url: {config.api_url}"

        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=1.0,
            )
            writer.close()
            await writer.wait_closed()
            return config, None
        except Exception:
            fallback_config = config.model_copy(update={"api_url": None})
            return fallback_config, f"configured MinerU api_url unavailable: {config.api_url}; used local auto-start"

    def _build_args(
        self,
        input_path: Path,
        output_dir: Path,
        config: MinerUConfig,
    ) -> list[str]:
        """Build CLI arguments for MinerU."""
        args = [
            "-p", str(input_path),
            "-o", str(output_dir),
            "-m", config.method.value,
        ]

        if config.api_url:
            args.extend(["--api-url", config.api_url])

        # Backend
        if config.backend:
            args.extend(["-b", config.backend.value])

        if config.vlm_url:
            args.extend(["-u", config.vlm_url])

        # Language
        if config.lang:
            args.extend(["-l", config.lang])

        # Page range
        if config.start_page is not None:
            args.extend(["-s", str(config.start_page)])
        if config.end_page is not None:
            args.extend(["-e", str(config.end_page)])

        # Features (requires boolean value in MinerU 3.x)
        args.extend(["--table", "true" if config.table else "false"])
        args.extend(["--formula", "true" if config.formula else "false"])

        return args

    def _build_env(self, config: MinerUConfig) -> dict[str, str]:
        """Build environment variables for MinerU."""
        env = {
            # Required for PyTorch 2.9+ to load model weights
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "0",
        }

        model_source = config.model_source or settings.mineru_model_source
        if model_source:
            env["MINERU_MODEL_SOURCE"] = model_source

        api_url = config.api_url or settings.mineru_api_url
        if api_url:
            env["MINERU_API_URL"] = api_url

        vlm_url = config.vlm_url or settings.mineru_vlm_url
        if vlm_url:
            env["MINERU_VL_SERVER_URL"] = vlm_url

        vlm_model_name = config.vlm_model_name or settings.mineru_vlm_model_name
        if vlm_model_name:
            env["MINERU_VL_MODEL_NAME"] = vlm_model_name

        vlm_api_key = config.vlm_api_key or settings.mineru_vlm_api_key
        if vlm_api_key:
            env["MINERU_VL_API_KEY"] = vlm_api_key

        if config.pdf_render_timeout:
            env["MINERU_PDF_RENDER_TIMEOUT"] = str(config.pdf_render_timeout)

        if config.pdf_render_threads:
            env["MINERU_PDF_RENDER_THREADS"] = str(config.pdf_render_threads)

        if config.table_merge_enable:
            env["MINERU_TABLE_MERGE_ENABLE"] = "true"
        else:
            env["MINERU_TABLE_MERGE_ENABLE"] = "false"

        env["MINERU_TABLE_ENABLE"] = "true" if config.table else "false"
        env["MINERU_FORMULA_ENABLE"] = "true" if config.formula else "false"

        if config.processing_window_size:
            env["MINERU_PROCESSING_WINDOW_SIZE"] = str(config.processing_window_size)

        if config.api_max_concurrent_requests:
            env["MINERU_API_MAX_CONCURRENT_REQUESTS"] = str(config.api_max_concurrent_requests)

        if config.local_api_startup_timeout_seconds:
            env["MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS"] = str(
                config.local_api_startup_timeout_seconds
            )

        if config.task_result_timeout_seconds:
            env["MINERU_TASK_RESULT_TIMEOUT_SECONDS"] = str(config.task_result_timeout_seconds)

        if config.task_result_download_timeout_seconds:
            env["MINERU_TASK_RESULT_DOWNLOAD_TIMEOUT_SECONDS"] = str(
                config.task_result_download_timeout_seconds
            )

        if config.intra_op_threads:
            env["MINERU_INTRA_OP_NUM_THREADS"] = str(config.intra_op_threads)

        if config.inter_op_threads:
            env["MINERU_INTER_OP_NUM_THREADS"] = str(config.inter_op_threads)

        return env

    def _find_outputs(self, input_path: Path, output_dir: Path) -> MinerUResult:
        """Find MinerU output files."""
        # MinerU creates: output_dir/{filename}/auto/{filename}.*
        filename = input_path.stem
        method_dir = output_dir / filename / "auto"

        if not method_dir.exists():
            # Try other method directories
            for method in ["auto", "txt", "ocr"]:
                method_dir = output_dir / filename / method
                if method_dir.exists():
                    break

        if not method_dir.exists():
            return MinerUResult(
                success=False,
                output_dir=output_dir,
                error=f"MinerU output directory not found: {method_dir}",
            )

        # Find content_list.json
        content_list_path = method_dir / f"{filename}_content_list.json"
        if not content_list_path.exists():
            # Try alternative naming
            for f in method_dir.glob("*_content_list.json"):
                content_list_path = f
                break

        # Find markdown
        markdown_path = method_dir / f"{filename}.md"
        if not markdown_path.exists():
            for f in method_dir.glob("*.md"):
                markdown_path = f
                break

        # Find middle.json (span/line-level info)
        middle_json = method_dir / f"{filename}_middle.json"
        if not middle_json.exists():
            for f in method_dir.glob("*_middle.json"):
                middle_json = f
                break

        # Find model.json (YOLO detection results)
        model_json = method_dir / f"{filename}_model.json"
        if not model_json.exists():
            for f in method_dir.glob("*_model.json"):
                model_json = f
                break

        # Find debug PDFs
        layout_pdf = method_dir / "layout.pdf"
        spans_pdf = method_dir / "spans.pdf"

        # Find images directory
        images_dir = method_dir / "images"

        # Compute stats
        stats = {}
        if content_list_path and content_list_path.exists():
            try:
                content = json.loads(content_list_path.read_text(encoding="utf-8"))
                stats["block_count"] = len(content) if isinstance(content, list) else 0
                stats["has_content_list"] = True
            except Exception:
                stats["has_content_list"] = False

        # Check model.json for YOLO stats
        if model_json.exists():
            try:
                model_data = json.loads(model_json.read_text(encoding="utf-8"))
                if isinstance(model_data, list):
                    stats["has_model_json"] = True
                    stats["page_count"] = len(model_data)
                    # Count detections
                    total_dets = sum(
                        len(page.get("layout_dets", []))
                        for page in model_data
                    )
                    stats["total_layout_detections"] = total_dets
            except Exception:
                stats["has_model_json"] = False

        return MinerUResult(
            success=True,
            output_dir=output_dir,
            content_list_path=content_list_path if content_list_path.exists() else None,
            middle_json_path=middle_json if middle_json.exists() else None,
            model_json_path=model_json if model_json.exists() else None,
            markdown_path=markdown_path if markdown_path.exists() else None,
            layout_pdf_path=layout_pdf if layout_pdf.exists() else None,
            spans_pdf_path=spans_pdf if spans_pdf.exists() else None,
            images_dir=images_dir if images_dir.exists() else None,
            stats=stats,
        )

    async def check_available(self) -> tuple[bool, str]:
        """Check if MinerU CLI is available."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.cli_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode == 0:
                version = stdout.decode().strip()
                return True, version
            else:
                return False, "MinerU returned non-zero exit code"

        except FileNotFoundError:
            return False, f"MinerU CLI not found: {self.cli_path}"
        except Exception as e:
            return False, str(e)

    async def get_version(self) -> str | None:
        """Get MinerU version string."""
        available, version_or_error = await self.check_available()
        if available:
            # Parse version from output like "mineru 1.0.0" or just "1.0.0"
            version = version_or_error.strip()
            if version.startswith("mineru"):
                version = version[6:].strip()
            return version
        return None

    @staticmethod
    def parse_version(version_str: str) -> tuple[int, int, int] | None:
        """Parse version string into tuple (major, minor, patch)."""
        import re
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return None


def parse_content_list(content_list_path: Path) -> list[dict[str, Any]]:
    """
    Parse MinerU content_list.json into structured blocks.

    content_list.json format:
    [
        {
            "type": "text",
            "text": "...",
            "text_level": 0,  # 0=body, 1/2/...=heading level
            "page_idx": 0,
            "bbox": [x0, y0, x1, y1]  # 0-1000 normalized
        },
        {
            "type": "image",
            "img_path": "images/xxx.png",
            "page_idx": 0,
            "bbox": [...],
            "img_caption": "...",
            "img_footnote": "..."
        },
        {
            "type": "table",
            "table_body": "...",  # HTML or text
            "page_idx": 0,
            "bbox": [...]
        },
        ...
    ]
    """
    content = json.loads(content_list_path.read_text(encoding="utf-8"))

    if not isinstance(content, list):
        raise ValueError("content_list.json should be a list")

    return content
