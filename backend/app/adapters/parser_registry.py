"""
Parser candidate registry used by the evaluation harness.

The production pipeline still uses the existing MinerU stage. This registry
keeps parser selection data in one place so benchmark jobs can compare
alternatives without hard-coding tool names across scripts.
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParserAdapterSpec:
    """Static metadata for one parser candidate."""

    parser_id: str
    display_name: str
    family: str
    license_summary: str
    python_package: str | None = None
    command_names: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    output_focus: tuple[str, ...] = ()
    strengths: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    open_source_default: bool = True
    reference_only: bool = False
    remote_endpoint_supported: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParserProbe:
    """Runtime availability result for one parser candidate."""

    parser_id: str
    display_name: str
    available: bool
    package_found: bool
    commands_found: dict[str, str | None] = field(default_factory=dict)
    remote_endpoint_supported: bool = False
    reference_only: bool = False
    license_summary: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PARSER_SPECS: tuple[ParserAdapterSpec, ...] = (
    ParserAdapterSpec(
        parser_id="mineru3_pipeline",
        display_name="MinerU 3.x pipeline",
        family="mineru",
        license_summary="MinerU Open Source License, Apache-2.0 based",
        python_package="mineru",
        command_names=("mineru", "mineru-api", "mineru-router"),
        modes=("pipeline", "auto", "ocr", "txt"),
        output_focus=("layout", "reading_order", "tables", "formulas", "debug_pdf"),
        strengths=(
            "Best continuity with the current codebase",
            "Structured content list and debug PDFs",
            "Good default for local and self-hosted workflows",
        ),
        caveats=(
            "VLM and pipeline structured outputs are not wire-compatible",
            "GPU/VLM deployment details must be captured in run manifests",
        ),
        remote_endpoint_supported=True,
    ),
    ParserAdapterSpec(
        parser_id="mineru3_vlm_http",
        display_name="MinerU 3.x VLM/hybrid HTTP",
        family="mineru",
        license_summary="MinerU Open Source License, Apache-2.0 based",
        python_package="mineru",
        command_names=("mineru", "mineru-openai-server"),
        modes=("vlm-http-client", "hybrid-http-client"),
        output_focus=("page_understanding", "tables", "figures", "charts"),
        strengths=(
            "Useful benchmark for end-to-end document VLM parsing",
            "Can use a remote OpenAI-compatible service",
        ),
        caveats=(
            "Requires separate VLM serving setup",
            "Output schema differs from pipeline backend",
        ),
        remote_endpoint_supported=True,
    ),
    ParserAdapterSpec(
        parser_id="docling_standard",
        display_name="Docling standard",
        family="docling",
        license_summary="MIT",
        python_package="docling",
        command_names=("docling",),
        modes=("standard", "vlm"),
        output_focus=("doc_model", "markdown", "json", "tables", "figures"),
        strengths=(
            "Strong unified document model",
            "Broad format support and RAG ecosystem integrations",
        ),
        caveats=(
            "Needs adapter work to preserve this project's asset index contract",
        ),
        remote_endpoint_supported=True,
    ),
    ParserAdapterSpec(
        parser_id="paddleocr_structure_v3",
        display_name="PaddleOCR PP-StructureV3 / PaddleOCR-VL",
        family="paddleocr",
        license_summary="Apache-2.0",
        python_package="paddleocr",
        command_names=("paddleocr",),
        modes=("ppstructure", "paddleocr-vl"),
        output_focus=("ocr", "cell_coordinates", "tables", "charts", "multilingual"),
        strengths=(
            "Fine-grained coordinates are valuable for evidence-first IR",
            "Small document VLM option for efficient local runs",
        ),
        caveats=(
            "May require separate Paddle runtime choices per platform",
        ),
    ),
    ParserAdapterSpec(
        parser_id="olmocr",
        display_name="olmOCR",
        family="olmocr",
        license_summary="Apache-2.0",
        python_package="olmocr",
        command_names=("olmocr",),
        modes=("markdown", "remote_server", "local_gpu"),
        output_focus=("linearized_markdown", "ocr", "reading_order"),
        strengths=(
            "Good OCR/linearized markdown baseline",
            "Supports remote OpenAI-compatible inference",
        ),
        caveats=(
            "Less suitable as the sole source for bbox-heavy asset recall",
        ),
        remote_endpoint_supported=True,
    ),
    ParserAdapterSpec(
        parser_id="marker_reference",
        display_name="Marker reference",
        family="marker",
        license_summary="GPL code and custom/commercial model terms",
        python_package="marker",
        command_names=("marker_single", "marker"),
        modes=("standard", "use_llm"),
        output_focus=("markdown", "json", "tables", "forms"),
        strengths=(
            "Useful reference point for PDF-to-markdown quality",
        ),
        caveats=(
            "Not a default dependency for permissive open-source packaging",
            "Commercial use needs separate review",
        ),
        open_source_default=False,
        reference_only=True,
        remote_endpoint_supported=True,
    ),
)


def get_parser_specs() -> tuple[ParserAdapterSpec, ...]:
    """Return all parser candidates in benchmark order."""
    return PARSER_SPECS


def probe_parser(spec: ParserAdapterSpec) -> ParserProbe:
    """Check whether a parser candidate appears available in this environment."""
    package_found = False
    if spec.python_package:
        package_found = importlib.util.find_spec(spec.python_package) is not None

    commands_found = {name: shutil.which(name) for name in spec.command_names}
    command_available = any(path for path in commands_found.values())
    available = package_found or command_available

    return ParserProbe(
        parser_id=spec.parser_id,
        display_name=spec.display_name,
        available=available,
        package_found=package_found,
        commands_found=commands_found,
        remote_endpoint_supported=spec.remote_endpoint_supported,
        reference_only=spec.reference_only,
        license_summary=spec.license_summary,
        notes=spec.notes,
    )


def probe_all_parsers() -> list[ParserProbe]:
    """Probe every parser candidate."""
    return [probe_parser(spec) for spec in PARSER_SPECS]


def parser_matrix() -> list[dict[str, Any]]:
    """Return static parser metadata as JSON-serializable dictionaries."""
    return [spec.to_dict() for spec in PARSER_SPECS]

