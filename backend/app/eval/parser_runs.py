"""Materialize parser smoke outputs into benchmarkable run artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from app.eval.parser_smoke import local_sample_sources
from app.eval.validation import validate_asset_entry, validate_source_map
from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.stages.normalize import NormalizeStage, save_document_ir
from app.pipeline.stages.package import PackageStage


@dataclass(frozen=True)
class ParserRunResult:
    parser_id: str
    sample_id: str
    ok: bool
    run_dir: str
    content_list_path: str | None = None
    document_ir_path: str | None = None
    manifest_path: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_id": self.parser_id,
            "sample_id": self.sample_id,
            "ok": self.ok,
            "run_dir": self.run_dir,
            "content_list_path": self.content_list_path,
            "document_ir_path": self.document_ir_path,
            "manifest_path": self.manifest_path,
            "outputs": self.outputs,
            "stats": self.stats,
            "error": self.error,
        }


def materialize_parser_run(
    workspace: Path,
    output_dir: Path,
    corpus_path: Path,
    sample_id: str,
    parser_id: str = "mineru3_pipeline",
) -> ParserRunResult:
    """Convert a successful parser smoke output into comparable run artifacts."""
    if parser_id == "mineru3_pipeline":
        return asyncio.run(
            _materialize_mineru_pipeline_run(
                workspace=workspace,
                output_dir=output_dir,
                corpus_path=corpus_path,
                sample_id=sample_id,
                parser_id=parser_id,
            )
        )
    if parser_id == "paddleocr_structure_v3":
        return asyncio.run(
            _materialize_paddleocr_run(
                workspace=workspace,
                output_dir=output_dir,
                corpus_path=corpus_path,
                sample_id=sample_id,
                parser_id=parser_id,
            )
        )
    return ParserRunResult(
        parser_id=parser_id,
        sample_id=sample_id,
        ok=False,
        run_dir=str(output_dir / "parser_runs" / parser_id / sample_id),
        error="parser-run materialization currently supports mineru3_pipeline and paddleocr_structure_v3 only",
    )


async def _materialize_paddleocr_run(
    workspace: Path,
    output_dir: Path,
    corpus_path: Path,
    sample_id: str,
    parser_id: str,
) -> ParserRunResult:
    run_dir = output_dir / "parser_runs" / parser_id / sample_id
    try:
        sample = _find_local_sample(workspace, corpus_path, sample_id)
        if not sample:
            return ParserRunResult(
                parser_id=parser_id,
                sample_id=sample_id,
                ok=False,
                run_dir=str(run_dir),
                error=f"local sample not found or source missing: {sample_id}",
            )

        source_path = Path(str(sample["source_path"]))
        result_json_path = _find_paddleocr_result_json(run_dir)
        if not result_json_path:
            return ParserRunResult(
                parser_id=parser_id,
                sample_id=sample_id,
                ok=False,
                run_dir=str(run_dir),
                error="PaddleOCR result JSON not found; run parser-smoke first",
            )

        paddle_result = json.loads(result_json_path.read_text(encoding="utf-8"))
        run_id = f"{parser_id}_{sample_id}"
        document_ir = _paddleocr_document_ir(
            doc_id=str(sample["doc_id"]),
            run_id=run_id,
            source_info=_source_info(source_path),
            result=paddle_result,
            output_dir=run_dir,
            output_dir_root=output_dir,
        )
        document_ir_path = save_document_ir(document_ir, run_dir)
        package_result = await PackageStage().run(
            doc_id=str(sample["doc_id"]),
            run_id=run_id,
            document_ir=document_ir,
            run_path=run_dir,
            parse_cache_path=run_dir,
            config_hash="dgx-benchmark",
        )
        if not package_result.success:
            return ParserRunResult(
                parser_id=parser_id,
                sample_id=sample_id,
                ok=False,
                run_dir=str(run_dir),
                content_list_path=str(result_json_path),
                document_ir_path=str(document_ir_path),
                error=f"package failed: {package_result.error}",
            )

        manifest_path = package_result.manifest_path
        if manifest_path:
            _augment_manifest(
                manifest_path=manifest_path,
                parser_id=parser_id,
                sample_id=sample_id,
                source_path=source_path,
                content_list_path=result_json_path,
                output_dir=output_dir,
                materialization="paddleocr_pp_structure_v3_output",
                package_name="paddleocr",
                mode="pp_structurev3",
            )

        outputs = {
            "dataset_md": str(package_result.dataset_md_path),
            "rag_md": str(package_result.rag_md_path),
            "assets_index": str(package_result.assets_index_path),
            "source_map": str(package_result.source_map_path),
            "quality": str(package_result.quality_path),
        }
        validation_errors = _validate_packaged_outputs(
            assets_index_path=package_result.assets_index_path,
            source_map_path=package_result.source_map_path,
        )
        stats = {
            "normalize": {
                "block_count": len(document_ir.blocks),
                "page_count": len(document_ir.pages),
                "by_type": document_ir.count_by_type(),
                "pages_rendered": False,
                "text_supplemented": 0,
            },
            "package": package_result.stats,
            "validation_error_count": len(validation_errors),
            "validation_errors": validation_errors,
        }
        return ParserRunResult(
            parser_id=parser_id,
            sample_id=sample_id,
            ok=not validation_errors,
            run_dir=str(run_dir),
            content_list_path=str(result_json_path),
            document_ir_path=str(document_ir_path),
            manifest_path=str(manifest_path) if manifest_path else None,
            outputs=outputs,
            stats=stats,
        )
    except Exception as exc:
        return ParserRunResult(
            parser_id=parser_id,
            sample_id=sample_id,
            ok=False,
            run_dir=str(run_dir),
            error=str(exc),
        )


async def _materialize_mineru_pipeline_run(
    workspace: Path,
    output_dir: Path,
    corpus_path: Path,
    sample_id: str,
    parser_id: str,
) -> ParserRunResult:
    run_dir = output_dir / "parser_runs" / parser_id / sample_id
    try:
        sample = _find_local_sample(workspace, corpus_path, sample_id)
        if not sample:
            return ParserRunResult(
                parser_id=parser_id,
                sample_id=sample_id,
                ok=False,
                run_dir=str(run_dir),
                error=f"local sample not found or source missing: {sample_id}",
            )

        source_path = Path(str(sample["source_path"]))
        content_list_path = _find_mineru_content_list(run_dir)
        if not content_list_path:
            return ParserRunResult(
                parser_id=parser_id,
                sample_id=sample_id,
                ok=False,
                run_dir=str(run_dir),
                error="MinerU content list not found; run parser-smoke first",
            )

        source_info = _source_info(source_path)
        run_id = f"{parser_id}_{sample_id}"
        normalize_result = await NormalizeStage().run(
            doc_id=str(sample["doc_id"]),
            run_id=run_id,
            content_list_path=content_list_path,
            source_info=source_info,
            render_pages=False,
            mineru_version=_package_version("mineru"),
        )
        if not normalize_result.success or not normalize_result.document_ir:
            return ParserRunResult(
                parser_id=parser_id,
                sample_id=sample_id,
                ok=False,
                run_dir=str(run_dir),
                content_list_path=str(content_list_path),
                error=f"normalize failed: {normalize_result.error}",
            )

        document_ir_path = save_document_ir(normalize_result.document_ir, run_dir)
        package_result = await PackageStage().run(
            doc_id=str(sample["doc_id"]),
            run_id=run_id,
            document_ir=normalize_result.document_ir,
            run_path=run_dir,
            parse_cache_path=content_list_path.parent,
            config_hash="dgx-benchmark",
        )
        if not package_result.success:
            return ParserRunResult(
                parser_id=parser_id,
                sample_id=sample_id,
                ok=False,
                run_dir=str(run_dir),
                content_list_path=str(content_list_path),
                document_ir_path=str(document_ir_path),
                error=f"package failed: {package_result.error}",
            )

        manifest_path = package_result.manifest_path
        if manifest_path:
            _augment_manifest(
                manifest_path=manifest_path,
                parser_id=parser_id,
                sample_id=sample_id,
                source_path=source_path,
                content_list_path=content_list_path,
                output_dir=output_dir,
            )

        outputs = {
            "dataset_md": str(package_result.dataset_md_path),
            "rag_md": str(package_result.rag_md_path),
            "assets_index": str(package_result.assets_index_path),
            "source_map": str(package_result.source_map_path),
            "quality": str(package_result.quality_path),
        }
        validation_errors = _validate_packaged_outputs(
            assets_index_path=package_result.assets_index_path,
            source_map_path=package_result.source_map_path,
        )
        stats = {
            "normalize": normalize_result.stats,
            "package": package_result.stats,
            "validation_error_count": len(validation_errors),
            "validation_errors": validation_errors,
        }
        return ParserRunResult(
            parser_id=parser_id,
            sample_id=sample_id,
            ok=not validation_errors,
            run_dir=str(run_dir),
            content_list_path=str(content_list_path),
            document_ir_path=str(document_ir_path),
            manifest_path=str(manifest_path) if manifest_path else None,
            outputs=outputs,
            stats=stats,
        )
    except Exception as exc:
        return ParserRunResult(
            parser_id=parser_id,
            sample_id=sample_id,
            ok=False,
            run_dir=str(run_dir),
            error=str(exc),
        )


def _find_local_sample(workspace: Path, corpus_path: Path, sample_id: str) -> dict[str, Any] | None:
    for sample in local_sample_sources(workspace, corpus_path):
        if sample.get("sample_id") == sample_id and sample.get("source_exists"):
            return sample
    return None


def _find_mineru_content_list(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.rglob("*_content_list.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(run_dir.rglob("content_list.json"))
    return candidates[0] if candidates else None


def _find_paddleocr_result_json(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("*_res.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(run_dir.rglob("*_res.json"))
    return candidates[0] if candidates else None


def _paddleocr_document_ir(
    doc_id: str,
    run_id: str,
    source_info: SourceInfo,
    result: dict[str, Any],
    output_dir: Path,
    output_dir_root: Path,
) -> DocumentIR:
    width = int(result.get("width") or 0)
    height = int(result.get("height") or 0)
    page_idx = int(result.get("page_index") or 0)
    blocks: list[Block] = []
    for index, item in enumerate(result.get("parsing_res_list", [])):
        block = _paddleocr_block(item, index, page_idx, width, height, output_dir)
        if block:
            blocks.append(block)

    blocks.sort(key=lambda block: (block.page_idx, block.reading_order))
    return DocumentIR(
        doc_id=doc_id,
        run_id=run_id,
        source=source_info,
        engine=EngineInfo(
            name="paddleocr",
            backend="pp_structurev3",
            version=_candidate_package_version(output_dir_root, "paddleocr_structure_v3", "paddleocr"),
            method="pp_structurev3",
            lang=None,
            table=True,
            formula=True,
        ),
        pages=[
            PageInfo(
                page_idx=page_idx,
                width_px=width or None,
                height_px=height or None,
                page_image_path=_relative_existing(output_dir, "original_0_preprocessed_img.png"),
            )
        ],
        blocks=blocks,
    )


def _paddleocr_block(
    item: dict[str, Any],
    index: int,
    fallback_page_idx: int,
    width: int,
    height: int,
    output_dir: Path,
) -> Block | None:
    label = str(item.get("block_label") or "").lower()
    content = item.get("block_content") or ""
    bbox = _normalize_bbox(item.get("block_bbox", []), width, height)
    block_id = f"p{index:06d}"
    reading_order = item.get("block_order")
    reading_order = int(reading_order) if isinstance(reading_order, int) else index

    if label == "table":
        table_html = _strip_html_shell(str(content))
        return Block(
            block_id=block_id,
            type=BlockType.TABLE,
            page_idx=fallback_page_idx,
            bbox_norm=bbox,
            reading_order=reading_order,
            payload={
                "table_body": table_html,
                "table_caption": None,
                "source_format": "paddleocr_pp_structurev3",
                "xlsx_path": _relative_existing(output_dir, "original_0_table_1.xlsx"),
                "html_path": _relative_existing(output_dir, "original_0_table_1.html"),
            },
        )

    if label in {"text", "figure_title", "title", "header", "footer"}:
        text = _plain_text(str(content))
        if not text:
            return None
        return Block(
            block_id=block_id,
            type=BlockType.TEXT,
            page_idx=fallback_page_idx,
            bbox_norm=bbox,
            reading_order=reading_order,
            payload={
                "text": text,
                "text_level": 1 if label in {"figure_title", "title"} else 0,
                "paddle_label": label,
            },
        )

    text = _plain_text(str(content))
    if text:
        return Block(
            block_id=block_id,
            type=BlockType.TEXT,
            page_idx=fallback_page_idx,
            bbox_norm=bbox,
            reading_order=reading_order,
            payload={"text": text, "text_level": 0, "paddle_label": label},
        )
    return None


def _normalize_bbox(bbox: Any, width: int, height: int) -> list[int]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return [0, 0, 0, 0]
    values = [float(value) for value in bbox]
    if width > 0 and height > 0:
        return [
            max(0, min(1000, int(values[0] / width * 1000))),
            max(0, min(1000, int(values[1] / height * 1000))),
            max(0, min(1000, int(values[2] / width * 1000))),
            max(0, min(1000, int(values[3] / height * 1000))),
        ]
    return [int(value) for value in values]


def _plain_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def _strip_html_shell(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^<html><body>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"</body></html>$", "", value, flags=re.IGNORECASE)
    return value.strip()


def _relative_existing(root: Path, filename: str) -> str | None:
    path = root / filename
    if path.exists():
        return filename
    return None


def _source_info(source_path: Path) -> SourceInfo:
    data = source_path.read_bytes()
    return SourceInfo(
        path=str(source_path),
        ext=source_path.suffix.lower().lstrip("."),
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _validate_packaged_outputs(
    assets_index_path: Path | None,
    source_map_path: Path | None,
) -> list[str]:
    errors: list[str] = []
    if assets_index_path and assets_index_path.exists():
        for idx, line in enumerate(assets_index_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"assets_index line {idx + 1}: invalid json: {exc}")
                continue
            for error in validate_asset_entry(entry):
                errors.append(f"assets_index line {idx + 1}: {error}")
    elif assets_index_path:
        errors.append(f"missing assets_index: {assets_index_path}")

    if source_map_path and source_map_path.exists():
        try:
            source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"source_map: invalid json: {exc}")
        else:
            for error in validate_source_map(source_map):
                errors.append(f"source_map: {error}")
    elif source_map_path:
        errors.append(f"missing source_map: {source_map_path}")
    return errors


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _augment_manifest(
    manifest_path: Path,
    parser_id: str,
    sample_id: str,
    source_path: Path,
    content_list_path: Path,
    output_dir: Path,
    materialization: str = "mineru_cli_smoke_output",
    package_name: str = "mineru",
    mode: str = "pipeline",
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    smoke = _smoke_entry(output_dir, parser_id, sample_id)
    manifest["benchmark"] = {
        "parser_id": parser_id,
        "sample_id": sample_id,
        "source_path": str(source_path),
        "content_list_path": str(content_list_path),
        "materialization": materialization,
        "smoke": smoke,
    }
    manifest.setdefault("engines", {})["parser_candidate"] = {
        "parser_id": parser_id,
        "package": package_name,
        "version": _candidate_package_version(output_dir, parser_id, package_name),
        "mode": mode,
        "endpoint": "local_cli",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_package_version(output_dir: Path, parser_id: str, package_name: str) -> str | None:
    inventory_path = output_dir / "candidate_reports" / parser_id / "environment_inventory.json"
    if inventory_path.exists():
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except Exception:
            inventory = {}
        version = inventory.get("python_packages", {}).get(package_name)
        if version:
            return str(version)
    return _package_version(package_name)


def _smoke_entry(output_dir: Path, parser_id: str, sample_id: str) -> dict[str, Any]:
    smoke_path = output_dir / "parser_smoke.json"
    if not smoke_path.exists():
        return {}
    report = json.loads(smoke_path.read_text(encoding="utf-8"))
    for command in report.get("commands", []):
        if command.get("parser_id") == parser_id and command.get("sample_id") == sample_id:
            return {
                "ok": command.get("ok"),
                "duration_seconds": command.get("duration_seconds"),
                "returncode": command.get("returncode"),
                "argv": command.get("argv", []),
            }
    return {}


def write_parser_run_reports(
    workspace: Path,
    output_dir: Path,
    corpus_path: Path,
    sample_id: str,
    parser_id: str = "mineru3_pipeline",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = materialize_parser_run(
        workspace=workspace,
        output_dir=output_dir,
        corpus_path=corpus_path,
        sample_id=sample_id,
        parser_id=parser_id,
    )
    report = {
        "workspace": str(workspace),
        "corpus": str(corpus_path),
        "result": result.to_dict(),
    }
    json_path = output_dir / "parser_run_materialization.json"
    markdown_path = output_dir / "parser_run_materialization.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate_path = output_dir / "parser_run_materializations.json"
    aggregate_report = _merge_materialization_report(
        aggregate_path=aggregate_path,
        workspace=workspace,
        corpus_path=corpus_path,
        new_result=result.to_dict(),
    )
    aggregate_path.write_text(json.dumps(aggregate_report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_parser_run_markdown(aggregate_report), encoding="utf-8")
    return {"json": json_path, "aggregate_json": aggregate_path, "markdown": markdown_path}


def _merge_materialization_report(
    aggregate_path: Path,
    workspace: Path,
    corpus_path: Path,
    new_result: dict[str, Any],
) -> dict[str, Any]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    if aggregate_path.exists():
        try:
            existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
            for result in existing.get("results", []):
                key = (str(result.get("parser_id")), str(result.get("sample_id")))
                results[key] = result
        except Exception:
            pass
    key = (str(new_result.get("parser_id")), str(new_result.get("sample_id")))
    results[key] = new_result
    return {
        "workspace": str(workspace),
        "corpus": str(corpus_path),
        "results": sorted(results.values(), key=lambda item: (str(item.get("sample_id")), str(item.get("parser_id")))),
    }


def render_parser_run_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Parser Run Materialization",
        "",
        f"- Workspace: `{report.get('workspace')}`",
        f"- Corpus: `{report.get('corpus')}`",
        "",
        "| parser | sample | ok | blocks | assets | anchors | validation errors | run directory |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    results = report.get("results", [])
    if not results and report.get("result"):
        results = [report["result"]]
    for result in results:
        stats = result.get("stats", {})
        normalize = stats.get("normalize", {})
        package = stats.get("package", {})
        lines.append(
            "| {parser} | {sample} | {ok} | {blocks} | {assets} | {anchors} | {errors} | `{run_dir}` |".format(
                parser=result.get("parser_id", ""),
                sample=result.get("sample_id", ""),
                ok="yes" if result.get("ok") else "no",
                blocks=normalize.get("block_count", ""),
                assets=package.get("asset_count", ""),
                anchors=package.get("anchor_count", ""),
                errors=stats.get("validation_error_count", ""),
                run_dir=result.get("run_dir", ""),
            )
        )
    lines.extend(["", "## Outputs", ""])
    for result in results:
        lines.append(f"### {result.get('parser_id')} / {result.get('sample_id')}")
        if result.get("error"):
            lines.append(f"- Error: {result.get('error')}")
        for name, path in result.get("outputs", {}).items():
            lines.append(f"- `{name}`: `{path}`")
        validation_errors = result.get("stats", {}).get("validation_errors", [])
        if validation_errors:
            lines.append("- Validation:")
            for error in validation_errors:
                lines.append(f"  - {error}")
        else:
            lines.append("- Validation: none")
    lines.append("")
    return "\n".join(lines)
