"""Parser candidate smoke plans and optional command execution."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParserSmokeCommand:
    parser_id: str
    sample_id: str
    source_path: str
    output_dir: str
    argv: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def local_sample_sources(workspace: Path, corpus_path: Path) -> list[dict[str, Any]]:
    corpus = read_json(corpus_path)
    samples: list[dict[str, Any]] = []
    for sample in corpus.get("local_samples", []):
        doc_id = sample.get("doc_id")
        if not doc_id:
            continue
        source_dir = workspace / "store" / "docs" / str(doc_id) / "source"
        source_path = next(iter(sorted(source_dir.glob("original.*"))), None)
        samples.append(
            {
                "sample_id": sample.get("sample_id"),
                "doc_id": doc_id,
                "source_path": str(source_path) if source_path else None,
                "source_exists": bool(source_path and source_path.exists()),
                "categories": sample.get("categories", []),
            }
        )
    return samples


def build_parser_smoke_commands(
    workspace: Path,
    output_dir: Path,
    corpus_path: Path,
    sample_id: str | None = None,
    parser_id: str | None = None,
) -> list[ParserSmokeCommand]:
    samples = [
        sample
        for sample in local_sample_sources(workspace, corpus_path)
        if sample.get("source_exists") and (sample_id is None or sample.get("sample_id") == sample_id)
    ]
    commands: list[ParserSmokeCommand] = []

    for sample in samples:
        source_path = str(sample["source_path"])
        sample_name = str(sample["sample_id"])

        commands.extend(
            [
                ParserSmokeCommand(
                    parser_id="mineru3_pipeline",
                    sample_id=sample_name,
                    source_path=source_path,
                    output_dir=str(output_dir / "parser_runs" / "mineru3_pipeline" / sample_name),
                    argv=[
                        ".venv-candidates/mineru3_pipeline/bin/mineru",
                        "-p",
                        source_path,
                        "-o",
                        str(output_dir / "parser_runs" / "mineru3_pipeline" / sample_name),
                        "-m",
                        "auto",
                        "-b",
                        "pipeline",
                        "-l",
                        "chinese_cht",
                        "-s",
                        "0",
                        "-e",
                        "0",
                    ],
                    notes=["one-page smoke; remove -s/-e for full-document benchmark"],
                ),
                ParserSmokeCommand(
                    parser_id="docling_standard",
                    sample_id=sample_name,
                    source_path=source_path,
                    output_dir=str(output_dir / "parser_runs" / "docling_standard" / sample_name),
                    argv=[
                        ".venv-candidates/docling_standard/bin/docling",
                        source_path,
                        "--to",
                        "md",
                        "--to",
                        "json",
                        "--image-export-mode",
                        "referenced",
                        "--pipeline",
                        "standard",
                        "--device",
                        "cuda",
                        "--output",
                        str(output_dir / "parser_runs" / "docling_standard" / sample_name),
                    ],
                    notes=["standard pipeline; may download model artifacts on first run"],
                ),
                ParserSmokeCommand(
                    parser_id="paddleocr_structure_v3",
                    sample_id=sample_name,
                    source_path=source_path,
                    output_dir=str(output_dir / "parser_runs" / "paddleocr_structure_v3" / sample_name),
                    argv=[
                        ".venv-candidates/paddleocr_structure_v3/bin/paddleocr",
                        "pp_structurev3",
                        "-i",
                        source_path,
                        "--save_path",
                        str(output_dir / "parser_runs" / "paddleocr_structure_v3" / sample_name),
                        "--device",
                        "gpu:0",
                    ],
                    notes=["PaddleOCR package is installed; runtime may still require Paddle backend/model downloads"],
                ),
                ParserSmokeCommand(
                    parser_id="olmocr",
                    sample_id=sample_name,
                    source_path=source_path,
                    output_dir=str(output_dir / "parser_runs" / "olmocr" / sample_name),
                    argv=[
                        ".venv-candidates/olmocr/bin/olmocr",
                        str(output_dir / "parser_runs" / "olmocr" / sample_name),
                        "--pdfs",
                        source_path,
                        "--markdown",
                        "--workers",
                        "1",
                    ],
                    notes=["requires local vLLM startup or --server OpenAI-compatible endpoint for real parsing"],
                ),
            ]
        )
    if parser_id:
        commands = [command for command in commands if command.parser_id == parser_id]
    return commands


def run_smoke_command(command: ParserSmokeCommand, timeout_seconds: int) -> dict[str, Any]:
    output_dir = Path(command.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command.argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        duration = time.monotonic() - started
        return {
            **command.to_dict(),
            "executed": True,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "duration_seconds": round(duration, 3),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "output_files": _list_output_files(output_dir),
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        return {
            **command.to_dict(),
            "executed": True,
            "ok": False,
            "timeout": True,
            "duration_seconds": round(duration, 3),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "output_files": _list_output_files(output_dir),
        }
    except Exception as exc:
        duration = time.monotonic() - started
        return {
            **command.to_dict(),
            "executed": True,
            "ok": False,
            "error": str(exc),
            "duration_seconds": round(duration, 3),
            "output_files": _list_output_files(output_dir),
        }


def _list_output_files(output_dir: Path, limit: int = 40) -> list[str]:
    if not output_dir.exists():
        return []
    files = [str(path.relative_to(output_dir)) for path in sorted(output_dir.rglob("*")) if path.is_file()]
    return files[:limit]


def write_parser_smoke_reports(
    workspace: Path,
    output_dir: Path,
    corpus_path: Path,
    sample_id: str | None = None,
    parser_id: str | None = None,
    execute: bool = False,
    timeout_seconds: int = 600,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_parser_smoke_commands(workspace, output_dir, corpus_path, sample_id, parser_id)
    if execute:
        results = [run_smoke_command(command, timeout_seconds) for command in commands]
    else:
        results = [{**command.to_dict(), "executed": False} for command in commands]

    if (sample_id or parser_id) and (output_dir / "parser_smoke.json").exists():
        results = _merge_existing_results(
            existing_path=output_dir / "parser_smoke.json",
            new_results=results,
        )

    report = {
        "workspace": str(workspace),
        "corpus": str(corpus_path),
        "sample_id": sample_id,
        "parser_id": parser_id,
        "execute": execute,
        "timeout_seconds": timeout_seconds,
        "commands": results,
    }

    json_path = output_dir / "parser_smoke.json"
    markdown_path = output_dir / "parser_smoke.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_parser_smoke_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _merge_existing_results(existing_path: Path, new_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
    except Exception:
        return new_results

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for command in existing.get("commands", []):
        key = (str(command.get("parser_id")), str(command.get("sample_id")))
        merged[key] = command
    for command in new_results:
        key = (str(command.get("parser_id")), str(command.get("sample_id")))
        merged[key] = command

    return sorted(merged.values(), key=lambda item: (str(item.get("sample_id")), str(item.get("parser_id"))))


def render_parser_smoke_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Parser Smoke",
        "",
        f"- Workspace: `{report.get('workspace')}`",
        f"- Corpus: `{report.get('corpus')}`",
        f"- Executed: {'yes' if report.get('execute') else 'no'}",
        "",
        "| parser | sample | executed | ok | output | command |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for command in report.get("commands", []):
        argv = " ".join(str(part) for part in command.get("argv", []))
        ok = command.get("ok")
        lines.append(
            "| {parser} | {sample} | {executed} | {ok} | {output} | `{argv}` |".format(
                parser=command.get("parser_id", ""),
                sample=command.get("sample_id", ""),
                executed="yes" if command.get("executed") else "no",
                ok="yes" if ok is True else "no" if ok is False else "",
                output=command.get("output_dir", ""),
                argv=argv,
            )
        )
    lines.append("")
    return "\n".join(lines)
