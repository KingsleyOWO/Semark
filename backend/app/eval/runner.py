"""Command-line entry point for pipeline evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.adapters.parser_registry import parser_matrix, probe_all_parsers
from app.eval.baseline import collect_workspace_baseline, write_baseline_reports
from app.eval.benchmark import write_benchmark_reports
from app.eval.parser_smoke import write_parser_smoke_reports
from app.eval.retrieval import run_retrieval_smoke_eval, write_retrieval_reports

DEFAULT_WORKSPACE = Path("workspace")


def environment_inventory() -> dict[str, Any]:
    """Collect a small environment inventory for reproducibility."""
    commands = {
        "python": shutil.which("python"),
        "uv": shutil.which("uv"),
        "mineru": shutil.which("mineru"),
        "mineru-api": shutil.which("mineru-api"),
        "docling": shutil.which("docling"),
        "paddleocr": shutil.which("paddleocr"),
        "olmocr": shutil.which("olmocr"),
        "ollama": shutil.which("ollama"),
        "vllm": shutil.which("vllm"),
        "nvidia-smi": shutil.which("nvidia-smi"),
        "nvcc": shutil.which("nvcc"),
    }
    packages = {
        name: _package_version(name)
        for name in (
            "mineru",
            "docling",
            "paddleocr",
            "olmocr",
            "marker",
            "torch",
            "torchvision",
            "paddlepaddle",
            "paddlepaddle-gpu",
            "vllm",
        )
    }
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "commands": commands,
        "command_versions": {
            "nvidia-smi": _run_version_command(["nvidia-smi"]) if commands["nvidia-smi"] else None,
            "nvcc": _run_version_command(["nvcc", "--version"]) if commands["nvcc"] else None,
            "ollama": _run_version_command(["ollama", "--version"]) if commands["ollama"] else None,
            "ollama_models": _run_version_command(["ollama", "list"]) if commands["ollama"] else None,
            "vllm": _run_version_command(["vllm", "--version"]) if commands["vllm"] else None,
        },
        "python_runtime": {
            "torch": _run_python_probe(
                "import torch; print(torch.__version__); print(torch.cuda.is_available())"
            )
            if packages.get("torch")
            else None,
            "paddle": _run_python_probe(
                "import paddle; print(paddle.__version__); print(paddle.device.get_device()); print(paddle.device.is_compiled_with_cuda())"
            )
            if packages.get("paddlepaddle")
            else None,
        },
        "python_packages": packages,
        "parser_probes": [probe.to_dict() for probe in probe_all_parsers()],
    }


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_version_command(argv: list[str], timeout_seconds: int = 10) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _run_python_probe(code: str, timeout_seconds: int = 10) -> dict[str, Any]:
    return _run_version_command([sys.executable, "-c", code], timeout_seconds=timeout_seconds)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_candidate_reports(output_dir: Path) -> dict[str, Path]:
    """Write parser candidate metadata and current availability probes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = parser_matrix()
    probes = [probe.to_dict() for probe in probe_all_parsers()]
    report = {"candidates": matrix, "probes": probes}

    json_path = output_dir / "parser_candidate_matrix.json"
    markdown_path = output_dir / "parser_candidate_matrix.md"
    write_json(json_path, report)
    markdown_path.write_text(render_candidate_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def render_candidate_markdown(report: dict[str, Any]) -> str:
    probes_by_id = {probe["parser_id"]: probe for probe in report.get("probes", [])}
    lines = [
        "# Parser Candidate Matrix",
        "",
        "| parser | license | default | available | remote | focus | caveats |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in report.get("candidates", []):
        probe = probes_by_id.get(candidate["parser_id"], {})
        focus = ", ".join(candidate.get("output_focus", []))
        caveats = "; ".join(candidate.get("caveats", []))
        lines.append(
            "| {name} | {license} | {default} | {available} | {remote} | {focus} | {caveats} |".format(
                name=candidate.get("display_name", ""),
                license=candidate.get("license_summary", ""),
                default="yes" if candidate.get("open_source_default") else "no",
                available="yes" if probe.get("available") else "no",
                remote="yes" if candidate.get("remote_endpoint_supported") else "no",
                focus=focus,
                caveats=caveats,
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Doc-VLM evaluation harness")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Workspace root containing store/docs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for reports; defaults to <workspace>/eval/reports",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("baseline", help="Collect metrics from existing run artifacts")
    subparsers.add_parser("candidates", help="Write parser candidate matrix and probes")
    subparsers.add_parser("inventory", help="Write environment inventory")
    retrieval_parser = subparsers.add_parser(
        "retrieval",
        help="Run lexical retrieval smoke checks over chunks and assets",
    )
    retrieval_parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/corpus.public.json"),
        help="Corpus manifest path",
    )
    retrieval_parser.add_argument(
        "--queries",
        type=Path,
        default=Path("benchmarks/retrieval_queries.jsonl"),
        help="Retrieval query manifest path",
    )
    retrieval_parser.add_argument("--top-k", type=int, default=5)
    subparsers.add_parser("all", help="Run baseline, candidate matrix, and inventory")
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Write DGX benchmark workflow and parser/model comparison reports",
    )
    benchmark_parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/corpus.public.json"),
        help="Corpus manifest path",
    )
    benchmark_parser.add_argument(
        "--queries",
        type=Path,
        default=Path("benchmarks/retrieval_queries.jsonl"),
        help="Retrieval query manifest path",
    )
    benchmark_parser.add_argument("--top-k", type=int, default=5)
    smoke_parser = subparsers.add_parser(
        "parser-smoke",
        help="Write or execute parser candidate smoke commands against local samples",
    )
    smoke_parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/corpus.public.json"),
        help="Corpus manifest path",
    )
    smoke_parser.add_argument("--sample-id", default=None)
    smoke_parser.add_argument("--parser-id", default=None)
    smoke_parser.add_argument("--execute", action="store_true")
    smoke_parser.add_argument("--timeout-seconds", type=int, default=600)
    run_parser = subparsers.add_parser(
        "parser-run",
        help="Materialize a parser smoke output into comparable run artifacts",
    )
    run_parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/corpus.public.json"),
        help="Corpus manifest path",
    )
    run_parser.add_argument("--sample-id", required=True)
    run_parser.add_argument("--parser-id", default="mineru3_pipeline")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or (args.workspace / "eval" / "reports")

    if args.command in {"baseline", "all"}:
        baseline = collect_workspace_baseline(args.workspace)
        paths = write_baseline_reports(baseline, output_dir)
        print(f"baseline json: {paths['json']}")
        print(f"baseline markdown: {paths['markdown']}")

    if args.command in {"candidates", "all"}:
        paths = write_candidate_reports(output_dir)
        print(f"candidate json: {paths['json']}")
        print(f"candidate markdown: {paths['markdown']}")

    if args.command in {"inventory", "all"}:
        inventory_path = output_dir / "environment_inventory.json"
        write_json(inventory_path, environment_inventory())
        print(f"environment inventory: {inventory_path}")

    if args.command in {"retrieval", "all"}:
        corpus_path = getattr(args, "corpus", Path("benchmarks/corpus.public.json"))
        queries_path = getattr(args, "queries", Path("benchmarks/retrieval_queries.jsonl"))
        top_k = getattr(args, "top_k", 5)
        retrieval = run_retrieval_smoke_eval(args.workspace, corpus_path, queries_path, top_k)
        paths = write_retrieval_reports(retrieval, output_dir)
        print(f"retrieval json: {paths['json']}")
        print(f"retrieval markdown: {paths['markdown']}")

    if args.command == "benchmark":
        paths = write_benchmark_reports(
            workspace=args.workspace,
            output_dir=output_dir,
            corpus_path=args.corpus,
            queries_path=args.queries,
            top_k=args.top_k,
        )
        print(f"benchmark workflow json: {paths['workflow_json']}")
        print(f"benchmark workflow markdown: {paths['workflow_markdown']}")
        print(f"parser/model comparison json: {paths['comparison_json']}")
        print(f"parser/model comparison markdown: {paths['comparison_markdown']}")
        print(f"parser contract comparison json: {paths['contract_json']}")
        print(f"parser contract comparison markdown: {paths['contract_markdown']}")

    if args.command == "parser-smoke":
        paths = write_parser_smoke_reports(
            workspace=args.workspace,
            output_dir=output_dir,
            corpus_path=args.corpus,
            sample_id=args.sample_id,
            parser_id=args.parser_id,
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
        )
        print(f"parser smoke json: {paths['json']}")
        print(f"parser smoke markdown: {paths['markdown']}")

    if args.command == "parser-run":
        from app.eval.parser_runs import write_parser_run_reports

        paths = write_parser_run_reports(
            workspace=args.workspace,
            output_dir=output_dir,
            corpus_path=args.corpus,
            sample_id=args.sample_id,
            parser_id=args.parser_id,
        )
        print(f"parser run json: {paths['json']}")
        print(f"parser run aggregate json: {paths['aggregate_json']}")
        print(f"parser run markdown: {paths['markdown']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
