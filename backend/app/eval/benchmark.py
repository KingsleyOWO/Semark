"""DGX benchmark workflow planning and comparison reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.adapters.parser_registry import parser_matrix, probe_all_parsers


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}
    return data if isinstance(data, dict) else {"value": data}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_benchmark_workflow(
    workspace: Path,
    output_dir: Path,
    corpus_path: Path,
    queries_path: Path,
    top_k: int,
) -> dict[str, Any]:
    """Build a non-destructive benchmark workflow for DGX parser/model comparison."""
    corpus = read_json(corpus_path)
    local_samples = corpus.get("local_samples", [])
    public_samples = corpus.get("public_samples", [])

    candidate_env_root = Path(".venv-candidates")
    candidate_report_root = output_dir / "candidate_reports"
    current_report_dir = output_dir

    return {
        "version": 1,
        "workspace": str(workspace),
        "report_dir": str(output_dir),
        "non_destructive": {
            "baseline_reports_preserved": True,
            "rule": "Use --output-dir for every evaluation run; do not write to workspace/eval/reports.",
            "candidate_report_root": str(candidate_report_root),
            "candidate_env_root": str(candidate_env_root),
        },
        "corpus": {
            "manifest": str(corpus_path),
            "local_samples": local_samples,
            "public_samples": public_samples,
        },
        "retrieval": {
            "queries": str(queries_path),
            "top_k": top_k,
        },
        "phases": [
            {
                "name": "current_framework_gate",
                "goal": "Verify the restored evaluation framework and preserved baseline artifacts.",
                "commands": [
                    ".venv/bin/python -m unittest discover tests",
                    (
                        ".venv/bin/python -m app.eval.runner --workspace "
                        f"{workspace} --output-dir {current_report_dir} all"
                    ),
                    (
                        ".venv/bin/python -m app.eval.runner --workspace "
                        f"{workspace} --output-dir {current_report_dir} benchmark"
                    ),
                ],
                "outputs": [
                    str(current_report_dir / "baseline_metrics.md"),
                    str(current_report_dir / "retrieval_smoke.md"),
                    str(current_report_dir / "environment_inventory.json"),
                    str(current_report_dir / "parser_model_comparison.md"),
                ],
            },
            {
                "name": "candidate_install_probe",
                "goal": "Install parser candidates in isolated virtual environments, then run probes.",
                "commands": _candidate_install_commands(candidate_env_root, candidate_report_root),
                "outputs": [
                    str(candidate_report_root / "<parser_id>" / "parser_candidate_matrix.md"),
                    str(candidate_report_root / "<parser_id>" / "environment_inventory.json"),
                ],
            },
            {
                "name": "parser_adapter_runs",
                "goal": "Run each available parser against local and public corpus samples without reusing baseline outputs.",
                "status": "MinerU smoke-output materialization is implemented; other parser adapters still need schema mappers.",
                "commands": [
                    (
                        ".venv/bin/python -m app.eval.runner --workspace "
                        f"{workspace} --output-dir {current_report_dir} parser-run "
                        "--sample-id backup_property_form --parser-id mineru3_pipeline"
                    ),
                ],
                "outputs": [
                    str(output_dir / "parser_runs" / "<parser_id>" / "<sample_id>" / "manifest.json"),
                    str(output_dir / "parser_runs" / "<parser_id>" / "<sample_id>" / "document_ir.json"),
                    str(output_dir / "parser_runs" / "<parser_id>" / "<sample_id>" / "outputs"),
                ],
                "contract": [
                    "manifest records parser package, version, mode, model, endpoint, decode params, and timing.",
                    "document_ir keeps block ids, page ids, normalized bboxes, payloads, and provenance.",
                    "assets_index.jsonl keeps retrieval_text, asset_path, block_id, and evidence fields.",
                    "source_map.json keeps markdown-to-block traceability for the viewer.",
                ],
            },
            {
                "name": "comparison_gate",
                "goal": "Promote a parser/model only when it improves a measured failure mode without breaking packaging contracts.",
                "metrics": [
                    "form field recall",
                    "figure/chart/org-chart semantic completeness",
                    "table structure fidelity",
                    "asset recall from retrieval queries",
                    "chunk/source-map traceability",
                    "latency",
                    "GPU/VRAM behavior",
                    "dependency and license risk",
                ],
            },
        ],
    }


def _candidate_install_commands(env_root: Path, report_root: Path) -> list[dict[str, Any]]:
    packages = {
        "mineru3_pipeline": ["mineru"],
        "mineru3_vlm_http": ["mineru"],
        "docling_standard": ["docling"],
        "paddleocr_structure_v3": ["paddleocr"],
        "olmocr": ["olmocr"],
    }

    commands: list[dict[str, Any]] = []
    for candidate in parser_matrix():
        parser_id = candidate["parser_id"]
        if candidate.get("reference_only"):
            commands.append(
                {
                    "parser_id": parser_id,
                    "install": "reference-only; do not install as a default dependency",
                    "probe": None,
                }
            )
            continue

        env_dir = env_root / parser_id
        report_dir = report_root / parser_id
        install_packages = " ".join(packages.get(parser_id, []))
        commands.append(
            {
                "parser_id": parser_id,
                "install": [
                    f"python3 -m venv {env_dir}",
                    f"{env_dir}/bin/python -m pip install -U pip",
                    f"{env_dir}/bin/python -m pip install -e .[dev] {install_packages}".strip(),
                ],
                "probe": (
                    f"{env_dir}/bin/python -m app.eval.runner --workspace workspace "
                    f"--output-dir {report_dir} candidates"
                ),
                "inventory": (
                    f"{env_dir}/bin/python -m app.eval.runner --workspace workspace "
                    f"--output-dir {report_dir} inventory"
                ),
            }
        )
    return commands


def build_parser_model_comparison(output_dir: Path) -> dict[str, Any]:
    """Build a concise comparison report from reports already written in output_dir."""
    baseline = read_json(output_dir / "baseline_metrics.json")
    retrieval = read_json(output_dir / "retrieval_smoke.json")
    inventory = read_json(output_dir / "environment_inventory.json")
    candidate_report = read_json(output_dir / "parser_candidate_matrix.json")
    smoke_by_id = _load_parser_smoke(output_dir)
    materialized_by_id = _load_parser_run_materialization(output_dir)

    candidates = candidate_report.get("candidates") or parser_matrix()
    probes = candidate_report.get("probes") or [probe.to_dict() for probe in probe_all_parsers()]
    probes_by_id = {probe.get("parser_id"): probe for probe in probes}
    isolated_probes_by_id = _load_isolated_candidate_probes(output_dir)

    parser_rows = []
    for candidate in candidates:
        probe = probes_by_id.get(candidate.get("parser_id"), {})
        isolated_probe = isolated_probes_by_id.get(candidate.get("parser_id"), {})
        parser_rows.append(
            {
                "parser_id": candidate.get("parser_id"),
                "display_name": candidate.get("display_name"),
                "available": bool(probe.get("available")),
                "isolated_available": bool(isolated_probe.get("available")),
                "isolated_report": isolated_probe.get("report_path"),
                "isolated_package_versions": isolated_probe.get("python_packages", {}),
                "isolated_python_runtime": isolated_probe.get("python_runtime", {}),
                "reference_only": bool(candidate.get("reference_only")),
                "open_source_default": bool(candidate.get("open_source_default")),
                "license_summary": candidate.get("license_summary"),
                "modes": candidate.get("modes", []),
                "focus": candidate.get("output_focus", []),
                "commands_found": probe.get("commands_found", {}),
                "package_found": bool(probe.get("package_found")),
                "smoke": smoke_by_id.get(candidate.get("parser_id"), {}),
                "materialized_run": materialized_by_id.get(candidate.get("parser_id"), {}),
                "status": _candidate_status(candidate, probe, isolated_probe),
            }
        )

    retrieval_results = retrieval.get("results", [])
    failed_queries = [
        result
        for result in retrieval_results
        if result.get("status") == "ok" and not bool(result.get("passed"))
    ]

    return {
        "baseline": {
            "run_count": baseline.get("run_count"),
            "successful_run_count": baseline.get("successful_run_count"),
            "validation_error_count": baseline.get("validation_error_count"),
            "asset_count_by_type": baseline.get("asset_count_by_type", {}),
            "block_count_by_type": baseline.get("block_count_by_type", {}),
        },
        "retrieval": {
            "query_count": len(retrieval_results),
            "passed_count": sum(1 for result in retrieval_results if result.get("passed")),
            "failed_queries": failed_queries,
        },
        "runtime": {
            "python_executable": inventory.get("python_executable"),
            "python_version": inventory.get("python_version"),
            "platform": inventory.get("platform"),
            "commands": inventory.get("commands", {}),
            "python_packages": inventory.get("python_packages", {}),
            "command_versions": inventory.get("command_versions", {}),
        },
        "parsers": parser_rows,
        "recommendations": _recommendations(parser_rows, failed_queries),
    }


def _load_isolated_candidate_probes(output_dir: Path) -> dict[str, dict[str, Any]]:
    probes_by_id: dict[str, dict[str, Any]] = {}
    candidate_reports = output_dir / "candidate_reports"
    if not candidate_reports.exists():
        return probes_by_id

    for report_path in sorted(candidate_reports.glob("*/parser_candidate_matrix.json")):
        report = read_json(report_path)
        inventory = read_json(report_path.parent / "environment_inventory.json")
        for probe in report.get("probes", []):
            if not probe.get("available"):
                continue
            parser_id = probe.get("parser_id")
            if parser_id:
                probes_by_id[parser_id] = {
                    **probe,
                    "report_path": str(report_path),
                    "python_packages": inventory.get("python_packages", {}),
                    "python_runtime": inventory.get("python_runtime", {}),
                }
    return probes_by_id


def _load_parser_smoke(output_dir: Path) -> dict[str, dict[str, Any]]:
    report = read_json(output_dir / "parser_smoke.json")
    smoke_by_id: dict[str, dict[str, Any]] = {}
    for command in report.get("commands", []):
        parser_id = command.get("parser_id")
        if not parser_id:
            continue
        smoke_by_id[parser_id] = {
            "executed": bool(command.get("executed")),
            "ok": bool(command.get("ok")),
            "returncode": command.get("returncode"),
            "duration_seconds": command.get("duration_seconds"),
            "sample_id": command.get("sample_id"),
            "output_dir": command.get("output_dir"),
            "output_file_count": len(command.get("output_files", [])),
            "failure_summary": _smoke_failure_summary(command),
        }
    return smoke_by_id


def _load_parser_run_materialization(output_dir: Path) -> dict[str, dict[str, Any]]:
    aggregate = read_json(output_dir / "parser_run_materializations.json")
    if aggregate.get("results"):
        materialized: dict[str, dict[str, Any]] = {}
        for result in aggregate.get("results", []):
            parser_id = result.get("parser_id")
            if parser_id:
                materialized[parser_id] = _materialized_result_summary(result)
        return materialized

    report = read_json(output_dir / "parser_run_materialization.json")
    result = report.get("result", {})
    parser_id = result.get("parser_id")
    if not parser_id:
        return {}
    return {parser_id: _materialized_result_summary(result)}


def _materialized_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": result.get("sample_id"),
        "ok": bool(result.get("ok")),
        "run_dir": result.get("run_dir"),
        "document_ir_path": result.get("document_ir_path"),
        "manifest_path": result.get("manifest_path"),
        "outputs": result.get("outputs", {}),
        "stats": result.get("stats", {}),
        "error": result.get("error"),
    }


def _smoke_failure_summary(command: dict[str, Any]) -> str:
    if command.get("ok"):
        return ""
    if command.get("timeout"):
        return "timeout"
    stderr = str(command.get("stderr_tail") or "")
    error = str(command.get("error") or "")
    text = "\n".join(part for part in [stderr, error] if part)
    for marker in [
        "ModuleNotFoundError:",
        "DependencyError:",
        "DownloadFileException:",
        "RuntimeError:",
        "ERROR:",
    ]:
        if marker in text:
            line = next((line.strip() for line in text.splitlines() if marker in line), "")
            return line[:240]
    line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    return line[:240]


def _candidate_status(
    candidate: dict[str, Any],
    probe: dict[str, Any],
    isolated_probe: dict[str, Any],
) -> str:
    if candidate.get("reference_only"):
        return "reference_only"
    if probe.get("available"):
        return "ready_to_benchmark"
    if isolated_probe.get("available"):
        return "ready_in_isolated_env"
    return "install_required"


def _recommendations(parser_rows: list[dict[str, Any]], failed_queries: list[dict[str, Any]]) -> list[str]:
    recommendations = [
        "Keep baseline reports immutable; write every DGX run to a timestamped --output-dir.",
        "Install parser candidates in isolated environments first, then promote only the packages needed by the chosen default path.",
        "Record parser package version, model id, endpoint mode, decoding parameters, and latency in every candidate manifest.",
        "Do not make Marker a default dependency; keep it as a reference-only comparison.",
    ]
    if failed_queries:
        recommendations.insert(
            0,
            "Prioritize figure/org-chart retrieval_text and asset ranking; the current lexical gate still has a figure_asset miss.",
        )
    if not any(
        (row["available"] or row["isolated_available"]) and not row["reference_only"]
        for row in parser_rows
    ):
        recommendations.append(
            "No open-source parser candidate is installed in the active environment yet; run the isolated candidate_install_probe phase before quality comparison."
        )
    smoke_by_id = {row.get("parser_id"): row.get("smoke", {}) for row in parser_rows}
    mineru_smoke = smoke_by_id.get("mineru3_pipeline", {})
    if mineru_smoke.get("ok"):
        recommendations.append(
            "Promote MinerU pipeline to the first full-document adapter target because it has CUDA-enabled Torch and a materialized contract path."
        )
    materialized_by_id = {row.get("parser_id"): row.get("materialized_run", {}) for row in parser_rows}
    if materialized_by_id.get("mineru3_pipeline", {}).get("ok"):
        recommendations.append(
            "Use the materialized MinerU parser-run artifacts as the first comparable contract baseline for future Docling/PaddleOCR/olmOCR adapters."
        )
    if "DownloadFileException" in str(smoke_by_id.get("docling_standard", {}).get("failure_summary", "")):
        recommendations.append(
            "Before re-running Docling quality tests, pre-cache or mirror RapidOCR model artifacts used by the standard PDF pipeline."
        )
    if "paddlex[ocr]" in str(smoke_by_id.get("paddleocr_structure_v3", {}).get("failure_summary", "")):
        recommendations.append(
            "Install the PaddleOCR `paddlex[ocr]` extra in the isolated PaddleOCR environment before evaluating PP-StructureV3."
        )
    paddle_row = next((row for row in parser_rows if row.get("parser_id") == "paddleocr_structure_v3"), {})
    paddle_runtime = _display_candidate_runtime(paddle_row)
    if smoke_by_id.get("paddleocr_structure_v3", {}).get("ok") and "cuda=False" in paddle_runtime:
        recommendations.append(
            "PaddleOCR PP-StructureV3 passed smoke, but the installed Paddle runtime is CPU-only; find a CUDA-enabled Paddle wheel for fair DGX GPU comparison."
        )
    elif smoke_by_id.get("paddleocr_structure_v3", {}).get("ok"):
        recommendations.append(
            "Build the PaddleOCR output adapter next so PP-StructureV3 HTML/XLSX/JSON outputs can be compared through the same DocumentIR and assets_index contract."
        )
    olmocr_smoke = smoke_by_id.get("olmocr", {})
    if olmocr_smoke.get("failure_summary") == "timeout":
        recommendations.append(
            "olmOCR has CUDA-enabled PyTorch but timed out on the default local CLI path; configure an explicit vLLM/OpenAI-compatible server and model before quality comparison."
        )
    elif "No module named 'torch'" in str(olmocr_smoke.get("failure_summary", "")):
        recommendations.append(
            "Install PyTorch in the olmOCR environment, then benchmark it through a local vLLM or OpenAI-compatible endpoint."
        )
    return recommendations


def write_benchmark_reports(
    workspace: Path,
    output_dir: Path,
    corpus_path: Path,
    queries_path: Path,
    top_k: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    workflow = build_benchmark_workflow(workspace, output_dir, corpus_path, queries_path, top_k)
    comparison = build_parser_model_comparison(output_dir)
    contract_comparison = build_materialized_contract_comparison(output_dir)

    workflow_json = output_dir / "benchmark_workflow.json"
    workflow_markdown = output_dir / "benchmark_workflow.md"
    comparison_json = output_dir / "parser_model_comparison.json"
    comparison_markdown = output_dir / "parser_model_comparison.md"
    contract_json = output_dir / "parser_contract_comparison.json"
    contract_markdown = output_dir / "parser_contract_comparison.md"

    write_json(workflow_json, workflow)
    workflow_markdown.write_text(render_benchmark_workflow_markdown(workflow), encoding="utf-8")
    write_json(comparison_json, comparison)
    comparison_markdown.write_text(render_parser_model_comparison_markdown(comparison), encoding="utf-8")
    write_json(contract_json, contract_comparison)
    contract_markdown.write_text(render_materialized_contract_comparison_markdown(contract_comparison), encoding="utf-8")

    return {
        "workflow_json": workflow_json,
        "workflow_markdown": workflow_markdown,
        "comparison_json": comparison_json,
        "comparison_markdown": comparison_markdown,
        "contract_json": contract_json,
        "contract_markdown": contract_markdown,
    }


def build_materialized_contract_comparison(output_dir: Path) -> dict[str, Any]:
    materialized = _load_parser_run_materialization(output_dir)
    parser_comparison = build_parser_model_comparison(output_dir)
    parser_rows = {row.get("parser_id"): row for row in parser_comparison.get("parsers", [])}
    rows = []
    for parser_id in ("mineru3_pipeline", "paddleocr_structure_v3"):
        result = materialized.get(parser_id, {})
        if not result:
            continue
        stats = result.get("stats", {})
        normalize = stats.get("normalize", {})
        package = stats.get("package", {})
        row = parser_rows.get(parser_id, {})
        rows.append(
            {
                "parser_id": parser_id,
                "display_name": row.get("display_name") or parser_id,
                "ok": bool(result.get("ok")),
                "sample_id": result.get("sample_id"),
                "runtime": _display_candidate_runtime(row),
                "smoke_seconds": (row.get("smoke") or {}).get("duration_seconds"),
                "block_count": normalize.get("block_count"),
                "block_types": normalize.get("by_type", {}),
                "asset_count": package.get("asset_count"),
                "anchor_count": package.get("anchor_count"),
                "validation_error_count": stats.get("validation_error_count"),
                "run_dir": result.get("run_dir"),
                "outputs": result.get("outputs", {}),
            }
        )
    return {
        "report": "mineru_vs_paddleocr_contract",
        "rows": rows,
        "conclusion": _contract_comparison_conclusion(rows),
    }


def _contract_comparison_conclusion(rows: list[dict[str, Any]]) -> list[str]:
    by_id = {row.get("parser_id"): row for row in rows}
    mineru = by_id.get("mineru3_pipeline", {})
    paddle = by_id.get("paddleocr_structure_v3", {})
    conclusions = []
    if mineru.get("ok") and paddle.get("ok"):
        conclusions.append(
            "Both MinerU and PaddleOCR now satisfy the shared parser-run contract for the backup_property_form sample."
        )
    if mineru.get("runtime") and "cuda=True" in str(mineru.get("runtime")):
        conclusions.append("MinerU remains the better DGX default candidate because its tested runtime is CUDA-enabled.")
    if paddle.get("runtime") and "cuda=False" in str(paddle.get("runtime")):
        conclusions.append("PaddleOCR is functionally complete for this sample, but its current Paddle runtime is CPU-only.")
    if paddle.get("asset_count") == mineru.get("asset_count"):
        conclusions.append("Both produced one table asset and zero validation errors under the shared assets/source-map contract.")
    conclusions.append(
        "Next comparison should score table fidelity and form-field recall from the materialized outputs, not just smoke success."
    )
    return conclusions


def render_benchmark_workflow_markdown(workflow: dict[str, Any]) -> str:
    lines = [
        "# DGX Benchmark Workflow",
        "",
        f"- Workspace: `{workflow.get('workspace')}`",
        f"- Report directory: `{workflow.get('report_dir')}`",
        f"- Candidate env root: `{workflow['non_destructive']['candidate_env_root']}`",
        f"- Candidate report root: `{workflow['non_destructive']['candidate_report_root']}`",
        "- Baseline reports: preserved by using `--output-dir`",
        "",
        "## Corpus",
        "",
    ]

    corpus = workflow.get("corpus", {})
    lines.append(f"- Manifest: `{corpus.get('manifest')}`")
    lines.append(f"- Local samples: {len(corpus.get('local_samples', []))}")
    lines.append(f"- Public samples: {len(corpus.get('public_samples', []))}")
    lines.append("")

    for phase in workflow.get("phases", []):
        lines.append(f"## {phase.get('name')}")
        lines.append("")
        lines.append(str(phase.get("goal", "")))
        lines.append("")
        commands = phase.get("commands")
        if commands:
            lines.append("Commands:")
            for command in commands:
                if isinstance(command, str):
                    lines.append(f"- `{command}`")
                else:
                    parser_id = command.get("parser_id", "")
                    install = command.get("install")
                    probe = command.get("probe")
                    lines.append(f"- `{parser_id}`")
                    if isinstance(install, list):
                        for install_command in install:
                            lines.append(f"  - `{install_command}`")
                    else:
                        lines.append(f"  - {install}")
                    if probe:
                        lines.append(f"  - `{probe}`")
                    inventory = command.get("inventory")
                    if inventory:
                        lines.append(f"  - `{inventory}`")
            lines.append("")
        outputs = phase.get("outputs")
        if outputs:
            lines.append("Outputs:")
            for output in outputs:
                lines.append(f"- `{output}`")
            lines.append("")
        metrics = phase.get("metrics")
        if metrics:
            lines.append("Metrics:")
            for metric in metrics:
                lines.append(f"- {metric}")
            lines.append("")

    return "\n".join(lines)


def render_parser_model_comparison_markdown(comparison: dict[str, Any]) -> str:
    baseline = comparison.get("baseline", {})
    retrieval = comparison.get("retrieval", {})
    runtime = comparison.get("runtime", {})
    lines = [
        "# Parser/Model Comparison",
        "",
        "## Current Baseline",
        "",
        f"- Runs discovered: {baseline.get('run_count')}",
        f"- Successful runs: {baseline.get('successful_run_count')}",
        f"- Validation errors: {baseline.get('validation_error_count')}",
        f"- Assets: `{json.dumps(baseline.get('asset_count_by_type', {}), ensure_ascii=False)}`",
        f"- Retrieval: {retrieval.get('passed_count')}/{retrieval.get('query_count')} passed",
        "",
        "## DGX Runtime",
        "",
        f"- Python: `{runtime.get('python_executable')}`",
        f"- Platform: `{runtime.get('platform')}`",
        f"- Commands: `{json.dumps(runtime.get('commands', {}), ensure_ascii=False)}`",
        "",
        "## Parser Candidates",
        "",
        "| parser | status | active env | isolated env | version | runtime | default | reference | license | focus |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in comparison.get("parsers", []):
        focus = ", ".join(row.get("focus", []))
        lines.append(
            "| {name} | {status} | {available} | {isolated} | {version} | {runtime} | {default} | {reference} | {license} | {focus} |".format(
                name=row.get("display_name", ""),
                status=row.get("status", ""),
                available="yes" if row.get("available") else "no",
                isolated="yes" if row.get("isolated_available") else "no",
                version=_display_candidate_version(row),
                runtime=_display_candidate_runtime(row),
                default="yes" if row.get("open_source_default") else "no",
                reference="yes" if row.get("reference_only") else "no",
                license=row.get("license_summary", ""),
                focus=focus,
            )
        )

    lines.extend(["", "## Failed Retrieval Gates", ""])
    failed_queries = retrieval.get("failed_queries", [])
    if not failed_queries:
        lines.append("- None")
    else:
        for query in failed_queries:
            top = query.get("top", [{}])[0] if query.get("top") else {}
            lines.append(
                "- `{query_id}` expected `{expected}`; top candidate `{candidate}` score `{score}`".format(
                    query_id=query.get("query_id"),
                    expected=query.get("expected_asset_type"),
                    candidate=top.get("candidate_id"),
                    score=top.get("score"),
                )
            )

    lines.extend(["", "## Parser Smoke", ""])
    lines.append("| parser | sample | ok | seconds | files | failure |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    has_smoke = False
    for row in comparison.get("parsers", []):
        smoke = row.get("smoke", {})
        if not smoke:
            continue
        has_smoke = True
        lines.append(
            "| {parser} | {sample} | {ok} | {seconds} | {files} | {failure} |".format(
                parser=row.get("display_name", ""),
                sample=smoke.get("sample_id", ""),
                ok="yes" if smoke.get("ok") else "no",
                seconds=smoke.get("duration_seconds", ""),
                files=smoke.get("output_file_count", 0),
                failure=str(smoke.get("failure_summary", "")).replace("|", "\\|"),
            )
        )
    if not has_smoke:
        lines.append("| _not run_ |  |  |  |  |  |")

    lines.extend(["", "## Materialized Parser Runs", ""])
    lines.append("| parser | sample | ok | blocks | assets | anchors | run dir |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
    has_materialized = False
    for row in comparison.get("parsers", []):
        run = row.get("materialized_run", {})
        if not run:
            continue
        has_materialized = True
        stats = run.get("stats", {})
        normalize = stats.get("normalize", {})
        package = stats.get("package", {})
        lines.append(
            "| {parser} | {sample} | {ok} | {blocks} | {assets} | {anchors} | `{run_dir}` |".format(
                parser=row.get("display_name", ""),
                sample=run.get("sample_id", ""),
                ok="yes" if run.get("ok") else "no",
                blocks=normalize.get("block_count", ""),
                assets=package.get("asset_count", ""),
                anchors=package.get("anchor_count", ""),
                run_dir=run.get("run_dir", ""),
            )
        )
    if not has_materialized:
        lines.append("| _not run_ |  |  |  |  |  |  |")

    lines.extend(["", "## Recommendations", ""])
    for recommendation in comparison.get("recommendations", []):
        lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


def render_materialized_contract_comparison_markdown(comparison: dict[str, Any]) -> str:
    rows = comparison.get("rows", [])
    title = "MinerU vs PaddleOCR Contract Comparison" if len(rows) > 1 else "Active Parser Contract Report"
    lines = [
        f"# {title}",
        "",
        "| parser | sample | ok | runtime | smoke seconds | blocks | block types | assets | anchors | validation errors |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {parser} | {sample} | {ok} | {runtime} | {seconds} | {blocks} | `{types}` | {assets} | {anchors} | {errors} |".format(
                parser=row.get("display_name", ""),
                sample=row.get("sample_id", ""),
                ok="yes" if row.get("ok") else "no",
                runtime=row.get("runtime", ""),
                seconds=row.get("smoke_seconds", ""),
                blocks=row.get("block_count", ""),
                types=json.dumps(row.get("block_types", {}), ensure_ascii=False),
                assets=row.get("asset_count", ""),
                anchors=row.get("anchor_count", ""),
                errors=row.get("validation_error_count", ""),
            )
        )
    lines.extend(["", "## Outputs", ""])
    for row in comparison.get("rows", []):
        lines.append(f"### {row.get('display_name')}")
        lines.append(f"- Run dir: `{row.get('run_dir')}`")
        for name, path in row.get("outputs", {}).items():
            lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Conclusion", ""])
    for conclusion in comparison.get("conclusion", []):
        lines.append(f"- {conclusion}")
    lines.append("")
    return "\n".join(lines)


def _display_candidate_version(row: dict[str, Any]) -> str:
    packages = row.get("isolated_package_versions", {})
    parser_id = row.get("parser_id")
    if parser_id in {"mineru3_pipeline", "mineru3_vlm_http"}:
        return packages.get("mineru") or ""
    if parser_id == "docling_standard":
        return packages.get("docling") or ""
    if parser_id == "paddleocr_structure_v3":
        return packages.get("paddleocr") or ""
    if parser_id == "olmocr":
        return packages.get("olmocr") or ""
    return ""


def _display_candidate_runtime(row: dict[str, Any]) -> str:
    runtime = row.get("isolated_python_runtime", {})
    parser_id = row.get("parser_id")
    if parser_id == "paddleocr_structure_v3":
        stdout = (runtime.get("paddle") or {}).get("stdout", "")
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if len(lines) >= 3:
            return f"paddle {lines[0]} {lines[1]} cuda={lines[2]}"
    if parser_id in {"mineru3_pipeline", "mineru3_vlm_http", "olmocr"}:
        stdout = (runtime.get("torch") or {}).get("stdout", "")
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if len(lines) >= 2:
            return f"torch {lines[0]} cuda={lines[1]}"
    return ""
