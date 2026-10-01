"""Isolated comparison of cached production IRs; never writes source runs/DB."""
import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from app.config import AgenticConfig, PipelineConfig, settings
from app.models.document_ir import DocumentIR
from app.pipeline.agentic import AgenticRepair
from app.pipeline.delivery import atomic_write_json
from app.pipeline.final_delivery import finalize_delivery, read_json
from app.pipeline.stages.chunk import ChunkStage
from app.pipeline.stages.enrich import EnrichStage
from app.pipeline.stages.normalize import save_document_ir
from app.pipeline.stages.package import PackageStage


def copy_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


async def package(ir, folder, cfg, cache):
    save_document_ir(ir, folder)
    result = await PackageStage(cfg).run(ir.doc_id, ir.run_id, ir, folder, cache)
    if not result.success:
        raise RuntimeError(result.error)
    chunks = await ChunkStage(cfg).run(ir.doc_id, ir.run_id, ir, folder)
    if not chunks.success:
        raise RuntimeError(chunks.error)
    return finalize_delivery(ir, folder, cfg)


async def verify_local_models(cfg: PipelineConfig) -> None:
    """Real-document pilots are restricted to locally installed Ollama models."""
    for model_cfg in (cfg.vlm, cfg.review_vlm):
        url = urlsplit(model_cfg.base_url)
        if (url.scheme != "http" or url.hostname != "127.0.0.1" or url.port != 11434
                or url.path.rstrip("/") != "/v1" or url.username or url.password
                or url.query or url.fragment or model_cfg.model.endswith(":cloud")):
            raise ValueError("Pilot permits only local Ollama at http://127.0.0.1:11434/v1")
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
        response = await client.get("http://127.0.0.1:11434/api/tags", timeout=10)
        response.raise_for_status()
        local = {m.get("name") for m in response.json().get("models", [])
                 if m.get("size", 0) > 0 and not m.get("remote_host")}
    if any(model_cfg.model not in local for model_cfg in (cfg.vlm, cfg.review_vlm)):
        raise ValueError("Pilot model is not verified as locally installed")


async def run(args):
    settings.workspace_path = args.workspace
    samples = json.loads(args.samples.read_text())
    db = sqlite3.connect(f"file:{args.workspace / 'doc_parser.db'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for sample in samples:
            source = settings.get_published_run_path(sample["doc_id"], sample["run_id"])
            destination = args.output / sample["doc_id"]
            destination.mkdir(parents=True, exist_ok=True)
            print(json.dumps({"event": "start", "doc_id": sample["doc_id"]}), flush=True)
            row = db.execute("SELECT config_json FROM runs WHERE run_id=?", (sample["run_id"],)).fetchone()
            cfg = PipelineConfig.model_validate(json.loads(row["config_json"]))
            if args.real_model:
                await verify_local_models(cfg)
            cfg.enrich.enable_vlm = False
            cfg.package.enable_semantic_repair = False
            cache_row = db.execute(
                "SELECT path FROM cache_entries WHERE doc_id=? AND stage='parse' ORDER BY created_at DESC LIMIT 1",
                (sample["doc_id"],),
            ).fetchone()
            cache = Path(cache_row["path"]) if cache_row else None
            if cache and not cache.is_absolute():
                cache = args.workspace / cache
            ir = DocumentIR.from_dict(json.loads((source / "document_ir.json").read_text()))
            ir.page_kind = "physical" if ir.has_physical_pages() else "logical"
            # Replay cached reconstruction outputs through the new acceptance
            # guard, matching enrich without spending on an entire OCR rerun.
            enrichments = source / "outputs/enrichments.jsonl"
            if enrichments.exists():
                for line in enrichments.read_text().splitlines():
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    block = ir.get_block(str(entry.get("block_id")))
                    if block and entry.get("kind") == "table_reconstruct":
                        EnrichStage._apply_table_reconstruction(block, entry.get("output"))
            old = destination / "before"
            shutil.copytree(source / "outputs", old / "outputs", dirs_exist_ok=True)
            old_status = finalize_delivery(ir, old, cfg)
            fixed = destination / "fixed"
            candidate = destination / "candidate"
            for folder in (fixed, candidate):
                (folder / "outputs").mkdir(parents=True, exist_ok=True)
                if enrichments.exists():
                    copy_file(enrichments, folder / "outputs/enrichments.jsonl")
                if (source / "assets").exists():
                    shutil.copytree(source / "assets", folder / "assets", dirs_exist_ok=True)
            for page in ir.pages:
                if page.page_image_path:
                    page.page_image_path = "assets/pages/" + Path(page.page_image_path).name
            started = time.monotonic()
            fixed_status = await package(ir.model_copy(deep=True), fixed, cfg, cache)
            fixed_seconds = time.monotonic() - started
            agent_cfg = cfg.model_copy(deep=True)
            agent_cfg.agentic = AgenticConfig(
                mode="apply" if args.real_model else "off", max_actions=args.max_actions,
                max_model_calls=args.max_model_calls, max_seconds=args.max_seconds,
                max_attempts_per_block=2, max_output_tokens=args.max_output_tokens,
            )
            started = time.monotonic()
            repaired, agent_report = await AgenticRepair(agent_cfg).run(ir, candidate, parse_cache_path=cache)
            candidate_status = await package(repaired, candidate, cfg, cache)
            candidate_seconds = time.monotonic() - started
            item = {
                **sample, "model": cfg.vlm.model, "before": old_status, "fixed": fixed_status,
                "candidate": candidate_status, "agent": agent_report,
                "seconds": {"fixed": round(fixed_seconds, 2), "candidate": round(candidate_seconds, 2)},
                "accepted_blocks": [
                    {"block_id": b.block_id, "page_idx": b.page_idx,
                     "semantics": b.payload["verified_semantics"]}
                    for b in repaired.blocks if b.payload.get("verified_semantics")
                ],
                "fixed_document_type": read_json(fixed / "outputs/document_plan.json", {}).get("document_type"),
            }
            results.append(item)
            atomic_write_json(args.output / "report.json", {
                "scope": f"{len(samples)} cached production documents; isolated packaging/chunking and bounded visual model calls",
                "production_modified": False, "results": results,
            }, ensure_ascii=False, indent=2)
            print(json.dumps({"doc_id": item["doc_id"], "type": item["fixed_document_type"],
                              "coverage_before": old_status["coverage"], "coverage_fixed": fixed_status["coverage"],
                              "coverage_candidate": candidate_status["coverage"], "agent": agent_report},
                             ensure_ascii=False), flush=True)
    finally:
        db.close()
        # Containers may need root to read owner-only source artifacts. Return
        # ownership of the isolated results to the owner of the mounted folder.
        owner = args.output.parent.stat()
        if os.geteuid() == 0:
            for path in [args.output, *args.output.rglob("*")]:
                if path.resolve().is_relative_to(args.output.resolve()):
                    os.chown(path, owner.st_uid, owner.st_gid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument("--max-actions", type=int, default=2)
    parser.add_argument("--max-model-calls", type=int, default=6)
    parser.add_argument("--max-seconds", type=float, default=240)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
