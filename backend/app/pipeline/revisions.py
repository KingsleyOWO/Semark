"""Durable stage checkpoints and atomic publication of a complete revision."""
import hashlib
import json
import os
import uuid
from pathlib import Path

from app.pipeline.delivery import atomic_write_json

PIPELINE_REVISION = "delivery-v20.26-word-text-stays-repairs-never-drop-20261016.2"


def execution_fingerprint(config: dict, source_sha: str) -> str:
    from app.adapters.ocr import ocr_runtime_identity
    raw = json.dumps([PIPELINE_REVISION, config, source_sha, ocr_runtime_identity()], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def begin_revision(root: Path, fingerprint: str, force: bool = False) -> tuple[Path, dict]:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text())
        revision = root / "revisions" / str(checkpoint.get("revision", ""))
        already_published = (root / "current").is_symlink() and (root / "current").resolve() == revision.resolve()
        if (not (force and already_published) and checkpoint.get("fingerprint") == fingerprint and revision.is_dir()
                and revision.resolve().parent == (root / "revisions").resolve()):
            return revision, checkpoint
    revision = root / "revisions" / uuid.uuid4().hex[:16]
    revision.mkdir(parents=True)
    checkpoint = {"schema_version": 1, "fingerprint": fingerprint,
                  "revision": revision.name, "completed": [], "stats": {}}
    atomic_write_json(checkpoint_path, checkpoint, indent=2)
    return revision, checkpoint


def save_checkpoint(root: Path, checkpoint: dict, ctx, stage: str) -> None:
    # Save IR before committing the completed-stage marker.
    from app.pipeline.stages.normalize import save_document_ir
    if ctx.document_ir:
        save_document_ir(ctx.document_ir, ctx.run_path)
    completed = list(checkpoint.get("completed", []))
    if stage not in completed:
        completed.append(stage)
    checkpoint.update({
        "completed": completed, "stats": ctx.stage_stats,
        "parse_cache_path": str(ctx.parse_cache_path) if ctx.parse_cache_path else None,
        "content_list_path": str(ctx.content_list_path) if ctx.content_list_path else None,
    })
    atomic_write_json(root / "checkpoint.json", checkpoint, ensure_ascii=False, indent=2)


def publish_revision(root: Path, revision: Path) -> None:
    """Check the exact bundle, then atomically switch one directory pointer."""
    manifest = json.loads((revision / "outputs/delivery_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = revision / name
        if not path.resolve().is_relative_to(revision.resolve()) or not path.is_file():
            raise ValueError(f"Invalid delivery member: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Delivery member changed after validation: {name}")
    temporary = root / f".current-{uuid.uuid4().hex}"
    temporary.symlink_to(Path("revisions") / revision.name, target_is_directory=True)
    try:
        os.replace(temporary, root / "current")
    finally:
        temporary.unlink(missing_ok=True)
    # API readers pin current.resolve() once per request. Compatibility aliases
    # preserve the historical workspace paths for offline consumers.
    for name in ("outputs", "assets", "document_ir.json", "source_map.json", "manifest.json"):
        if not (revision / name).exists():
            continue
        alias = root / name
        if alias.exists() and not alias.is_symlink():
            legacy = root / "legacy"
            legacy.mkdir(exist_ok=True)
            os.replace(alias, legacy / f"{name}-{uuid.uuid4().hex[:8]}")
        link = root / f".alias-{uuid.uuid4().hex}"
        link.symlink_to(Path("current") / name, target_is_directory=(revision / name).is_dir())
        os.replace(link, alias)
