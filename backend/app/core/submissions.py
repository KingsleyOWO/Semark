"""Persistent execution intent, separate from merely creating a pending Run."""
import json
from pathlib import Path

from app.config import settings
from app.pipeline.delivery import atomic_write_json


def submission_path(run) -> Path:
    return settings.get_run_path(run.doc_id, run.run_id) / "execution.json"


def has_submission(run) -> bool:
    if not getattr(run, "doc_id", None):
        return False
    path = submission_path(run)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return data.get("run_id") == run.run_id and data.get("config_hash") == run.config_hash


def save_submission(run) -> Path:
    path = submission_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"run_id": run.run_id, "doc_id": run.doc_id, "config_hash": run.config_hash})
    return path
