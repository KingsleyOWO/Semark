"""A run can send its model calls to a second machine serving the same model.

Batch 2026-10-08: dozens of documents at up to 6 hours each, and Ollama runs this
model architecture one request at a time, so one machine cannot go faster. A second
machine halves the batch only if a run can be pointed at its Ollama; the run's
database, files and delivery stay on this machine.
"""
import pytest

from app.api.routes import runs as runs_routes
from app.config import settings
from app.db.database import Database
from app.db.repositories import DocRepository, RunRepository
from app.models.api import RunCreateRequest
from app.models.entities import DocCreate

SECOND_MACHINE = "http://10.0.0.2:11434/v1"


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "workspace_path", tmp_path)
    database = Database(tmp_path / "test.db")
    await database.connect()
    await DocRepository(database).create("doc-test", DocCreate(
        source_path="示範報告.docx", ext="docx", sha256="test", size_bytes=0))
    try:
        yield database
    finally:
        await database.disconnect()


async def _config(db, **request):
    response = await runs_routes.create_run(RunCreateRequest(doc_id="doc-test", **request), db=db)
    return (await RunRepository(db).get(response.run_id)).config


async def test_a_run_sent_to_a_second_machine_reads_and_reviews_there(db):
    local = await _config(db)
    remote = await _config(db, vlm_base_url=SECOND_MACHINE)

    assert local["vlm"]["base_url"] == local["review_vlm"]["base_url"] != SECOND_MACHINE
    assert remote["vlm"]["base_url"] == remote["review_vlm"]["base_url"] == SECOND_MACHINE
    # Only the address moves: same model, same decoding.
    for role in ("vlm", "review_vlm"):
        assert {k: v for k, v in remote[role].items() if k != "base_url"} == \
            {k: v for k, v in local[role].items() if k != "base_url"}


async def test_a_reviewer_configured_on_its_own_server_stays_there(db, monkeypatch):
    own_reviewer = "http://10.0.0.3:8000/v1"
    get_review = runs_routes.get_review_vlm_config

    async def review_elsewhere(database):
        return {**await get_review(database), "base_url": own_reviewer}

    monkeypatch.setattr(runs_routes, "get_review_vlm_config", review_elsewhere)

    config = await _config(db, vlm_base_url=SECOND_MACHINE)

    assert config["vlm"]["base_url"] == SECOND_MACHINE
    assert config["review_vlm"]["base_url"] == own_reviewer
