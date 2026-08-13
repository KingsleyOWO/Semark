"""
Verified defect: deleting a document left its surplus runs behind.

Both delete paths in backend/app/api/routes/docs.py --
``DELETE /api/docs/{doc_id}`` and ``POST /api/docs/batch-delete`` -- were
written as:

    # Delete all runs (this also deletes run_stages)
    runs = await run_repo.list_by_doc(doc_id)
    for run in runs:
        await run_repo.delete(run.run_id)

``RunRepository.list_by_doc`` (repositories.py) defaults to ``limit=50``,
which is the right default for the listing screens it was written for and
silently wrong here: the comment and the endpoint docstring both promise
"all its runs", but a document re-processed more than 50 times kept the
surplus rows after its own row was deleted.

The fix drains in batches via ``_delete_all_runs_for_doc``. These tests seed
a document with more runs than that default and assert nothing is left, so
they fail against the pre-fix code with a genuine row count rather than an
ImportError.
"""

import pytest
from contextlib import asynccontextmanager
from pathlib import Path

from app.api.routes.docs import _delete_all_runs_for_doc
from app.db.database import Database
from app.db.repositories import DocRepository, RunRepository
from app.models.entities import DocCreate, RunStatus

# More runs than RunRepository.list_by_doc's default page, by enough that a
# single undrained page is unmistakable in the assertion message.
RUNS_PER_DOC = 63


@asynccontextmanager
async def _open_db(tmp_path: Path):
    db = Database(db_path=tmp_path / "test.db")
    await db.connect()
    try:
        yield db
    finally:
        await db.disconnect()


async def _seed_doc_with_runs(db: Database, doc_id: str, run_count: int) -> None:
    await DocRepository(db).create(
        doc_id,
        DocCreate(source_path=f"{doc_id}.pdf", sha256=doc_id, ext=".pdf", size_bytes=10),
    )
    for index in range(run_count):
        # Fixed run_id/created_at rather than RunRepository.create's generated
        # ULID so the ordering the delete walks is deterministic.
        stamp = f"2026-08-13T00:{index // 60:02d}:{index % 60:02d}"
        await db.connection.execute(
            """
            INSERT INTO runs (run_id, doc_id, profile, config_json, config_hash,
                            status, use_cache, force_stages, created_at, updated_at)
            VALUES (?, ?, 'accurate', '{}', 'hash', ?, 1, NULL, ?, ?)
            """,
            (f"{doc_id}_run{index:04d}", doc_id, RunStatus.SUCCEEDED.value, stamp, stamp),
        )
    await db.connection.commit()


async def _remaining_runs(db: Database, doc_id: str) -> int:
    async with db.connection.execute(
        "SELECT COUNT(*) FROM runs WHERE doc_id = ?", (doc_id,)
    ) as cursor:
        row = await cursor.fetchone()
        return row[0]


@pytest.mark.asyncio
async def test_delete_drains_every_run_past_the_list_page_default(tmp_path):
    async with _open_db(tmp_path) as db:
        await _seed_doc_with_runs(db, "doc_over_page", RUNS_PER_DOC)
        run_repo = RunRepository(db)

        # The bug in one line: the listing helper the delete paths relied on
        # hands back only its default page, not the whole set.
        assert len(await run_repo.list_by_doc("doc_over_page")) < RUNS_PER_DOC

        deleted = await _delete_all_runs_for_doc(run_repo, "doc_over_page")

        assert deleted == RUNS_PER_DOC
        assert await _remaining_runs(db, "doc_over_page") == 0


@pytest.mark.asyncio
async def test_delete_touches_only_the_named_document(tmp_path):
    async with _open_db(tmp_path) as db:
        await _seed_doc_with_runs(db, "doc_target", RUNS_PER_DOC)
        await _seed_doc_with_runs(db, "doc_bystander", 5)

        await _delete_all_runs_for_doc(RunRepository(db), "doc_target")

        assert await _remaining_runs(db, "doc_target") == 0
        assert await _remaining_runs(db, "doc_bystander") == 5


@pytest.mark.asyncio
async def test_delete_on_a_document_with_no_runs_is_a_no_op(tmp_path):
    async with _open_db(tmp_path) as db:
        await _seed_doc_with_runs(db, "doc_empty", 0)

        assert await _delete_all_runs_for_doc(RunRepository(db), "doc_empty") == 0
