"""
Verified pagination defect: GET /api/runs/outputs-summary
(backend/app/api/routes/runs.py:list_outputs_summary).

With has_documents_only=True (the endpoint's default, and what the
frontend's "download all documents" flow relies on via getAllOutputsSummary
in frontend/src/lib/api.ts), the pre-fix handler did this:

    candidate_limit = min(max(limit * 3, limit), 500)
    candidate_offset = offset if not has_documents_only else 0
    runs = await run_repo.list_all(limit=candidate_limit, offset=candidate_offset, ...)
    ... skip runs with no split documents ...
    if has_documents_only:
        total = len(items)
        items = items[offset:offset + limit]

It only ever inspected a single capped window of at most 500 of the newest
runs (offset forced to 0) and reported `total` as the document-bearing count
WITHIN that window. Once an install accumulates more runs than that window,
document-bearing runs older than the window are silently dropped from
"download all" lists, and `total` under-counts -- which also makes the
frontend's `while (page.runs.length > 0 && runs.length < page.total)` paging
loop stop after a single page.

These tests pin two things:

1. `_collect_outputs_summary` (runs.py): the route's logic, factored out
   into a helper that takes an internal `scan_batch_size` (default 500, same
   as the old hard cap) so tests can shrink the scan window far below the
   real default without fabricating hundreds of fixture runs to reproduce a
   multi-batch scan.
2. The `run_id DESC` tiebreaker added to RunRepository.list_all /
   list_by_doc (repositories.py), which makes paging deterministic when
   multiple runs share the same `created_at` (e.g. batch-created runs).

`test_regression_matches_ticket_scenario_through_public_route` and
`test_run_repository_list_all_orders_stably_on_created_at_tie` exercise the
real, unchanged-signature public functions directly (no test-only helper
needed) so they fail with genuine assertion errors against the pre-fix code,
not just an ImportError from the new helper not existing yet.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from app.api.routes.runs import list_outputs_summary
from app.config import settings
from app.db.database import Database
from app.db.repositories import DocRepository, RunRepository
from app.models.entities import DocCreate, RunStatus

# ---------------------------------------------------------------------------
# Fixtures / helpers (mirrors backend/tests/test_enrich_cache_key.py)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _open_db(tmp_path: Path):
    db = Database(db_path=tmp_path / "test.db")
    await db.connect()
    try:
        yield db
    finally:
        await db.disconnect()


@pytest.fixture
def fake_workspace(tmp_path, monkeypatch):
    """Point global settings workspace at tmp so get_run_path() resolves
    under a throwaway directory instead of the real workspace/store."""
    workspace = tmp_path / "ws"
    monkeypatch.setattr(settings, "workspace_path", workspace)
    return workspace


async def _seed_doc(db: Database, doc_id: str) -> None:
    await DocRepository(db).create(
        doc_id,
        DocCreate(source_path=f"{doc_id}.pdf", sha256=doc_id, ext=".pdf", size_bytes=10),
    )


async def _seed_run(
    db: Database,
    *,
    run_id: str,
    doc_id: str,
    created_at: str,
    status: RunStatus = RunStatus.SUCCEEDED,
) -> None:
    """Insert a run row with a fully-controlled run_id/created_at, bypassing
    RunRepository.create's auto-generated ULID/timestamp so paging and tie
    scenarios are deterministic instead of depending on wall-clock timing."""
    await db.connection.execute(
        """
        INSERT INTO runs (run_id, doc_id, profile, config_json, config_hash,
                        status, use_cache, force_stages, created_at, updated_at)
        VALUES (?, ?, 'accurate', '{}', 'hash', ?, 1, NULL, ?, ?)
        """,
        (run_id, doc_id, status.value, created_at, created_at),
    )
    await db.connection.commit()


def _write_documents(doc_id: str, run_id: str, count: int = 1) -> None:
    """Give a run a non-empty documents_index.json (document-bearing)."""
    outputs_dir = settings.get_run_path(doc_id, run_id) / "outputs"
    documents_dir = outputs_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    for i in range(count):
        filename = f"doc_{i}.md"
        (documents_dir / filename).write_text(f"# {filename}", encoding="utf-8")
        docs.append(
            {
                "document_id": f"doc_{i}",
                "kind": "main" if i == 0 else "form",
                "file": str(documents_dir / filename),
            }
        )
    (outputs_dir / "documents_index.json").write_text(
        json.dumps(docs, ensure_ascii=False), encoding="utf-8"
    )


def _timestamp(index: int) -> str:
    """Strictly increasing created_at: index 0 is oldest, higher is newer."""
    return f"2024-01-01T00:{index // 60:02d}:{index % 60:02d}"


async def _seed_timeline(
    db: Database,
    doc_id: str,
    count: int,
    documents_at: set[int],
    run_id_fmt: str = "01RUN{:04d}",
) -> list[str]:
    """Seed `count` runs on a strictly increasing created_at timeline
    (creation-index 0 oldest .. count-1 newest). Runs at `documents_at`
    creation-indexes are document-bearing. Returns run_ids in creation-index
    order (run_ids[0] is oldest)."""
    run_ids = [run_id_fmt.format(i) for i in range(count)]
    for i, run_id in enumerate(run_ids):
        await _seed_run(db, run_id=run_id, doc_id=doc_id, created_at=_timestamp(i))
        if i in documents_at:
            _write_documents(doc_id, run_id)
    return run_ids


# ---------------------------------------------------------------------------
# 1. Regression reproduced through the real public route (no helper needed)
# ---------------------------------------------------------------------------


async def test_regression_matches_ticket_scenario_through_public_route(
    tmp_path, fake_workspace
):
    """Reproduces the exact pre-fix symptom through the real, public
    `list_outputs_summary` route: a small `limit` used to shrink the pre-fix
    scan window (candidate_limit = min(max(limit*3, limit), 500)) below the
    run count, so a document-bearing run outside that window vanished and
    `total` under-counted. Post-fix, the internal scan window is decoupled
    from the caller's `limit`, so a small `limit` must not affect
    correctness. Also pins the response shape: {"runs": [...], "total": N}
    with the same per-item fields as before the refactor.
    """
    async with _open_db(tmp_path) as db:
        doc_id = "doc-c"
        await _seed_doc(db, doc_id)
        # 5 runs; only the OLDEST (creation-index 0) has documents. Pre-fix,
        # limit=1 shrinks candidate_limit to min(max(1*3,1),500)=3, which
        # (offset forced to 0) only reaches creation-indexes 4,3,2 in DESC
        # order -- never reaching index 0.
        run_ids = await _seed_timeline(db, doc_id, count=5, documents_at={0})

        page = await list_outputs_summary(
            status=RunStatus.SUCCEEDED,
            limit=1,
            offset=0,
            include_hidden=True,
            has_documents_only=True,
            db=db,
        )

        assert page["total"] == 1
        assert [item["run_id"] for item in page["runs"]] == [run_ids[0]]

        assert set(page.keys()) == {"runs", "total"}
        item = page["runs"][0]
        assert set(item.keys()) == {
            "run_id", "doc_id", "profile", "status", "created_at", "updated_at",
            "source_path", "source_name", "documents_total", "main_document_count",
            "extracted_document_count", "documents", "quality_gate_status",
            "quality_score", "quality_issue_count",
        }


# ---------------------------------------------------------------------------
# 2. Multi-batch paging: non-overlapping pages, correct total on every page
# ---------------------------------------------------------------------------


async def test_paging_returns_non_overlapping_pages_covering_all_document_bearing_runs(
    tmp_path, fake_workspace
):
    """limit smaller than the filtered total, scan_batch_size smaller than
    the run count: pages must not overlap, their union must be exactly the
    document-bearing runs, and total must be identical and correct on every
    page."""
    from app.api.routes.runs import _collect_outputs_summary

    async with _open_db(tmp_path) as db:
        doc_id = "doc-a"
        await _seed_doc(db, doc_id)
        # 10 runs, only even creation-indexes (0,2,4,6,8) are document-bearing.
        run_ids = await _seed_timeline(db, doc_id, count=10, documents_at={0, 2, 4, 6, 8})

        # DESC (newest-first) order is creation-index 9..0; filtered to
        # document-bearing indexes preserves relative order: 8,6,4,2,0.
        expected_order = [run_ids[i] for i in (8, 6, 4, 2, 0)]

        async def _page(offset: int):
            return await _collect_outputs_summary(
                db,
                status=RunStatus.SUCCEEDED,
                limit=2,
                offset=offset,
                include_hidden=True,
                has_documents_only=True,
                scan_batch_size=3,  # < 10 runs: forces multiple internal batches
            )

        page1, page2, page3 = await _page(0), await _page(2), await _page(4)

        assert page1["total"] == page2["total"] == page3["total"] == 5

        ids1 = [item["run_id"] for item in page1["runs"]]
        ids2 = [item["run_id"] for item in page2["runs"]]
        ids3 = [item["run_id"] for item in page3["runs"]]

        assert ids1 == expected_order[0:2]
        assert ids2 == expected_order[2:4]
        assert ids3 == expected_order[4:5]
        assert ids1 + ids2 + ids3 == expected_order
        assert len(set(ids1) | set(ids2) | set(ids3)) == 5  # no duplicates/overlap


# ---------------------------------------------------------------------------
# 3. Document-bearing runs entirely beyond the first scan batch (regression)
# ---------------------------------------------------------------------------


async def test_total_reflects_document_bearing_runs_beyond_first_scan_batch(
    tmp_path, fake_workspace
):
    """If only runs OLDER than the first scan batch are document-bearing,
    pre-fix code (which only ever inspected a single capped window) would
    report total=0 for this window. The fix must keep scanning subsequent
    batches until exhausted, so total (and the page) reflect the full
    filtered sequence."""
    from app.api.routes.runs import _collect_outputs_summary

    async with _open_db(tmp_path) as db:
        doc_id = "doc-b"
        await _seed_doc(db, doc_id)
        # 8 runs; only the 3 OLDEST (creation-index 0,1,2) are
        # document-bearing. DESC order is index 7..0, so the newest 5
        # (indexes 7..3) -- which entirely fill the first two
        # scan_batch_size=3 batches without a single document -- come
        # first; the document-bearing runs are only reached afterward.
        run_ids = await _seed_timeline(db, doc_id, count=8, documents_at={0, 1, 2})

        page = await _collect_outputs_summary(
            db,
            status=RunStatus.SUCCEEDED,
            limit=10,
            offset=0,
            include_hidden=True,
            has_documents_only=True,
            scan_batch_size=3,
        )

        assert page["total"] == 3
        assert [item["run_id"] for item in page["runs"]] == [
            run_ids[2],
            run_ids[1],
            run_ids[0],
        ]


# ---------------------------------------------------------------------------
# 4. offset beyond total
# ---------------------------------------------------------------------------


async def test_offset_beyond_total_returns_empty_runs_with_correct_total(
    tmp_path, fake_workspace
):
    from app.api.routes.runs import _collect_outputs_summary

    async with _open_db(tmp_path) as db:
        doc_id = "doc-d"
        await _seed_doc(db, doc_id)
        await _seed_timeline(db, doc_id, count=6, documents_at={0, 1, 2})  # 3 document-bearing

        page = await _collect_outputs_summary(
            db,
            status=RunStatus.SUCCEEDED,
            limit=10,
            offset=100,
            include_hidden=True,
            has_documents_only=True,
            scan_batch_size=2,
        )

        assert page["runs"] == []
        assert page["total"] == 3


# ---------------------------------------------------------------------------
# 5. Ordering stability across calls when runs share created_at
# ---------------------------------------------------------------------------


async def test_ordering_is_stable_across_calls_when_runs_share_created_at(
    tmp_path, fake_workspace
):
    """Multiple runs inserted with the identical created_at (as happens with
    batch-created runs) must be returned in the same order on every call --
    proving the `run_id DESC` tiebreaker added to RunRepository.list_all
    makes paging deterministic instead of depending on unspecified SQLite
    tie order."""
    from app.api.routes.runs import _collect_outputs_summary

    async with _open_db(tmp_path) as db:
        doc_id = "doc-e"
        await _seed_doc(db, doc_id)
        tied_created_at = "2024-06-01T00:00:00"
        # Deliberately out-of-order run_id insertion so a correct fix must
        # actively sort by run_id, not coincidentally preserve insert order.
        run_ids = [
            "01RUNTIE-CCCC",
            "01RUNTIE-AAAA",
            "01RUNTIE-EEEE",
            "01RUNTIE-BBBB",
            "01RUNTIE-DDDD",
        ]
        for run_id in run_ids:
            await _seed_run(db, run_id=run_id, doc_id=doc_id, created_at=tied_created_at)
            _write_documents(doc_id, run_id)

        expected_order = sorted(run_ids, reverse=True)  # run_id DESC tiebreak

        results = []
        for _ in range(3):
            page = await _collect_outputs_summary(
                db,
                status=RunStatus.SUCCEEDED,
                limit=10,
                offset=0,
                include_hidden=True,
                has_documents_only=True,
                scan_batch_size=2,  # forces multiple batches even among tied rows
            )
            results.append([item["run_id"] for item in page["runs"]])

        assert results[0] == expected_order
        assert results[1] == expected_order
        assert results[2] == expected_order


# ---------------------------------------------------------------------------
# 6. Direct repositories.py pin (independent of the runs.py helper)
# ---------------------------------------------------------------------------


async def test_run_repository_list_all_orders_stably_on_created_at_tie(
    tmp_path, fake_workspace
):
    """Pins the repositories.py fix directly: RunRepository.list_all must
    order ties on created_at deterministically (run_id DESC) rather than
    relying on unspecified SQLite tie order."""
    async with _open_db(tmp_path) as db:
        doc_id = "doc-h"
        await _seed_doc(db, doc_id)
        tied_created_at = "2024-06-02T00:00:00"
        run_ids = ["01Z-CCCC", "01Z-AAAA", "01Z-BBBB"]
        for run_id in run_ids:
            await _seed_run(db, run_id=run_id, doc_id=doc_id, created_at=tied_created_at)

        repo = RunRepository(db)
        first = await repo.list_all(limit=10, offset=0)
        second = await repo.list_all(limit=10, offset=0)

        expected = sorted(run_ids, reverse=True)
        assert [r.run_id for r in first] == expected
        assert [r.run_id for r in second] == expected


# ---------------------------------------------------------------------------
# 7. Bonus safety net: has_documents_only=False branch is untouched
# ---------------------------------------------------------------------------


async def test_has_documents_only_false_preserves_existing_candidate_behavior(
    tmp_path, fake_workspace
):
    """has_documents_only=False is explicitly out of scope for this fix;
    pin its pre-existing candidate_limit heuristic (items are NOT sliced to
    `limit`; total comes from a separate COUNT query) so the refactor into
    _collect_outputs_summary doesn't accidentally change it."""
    from app.api.routes.runs import _collect_outputs_summary

    async with _open_db(tmp_path) as db:
        doc_id = "doc-g"
        await _seed_doc(db, doc_id)
        # No documents on any run -- has_documents_only=False must still
        # return them (they are not filtered out by document presence).
        await _seed_timeline(db, doc_id, count=5, documents_at=set())

        page = await _collect_outputs_summary(
            db,
            status=RunStatus.SUCCEEDED,
            limit=1,
            offset=0,
            include_hidden=True,
            has_documents_only=False,
        )

        # candidate_limit = min(max(1*3, 1), 500) = 3: unsliced, so 3 items
        # come back even though the caller asked for limit=1.
        assert len(page["runs"]) == 3
        assert page["total"] == 5
