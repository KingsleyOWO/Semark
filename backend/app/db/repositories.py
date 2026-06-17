"""
Repository classes for database operations.
"""

import json
from datetime import datetime
from typing import Any

import aiosqlite
import ulid

from app.db.database import Database
from app.models.entities import (
    CacheEntry,
    CacheEntryCreate,
    Doc,
    DocCreate,
    EnrichEntry,
    EnrichEntryCreate,
    Run,
    RunCreate,
    RunStage,
    RunStageCreate,
    RunStatus,
    RunWithStages,
    StageName,
    StageStatus,
)


def _parse_json(value: str | None) -> dict[str, Any] | None:
    """Parse JSON string to dict."""
    if value is None:
        return None
    return json.loads(value)


def _to_json(value: dict[str, Any] | list | None) -> str | None:
    """Convert dict/list to JSON string."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


class DocRepository:
    """Repository for document operations."""

    def __init__(self, db: Database):
        self.db = db

    async def create(self, doc_id: str, data: DocCreate) -> Doc:
        """Create a new document."""
        now = datetime.now().isoformat()
        await self.db.connection.execute(
            """
            INSERT INTO docs (doc_id, source_path, sha256, ext, size_bytes, created_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                data.source_path,
                data.sha256,
                data.ext,
                data.size_bytes,
                now,
                _to_json(data.meta),
            ),
        )
        await self.db.connection.commit()
        return await self.get(doc_id)  # type: ignore

    async def get(self, doc_id: str) -> Doc | None:
        """Get a document by ID."""
        async with self.db.connection.execute(
            "SELECT * FROM docs WHERE doc_id = ?", (doc_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return Doc(
                doc_id=row["doc_id"],
                source_path=row["source_path"],
                sha256=row["sha256"],
                ext=row["ext"],
                size_bytes=row["size_bytes"],
                created_at=datetime.fromisoformat(row["created_at"]),
                meta=_parse_json(row["meta_json"]),
            )

    async def get_by_sha256(self, sha256: str) -> Doc | None:
        """Get a document by SHA256 hash."""
        async with self.db.connection.execute(
            "SELECT * FROM docs WHERE sha256 = ?", (sha256,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return Doc(
                doc_id=row["doc_id"],
                source_path=row["source_path"],
                sha256=row["sha256"],
                ext=row["ext"],
                size_bytes=row["size_bytes"],
                created_at=datetime.fromisoformat(row["created_at"]),
                meta=_parse_json(row["meta_json"]),
            )

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Doc]:
        """List all documents."""
        async with self.db.connection.execute(
            "SELECT * FROM docs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                Doc(
                    doc_id=row["doc_id"],
                    source_path=row["source_path"],
                    sha256=row["sha256"],
                    ext=row["ext"],
                    size_bytes=row["size_bytes"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    meta=_parse_json(row["meta_json"]),
                )
                for row in rows
            ]

    async def count(self) -> int:
        """Count total documents."""
        async with self.db.connection.execute("SELECT COUNT(*) FROM docs") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        cursor = await self.db.connection.execute(
            "DELETE FROM docs WHERE doc_id = ?", (doc_id,)
        )
        await self.db.connection.commit()
        return cursor.rowcount > 0


class RunRepository:
    """Repository for run operations."""

    def __init__(self, db: Database):
        self.db = db

    def _generate_run_id(self) -> str:
        """Generate a new run ID using ULID."""
        return str(ulid.new())

    async def create(self, data: RunCreate) -> Run:
        """Create a new run."""
        run_id = self._generate_run_id()
        now = datetime.now().isoformat()

        await self.db.connection.execute(
            """
            INSERT INTO runs (run_id, doc_id, profile, config_json, config_hash,
                            status, use_cache, force_stages, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                data.doc_id,
                data.profile,
                _to_json(data.config),
                data.config_hash,
                RunStatus.PENDING.value,
                1 if data.use_cache else 0,
                _to_json([s.value for s in data.force_stages] if data.force_stages else None),
                now,
                now,
            ),
        )
        await self.db.connection.commit()
        return await self.get(run_id)  # type: ignore

    async def get(self, run_id: str) -> Run | None:
        """Get a run by ID."""
        async with self.db.connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_run(row)

    async def get_with_stages(self, run_id: str) -> RunWithStages | None:
        """Get a run with its stages."""
        run = await self.get(run_id)
        if not run:
            return None

        stages = await RunStageRepository(self.db).list_by_run(run_id)
        return RunWithStages(**run.model_dump(), stages=stages)

    async def list_by_doc(self, doc_id: str, limit: int = 50) -> list[Run]:
        """List runs for a document."""
        async with self.db.connection.execute(
            "SELECT * FROM runs WHERE doc_id = ? ORDER BY created_at DESC LIMIT ?",
            (doc_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_run(row) for row in rows]

    async def list_all(
        self,
        status: RunStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        exclude_run_ids: set[str] | None = None,
    ) -> list[Run]:
        """List all runs with optional status filter and hidden-run exclusion."""
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if exclude_run_ids:
            placeholders = ", ".join("?" for _ in exclude_run_ids)
            clauses.append(f"run_id NOT IN ({placeholders})")
            params.extend(sorted(exclude_run_ids))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self.db.connection.execute(query, tuple(params)) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_run(row) for row in rows]

    async def count(
        self,
        status: RunStatus | None = None,
        exclude_run_ids: set[str] | None = None,
    ) -> int:
        """Count total runs with optional status filter and hidden-run exclusion."""
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if exclude_run_ids:
            placeholders = ", ".join("?" for _ in exclude_run_ids)
            clauses.append(f"run_id NOT IN ({placeholders})")
            params.extend(sorted(exclude_run_ids))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT COUNT(*) FROM runs {where}"

        async with self.db.connection.execute(query, tuple(params)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def update_status(self, run_id: str, status: RunStatus) -> None:
        """Update run status."""
        now = datetime.now().isoformat()
        await self.db.connection.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
            (status.value, now, run_id),
        )
        await self.db.connection.commit()

    async def delete(self, run_id: str) -> bool:
        """Delete a run and its stages."""
        # Delete stages first (foreign key)
        await self.db.connection.execute(
            "DELETE FROM run_stages WHERE run_id = ?", (run_id,)
        )
        cursor = await self.db.connection.execute(
            "DELETE FROM runs WHERE run_id = ?", (run_id,)
        )
        await self.db.connection.commit()
        return cursor.rowcount > 0

    def _row_to_run(self, row: aiosqlite.Row) -> Run:
        """Convert database row to Run model."""
        force_stages_raw = _parse_json(row["force_stages"])
        force_stages = (
            [StageName(s) for s in force_stages_raw] if force_stages_raw else None
        )
        return Run(
            run_id=row["run_id"],
            doc_id=row["doc_id"],
            profile=row["profile"],
            config=_parse_json(row["config_json"]) or {},
            config_hash=row["config_hash"],
            status=RunStatus(row["status"]),
            use_cache=bool(row["use_cache"]),
            force_stages=force_stages,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


class RunStageRepository:
    """Repository for run stage operations."""

    def __init__(self, db: Database):
        self.db = db

    async def create(self, data: RunStageCreate) -> RunStage:
        """Create a new run stage."""
        await self.db.connection.execute(
            "INSERT INTO run_stages (run_id, stage, status) VALUES (?, ?, ?)",
            (data.run_id, data.stage.value, StageStatus.PENDING.value),
        )
        await self.db.connection.commit()
        return await self.get(data.run_id, data.stage)  # type: ignore

    async def create_all_stages(self, run_id: str) -> list[RunStage]:
        """Create all pipeline stages for a run."""
        stages = []
        for stage in StageName:
            stage_obj = await self.create(RunStageCreate(run_id=run_id, stage=stage))
            stages.append(stage_obj)
        return stages

    async def get(self, run_id: str, stage: StageName) -> RunStage | None:
        """Get a specific stage for a run."""
        async with self.db.connection.execute(
            "SELECT * FROM run_stages WHERE run_id = ? AND stage = ?",
            (run_id, stage.value),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_stage(row)

    async def list_by_run(self, run_id: str) -> list[RunStage]:
        """List all stages for a run."""
        async with self.db.connection.execute(
            "SELECT * FROM run_stages WHERE run_id = ? ORDER BY id",
            (run_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_stage(row) for row in rows]

    async def update_status(
        self,
        run_id: str,
        stage: StageName,
        status: StageStatus,
        error: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Update stage status with optional error/stats."""
        now = datetime.now().isoformat()

        if status == StageStatus.RUNNING:
            await self.db.connection.execute(
                "UPDATE run_stages SET status = ?, started_at = ? WHERE run_id = ? AND stage = ?",
                (status.value, now, run_id, stage.value),
            )
        elif status in (StageStatus.SUCCEEDED, StageStatus.FAILED, StageStatus.CANCELED):
            await self.db.connection.execute(
                """
                UPDATE run_stages
                SET status = ?, finished_at = ?, error_json = ?, stats_json = ?
                WHERE run_id = ? AND stage = ?
                """,
                (status.value, now, _to_json(error), _to_json(stats), run_id, stage.value),
            )
        else:
            await self.db.connection.execute(
                "UPDATE run_stages SET status = ? WHERE run_id = ? AND stage = ?",
                (status.value, run_id, stage.value),
            )

        await self.db.connection.commit()

    async def update_running_stats(
        self,
        run_id: str,
        stage: StageName,
        stats: dict[str, Any],
    ) -> None:
        """Update stats for an in-flight running stage without resetting timers."""
        await self.db.connection.execute(
            """
            UPDATE run_stages
            SET stats_json = ?
            WHERE run_id = ? AND stage = ? AND status = ?
            """,
            (_to_json(stats), run_id, stage.value, StageStatus.RUNNING.value),
        )
        await self.db.connection.commit()

    def _row_to_stage(self, row: aiosqlite.Row) -> RunStage:
        """Convert database row to RunStage model."""
        return RunStage(
            id=row["id"],
            run_id=row["run_id"],
            stage=StageName(row["stage"]),
            status=StageStatus(row["status"]),
            started_at=(
                datetime.fromisoformat(row["started_at"]) if row["started_at"] else None
            ),
            finished_at=(
                datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None
            ),
            error=_parse_json(row["error_json"]),
            stats=_parse_json(row["stats_json"]),
        )


class CacheRepository:
    """Repository for cache operations."""

    def __init__(self, db: Database):
        self.db = db

    async def get(self, cache_key: str) -> CacheEntry | None:
        """Get a cache entry by key."""
        async with self.db.connection.execute(
            "SELECT * FROM cache_entries WHERE cache_key = ?", (cache_key,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return CacheEntry(
                cache_key=row["cache_key"],
                doc_id=row["doc_id"],
                stage=StageName(row["stage"]),
                config_hash=row["config_hash"],
                path=row["path"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    async def set(self, data: CacheEntryCreate) -> CacheEntry:
        """Create or update a cache entry."""
        now = datetime.now().isoformat()
        await self.db.connection.execute(
            """
            INSERT OR REPLACE INTO cache_entries
            (cache_key, doc_id, stage, config_hash, path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                data.cache_key,
                data.doc_id,
                data.stage.value,
                data.config_hash,
                data.path,
                now,
            ),
        )
        await self.db.connection.commit()
        return await self.get(data.cache_key)  # type: ignore

    async def invalidate_by_doc_stage(self, doc_id: str, stage: StageName) -> int:
        """Invalidate cache entries for a document and stage."""
        cursor = await self.db.connection.execute(
            "DELETE FROM cache_entries WHERE doc_id = ? AND stage = ?",
            (doc_id, stage.value),
        )
        await self.db.connection.commit()
        return cursor.rowcount

    async def invalidate_by_doc(self, doc_id: str) -> int:
        """Invalidate all cache entries for a document."""
        cursor = await self.db.connection.execute(
            "DELETE FROM cache_entries WHERE doc_id = ?", (doc_id,)
        )
        await self.db.connection.commit()
        return cursor.rowcount


class EnrichRepository:
    """Repository for enrichment cache operations."""

    def __init__(self, db: Database):
        self.db = db

    async def get(
        self,
        doc_id: str,
        block_id: str,
        vlm_config_hash: str,
        prompt_version: str,
    ) -> EnrichEntry | None:
        """Get an enrichment entry."""
        async with self.db.connection.execute(
            """
            SELECT * FROM enrich_entries
            WHERE doc_id = ? AND block_id = ? AND vlm_config_hash = ? AND prompt_version = ?
            """,
            (doc_id, block_id, vlm_config_hash, prompt_version),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return EnrichEntry(
                id=row["id"],
                doc_id=row["doc_id"],
                block_id=row["block_id"],
                vlm_config_hash=row["vlm_config_hash"],
                prompt_version=row["prompt_version"],
                output=_parse_json(row["output_json"]) or {},
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    async def set(self, data: EnrichEntryCreate) -> EnrichEntry:
        """Create or update an enrichment entry."""
        now = datetime.now().isoformat()
        await self.db.connection.execute(
            """
            INSERT OR REPLACE INTO enrich_entries
            (doc_id, block_id, vlm_config_hash, prompt_version, output_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                data.doc_id,
                data.block_id,
                data.vlm_config_hash,
                data.prompt_version,
                _to_json(data.output),
                now,
            ),
        )
        await self.db.connection.commit()
        return await self.get(
            data.doc_id, data.block_id, data.vlm_config_hash, data.prompt_version
        )  # type: ignore

    async def list_by_doc(self, doc_id: str) -> list[EnrichEntry]:
        """List all enrichment entries for a document."""
        async with self.db.connection.execute(
            "SELECT * FROM enrich_entries WHERE doc_id = ? ORDER BY created_at",
            (doc_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                EnrichEntry(
                    id=row["id"],
                    doc_id=row["doc_id"],
                    block_id=row["block_id"],
                    vlm_config_hash=row["vlm_config_hash"],
                    prompt_version=row["prompt_version"],
                    output=_parse_json(row["output_json"]) or {},
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

    async def invalidate_by_doc(self, doc_id: str) -> int:
        """Invalidate all enrichment entries for a document."""
        cursor = await self.db.connection.execute(
            "DELETE FROM enrich_entries WHERE doc_id = ?", (doc_id,)
        )
        await self.db.connection.commit()
        return cursor.rowcount


class SettingsRepository:
    """Repository for app settings (key-value store)."""

    def __init__(self, db: Database):
        self.db = db

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get a setting by key."""
        async with self.db.connection.execute(
            "SELECT value_json FROM app_settings WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return _parse_json(row["value_json"])

    async def set(self, key: str, value: dict[str, Any]) -> None:
        """Set a setting."""
        now = datetime.now().isoformat()
        await self.db.connection.execute(
            """
            INSERT OR REPLACE INTO app_settings (key, value_json, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, _to_json(value), now),
        )
        await self.db.connection.commit()

    async def delete(self, key: str) -> bool:
        """Delete a setting."""
        cursor = await self.db.connection.execute(
            "DELETE FROM app_settings WHERE key = ?", (key,)
        )
        await self.db.connection.commit()
        return cursor.rowcount > 0

    async def list_all(self) -> dict[str, Any]:
        """List all settings."""
        async with self.db.connection.execute(
            "SELECT key, value_json FROM app_settings"
        ) as cursor:
            rows = await cursor.fetchall()
            return {row["key"]: _parse_json(row["value_json"]) for row in rows}
