"""
SQLite database connection and schema management.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from app.config import settings

SCHEMA_VERSION = 2

SCHEMA_SQL = """
-- Documents table
CREATE TABLE IF NOT EXISTS docs (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    ext TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    meta_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_docs_sha256 ON docs(sha256);

-- Runs table
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    use_cache INTEGER NOT NULL DEFAULT 1,
    force_stages TEXT,  -- JSON array of stages to force re-run
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (doc_id) REFERENCES docs(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_doc_id ON runs(doc_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);

-- Run stages table
CREATE TABLE IF NOT EXISTS run_stages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT,
    finished_at TEXT,
    error_json TEXT,
    stats_json TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    UNIQUE(run_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_run_stages_run_id ON run_stages(run_id);

-- Cache entries table
CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (doc_id) REFERENCES docs(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_cache_doc_stage ON cache_entries(doc_id, stage);

-- Enrich entries table (VLM enrichments cache)
CREATE TABLE IF NOT EXISTS enrich_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    vlm_config_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (doc_id) REFERENCES docs(doc_id),
    UNIQUE(doc_id, block_id, vlm_config_hash, prompt_version)
);

CREATE INDEX IF NOT EXISTS idx_enrich_doc_block ON enrich_entries(doc_id, block_id);

-- App settings table (key-value store for runtime config)
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Schema version table
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


class Database:
    """Async SQLite database manager."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.database_path
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Initialize database connection and create schema if needed."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row

        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys = ON")

        # Initialize schema
        await self._init_schema()

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        assert self._connection is not None

        # Check current schema version
        try:
            async with self._connection.execute(
                "SELECT version FROM schema_version LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
                current_version = row["version"] if row else 0
        except aiosqlite.OperationalError:
            current_version = 0

        if current_version < SCHEMA_VERSION:
            # Apply schema
            await self._connection.executescript(SCHEMA_SQL)

            # Update version
            await self._connection.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            await self._connection.commit()

    @property
    def connection(self) -> aiosqlite.Connection:
        """Get the active connection."""
        if self._connection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._connection

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager for transactions."""
        conn = self.connection
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


# Global database instance
db = Database()


async def get_db() -> Database:
    """Dependency for FastAPI routes."""
    return db
