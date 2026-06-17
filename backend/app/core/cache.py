"""
Cache management for pipeline stages.

Cache keys:
- Parse: (doc_id, mineru_config_hash, mineru_version)
- Enrich: (doc_id, block_id, vlm_config_hash, prompt_version)
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from app.config import MinerUConfig, settings
from app.db.database import Database
from app.db.repositories import CacheRepository, EnrichRepository
from app.models.entities import CacheEntryCreate, EnrichEntryCreate, StageName


def compute_config_hash(config: dict[str, Any] | MinerUConfig, version: str | None = None) -> str:
    """
    Compute hash of configuration for cache key.

    Args:
        config: MinerU config or dict
        version: Optional MinerU version to include in hash
    """
    if isinstance(config, MinerUConfig):
        config = config.model_dump()

    # Include version in hash if provided
    if version:
        config = {**config, "_mineru_version": version}

    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class CacheManager:
    """
    Manages caching for pipeline stages.

    Parse cache is stored in:
      workspace/store/docs/{doc_id}/cache/parse/{cache_key}/

    Enrich cache is stored in database (enrich_entries table).
    """

    def __init__(self, db: Database):
        self.db = db
        self.cache_repo = CacheRepository(db)
        self.enrich_repo = EnrichRepository(db)

    # ==================== Parse Cache ====================

    def get_parse_cache_key(
        self, doc_id: str, config: MinerUConfig, version: str | None = None
    ) -> str:
        """
        Generate cache key for parse stage.

        Args:
            doc_id: Document ID
            config: MinerU configuration
            version: MinerU version (recommended for cache invalidation on upgrades)
        """
        config_hash = compute_config_hash(config, version=version)
        return f"{doc_id}_{config_hash}"

    def get_parse_cache_path(self, doc_id: str, cache_key: str) -> Path:
        """Get filesystem path for parse cache."""
        return settings.get_doc_path(doc_id) / "cache" / "parse" / cache_key

    async def get_parse_cache(
        self,
        doc_id: str,
        config: MinerUConfig,
        version: str | None = None,
    ) -> Path | None:
        """
        Check if parse cache exists and return path.

        Args:
            doc_id: Document ID
            config: MinerU configuration
            version: MinerU version (for cache key)

        Returns None if cache miss.
        """
        cache_key = self.get_parse_cache_key(doc_id, config, version=version)

        # Check database
        entry = await self.cache_repo.get(cache_key)
        if not entry:
            return None

        # Verify path exists
        cache_path = Path(entry.path)
        if not cache_path.exists():
            # Cache entry exists but files are gone, invalidate
            await self.cache_repo.invalidate_by_doc_stage(doc_id, StageName.PARSE)
            return None

        return cache_path

    async def set_parse_cache(
        self,
        doc_id: str,
        config: MinerUConfig,
        source_path: Path,
        version: str | None = None,
    ) -> Path:
        """
        Store parse output in cache.

        Args:
            doc_id: Document ID
            config: MinerU configuration
            source_path: Path to MinerU output to cache
            version: MinerU version (for cache key)

        Copies source_path contents to cache directory.
        Returns the cache path.
        """
        cache_key = self.get_parse_cache_key(doc_id, config, version=version)
        cache_path = self.get_parse_cache_path(doc_id, cache_key)

        # Ensure cache directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy output to cache (or move if same filesystem)
        if cache_path.exists():
            shutil.rmtree(cache_path)

        shutil.copytree(source_path, cache_path)

        # Store in database
        await self.cache_repo.set(
            CacheEntryCreate(
                cache_key=cache_key,
                doc_id=doc_id,
                stage=StageName.PARSE,
                config_hash=compute_config_hash(config, version=version),
                path=str(cache_path),
            )
        )

        return cache_path

    async def invalidate_parse_cache(self, doc_id: str) -> int:
        """Invalidate all parse cache for a document."""
        count = await self.cache_repo.invalidate_by_doc_stage(doc_id, StageName.PARSE)

        # Also delete files
        cache_dir = settings.get_doc_path(doc_id) / "cache" / "parse"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        return count

    # ==================== Enrich Cache ====================

    async def get_enrich_cache(
        self,
        doc_id: str,
        block_id: str,
        vlm_config_hash: str,
        prompt_version: str,
    ) -> dict[str, Any] | None:
        """
        Get cached enrichment for a block.

        Returns None if cache miss.
        """
        entry = await self.enrich_repo.get(
            doc_id=doc_id,
            block_id=block_id,
            vlm_config_hash=vlm_config_hash,
            prompt_version=prompt_version,
        )

        if entry:
            return entry.output

        return None

    async def set_enrich_cache(
        self,
        doc_id: str,
        block_id: str,
        vlm_config_hash: str,
        prompt_version: str,
        output: dict[str, Any],
    ) -> None:
        """Store enrichment in cache."""
        await self.enrich_repo.set(
            EnrichEntryCreate(
                doc_id=doc_id,
                block_id=block_id,
                vlm_config_hash=vlm_config_hash,
                prompt_version=prompt_version,
                output=output,
            )
        )

    async def list_enrich_cache(self, doc_id: str) -> list[dict[str, Any]]:
        """List all enrichment cache entries for a document."""
        entries = await self.enrich_repo.list_by_doc(doc_id)
        return [
            {
                "block_id": e.block_id,
                "vlm_config_hash": e.vlm_config_hash,
                "prompt_version": e.prompt_version,
                "output": e.output,
                "created_at": e.created_at.isoformat(),
            }
            for e in entries
        ]

    async def invalidate_enrich_cache(self, doc_id: str) -> int:
        """Invalidate all enrich cache for a document."""
        return await self.enrich_repo.invalidate_by_doc(doc_id)

    # ==================== General ====================

    async def invalidate_all(self, doc_id: str) -> dict[str, int]:
        """Invalidate all cache for a document."""
        parse_count = await self.invalidate_parse_cache(doc_id)
        enrich_count = await self.invalidate_enrich_cache(doc_id)

        return {"parse": parse_count, "enrich": enrich_count}

    async def invalidate_stages(
        self,
        doc_id: str,
        stages: list[str],
    ) -> dict[str, int]:
        """Invalidate cache for specific stages."""
        result: dict[str, int] = {}

        for stage in stages:
            if stage == "parse":
                result["parse"] = await self.invalidate_parse_cache(doc_id)
            elif stage == "enrich":
                result["enrich"] = await self.invalidate_enrich_cache(doc_id)

        return result

    async def get_cache_stats(self, doc_id: str) -> dict[str, Any]:
        """Get cache statistics for a document."""
        # Check parse cache
        parse_cache_dir = settings.get_doc_path(doc_id) / "cache" / "parse"
        parse_entries = list(parse_cache_dir.glob("*")) if parse_cache_dir.exists() else []

        # Check enrich cache
        enrich_entries = await self.enrich_repo.list_by_doc(doc_id)

        return {
            "parse": {
                "entries": len(parse_entries),
                "keys": [p.name for p in parse_entries],
            },
            "enrich": {
                "entries": len(enrich_entries),
                "blocks": list(set(e.block_id for e in enrich_entries)),
            },
        }
