"""
MinerU hybrid-backend `effort` passthrough tests.

Ground truth verified against installed mineru 3.4.2
(.venv/lib/python3.12/site-packages/mineru/cli/):
- CLI flag: `--effort [medium|high]`, default "medium"
  (client.py; HYBRID_EFFORT_CHOICES / DEFAULT_HYBRID_EFFORT in backend_options.py)
- Applies only to hybrid-* backends; Semark's "hybrid-auto-engine" is accepted
  via MinerU's LEGACY_BACKEND_ALIASES -> "hybrid-engine".
"""

import unittest
from pathlib import Path
from unittest import mock

from app.adapters.mineru import MinerUAdapter
from app.config import (
    PROFILES,
    MinerUBackend,
    MinerUConfig,
    MinerUEffort,
    PipelineConfig,
    ProfileName,
    VLMConfig,
)
from app.core.cache import compute_config_hash


def _build_args(config: MinerUConfig) -> list[str]:
    adapter = MinerUAdapter(config)
    return adapter._build_args(Path("/tmp/in.pdf"), Path("/tmp/out"), config)


def _effort_values(args: list[str]) -> list[str]:
    return [args[i + 1] for i, arg in enumerate(args) if arg == "--effort"]


class MinerUEffortCliArgsTest(unittest.TestCase):
    """CLI arg construction in MinerUAdapter._build_args."""

    def test_pipeline_backend_with_effort_set_omits_flag(self):
        """Non-hybrid backend + effort set: no --effort flag, no crash, debug log."""
        config = MinerUConfig(backend=MinerUBackend.PIPELINE, effort=MinerUEffort.HIGH)
        with self.assertLogs("app.adapters.mineru", level="DEBUG") as logs:
            args = _build_args(config)
        self.assertNotIn("--effort", args)
        self.assertTrue(any("effort" in line for line in logs.output))

    def test_hybrid_backend_with_effort_high_emits_flag(self):
        config = MinerUConfig(
            backend=MinerUBackend.HYBRID_AUTO_ENGINE, effort=MinerUEffort.HIGH
        )
        self.assertEqual(_effort_values(_build_args(config)), ["high"])

    def test_hybrid_http_client_with_effort_medium_emits_flag(self):
        config = MinerUConfig(backend=MinerUBackend.HYBRID_HTTP_CLIENT, effort="medium")
        self.assertEqual(_effort_values(_build_args(config)), ["medium"])

    def test_hybrid_backend_without_effort_omits_flag(self):
        config = MinerUConfig(backend=MinerUBackend.HYBRID_AUTO_ENGINE)
        self.assertIsNone(config.effort)
        self.assertNotIn("--effort", _build_args(config))

    def test_vlm_backend_with_effort_set_omits_flag(self):
        config = MinerUConfig(
            backend=MinerUBackend.VLM_AUTO_ENGINE, effort=MinerUEffort.HIGH
        )
        self.assertNotIn("--effort", _build_args(config))

    def test_effort_rejects_values_unknown_to_mineru_cli(self):
        """Mirror mineru 3.4.2 validate_effort: only medium|high are accepted."""
        with self.assertRaises(ValueError):
            MinerUConfig(backend=MinerUBackend.HYBRID_AUTO_ENGINE, effort="ultra")


class MinerUEffortCacheKeyTest(unittest.TestCase):
    """Switching effort must invalidate the parse cache."""

    def test_parse_config_hash_changes_with_effort(self):
        base = MinerUConfig(backend=MinerUBackend.HYBRID_AUTO_ENGINE)
        medium = MinerUConfig(
            backend=MinerUBackend.HYBRID_AUTO_ENGINE, effort=MinerUEffort.MEDIUM
        )
        high = MinerUConfig(
            backend=MinerUBackend.HYBRID_AUTO_ENGINE, effort=MinerUEffort.HIGH
        )
        hashes = {compute_config_hash(c) for c in (base, medium, high)}
        self.assertEqual(len(hashes), 3)


class MinerUEffortRunConfigTest(unittest.TestCase):
    """Saved MinerU runtime settings must carry effort into run configs."""

    def test_runtime_settings_forward_effort_into_run_config(self):
        from app.api.routes.runs import _apply_runtime_settings, merge_config

        merged = merge_config(PROFILES[ProfileName.FAST], None)
        vlm_settings = VLMConfig().model_dump()
        mineru_settings = {
            **MinerUConfig().model_dump(),
            "backend": MinerUBackend.HYBRID_AUTO_ENGINE.value,
            "effort": MinerUEffort.HIGH.value,
        }

        _apply_runtime_settings(merged, mineru_settings, vlm_settings, vlm_settings)

        self.assertEqual(merged["mineru"]["effort"], "high")
        # Orchestrator revalidates run.config through PipelineConfig.
        validated = PipelineConfig(**merged)
        self.assertEqual(validated.mineru.backend, MinerUBackend.HYBRID_AUTO_ENGINE)
        self.assertEqual(validated.mineru.effort, MinerUEffort.HIGH)


class _StubSettingsRepo:
    """In-memory stand-in for SettingsRepository in settings routes."""

    store: dict[str, dict] = {}

    def __init__(self, db):
        pass

    async def get(self, key):
        return self.__class__.store.get(key)

    async def set(self, key, value):
        self.__class__.store[key] = value


class MinerUEffortSettingsRouteTest(unittest.IsolatedAsyncioTestCase):
    """GET/PUT /settings/mineru expose effort explicitly."""

    def setUp(self):
        _StubSettingsRepo.store = {}
        from app.api.routes import settings as settings_routes

        self.settings_routes = settings_routes
        patcher = mock.patch.object(
            settings_routes, "SettingsRepository", _StubSettingsRepo
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_get_mineru_config_defaults_effort_to_none(self):
        config = await self.settings_routes.get_mineru_config(db=None)
        self.assertIn("effort", config)
        self.assertIsNone(config["effort"])

    async def test_put_then_get_roundtrips_effort(self):
        update = self.settings_routes.MinerUSettingsUpdate(effort="high")
        await self.settings_routes.update_mineru_settings(update, db=None)

        config = await self.settings_routes.get_mineru_config(db=None)
        self.assertEqual(config["effort"], "high")

    async def test_put_rejects_invalid_effort(self):
        from fastapi import HTTPException

        update = self.settings_routes.MinerUSettingsUpdate(effort="ultra")
        with self.assertRaises(HTTPException) as ctx:
            await self.settings_routes.update_mineru_settings(update, db=None)
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_get_settings_endpoint_lists_available_efforts(self):
        response = await self.settings_routes.get_mineru_settings(db=None)
        self.assertEqual(response["available_efforts"], ["medium", "high"])


if __name__ == "__main__":
    unittest.main()
