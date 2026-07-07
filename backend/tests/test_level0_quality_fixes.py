"""Level 0 output-quality fixes: defaults and behaviors that silently degraded
VLM enrichment (image transport, token budgets, truncation, caching, profiles,
render resolution, structured outputs)."""

import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

if importlib.util.find_spec("openai") is None:
    raise unittest.SkipTest("openai package is required to import VLMAdapter")

from app.adapters.vlm import VLMAdapter
from app.config import (
    PROFILES,
    MinerUMethod,
    ProfileName,
    VLMApiMode,
    VLMConfig,
    VLMDecodeParams,
    VLMImageMode,
)

PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000002000148afa4710000000049454e44ae426082"
)


def _write_png(tmp_dir: Path) -> Path:
    path = tmp_dir / "crop.png"
    path.write_bytes(PNG_BYTES)
    return path


class ImageModeDefaultsTest(unittest.TestCase):
    def test_default_image_mode_is_data_uri(self):
        # STATIC_URL by default breaks Ollama (base64-only) and any endpoint
        # that cannot reach the backend; data URI is the universally safe mode.
        self.assertEqual(VLMConfig().image_mode, VLMImageMode.DATA_URI)

    def test_ollama_forces_data_uri_even_when_static_url_configured(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            image_path = _write_png(Path(tmp))
            adapter = VLMAdapter(
                VLMConfig(image_mode=VLMImageMode.STATIC_URL, api_mode=VLMApiMode.OLLAMA)
            )
            url = adapter._build_image_url(image_path, doc_id="d1", run_id="r1")
            self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_static_url_preserved_for_non_ollama_modes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "assets" / "figures" / "crop.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(PNG_BYTES)
            adapter = VLMAdapter(
                VLMConfig(image_mode=VLMImageMode.STATIC_URL, api_mode=VLMApiMode.VLLM)
            )
            url = adapter._build_image_url(image_path, doc_id="d1", run_id="r1")
            self.assertTrue(url.startswith("http"))
            self.assertIn("/d1/r1/assets/", url)


class TokenBudgetTest(unittest.TestCase):
    def test_task_caps_apply_with_default_global_config(self):
        # Per-kind caps are authoritative; the old min(global, cap) collapsed
        # every task to the 1024 default and truncated dense forms.
        adapter = VLMAdapter(VLMConfig())
        self.assertEqual(adapter._max_tokens_for_kind("form_asset"), 8192)
        # 12288 was unfinishable within the request timeout on thinking models
        # (observed 2x600s timeout+retry on a 35B MoE); 8192 completes.
        self.assertEqual(adapter._max_tokens_for_kind("semantic_repair"), 8192)
        self.assertEqual(adapter._max_tokens_for_kind("table_summary"), 512)

    def test_global_max_tokens_used_for_unknown_kinds(self):
        adapter = VLMAdapter(
            VLMConfig(decode_params=VLMDecodeParams(max_tokens=4096))
        )
        self.assertEqual(adapter._max_tokens_for_kind("custom_experiment"), 4096)


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def _fake_response(text: str, finish_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(total_tokens=42),
    )


class TruncationRetryTest(unittest.IsolatedAsyncioTestCase):
    def _adapter_with(self, responses) -> tuple[VLMAdapter, _FakeCompletions]:
        adapter = VLMAdapter(VLMConfig())
        fake = _FakeCompletions(responses)
        adapter.client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake)
        )
        return adapter, fake

    async def test_length_finish_reason_triggers_one_retry_with_larger_budget(self):
        valid = json.dumps(
            {
                "summary_zh": "表格摘要",
                "keywords": ["假別"],
                "needs_review": False,
            }
        )
        adapter, fake = self._adapter_with(
            [
                _fake_response('{"summary_zh": "被截斷的', "length"),
                _fake_response(valid, "stop"),
            ]
        )

        result = await adapter._enrich(None, "table_summary", context="ctx")

        self.assertEqual(len(fake.calls), 2)
        self.assertGreater(fake.calls[1]["max_tokens"], fake.calls[0]["max_tokens"])
        self.assertTrue(result.success)

    async def test_no_retry_when_finish_reason_is_stop(self):
        valid = json.dumps(
            {
                "summary_zh": "表格摘要",
                "keywords": ["假別"],
                "needs_review": False,
            }
        )
        adapter, fake = self._adapter_with([_fake_response(valid, "stop")])

        result = await adapter._enrich(None, "table_summary", context="ctx")

        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(result.success)

    async def test_still_truncated_after_retry_is_flagged_for_review(self):
        adapter, fake = self._adapter_with(
            [
                _fake_response('{"summary_zh": "被截斷的', "length"),
                _fake_response('{"summary_zh": "還是被截斷', "length"),
            ]
        )

        result = await adapter._enrich(None, "table_summary", context="ctx")

        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(result.needs_review)


class JsonSchemaFallbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_schema_rejected_by_provider_falls_back_to_unconstrained(self):
        valid = json.dumps(
            {
                "summary_zh": "表格摘要",
                "keywords": ["假別"],
                "needs_review": False,
            }
        )

        class _SchemaRejectingCompletions(_FakeCompletions):
            async def create(self, **kwargs):
                self.calls.append(kwargs)
                if "response_format" in kwargs:
                    raise RuntimeError("response_format is not supported")
                return _fake_response(valid, "stop")

        adapter = VLMAdapter(VLMConfig())
        fake = _SchemaRejectingCompletions([])
        adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

        result = await adapter._enrich(None, "table_summary", context="ctx")

        self.assertTrue(result.success)
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("response_format", fake.calls[0])
        self.assertNotIn("response_format", fake.calls[1])


class EnrichCachePolicyTest(unittest.TestCase):
    def test_needs_review_results_are_not_cached(self):
        from app.pipeline.stages.enrich import should_cache_enrichment

        good = SimpleNamespace(success=True, needs_review=False, output={"a": 1})
        salvaged = SimpleNamespace(success=True, needs_review=True, output={"a": 1})
        failed = SimpleNamespace(success=False, needs_review=False, output=None)

        self.assertTrue(should_cache_enrichment(good))
        self.assertFalse(should_cache_enrichment(salvaged))
        self.assertFalse(should_cache_enrichment(failed))


class AccurateProfileTest(unittest.TestCase):
    def test_accurate_profile_does_not_force_ocr(self):
        # method=ocr re-recognizes born-digital text with the chinese_cht model
        # and injects recognition errors; AUTO routes scanned pages to OCR only.
        self.assertEqual(
            PROFILES[ProfileName.ACCURATE].mineru.method, MinerUMethod.AUTO
        )


class RenderResolutionTest(unittest.TestCase):
    def test_page_render_dpi_matches_mineru_internal(self):
        from app.pipeline.stages.normalize import NormalizeStage

        self.assertGreaterEqual(NormalizeStage.PAGE_RENDER_DPI, 200)

    def test_vlm_crop_limits_allow_legible_zh_labels(self):
        from app.pipeline.stages import enrich

        self.assertGreaterEqual(enrich.VLM_IMAGE_MAX_SIZE, 2048)
        self.assertGreaterEqual(enrich.VLM_CROP_MAX_ZOOM, 4.0)


class StructuredOutputDefaultTest(unittest.TestCase):
    def test_json_schema_constraint_enabled_by_default(self):
        self.assertTrue(VLMConfig().use_json_schema)


class ClientRetryPolicyTest(unittest.TestCase):
    def test_client_does_not_auto_retry_timed_out_requests(self):
        # A timed-out 600s VLM call auto-retried by the OpenAI client doubles
        # the waste (observed 1082s wall for one semantic_repair). Recovery is
        # handled explicitly in _create_with_recovery instead.
        self.assertEqual(VLMAdapter().client.max_retries, 0)
