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
        # every task above the 1024 default back to 1024 and truncated dense
        # forms. Caps above the default must survive (scalable long-output
        # kinds are covered separately below).
        adapter = VLMAdapter(VLMConfig())  # default max_tokens = 1024
        self.assertEqual(adapter._max_tokens_for_kind("figure_caption"), 2048)
        self.assertEqual(adapter._max_tokens_for_kind("structured_table_records"), 4096)
        self.assertEqual(adapter._max_tokens_for_kind("table_summary"), 512)

    def test_scalable_kinds_get_adequate_budget_at_default_config(self):
        # Regression (prod 2-10, 2026-07-08): at the small default max_tokens the
        # scalable long-output kinds were capped at 8192, which truncated a
        # whole-document rewrite of an 8-page zh-TW doc and dropped ~20% of facts
        # (fact_survival 0.79 -> reviewer repair rejected by the fact guard).
        # These kinds must floor at a budget that fits a multi-page rewrite
        # without requiring the operator to hand-raise max_tokens.
        adapter = VLMAdapter(VLMConfig())  # default max_tokens = 1024
        self.assertGreaterEqual(adapter._max_tokens_for_kind("semantic_repair"), 24000)
        self.assertGreaterEqual(adapter._max_tokens_for_kind("form_asset"), 24000)
        self.assertGreaterEqual(adapter._max_tokens_for_kind("form_guide"), 24000)
        # non-scalable small tasks stay tightly capped
        self.assertEqual(adapter._max_tokens_for_kind("table_summary"), 512)

    def test_global_max_tokens_used_for_unknown_kinds(self):
        adapter = VLMAdapter(
            VLMConfig(decode_params=VLMDecodeParams(max_tokens=4096))
        )
        self.assertEqual(adapter._max_tokens_for_kind("custom_experiment"), 4096)

    def test_long_output_tasks_scale_with_raised_budget(self):
        # Above the default floor, an operator-raised max_tokens lifts the
        # long-output kinds further (capped at 32768); small tasks stay capped.
        adapter = VLMAdapter(VLMConfig(decode_params=VLMDecodeParams(max_tokens=30000)))
        self.assertEqual(adapter._max_tokens_for_kind("semantic_repair"), 30000)
        self.assertEqual(adapter._max_tokens_for_kind("form_asset"), 30000)
        # small tasks stay capped regardless of the raised global budget
        self.assertEqual(adapter._max_tokens_for_kind("table_summary"), 512)
        self.assertEqual(adapter._max_tokens_for_kind("figure_caption"), 2048)

    def test_long_output_scaling_is_ceilinged(self):
        adapter = VLMAdapter(VLMConfig(decode_params=VLMDecodeParams(max_tokens=100000)))
        self.assertEqual(adapter._max_tokens_for_kind("semantic_repair"), 32768)


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

        # recovery pair + the ollama no-thinking parse retry (fake replays the
        # truncated reply, so the salvage path still wins and stays flagged)
        self.assertEqual(len(fake.calls), 3)
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


class ParseFailureRetryTest(unittest.IsolatedAsyncioTestCase):
    def _adapter_with(self, responses) -> tuple[VLMAdapter, _FakeCompletions]:
        adapter = VLMAdapter(VLMConfig())
        fake = _FakeCompletions(responses)
        adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        return adapter, fake

    async def test_unparseable_json_reply_gets_one_parse_retry_then_succeeds(self):
        """A completed reply that is not valid JSON (finish_reason=stop, i.e. not
        a length truncation) triggers exactly one fresh attempt; the clean retry
        output is adopted instead of degrading to salvage text with _error. This
        is the b000014 vlm_enrichment_parse_failed class of intermittent flake."""
        valid = json.dumps({"summary_zh": "表格摘要", "keywords": ["假別"], "needs_review": False})
        adapter, fake = self._adapter_with([
            _fake_response("Sure! Here is the description, not JSON.", "stop"),
            _fake_response(valid, "stop"),
        ])

        result = await adapter._enrich(None, "table_summary", context="ctx")

        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(result.success)
        self.assertNotIn("_error", result.output)

    async def test_persistent_unparseable_json_keeps_error_after_single_retry(self):
        """If both attempts fail to parse, stop after one retry (no loop) and keep
        the salvaged output flagged for review."""
        adapter, fake = self._adapter_with([
            _fake_response("still not json", "stop"),
            _fake_response("also not json", "stop"),
        ])

        result = await adapter._enrich(None, "table_summary", context="ctx")

        self.assertEqual(len(fake.calls), 2)
        self.assertIn("_error", result.output)
        self.assertIn("JSON_PARSE_FAILED", result.output["_error"])
        self.assertTrue(result.needs_review)

    async def test_parse_retry_disables_thinking_for_ollama(self):
        """Local thinking models can burn the whole budget ruminating on a
        low-information crop (deterministic at temperature 0.1 — b000031/fig0006:
        9562 thinking chars, 0 content). A same-settings resample replays the same
        failure, so the ollama-mode parse retry must ask for no reasoning instead."""
        valid = json.dumps({"summary_zh": "表格摘要", "keywords": ["假別"], "needs_review": False})
        adapter, fake = self._adapter_with([
            _fake_response("这个请求要求我分析一张图片……(纯思考过程，无JSON)", "stop"),
            _fake_response(valid, "stop"),
        ])

        result = await adapter._enrich(None, "table_summary", context="ctx")

        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[1].get("extra_body"), {"reasoning_effort": "none"})
        self.assertTrue(result.success)
        self.assertNotIn("_error", result.output)

    async def test_length_truncated_parse_failure_gets_thinking_disabled_retry(self):
        """Rumination usually ends in finish_reason=length: recovery doubles the
        budget once (thinking burns that too), then the ollama no-thinking retry
        is the one that actually recovers content. Exactly one extra call, no loop."""
        valid = json.dumps({"summary_zh": "表格摘要", "keywords": ["假別"], "needs_review": False})
        adapter, fake = self._adapter_with([
            _fake_response('{"summary_zh": "被截斷', "length"),
            _fake_response('{"summary_zh": "還是被截斷', "length"),
            _fake_response(valid, "stop"),
        ])

        result = await adapter._enrich(None, "table_summary", context="ctx")

        self.assertEqual(len(fake.calls), 3)  # recovery pair + one no-thinking retry
        self.assertEqual(fake.calls[2].get("extra_body"), {"reasoning_effort": "none"})
        self.assertTrue(result.success)
        self.assertNotIn("_error", result.output)

    async def test_nothink_retry_failure_keeps_salvaged_output(self):
        """If the no-thinking retry itself errors (e.g. a model that rejects the
        reasoning_effort field), keep the salvaged first reply instead of losing
        the enrichment to a hard failure."""

        class _RetryRejectingCompletions(_FakeCompletions):
            async def create(self, **kwargs):
                self.calls.append(kwargs)
                if "extra_body" in kwargs:
                    raise RuntimeError("unknown field: reasoning_effort")
                return _fake_response("still not json", "stop")

        adapter = VLMAdapter(VLMConfig())
        fake = _RetryRejectingCompletions([])
        adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

        result = await adapter._enrich(None, "table_summary", context="ctx")

        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(result.success)
        self.assertIn("_error", result.output)
        self.assertTrue(result.needs_review)

    async def test_non_ollama_parse_retry_keeps_plain_resample(self):
        """Cloud/vLLM providers may reject the reasoning_effort field outright, so
        outside ollama mode the parse retry stays a plain same-settings resample
        and still never fires on top of a length truncation."""
        from app.config import VLMApiMode

        valid = json.dumps({"summary_zh": "表格摘要", "keywords": ["假別"], "needs_review": False})
        adapter = VLMAdapter(VLMConfig(api_mode=VLMApiMode.OPENAI))
        fake = _FakeCompletions([
            _fake_response("not json at all", "stop"),
            _fake_response(valid, "stop"),
        ])
        adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

        result = await adapter._enrich(None, "table_summary", context="ctx")

        self.assertEqual(len(fake.calls), 2)
        self.assertNotIn("extra_body", fake.calls[1])
        self.assertTrue(result.success)

        adapter2 = VLMAdapter(VLMConfig(api_mode=VLMApiMode.OPENAI))
        fake2 = _FakeCompletions([
            _fake_response('{"summary_zh": "被截斷', "length"),
            _fake_response('{"summary_zh": "還是被截斷', "length"),
        ])
        adapter2.client = SimpleNamespace(chat=SimpleNamespace(completions=fake2))

        result2 = await adapter2._enrich(None, "table_summary", context="ctx")

        self.assertEqual(len(fake2.calls), 2)  # recovery pair only, no extra retry
        self.assertTrue(result2.needs_review)


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


class FigureEnrichmentLanguageTest(unittest.IsolatedAsyncioTestCase):
    """#6: the figure-caption prompt must force the resolved output language, so the
    local qwen VLM stops emitting English semantic_caption/facts for zh-TW documents
    (observed on 204電子白板 / 信箱封存: every caption + fact came back in English)."""

    def _adapter_with(self, responses) -> tuple[VLMAdapter, _FakeCompletions]:
        adapter = VLMAdapter(VLMConfig())
        fake = _FakeCompletions(responses)
        adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        return adapter, fake

    async def test_figure_prompt_requests_traditional_chinese_when_zh_tw(self):
        valid = json.dumps(
            {"semantic_caption": "說明", "facts": [], "keywords": [], "needs_review": False}
        )
        adapter, fake = self._adapter_with([_fake_response(valid, "stop")])
        await adapter._enrich(
            None,
            "figure_caption",
            context="ctx",
            extra_vars={"semantic_output_language": "zh-TW"},
        )
        sent = json.dumps(fake.calls[0]["messages"], ensure_ascii=False)
        # The strong zh-TW directive from prompt_language_instruction must reach the model.
        self.assertIn("繁體中文", sent)

    async def test_enrich_figure_threads_output_language_into_prompt(self):
        """The public enrich_figure path (used by the enrich stage) must forward the
        resolved language, not silently default to the weak 'auto' instruction."""
        valid = json.dumps(
            {"semantic_caption": "說明", "facts": [], "keywords": [], "needs_review": False}
        )
        adapter, fake = self._adapter_with([_fake_response(valid, "stop")])
        await adapter.enrich_figure(
            None,  # type: ignore[arg-type]
            context_text="ctx",
            extra_vars={"semantic_output_language": "zh-TW"},
        )
        sent = json.dumps(fake.calls[0]["messages"], ensure_ascii=False)
        self.assertIn("繁體中文", sent)

    def test_salvage_never_uses_reasoning_dump_as_caption(self):
        """A local VLM sometimes replies with plain-text chain-of-thought (no <think>
        tags, no JSON). The salvage path used to put that WHOLE dump into
        semantic_caption (observed live: a 9105-char simplified-Chinese reasoning dump
        woven into rag.md). A caption that is just the raw reply must be dropped."""
        dump = "这个任务需要我分析一张图片，并按照指定的JSON格式输出。" + "让我们仔细看Context里的文本，这些看起来像是界面元素。" * 60
        adapter = VLMAdapter(VLMConfig())
        result = adapter._salvage_figure_jsonish_response(dump, None)
        self.assertEqual(result.get("semantic_caption", ""), "")
        self.assertTrue(result.get("_error"))

    def test_parse_failure_predicate_covers_unknown_error(self):
        """UNKNOWN_ERROR salvages are parse failures too — they must get the same
        single resample as JSON_PARSE_FAILED (observed live: an UNKNOWN_ERROR reply
        skipped the retry because the predicate only matched JSON_PARSE_FAILED)."""
        self.assertTrue(VLMAdapter._is_parse_failure({"_error": "UNKNOWN_ERROR"}))
        self.assertTrue(VLMAdapter._is_parse_failure({"_error": "JSON_PARSE_FAILED: x"}))
        self.assertFalse(VLMAdapter._is_parse_failure({"semantic_caption": "ok"}))

    async def test_figure_prompt_forbids_transcribing_incidental_personal_content(self):
        """#4 privacy: screenshots in how-to guides often catch incidental personal
        content (an inbox's subjects/senders, chat messages). The prompt must tell the
        VLM not to transcribe those items — observed leak: real email subjects and
        senders from an Outlook inbox screenshot landed in the RAG corpus."""
        valid = json.dumps(
            {"semantic_caption": "說明", "facts": [], "keywords": [], "needs_review": False}
        )
        adapter, fake = self._adapter_with([_fake_response(valid, "stop")])
        await adapter._enrich(None, "figure_caption", context="ctx")
        sent = json.dumps(fake.calls[0]["messages"], ensure_ascii=False)
        self.assertIn("incidental personal content", sent)
