"""A reachable endpoint that does not serve the configured model is not
"available" — enrich must skip loudly instead of failing every call.

``check_available`` returned True whenever the HTTP endpoint answered, even
when it had just reported that the configured model was absent:

    True | Connected, but model qwen2.5-vl:7b not in list:
           ['qwen3-embedding:0.6b', 'qwen3.6:35b-a3b-q8_0']

``enrich.py`` only reads the boolean (``if not vlm_ok: return skipped``), so a
typo'd or unpulled model let the stage proceed and then fail on every single
request, instead of stopping once with a reason a human can act on.

The over-correction to avoid is failing closed on endpoints that simply do not
enumerate their models: an empty list is "I cannot tell", not "the model is
missing". Ollama's implicit ``:latest`` tag matters for the same reason —
``llama3`` configured against a served ``llama3:latest`` is the same model.
"""

import importlib.util
import unittest
from types import SimpleNamespace

if importlib.util.find_spec("openai") is None:
    raise unittest.SkipTest("openai package is required to import VLMAdapter")

from app.adapters.vlm import VLMAdapter
from app.config import VLMConfig

ASYNC_RUN = __import__("asyncio").run


def _adapter(model: str, served: list[str] | None, *, error: str | None = None) -> VLMAdapter:
    """An adapter whose endpoint serves ``served`` (None ⇒ unreachable)."""
    adapter = VLMAdapter(VLMConfig(model=model))

    async def _list():
        if error is not None:
            raise RuntimeError(error)
        return SimpleNamespace(data=[SimpleNamespace(id=name) for name in served or []])

    adapter.client = SimpleNamespace(models=SimpleNamespace(list=_list))

    async def _vision():
        return True

    adapter._check_vision_support = _vision
    return adapter


class CheckAvailableTest(unittest.TestCase):
    def test_a_listed_endpoint_without_the_configured_model_is_not_available(self):
        """The live case: Ollama answered, but 7b had never been pulled."""
        ok, _ = ASYNC_RUN(
            _adapter(
                "qwen2.5-vl:7b", ["qwen3-embedding:0.6b", "qwen3.6:35b-a3b-q8_0"]
            ).check_available()
        )

        self.assertFalse(ok)

    def test_the_message_names_the_missing_model_and_what_is_served(self):
        """It lands in the stage's skip reason, so it has to be actionable."""
        _, message = ASYNC_RUN(
            _adapter(
                "qwen2.5-vl:7b", ["qwen3-embedding:0.6b", "qwen3.6:35b-a3b-q8_0"]
            ).check_available()
        )

        self.assertIn("qwen2.5-vl:7b", message)
        self.assertIn("qwen3.6:35b-a3b-q8_0", message)

    def test_the_configured_model_being_served_is_available(self):
        ok, message = ASYNC_RUN(
            _adapter("qwen3.6:35b-a3b-q8_0", ["qwen3.6:35b-a3b-q8_0"]).check_available()
        )

        self.assertTrue(ok)
        self.assertIn("qwen3.6:35b-a3b-q8_0", message)

    def test_an_endpoint_that_enumerates_nothing_stays_available(self):
        """Anti-over-correction: vLLM/LMDeploy behind a gateway may serve an
        empty list. Unknown is not the same as missing, so do not fail closed."""
        ok, _ = ASYNC_RUN(_adapter("some-model", []).check_available())

        self.assertTrue(ok)

    def test_an_unreachable_endpoint_stays_unavailable_with_its_error(self):
        ok, message = ASYNC_RUN(
            _adapter("qwen3.6:35b-a3b-q8_0", None, error="Connection refused").check_available()
        )

        self.assertFalse(ok)
        self.assertIn("Connection refused", message)


class ImplicitLatestTagTest(unittest.TestCase):
    """Ollama resolves a bare name to ``:latest``; an exact string compare
    would have called a working configuration broken."""

    def test_a_bare_name_matches_the_served_latest_tag(self):
        probe = ASYNC_RUN(_adapter("llama3", ["llama3:latest"]).probe_capabilities())

        self.assertTrue(probe.model_found)

    def test_an_explicit_latest_tag_matches_a_bare_served_name(self):
        probe = ASYNC_RUN(_adapter("llama3:latest", ["llama3"]).probe_capabilities())

        self.assertTrue(probe.model_found)

    def test_a_different_tag_is_still_a_different_model(self):
        """Only ``:latest`` is implicit — 7b and 70b must not be conflated."""
        probe = ASYNC_RUN(_adapter("llama3:70b", ["llama3:7b"]).probe_capabilities())

        self.assertFalse(probe.model_found)


class ProbeStaysHonestTest(unittest.TestCase):
    def test_model_found_reports_the_literal_answer_for_the_settings_ui(self):
        """``/api/settings/vlm/probe`` shows this; the permissiveness belongs in
        check_available, not in a probe that claims to have found a model."""
        probe = ASYNC_RUN(_adapter("some-model", []).probe_capabilities())

        self.assertTrue(probe.available)
        self.assertFalse(probe.model_found)


if __name__ == "__main__":
    unittest.main()
