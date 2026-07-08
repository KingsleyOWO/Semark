"""use_json_schema must be settable per role via the settings API so a reviewer
that degenerates under strict constrained decoding can run unconstrained."""

import unittest

from app.api.routes.settings import VLMSettingsUpdate, _update_vlm_settings_key


class _FakeRepo:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return dict(self.store.get(key, {}))

    async def set(self, key, value):
        self.store[key] = dict(value)


class _FakeDB:
    pass


class JsonSchemaToggleTest(unittest.IsolatedAsyncioTestCase):
    async def test_use_json_schema_persists_via_update(self):
        import app.api.routes.settings as settings_mod

        repo = _FakeRepo()
        orig = settings_mod.SettingsRepository
        settings_mod.SettingsRepository = lambda db: repo
        try:
            result = await _update_vlm_settings_key(
                "review_vlm", VLMSettingsUpdate(use_json_schema=False), _FakeDB()
            )
        finally:
            settings_mod.SettingsRepository = orig
        self.assertFalse(result["use_json_schema"])
        self.assertFalse(repo.store["review_vlm"]["use_json_schema"])


if __name__ == "__main__":
    unittest.main()
