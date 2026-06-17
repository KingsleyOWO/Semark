import importlib.util
import unittest

if importlib.util.find_spec("openai") is None:
    raise unittest.SkipTest("openai package is required to import VLMAdapter")

from app.adapters.vlm import VLMAdapter
from app.config import VLMConfig, VLMDecodeParams


class VLMResponseParsingTest(unittest.TestCase):
    def test_truncated_form_json_is_salvaged_with_fields(self):
        raw = r'''{
  "title": "表三國內（外）出差單",
  "document_type": "form",
  "triggers": ["出差單", "差旅申請"],
  "all_text": [
    "申請單位",
    "申請人",
    "出差地點",
    "出差事由",
    "單位主管",
    "出差人簽名",
    "註：本單應於出差前填寫"
  ],
  "filling_guide": "## 表單用途\n用於申請國內外出差。\n## 填寫重點\n填寫申請單位、申請人、出差地點與出差事由。'''

        output = VLMAdapter()._parse_response(raw, "form_asset")

        self.assertTrue(output["needs_review"])
        self.assertTrue(output["_salvaged"])
        self.assertEqual(output["title"], "表三國內（外）出差單")
        names = {field["name"] for field in output["field_schema"]}
        self.assertIn("申請單位", names)
        self.assertIn("出差地點", names)
        self.assertIn("單位主管", names)
        self.assertIn("出差人簽名", names)
        self.assertIn("表三國內（外）出差單", output["retrieval_text"])

    def test_reference_table_salvage_does_not_emit_form_fields(self):
        raw = r'''{
  "title": "國內出差旅費報支數額表",
  "document_type": "reference_table",
  "all_text": ["職稱/職級別", "宿費(平日)", "雜費"],
  "field_schema": [{"name":"職稱/職級別","type":"text","required":false}],
  "filling_guide": "這是查詢職級對應交通費、宿費與雜費上限的標準表。"
'''

        output = VLMAdapter()._parse_response(raw, "form_asset")

        self.assertEqual(output["document_type"], "reference_table")
        self.assertEqual(output["field_schema"], [])
        self.assertIn("國內出差旅費報支數額表", output["retrieval_text"])

    def test_form_token_cap_allows_complex_forms_when_configured_high(self):
        adapter = VLMAdapter(VLMConfig(decode_params=VLMDecodeParams(max_tokens=128000)))

        self.assertEqual(adapter._max_tokens_for_kind("form_asset"), 8192)


if __name__ == "__main__":
    unittest.main()
