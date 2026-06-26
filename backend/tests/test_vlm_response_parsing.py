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

    def test_truncated_figure_jsonish_response_is_salvaged(self):
        raw = r'''```json
{
  "semantic_caption": "性騷擾申訴流程依適用法律分流",
  "image_type": "flowchart",
  "structured_content": [
    "被害人提出申訴 > 判斷適用法律",
    "性騷擾防治法 > 行為人是機關人員 > 是否有不予受理之情形"
  ],
  "all_text": ["被害人提出申訴", "判斷適用法律"]
'''

        output = VLMAdapter()._parse_response(raw, "figure_description")

        self.assertTrue(output["needs_review"])
        self.assertTrue(output["_salvaged"])
        self.assertEqual(output["image_type"], "flowchart")
        self.assertIn("被害人提出申訴 > 判斷適用法律", output["structured_content"])
        self.assertIn("判斷適用法律", output["all_text"])


    def test_form_token_cap_allows_complex_forms_when_configured_high(self):
        adapter = VLMAdapter(VLMConfig(decode_params=VLMDecodeParams(max_tokens=128000)))

        self.assertEqual(adapter._max_tokens_for_kind("form_asset"), 8192)


    def test_semantic_repair_json_is_validated(self):
        raw = (
            '{"status":"repaired","repaired_markdown":"# Form\\n\\n## Purpose\\nUseful repaired text.",'
            '"summary":"rewritten","applied_repairs":["split_merged_fields"],'
            '"confidence":0.9,"needs_review":false}'
        )

        output = VLMAdapter()._parse_response(raw, "semantic_repair")

        self.assertEqual(output["status"], "repaired")
        self.assertIn("Useful repaired text", output["repaired_markdown"])
        self.assertEqual(output["applied_repairs"], ["split_merged_fields"])



    def test_semantic_repair_markdown_response_is_salvaged(self):
        raw = """# Visa Checklist

## Required Documents
- Passport copy.
- Proof of financial resources of at least NT$100,000.

## Applicant Signature
The applicant signs the checklist after confirming the required documents are prepared.
"""

        output = VLMAdapter()._parse_response(raw, "semantic_repair")

        self.assertEqual(output["status"], "repaired")
        self.assertTrue(output["needs_review"])
        self.assertTrue(output["_salvaged"])
        self.assertIn("NT$100,000", output["repaired_markdown"])


    def test_invalid_semantic_repair_json_does_not_emit_markdown(self):
        raw = '{"status":"repaired","repaired_markdown":"# Form SSA-827\n- unfinished'

        output = VLMAdapter()._parse_response(raw, "semantic_repair")

        self.assertEqual(output["status"], "uncertain")
        self.assertEqual(output["repaired_markdown"], "")
        self.assertIn("JSON_PARSE_FAILED", output["_error"])
        self.assertIn("raw_response_preview", output)


    def test_json_payload_after_thinking_is_parsed(self):
        raw = """<think>Need to inspect the source.</think>
```json
{
  "semantic_caption": "流程圖描述申訴後依適用法律分流。",
  "image_type": "flowchart",
  "structured_content": "起點：被害人提出申訴；判斷：適用法律；分支：性別平等工作法、性別平等教育法、性騷擾防治法。",
  "all_text": ["被害人提出申訴", "判斷適用法律"],
  "facts": ["申訴流程依場域與身分關係分流"],
  "keywords": ["性騷擾申訴", "流程圖"],
  "needs_review": false
}
```
"""

        output = VLMAdapter()._parse_response(raw, "figure_description")

        self.assertEqual(output["image_type"], "flowchart")
        self.assertIn("被害人提出申訴", output["structured_content"])
        self.assertNotIn("_error", output)


    def test_json_payload_embedded_in_prose_is_parsed(self):
        raw = r"""Here is the JSON:
{"status":"repaired","repaired_markdown":"# Flowchart\n\n## Process\nVictim files complaint.\n\n## Branches\nEmployment, education, or prevention law.","summary":"rewritten","applied_repairs":["flowchart_structure"],"confidence":0.8,"needs_review":false}
Done."""

        output = VLMAdapter()._parse_response(raw, "semantic_repair")

        self.assertEqual(output["status"], "repaired")
        self.assertIn("Victim files complaint", output["repaired_markdown"])


    def test_semantic_repair_markdown_after_thinking_is_salvaged(self):
        raw = """<think>I will rewrite this as Markdown.</think>
# Flowchart

## Process
The victim files a complaint, then the case is routed by scenario and party relationship.

## Legal Routing
- Employment context routes to the employment equality law path.
- Campus gender event routes to the education equality law path.
- Other cases route to the sexual harassment prevention law path.
"""

        output = VLMAdapter()._parse_response(raw, "semantic_repair")

        self.assertEqual(output["status"], "repaired")
        self.assertTrue(output["_salvaged"])
        self.assertIn("Legal Routing", output["repaired_markdown"])


    def test_semantic_repair_token_cap_allows_rewrite(self):
        adapter = VLMAdapter(VLMConfig(decode_params=VLMDecodeParams(max_tokens=128000)))

        self.assertEqual(adapter._max_tokens_for_kind("semantic_repair"), 12288)


if __name__ == "__main__":
    unittest.main()
