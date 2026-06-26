import unittest

from app.eval.validation import (
    validate_asset_entry,
    validate_chunk_entry,
    validate_source_map,
)


class ValidationTest(unittest.TestCase):
    def test_asset_entry_requires_retrieval_text(self):
        errors = validate_asset_entry(
            {
                "type": "form_asset",
                "asset_id": "form0001",
                "doc_id": "doc",
                "run_id": "run",
                "title": "Form",
                "page_idx": 0,
                "asset_path": "assets/forms/form.png",
                "block_id": "form_page_0000",
                "retrieval_text": "",
                "needs_review": False,
            }
        )
        self.assertIn("retrieval_text must be non-empty", errors)

    def test_chunk_entry_accepts_required_shape(self):
        errors = validate_chunk_entry(
            {
                "chunk_id": "c000001",
                "doc_id": "doc",
                "run_id": "run",
                "view": "rag",
                "content": "content",
                "block_ids": ["b1"],
                "page_indices": [0],
                "attachments": [],
                "metadata": {},
            }
        )
        self.assertEqual(errors, [])

    def test_source_map_checks_anchor_ranges(self):
        errors = validate_source_map(
            {
                "md_anchors": [
                    {"anchor_id": "a1", "md_range": [10, 1], "block_ids": ["b1"]}
                ]
            }
        )
        self.assertEqual(errors, ["anchor 0 has invalid md_range"])


if __name__ == "__main__":
    unittest.main()

