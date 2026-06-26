import json
import tempfile
import unittest
from pathlib import Path

from app.eval.baseline import collect_workspace_baseline


class BaselineCollectorTest(unittest.TestCase):
    def test_collects_metrics_from_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_path = workspace / "store" / "docs" / "doc-a" / "runs" / "run-a"
            outputs = run_path / "outputs"
            outputs.mkdir(parents=True)

            (run_path / "document_ir.json").write_text("{}", encoding="utf-8")
            (run_path / "manifest.json").write_text(
                json.dumps(
                    {
                        "engines": {
                            "mineru": {"backend": "pipeline", "method": "auto"},
                            "vlm": {"model": "mock-vlm", "api_mode": "mock"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_path / "source_map.json").write_text(
                json.dumps(
                    {
                        "md_anchors": [
                            {
                                "anchor_id": "b000001",
                                "md_range": [0, 12],
                                "block_ids": ["b000001"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (outputs / "dataset.md").write_text("# Dataset\n", encoding="utf-8")
            (outputs / "rag.md").write_text("hello [[asset:form0001]]\n", encoding="utf-8")
            (outputs / "quality.json").write_text(
                json.dumps({"block_counts": {"text": 1}, "page_count": 1}),
                encoding="utf-8",
            )
            (outputs / "assets_index.jsonl").write_text(
                json.dumps(
                    {
                        "type": "form_asset",
                        "asset_id": "form0001",
                        "doc_id": "doc-a",
                        "run_id": "run-a",
                        "title": "Mock form",
                        "triggers": ["mock"],
                        "page_idx": 0,
                        "asset_path": "assets/forms/form.png",
                        "block_id": "form_page_0000",
                        "retrieval_text": "Mock form",
                        "needs_review": False,
                        "field_schema": [{"name": "Field"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (outputs / "chunks.jsonl").write_text(
                json.dumps(
                    {
                        "chunk_id": "c000001",
                        "doc_id": "doc-a",
                        "run_id": "run-a",
                        "view": "rag",
                        "content": "hello",
                        "block_ids": ["b000001"],
                        "page_indices": [0],
                        "attachments": ["asset://assets/forms/form.png"],
                        "metadata": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = collect_workspace_baseline(workspace)

        self.assertEqual(report["run_count"], 1)
        self.assertEqual(report["successful_run_count"], 1)
        run = report["runs"][0]
        self.assertEqual(run["asset_count"], 1)
        self.assertEqual(run["field_schema_count"], 1)
        self.assertEqual(run["asset_token_count_in_rag"], 1)
        self.assertEqual(run["chunk_attachment_count"], 1)
        self.assertEqual(run["validation_errors"], [])


if __name__ == "__main__":
    unittest.main()

