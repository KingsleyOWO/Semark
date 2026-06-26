import json
import tempfile
import unittest
from pathlib import Path

from app.eval.parser_runs import materialize_parser_run


class ParserRunMaterializationTest(unittest.TestCase):
    def test_materialize_mineru_smoke_output_writes_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            output_dir = workspace / "eval" / "reports_dgx_test"
            source_dir = workspace / "store" / "docs" / "doc-a" / "source"
            source_dir.mkdir(parents=True)
            source_path = source_dir / "original.pdf"
            source_path.write_bytes(b"%PDF-1.4\n")

            corpus = root / "corpus.json"
            corpus.write_text(
                json.dumps(
                    {
                        "local_samples": [
                            {
                                "sample_id": "sample-a",
                                "doc_id": "doc-a",
                                "categories": ["table"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            raw_dir = output_dir / "parser_runs" / "mineru3_pipeline" / "sample-a" / "original" / "auto"
            raw_dir.mkdir(parents=True)
            inventory_dir = output_dir / "candidate_reports" / "mineru3_pipeline"
            inventory_dir.mkdir(parents=True)
            (inventory_dir / "environment_inventory.json").write_text(
                json.dumps({"python_packages": {"mineru": "3.1.14"}}),
                encoding="utf-8",
            )
            (raw_dir / "original_content_list.json").write_text(
                json.dumps(
                    [
                        {
                            "type": "text",
                            "text": "財產增加單",
                            "bbox": [0, 0, 100, 20],
                            "page_idx": 0,
                        },
                        {
                            "type": "table",
                            "table_body": "<table><tr><td>項次</td></tr></table>",
                            "table_caption": ["財產"],
                            "bbox": [0, 30, 100, 80],
                            "page_idx": 0,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            result = materialize_parser_run(
                workspace=workspace,
                output_dir=output_dir,
                corpus_path=corpus,
                sample_id="sample-a",
            )

            run_dir = output_dir / "parser_runs" / "mineru3_pipeline" / "sample-a"
            self.assertTrue(result.ok, result.error)
            self.assertEqual(result.stats["validation_error_count"], 0)
            self.assertTrue((run_dir / "document_ir.json").exists())
            self.assertTrue((run_dir / "outputs" / "assets_index.jsonl").exists())
            self.assertTrue((run_dir / "source_map.json").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["benchmark"]["parser_id"], "mineru3_pipeline")
            self.assertEqual(manifest["benchmark"]["sample_id"], "sample-a")
            self.assertEqual(manifest["engines"]["parser_candidate"]["version"], "3.1.14")

    def test_materialize_rejects_unsupported_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = materialize_parser_run(
                workspace=Path(tmp) / "workspace",
                output_dir=Path(tmp) / "reports",
                corpus_path=Path(tmp) / "corpus.json",
                sample_id="sample-a",
                parser_id="docling_standard",
            )

        self.assertFalse(result.ok)
        self.assertIn("supports mineru3_pipeline and paddleocr_structure_v3 only", result.error)

    def test_materialize_paddleocr_smoke_output_writes_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            output_dir = workspace / "eval" / "reports_dgx_test"
            source_dir = workspace / "store" / "docs" / "doc-a" / "source"
            source_dir.mkdir(parents=True)
            (source_dir / "original.pdf").write_bytes(b"%PDF-1.4\n")
            corpus = root / "corpus.json"
            corpus.write_text(
                json.dumps({"local_samples": [{"sample_id": "sample-a", "doc_id": "doc-a"}]}),
                encoding="utf-8",
            )
            run_dir = output_dir / "parser_runs" / "paddleocr_structure_v3" / "sample-a"
            run_dir.mkdir(parents=True)
            (run_dir / "original_0_res.json").write_text(
                json.dumps(
                    {
                        "page_index": 0,
                        "page_count": 1,
                        "width": 1000,
                        "height": 1000,
                        "parsing_res_list": [
                            {
                                "block_label": "text",
                                "block_content": "財產增加單",
                                "block_bbox": [0, 0, 100, 20],
                                "block_id": 0,
                                "block_order": 1,
                            },
                            {
                                "block_label": "table",
                                "block_content": "<html><body><table><tr><td>項次</td></tr></table></body></html>",
                                "block_bbox": [0, 30, 100, 80],
                                "block_id": 1,
                                "block_order": 2,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = materialize_parser_run(
                workspace=workspace,
                output_dir=output_dir,
                corpus_path=corpus,
                sample_id="sample-a",
                parser_id="paddleocr_structure_v3",
            )

            self.assertTrue(result.ok, result.error)
            self.assertEqual(result.stats["validation_error_count"], 0)
            self.assertTrue((run_dir / "document_ir.json").exists())
            self.assertTrue((run_dir / "outputs" / "assets_index.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
