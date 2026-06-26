import json
import tempfile
import unittest
from pathlib import Path

from app.eval.benchmark import build_benchmark_workflow, build_parser_model_comparison


class BenchmarkWorkflowTest(unittest.TestCase):
    def test_workflow_uses_output_dir_and_candidate_envs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.json"
            corpus.write_text(
                json.dumps(
                    {
                        "local_samples": [{"sample_id": "local-a", "doc_id": "doc-a"}],
                        "public_samples": [{"sample_id": "public-a", "source_url": "https://example.test/a.pdf"}],
                    }
                ),
                encoding="utf-8",
            )

            workflow = build_benchmark_workflow(
                workspace=Path("workspace"),
                output_dir=Path("workspace/eval/reports_dgx_test"),
                corpus_path=corpus,
                queries_path=Path("benchmarks/retrieval_queries.jsonl"),
                top_k=5,
            )

        self.assertTrue(workflow["non_destructive"]["baseline_reports_preserved"])
        self.assertEqual(workflow["corpus"]["local_samples"][0]["sample_id"], "local-a")
        first_phase_commands = workflow["phases"][0]["commands"]
        self.assertTrue(any("--output-dir workspace/eval/reports_dgx_test" in c for c in first_phase_commands))

    def test_comparison_marks_missing_parsers_install_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            (report_dir / "baseline_metrics.json").write_text(
                json.dumps(
                    {
                        "run_count": 1,
                        "successful_run_count": 1,
                        "validation_error_count": 0,
                        "asset_count_by_type": {"form_asset": 1},
                        "block_count_by_type": {"text": 1},
                    }
                ),
                encoding="utf-8",
            )
            (report_dir / "retrieval_smoke.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "query_id": "q1",
                                "status": "ok",
                                "passed": False,
                                "expected_asset_type": "figure_asset",
                                "top_candidates": [{"id": "c1", "score": 2}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (report_dir / "environment_inventory.json").write_text(
                json.dumps({"commands": {}, "python_packages": {}, "command_versions": {}}),
                encoding="utf-8",
            )
            (report_dir / "parser_candidate_matrix.json").write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "parser_id": "docling_standard",
                                "display_name": "Docling standard",
                                "reference_only": False,
                                "open_source_default": True,
                                "license_summary": "MIT",
                                "modes": ["standard"],
                                "output_focus": ["tables"],
                            }
                        ],
                        "probes": [
                            {
                                "parser_id": "docling_standard",
                                "available": False,
                                "package_found": False,
                                "commands_found": {"docling": None},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (report_dir / "parser_smoke.json").write_text(
                json.dumps(
                    {
                        "commands": [
                            {
                                "parser_id": "docling_standard",
                                "sample_id": "local-a",
                                "executed": True,
                                "ok": False,
                                "returncode": 1,
                                "duration_seconds": 2.5,
                                "output_files": [],
                                "stderr_tail": "DownloadFileException: Failed to download model",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (report_dir / "parser_run_materialization.json").write_text(
                json.dumps(
                    {
                        "result": {
                            "parser_id": "docling_standard",
                            "sample_id": "local-a",
                            "ok": False,
                            "run_dir": "reports/parser_runs/docling_standard/local-a",
                            "stats": {},
                            "error": "unsupported",
                        }
                    }
                ),
                encoding="utf-8",
            )

            comparison = build_parser_model_comparison(report_dir)

        self.assertEqual(comparison["parsers"][0]["status"], "install_required")
        self.assertFalse(comparison["parsers"][0]["smoke"]["ok"])
        self.assertFalse(comparison["parsers"][0]["materialized_run"]["ok"])
        self.assertIn("DownloadFileException", comparison["parsers"][0]["smoke"]["failure_summary"])
        self.assertEqual(comparison["retrieval"]["passed_count"], 0)
        self.assertIn("figure/org-chart", comparison["recommendations"][0])


if __name__ == "__main__":
    unittest.main()
