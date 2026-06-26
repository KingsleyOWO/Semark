import json
import tempfile
import unittest
from pathlib import Path

from app.eval.parser_smoke import (
    build_parser_smoke_commands,
    local_sample_sources,
    write_parser_smoke_reports,
)


class ParserSmokeTest(unittest.TestCase):
    def test_local_sample_sources_resolve_original_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            source_dir = workspace / "store" / "docs" / "doc-a" / "source"
            source_dir.mkdir(parents=True)
            (source_dir / "original.pdf").write_bytes(b"%PDF")
            corpus = root / "corpus.json"
            corpus.write_text(
                json.dumps({"local_samples": [{"sample_id": "sample-a", "doc_id": "doc-a"}]}),
                encoding="utf-8",
            )

            samples = local_sample_sources(workspace, corpus)

        self.assertEqual(samples[0]["sample_id"], "sample-a")
        self.assertTrue(samples[0]["source_exists"])
        self.assertTrue(samples[0]["source_path"].endswith("original.pdf"))

    def test_build_parser_smoke_commands_are_non_destructive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            source_dir = workspace / "store" / "docs" / "doc-a" / "source"
            source_dir.mkdir(parents=True)
            (source_dir / "original.pdf").write_bytes(b"%PDF")
            corpus = root / "corpus.json"
            corpus.write_text(
                json.dumps({"local_samples": [{"sample_id": "sample-a", "doc_id": "doc-a"}]}),
                encoding="utf-8",
            )

            commands = build_parser_smoke_commands(
                workspace=workspace,
                output_dir=Path("workspace/eval/reports_dgx_test"),
                corpus_path=corpus,
                sample_id="sample-a",
            )

        self.assertEqual({command.parser_id for command in commands}, {
            "mineru3_pipeline",
            "docling_standard",
            "paddleocr_structure_v3",
            "olmocr",
        })
        for command in commands:
            self.assertIn("workspace/eval/reports_dgx_test/parser_runs", command.output_dir)
            self.assertNotIn("workspace/eval/reports/", command.output_dir)

    def test_build_parser_smoke_commands_can_filter_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            source_dir = workspace / "store" / "docs" / "doc-a" / "source"
            source_dir.mkdir(parents=True)
            (source_dir / "original.pdf").write_bytes(b"%PDF")
            corpus = root / "corpus.json"
            corpus.write_text(
                json.dumps({"local_samples": [{"sample_id": "sample-a", "doc_id": "doc-a"}]}),
                encoding="utf-8",
            )

            commands = build_parser_smoke_commands(
                workspace=workspace,
                output_dir=Path("workspace/eval/reports_dgx_test"),
                corpus_path=corpus,
                parser_id="mineru3_pipeline",
            )

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].parser_id, "mineru3_pipeline")

    def test_filtered_smoke_report_merges_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            output_dir = root / "reports"
            source_dir = workspace / "store" / "docs" / "doc-a" / "source"
            source_dir.mkdir(parents=True)
            (source_dir / "original.pdf").write_bytes(b"%PDF")
            corpus = root / "corpus.json"
            corpus.write_text(
                json.dumps({"local_samples": [{"sample_id": "sample-a", "doc_id": "doc-a"}]}),
                encoding="utf-8",
            )
            output_dir.mkdir()
            (output_dir / "parser_smoke.json").write_text(
                json.dumps(
                    {
                        "commands": [
                            {
                                "parser_id": "docling_standard",
                                "sample_id": "sample-a",
                                "executed": True,
                                "ok": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            write_parser_smoke_reports(
                workspace=workspace,
                output_dir=output_dir,
                corpus_path=corpus,
                sample_id="sample-a",
                parser_id="mineru3_pipeline",
                execute=False,
            )
            report = json.loads((output_dir / "parser_smoke.json").read_text(encoding="utf-8"))

        parser_ids = {command["parser_id"] for command in report["commands"]}
        self.assertEqual(parser_ids, {"docling_standard", "mineru3_pipeline"})


if __name__ == "__main__":
    unittest.main()
