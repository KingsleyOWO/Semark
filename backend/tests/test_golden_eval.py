"""Golden-set evaluation harness tests.

These tests drive ``app.eval.golden`` with tiny synthetic outputs
directories (never real pipeline runs) so quality scoring stays fast and
deterministic.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.eval import runner
from app.eval.golden import (
    GoldenDoc,
    GoldenExpectations,
    GoldenManifest,
    compare_runs,
    load_manifest,
    render_comparison_markdown,
    render_markdown,
    score_run,
)
from app.pipeline.stages.chunk import estimate_tokens

BACKEND_DIR = Path(__file__).resolve().parents[1]
GOLDEN_MANIFEST_DIR = BACKEND_DIR / "examples" / "eval" / "golden"
RESIDUE_MARKERS = ["來源抽取文字", "<table", "$\\square$"]


def make_manifest(**expectations) -> GoldenManifest:
    return GoldenManifest(
        doc=GoldenDoc(name="synthetic-doc", source_pdf="/tmp/synthetic.pdf"),
        expectations=GoldenExpectations(**expectations),
    )


def write_outputs(
    base: Path,
    *,
    structured_rag: str | None = None,
    documents: dict[str, str] | None = None,
    chunks: list[dict] | None = None,
    structured_chunks: list[dict] | None = None,
    quality_gate: dict | None = None,
) -> Path:
    outputs = base / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    if structured_rag is not None:
        (outputs / "structured_rag.md").write_text(structured_rag, encoding="utf-8")
    for name, text in (documents or {}).items():
        docs_dir = outputs / "documents"
        docs_dir.mkdir(exist_ok=True)
        (docs_dir / name).write_text(text, encoding="utf-8")
    if chunks is not None:
        (outputs / "chunks.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in chunks),
            encoding="utf-8",
        )
    if structured_chunks is not None:
        (outputs / "structured_chunks.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in structured_chunks
            ),
            encoding="utf-8",
        )
    if quality_gate is not None:
        (outputs / "quality_gate.json").write_text(
            json.dumps(quality_gate, ensure_ascii=False), encoding="utf-8"
        )
    return outputs


def chunk_record(chunk_id: str, content: str, heading_path: list[str] | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "content": content,
        "metadata": {"heading_path": heading_path or []},
    }


class TestManifestSchema:
    def test_manifest_loads_from_json_file(self, tmp_path):
        path = tmp_path / "doc.golden.json"
        path.write_text(
            json.dumps(
                {
                    "doc": {"name": "請假辦法", "source_pdf": "/abs/leave.pdf"},
                    "expectations": {
                        "must_include": ["示範研究院"],
                        "must_not_include": ["<table"],
                        "fact_tokens": ["114.12.11", "第二十六條"],
                        "min_chunk_count": 3,
                        "max_chunk_tokens": 1024,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        manifest = load_manifest(path)

        assert manifest.doc.name == "請假辦法"
        assert manifest.doc.source_pdf == "/abs/leave.pdf"
        assert manifest.expectations.fact_tokens == ["114.12.11", "第二十六條"]
        assert manifest.expectations.min_chunk_count == 3
        assert manifest.expectations.max_chunk_tokens == 1024

    def test_manifest_defaults_are_minimal(self):
        manifest = GoldenManifest(
            doc=GoldenDoc(name="x", source_pdf="/x.pdf"),
            expectations=GoldenExpectations(),
        )
        assert manifest.expectations.must_include == []
        assert manifest.expectations.must_not_include == []
        assert manifest.expectations.fact_tokens == []
        assert manifest.expectations.min_chunk_count == 0
        assert manifest.expectations.max_chunk_tokens is None

    def test_manifest_rejects_unknown_keys(self):
        with pytest.raises(ValidationError):
            GoldenExpectations(must_inclde=["typo"])


class TestFactRecall:
    def test_fact_counts_present_in_structured_rag_or_documents(self, tmp_path):
        outputs = write_outputs(
            tmp_path,
            structured_rag="規章名稱：示範研究院人員請假辦法",
            documents={"main.md": "第 二十六 條 本辦法經院長核定後實施"},
            chunks=[chunk_record("c1", "text", ["h"])],
        )
        manifest = make_manifest(
            fact_tokens=["示範研究院人員請假辦法", "第二十六條", "不存在的詞彙"]
        )

        report = score_run(manifest, outputs)

        assert report.fact_token_count == 3
        assert report.fact_tokens_found == 2
        assert report.fact_recall == pytest.approx(2 / 3)
        assert report.missing_fact_tokens == ["不存在的詞彙"]

    def test_fact_matching_is_whitespace_and_fullwidth_insensitive(self, tmp_path):
        outputs = write_outputs(
            tmp_path,
            structured_rag="修正日期：１１４.１２.１１\n特別休假 三十四 日",
            chunks=[chunk_record("c1", "text")],
        )
        manifest = make_manifest(fact_tokens=["114.12.11", "特別休假三十四日"])

        report = score_run(manifest, outputs)

        assert report.fact_recall == 1.0
        assert report.missing_fact_tokens == []

    def test_fact_tokens_do_not_leak_across_file_boundaries(self, tmp_path):
        # "114." at the end of one file and "12.11" at the start of another
        # must not be stitched into a fake "114.12.11" hit.
        outputs = write_outputs(
            tmp_path,
            structured_rag="修正 114.",
            documents={"main.md": "12.11 施行"},
        )
        manifest = make_manifest(fact_tokens=["114.12.11"])

        report = score_run(manifest, outputs)

        assert report.fact_tokens_found == 0

    def test_empty_fact_token_list_scores_full_recall(self, tmp_path):
        outputs = write_outputs(tmp_path, structured_rag="任何內容")
        report = score_run(make_manifest(), outputs)
        assert report.fact_recall == 1.0
        assert report.fact_token_count == 0


class TestIncludeRules:
    def test_must_include_hit_rate_and_missing(self, tmp_path):
        outputs = write_outputs(
            tmp_path,
            structured_rag="示範研究院人員訓練辦法\n範-12",
        )
        manifest = make_manifest(
            must_include=["示範研究院人員訓練辦法", "範-12", "附件三"]
        )

        report = score_run(manifest, outputs)

        assert report.must_include_count == 3
        assert report.must_include_found == 2
        assert report.must_include_rate == pytest.approx(2 / 3)
        assert report.missing_must_include == ["附件三"]

    def test_must_not_include_reports_violations(self, tmp_path):
        outputs = write_outputs(
            tmp_path,
            structured_rag="<table><tr><td>殘留</td></tr></table>",
            documents={"main.md": "選項 $\\square$ 是"},
        )
        manifest = make_manifest(
            must_not_include=["來源抽取文字", "<table", "$\\square$"]
        )

        report = score_run(manifest, outputs)

        assert report.must_not_include_violations == ["<table", "$\\square$"]


class TestChunkStats:
    def test_chunk_stats_token_sizes_and_heading_coverage(self, tmp_path):
        long_text = "很長的內容" * 40
        outputs = write_outputs(
            tmp_path,
            structured_rag="內容",
            chunks=[
                chunk_record("c1", "短內容", ["第一條"]),
                chunk_record("c2", long_text, ["第二條", "第一項"]),
                chunk_record("c3", "另一段", None),
            ],
        )
        manifest = make_manifest(min_chunk_count=2, max_chunk_tokens=50)

        report = score_run(manifest, outputs)

        assert report.chunks.chunks_file == "chunks.jsonl"
        assert report.chunks.count == 3
        assert report.chunks.max_tokens == estimate_tokens(long_text)
        assert report.chunks.heading_path_coverage == pytest.approx(2 / 3)
        assert report.chunks.min_chunk_count_ok is True
        assert report.chunks.max_chunk_tokens_ok is False
        assert report.chunks.oversized_chunk_ids == ["c2"]

    def test_chunk_stats_fall_back_to_structured_chunks(self, tmp_path):
        outputs = write_outputs(
            tmp_path,
            structured_rag="內容",
            structured_chunks=[chunk_record("sr1", "紀錄一"), chunk_record("sr2", "紀錄二")],
        )
        manifest = make_manifest(min_chunk_count=3)

        report = score_run(manifest, outputs)

        assert report.chunks.chunks_file == "structured_chunks.jsonl"
        assert report.chunks.count == 2
        assert report.chunks.min_chunk_count_ok is False

    def test_missing_chunk_files_score_zero_chunks(self, tmp_path):
        outputs = write_outputs(tmp_path, structured_rag="內容")
        report = score_run(make_manifest(), outputs)
        assert report.chunks.chunks_file is None
        assert report.chunks.count == 0
        assert report.chunks.heading_path_coverage == 0.0


class TestQualityGatePassthrough:
    def test_quality_gate_status_and_score_passthrough(self, tmp_path):
        outputs = write_outputs(
            tmp_path,
            structured_rag="內容",
            quality_gate={"status": "pass", "score": 0.91, "issues": []},
        )
        report = score_run(make_manifest(), outputs)
        assert report.quality_gate.present is True
        assert report.quality_gate.status == "pass"
        assert report.quality_gate.score == pytest.approx(0.91)

    def test_missing_quality_gate_is_reported_absent(self, tmp_path):
        outputs = write_outputs(tmp_path, structured_rag="內容")
        report = score_run(make_manifest(), outputs)
        assert report.quality_gate.present is False
        assert report.quality_gate.status is None
        assert report.quality_gate.score is None


class TestReportOutput:
    def test_score_run_requires_existing_outputs_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            score_run(make_manifest(), tmp_path / "missing" / "outputs")

    def test_report_is_json_serializable(self, tmp_path):
        outputs = write_outputs(
            tmp_path,
            structured_rag="示範研究院",
            chunks=[chunk_record("c1", "內容", ["h"])],
            quality_gate={"status": "pass", "score": 1.0},
        )
        report = score_run(make_manifest(fact_tokens=["示範研究院"]), outputs)

        payload = json.dumps(report.to_dict(), ensure_ascii=False)

        decoded = json.loads(payload)
        assert decoded["fact_recall"] == 1.0
        assert decoded["chunks"]["count"] == 1
        assert decoded["quality_gate"]["status"] == "pass"

    def test_render_markdown_is_a_compact_table(self, tmp_path):
        outputs = write_outputs(
            tmp_path,
            structured_rag="示範研究院",
            chunks=[chunk_record("c1", "內容", ["h"])],
        )
        report = score_run(
            make_manifest(fact_tokens=["示範研究院", "找不到的詞"]), outputs
        )

        markdown = render_markdown(report)

        assert "synthetic-doc" in markdown
        assert "| fact recall |" in markdown
        assert "找不到的詞" in markdown


class TestCompareRuns:
    def _build_pair(self, tmp_path):
        outputs_a = write_outputs(
            tmp_path / "a",
            structured_rag="只有一個 114.12.11 <table",
            chunks=[chunk_record("c1", "內容一", ["h"]), chunk_record("c2", "內容二")],
            quality_gate={"status": "warn", "score": 0.5},
        )
        outputs_b = write_outputs(
            tmp_path / "b",
            structured_rag="包含 114.12.11 也包含 第二十六條",
            chunks=[
                chunk_record("c1", "內容一", ["h"]),
                chunk_record("c2", "內容二", ["h"]),
                chunk_record("c3", "內容三", ["h"]),
            ],
            quality_gate={"status": "pass", "score": 0.9},
        )
        manifest = make_manifest(
            fact_tokens=["114.12.11", "第二十六條"],
            must_not_include=["<table"],
        )
        return manifest, outputs_a, outputs_b

    def test_compare_runs_reports_per_metric_deltas(self, tmp_path):
        manifest, outputs_a, outputs_b = self._build_pair(tmp_path)

        comparison = compare_runs(manifest, outputs_a, outputs_b)

        assert comparison["doc_name"] == "synthetic-doc"
        assert comparison["a"]["outputs_dir"] == str(outputs_a)
        assert comparison["b"]["outputs_dir"] == str(outputs_b)
        delta = comparison["delta"]
        assert delta["fact_recall"] == pytest.approx(0.5)
        assert delta["fact_tokens_found"] == 1
        assert delta["must_not_include_violations"] == -1
        assert delta["chunk_count"] == 1
        assert delta["heading_path_coverage"] == pytest.approx(0.5)
        assert delta["quality_gate_score"] == pytest.approx(0.4)

    def test_compare_runs_renders_markdown(self, tmp_path):
        manifest, outputs_a, outputs_b = self._build_pair(tmp_path)
        comparison = compare_runs(manifest, outputs_a, outputs_b)

        markdown = render_comparison_markdown(comparison)

        assert "| metric |" in markdown
        assert "fact recall" in markdown


class TestGoldenCli:
    def _write_manifest(self, tmp_path) -> Path:
        path = tmp_path / "doc.golden.json"
        path.write_text(
            json.dumps(
                {
                    "doc": {"name": "cli-doc", "source_pdf": "/abs/doc.pdf"},
                    "expectations": {
                        "fact_tokens": ["114.12.11"],
                        "must_not_include": ["<table"],
                        "min_chunk_count": 1,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_cli_scores_run_and_writes_json(self, tmp_path, capsys):
        manifest_path = self._write_manifest(tmp_path)
        outputs = write_outputs(
            tmp_path / "run",
            structured_rag="核定 114.12.11",
            chunks=[chunk_record("c1", "內容", ["h"])],
            quality_gate={"status": "pass", "score": 1.0},
        )
        json_path = tmp_path / "golden_report.json"

        exit_code = runner.main(
            [
                "golden",
                "--manifest",
                str(manifest_path),
                "--run-dir",
                str(outputs),
                "--json",
                str(json_path),
            ]
        )

        assert exit_code == 0
        printed = capsys.readouterr().out
        assert "cli-doc" in printed
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["report"]["fact_recall"] == 1.0
        assert payload["comparison"] is None

    def test_cli_compares_against_baseline_dir(self, tmp_path, capsys):
        manifest_path = self._write_manifest(tmp_path)
        baseline = write_outputs(
            tmp_path / "baseline",
            structured_rag="沒有事實記號 <table",
            chunks=[chunk_record("c1", "內容")],
        )
        candidate = write_outputs(
            tmp_path / "candidate",
            structured_rag="核定 114.12.11",
            chunks=[chunk_record("c1", "內容", ["h"])],
        )
        json_path = tmp_path / "golden_compare.json"

        exit_code = runner.main(
            [
                "golden",
                "--manifest",
                str(manifest_path),
                "--run-dir",
                str(candidate),
                "--baseline-dir",
                str(baseline),
                "--json",
                str(json_path),
            ]
        )

        assert exit_code == 0
        printed = capsys.readouterr().out
        assert "| metric |" in printed
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["comparison"]["delta"]["fact_recall"] == pytest.approx(1.0)
        assert payload["comparison"]["delta"]["must_not_include_violations"] == -1


class TestShippedManifests:
    def test_golden_manifests_exist_and_parse(self):
        manifest_paths = sorted(GOLDEN_MANIFEST_DIR.glob("*.json"))
        assert len(manifest_paths) >= 3, f"expected golden manifests in {GOLDEN_MANIFEST_DIR}"

        manifests = [load_manifest(path) for path in manifest_paths]
        for manifest in manifests:
            assert manifest.doc.name
            assert manifest.doc.source_pdf.startswith("/")
            for marker in RESIDUE_MARKERS:
                assert marker in manifest.expectations.must_not_include

        # The two text-layer corpus documents carry discriminative fact tokens.
        rich = [m for m in manifests if len(m.expectations.fact_tokens) >= 10]
        assert len(rich) >= 2
