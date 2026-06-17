import json

from app.api.routes.runs import _load_quality_gate_summary, _load_split_document_summary


def test_load_split_document_summary_counts_main_and_extracted(tmp_path):
    outputs = tmp_path / "outputs"
    documents = outputs / "documents"
    documents.mkdir(parents=True)
    (outputs / "documents_index.json").write_text(
        json.dumps(
            [
                {"document_id": "main", "kind": "main", "file": str(documents / "main.md")},
                {"document_id": "form_0001", "kind": "form", "file": str(documents / "form_0001.md")},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = _load_split_document_summary(outputs)

    assert summary["documents_total"] == 2
    assert summary["main_document_count"] == 1
    assert summary["extracted_document_count"] == 1
    assert summary["documents"][0]["filename"] == "main.md"
    assert "file" not in summary["documents"][0]


def test_load_quality_gate_summary_is_compact(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "quality_gate.json").write_text(
        json.dumps({"status": "warning", "score": 0.88, "issues": [{"code": "x"}]}),
        encoding="utf-8",
    )

    summary = _load_quality_gate_summary(outputs)

    assert summary == {
        "quality_gate_status": "warning",
        "quality_score": 0.88,
        "quality_issue_count": 1,
    }
