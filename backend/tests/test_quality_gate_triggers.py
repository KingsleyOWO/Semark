"""Level 1 trigger semantics for the quality gate.

Covers two areas:
- structured_output_empty must only fire when the document shows structure
  signals (tables, form/figure enriched images, form-like text); plain prose
  documents legitimately have empty structured output.
- Symmetric language-noise checks: a lowered english_noise_high floor for
  zh-TW output and a new zh_noise_in_english check for en output.
"""

import asyncio
from types import SimpleNamespace

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.quality_gate import _check_language_noise, run_quality_gate


def _empty_structured_output(document_type: str = "generic_document") -> SimpleNamespace:
    return SimpleNamespace(
        plan=SimpleNamespace(document_type=document_type),
        records=[],
        chunks=[],
        rag_markdown="",
    )


def _make_ir(path: str, ext: str, blocks: list[Block]) -> DocumentIR:
    return DocumentIR(
        doc_id="doc-trigger",
        run_id="run-trigger",
        source=SourceInfo(path=path, ext=ext, sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png")],
        blocks=blocks,
    )


def _run_gate(document_ir, structured_output, source_md, enrichments=None, tmp_path=None, **kwargs):
    return asyncio.run(
        run_quality_gate(
            document_ir=document_ir,
            source_md=source_md,
            assets=[],
            structured_output=structured_output,
            enrichments=enrichments or {},
            run_path=tmp_path,
            vlm_adapter=None,
            max_vlm_audits=0,
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# structured_output_empty requires structure signals
# ---------------------------------------------------------------------------

def test_structured_output_empty_skipped_for_prose_document_without_structure_signals(tmp_path):
    document_ir = _make_ir(
        "employee-leave-regulation.pdf",
        "pdf",
        [
            Block(
                block_id="t1",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "本辦法依勞動基準法規定訂定之。員工休假管理悉依本辦法辦理。"},
            ),
            Block(
                block_id="t2",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "寒暑假期間之出勤另依學校行事曆調整，未盡事宜由人事室解釋之。"},
            ),
        ],
    )

    result = _run_gate(
        document_ir,
        _empty_structured_output(),
        source_md="本辦法依勞動基準法規定訂定之。員工休假管理悉依本辦法辦理。寒暑假期間之出勤另依學校行事曆調整，未盡事宜由人事室解釋之。",
        tmp_path=tmp_path,
    )

    codes = {issue.code for issue in result.issues}
    assert "structured_output_empty" not in codes
    assert result.status == "pass"


def test_structured_output_empty_fires_for_table_document_with_empty_output(tmp_path):
    document_ir = _make_ir(
        "unit-price-list.pdf",
        "pdf",
        [
            Block(
                block_id="tbl",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": "<table><tr><td>項目</td><td>數量</td><td>單價</td></tr>"
                    "<tr><td>紙張</td><td>10</td><td>50</td></tr></table>",
                },
            )
        ],
    )

    result = _run_gate(
        document_ir,
        _empty_structured_output(),
        source_md="採購品項一覽包含紙張與其他文具，數量與單價均已列示於上方表格中，供各單位查閱與比價使用。",
        tmp_path=tmp_path,
    )

    issue = next(issue for issue in result.issues if issue.code == "structured_output_empty")
    assert issue.severity == "high"
    assert result.status == "needs_review"
    for key in ("source_text_length", "structured_document_type", "record_count", "chunk_count", "has_table_or_image"):
        assert key in issue.evidence
    assert issue.evidence["has_table_or_image"] is True


def test_structured_output_empty_ignores_unenriched_decorative_image(tmp_path):
    document_ir = _make_ir(
        "campus-announcement.pdf",
        "pdf",
        [
            Block(
                block_id="t1",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "各單位辦理性別平等教育宣導時，應依本流程辦理相關通報作業。"},
            ),
            Block(
                block_id="logo",
                type=BlockType.IMAGE,
                page_idx=0,
                payload={"img_path": "images/logo.png"},
            ),
        ],
    )

    result = _run_gate(
        document_ir,
        _empty_structured_output(),
        source_md="各單位辦理性別平等教育宣導時，應依本流程辦理相關通報作業，並於期限內完成結案報告以利彙整。",
        tmp_path=tmp_path,
    )

    codes = {issue.code for issue in result.issues}
    assert "structured_output_empty" not in codes
    assert result.status == "pass"


def test_structured_output_empty_fires_for_image_with_figure_enrichment(tmp_path):
    document_ir = _make_ir(
        "sop-flowchart.pdf",
        "pdf",
        [
            Block(
                block_id="t1",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "各單位辦理性別平等教育宣導時，應依本流程辦理相關通報作業。"},
            ),
            Block(
                block_id="fig",
                type=BlockType.IMAGE,
                page_idx=0,
                payload={"img_path": "images/flowchart.jpg"},
            ),
        ],
    )

    result = _run_gate(
        document_ir,
        _empty_structured_output(),
        source_md="各單位辦理性別平等教育宣導時，應依本流程辦理相關通報作業，並於期限內完成結案報告以利彙整。",
        enrichments={
            "fig": {
                "kind": "figure_caption",
                "input": {"page_idx": 0},
                "output": {"semantic_caption": "通報作業流程圖"},
            }
        },
        tmp_path=tmp_path,
    )

    issue = next(issue for issue in result.issues if issue.code == "structured_output_empty")
    assert issue.severity == "high"
    assert result.status == "needs_review"


def test_structured_output_empty_fires_for_form_like_text_document(tmp_path):
    document_ir = _make_ir(
        "出差旅費申請單.xls",
        "xls",
        [
            Block(
                block_id="t1",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "申請日期：　　事由：　　金額："},
            ),
        ],
    )

    result = _run_gate(
        document_ir,
        _empty_structured_output(),
        source_md="出差旅費申請單需填寫申請日期、事由與金額，並依內部作業規範送交權責單位完成審核流程後始得報支。",
        tmp_path=tmp_path,
    )

    issue = next(issue for issue in result.issues if issue.code == "structured_output_empty")
    assert issue.severity == "high"
    assert result.status == "needs_review"


# ---------------------------------------------------------------------------
# Language noise checks: lowered english_noise_high floor
# ---------------------------------------------------------------------------

def test_english_noise_high_fires_below_old_floor():
    zh_part = "中文語意內容" * 9  # 54 CJK chars, below the old floor of 80
    text = zh_part + " English caption noise from VLM output here"  # 36 ascii letters > 0.6 * 54

    issues = _check_language_noise("", "zh-TW", structured_text=text)

    assert [issue.code for issue in issues] == ["english_noise_high"]
    assert issues[0].severity == "medium"


def test_english_noise_high_requires_min_cjk():
    zh_part = "中文" * 15  # 30 CJK chars, below the 40-char floor
    text = zh_part + " This document is essentially English content with many letters"

    issues = _check_language_noise("", "zh-TW", structured_text=text)

    assert issues == []


def test_english_noise_high_not_fired_for_mostly_chinese():
    zh_part = "中文內容穩定" * 10  # 60 CJK chars
    text = zh_part + " short English tail"  # 16 ascii letters <= 0.6 * 60

    issues = _check_language_noise("", "zh-TW", structured_text=text)

    assert issues == []

    boundary_zh = "中文字元" * 12 + "五十"  # 50 CJK chars
    boundary = boundary_zh + " abcdefghij abcdefghij abcdefghij"  # exactly 30 = 0.6 * 50 letters

    assert _check_language_noise("", "zh-TW", structured_text=boundary) == []


# ---------------------------------------------------------------------------
# Language noise checks: zh_noise_in_english
# ---------------------------------------------------------------------------

def test_zh_noise_in_english_fires_at_15_percent():
    text = "Employee travel reimbursement rules and workflow. 出差費用核銷規定與流程說明。"

    issues = _check_language_noise("", "en", structured_text=text)

    zh_noise = [issue for issue in issues if issue.code == "zh_noise_in_english"]
    assert len(zh_noise) == 1
    assert zh_noise[0].severity == "medium"


def test_zh_noise_in_english_excludes_code_blocks():
    body = "English guide for the deployment process.\nAll remaining text is English only.\n"
    cjk = "中文註解內容僅出現在程式碼區塊中"

    fenced = body + "```\n" + cjk + "\n```\n"
    unfenced = body + cjk + "\n"

    fenced_codes = [issue.code for issue in _check_language_noise("", "en", structured_text=fenced)]
    unfenced_codes = [issue.code for issue in _check_language_noise("", "en", structured_text=unfenced)]

    assert "zh_noise_in_english" not in fenced_codes
    assert "zh_noise_in_english" in unfenced_codes


def test_zh_noise_in_english_below_threshold_only_flags_language_mismatch():
    text = (
        "This procurement policy applies to every department and describes the "
        "review workflow, budget limits, delegation rules, and record retention "
        "requirements in detail for auditors. 備註說明"
    )

    issues = _check_language_noise("", "en", structured_text=text)
    codes = [issue.code for issue in issues]

    assert "target_language_mismatch" in codes
    assert "zh_noise_in_english" not in codes


def test_zh_noise_in_english_reported_through_quality_gate(tmp_path):
    document_ir = _make_ir(
        "travel-notes.pdf",
        "pdf",
        [
            Block(
                block_id="t1",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "Travel expense policy"},
            ),
        ],
    )
    structured_output = SimpleNamespace(
        plan=SimpleNamespace(document_type="generic_document"),
        records=[],
        chunks=[],
        rag_markdown=(
            "## Travel Expense Rules\nEmployees must submit receipts within 30 days.\n\n"
            "出差費用核銷規定：員工應於三十日內檢附單據辦理核銷作業。"
        ),
    )

    result = _run_gate(
        document_ir,
        structured_output,
        source_md="Employees must submit travel expense receipts within thirty days of returning.",
        tmp_path=tmp_path,
        semantic_output_language="en",
    )

    codes = {issue.code for issue in result.issues}
    assert {"target_language_mismatch", "zh_noise_in_english"} <= codes
    assert result.status == "warning"
    assert result.stats["semantic_quality"]["correctness_score"] == 0.76

    payload = result.to_dict()
    zh_noise = next(issue for issue in payload["issues"] if issue["code"] == "zh_noise_in_english")
    assert not any("一" <= char <= "鿿" for char in zh_noise["message"])
