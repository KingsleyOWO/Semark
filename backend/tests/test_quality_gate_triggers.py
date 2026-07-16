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


def test_structured_output_empty_skipped_when_table_content_is_woven_into_rag_body(tmp_path):
    # A generic (default_chunks) document delivers table semantics inside
    # rag.md, not as structured records — same rationale as the existing
    # enriched-image exemption. Live: 5 slide-deck guides were pushed to
    # needs_review 0.0-0.25 over tables whose content sat right there in
    # rag.md.
    document_ir = _make_ir(
        "meeting-room-guide.pdf",
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
        source_md=(
            "# 採購說明\n\n表格名稱：採購品項一覽\n\n### 紙張\n- 項目：紙張\n- 數量：10\n- 單價：50\n\n"
            "以上供各單位查閱與比價使用。"
        ),
        tmp_path=tmp_path,
    )

    codes = {issue.code for issue in result.issues}
    assert "structured_output_empty" not in codes


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


def test_structured_output_empty_skipped_for_figure_enrichment_in_generic_document(tmp_path):
    """A generic (prose) document weaves figure enrichments into the RAG
    markdown body, not into structured records, so an enriched figure alone is
    not evidence that structured output should exist. This is the 208-style
    meeting-room operation guide (photos/screenshots + prose) that legitimately
    has no rows; flagging it structured_output_empty is a false positive."""
    document_ir = _make_ir(
        "208-meeting-room-operation.pdf",
        "pdf",
        [
            Block(
                block_id="t1",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "會議室情境操作說明：請依畫面按鈕操作投影機與音響系統。"},
            ),
            Block(
                block_id="fig",
                type=BlockType.IMAGE,
                page_idx=0,
                payload={"img_path": "images/control-panel.jpg"},
            ),
        ],
    )

    result = _run_gate(
        document_ir,
        _empty_structured_output(),  # generic_document, empty records/chunks
        source_md="會議室情境操作說明：請依畫面按鈕操作投影機與音響系統，包含情境模式、音量與黑幕升降控制。",
        enrichments={
            "fig": {
                "kind": "figure_caption",
                "input": {"page_idx": 0},
                "output": {"semantic_caption": "會議室環控面板"},
            }
        },
        tmp_path=tmp_path,
    )

    codes = {issue.code for issue in result.issues}
    assert "structured_output_empty" not in codes
    assert result.status == "pass"


def test_structured_output_empty_fires_for_figure_enrichment_in_form_document(tmp_path):
    """The enriched-figure structure signal still fires structured_output_empty
    for form-type documents, where empty structured records really is a defect.
    Only the generic/prose case is exempt; the figure-signal path is preserved."""
    document_ir = _make_ir(
        "signature-form-scan.pdf",
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
        _empty_structured_output("form_document"),
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


def test_form_page_enrichment_counts_as_structure_signal():
    # A detected form page (full-page enrichment keyed form_page_*) is not an
    # IR IMAGE block, but it is hard evidence of structured content: the check
    # must fire when structured output is empty. Live gap: 2-10 請假辦法 has a
    # form page + form_asset enrichment yet shipped 0-byte structured output.
    from app.pipeline.quality_gate import _check_structured_output_presence

    document_ir = DocumentIR(
        doc_id="doc-form-page",
        run_id="run-1",
        source=SourceInfo(path="請假辦法.pdf", ext=".pdf", sha256="x", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[
            Block(
                block_id="t-0",
                type=BlockType.TEXT,
                page_idx=0,
                bbox_norm=[0, 0, 1000, 100],
                payload={"text": "第一條 本辦法依本院人員管理辦法訂定之。"},
            )
        ],
    )
    issues = _check_structured_output_presence(
        document_ir,
        structured_output=None,
        structured_text="",
        source_md="來源文字" * 20,
        enrichments={"form_page_0011": {"kind": "form_asset", "output": {"title": "請假單"}}},
    )

    assert [i.code for i in issues] == ["structured_output_empty"]
    assert "form_page_enrichment" in issues[0].evidence["structure_signals"]


# ---------------------------------------------------------------------------
# #7: former gate blind spots — broken embeds, silent text drop, watermark leak,
# figure-semantics language. All four passed 0.95-1.0 on visibly broken outputs
# before these checks existed.
# ---------------------------------------------------------------------------

def _prose_blocks(texts, page_idx=0):
    return [
        Block(block_id=f"t{i}", type=BlockType.TEXT, page_idx=page_idx, payload={"text": t})
        for i, t in enumerate(texts)
    ]


_GUIDE_STEPS = [
    "1. 開啟系統設定頁面，點選帳戶管理。",
    "2. 於帳戶清單中選擇要調整的帳戶。",
    "3. 點選權限分頁並修改角色設定。",
    "4. 儲存變更後重新登入系統。",
    "提醒：變更權限後三十分鐘內生效。",
]


def test_broken_image_embed_fires_on_markdown_image_link(tmp_path):
    ir = _make_ir("guide.pdf", "pdf", _prose_blocks(["操作說明如下，請依序執行。"]))
    result = _run_gate(
        ir,
        _empty_structured_output(),
        "# 說明\n\n![外接示意](images/abc123.jpg)\n\n操作說明如下，請依序執行。",
        tmp_path=tmp_path,
    )
    codes = {issue.code: issue.severity for issue in result.issues}
    assert codes.get("broken_image_embed") == "high", codes


def test_unresolved_asset_token_fires_as_broken_embed(tmp_path):
    # [[asset:...]] tokens are internal anchors; one that survives into the
    # final semantic text is unreadable noise (observed: table placeholders
    # in rag.md on gate-passing docs).
    ir = _make_ir("guide.pdf", "pdf", _prose_blocks(["操作說明如下，請依序執行。"]))
    result = _run_gate(
        ir,
        _empty_structured_output(),
        "# 說明\n\n操作說明如下，請依序執行。\n\n[[asset:tbl0000]]\n\n後續內容。",
        tmp_path=tmp_path,
    )
    codes = {issue.code: issue.severity for issue in result.issues}
    assert codes.get("broken_image_embed") == "high", codes


def test_authored_text_dropped_fires_when_steps_missing_from_output(tmp_path):
    ir = _make_ir("guide.pdf", "pdf", _prose_blocks(_GUIDE_STEPS))
    # Output silently kept only step 1 (the mailbox-archive failure shape).
    result = _run_gate(
        ir,
        _empty_structured_output(),
        "# 指南\n\n1. 開啟系統設定頁面，點選帳戶管理。",
        tmp_path=tmp_path,
    )
    hits = [issue for issue in result.issues if issue.code == "authored_text_dropped"]
    assert hits, [issue.code for issue in result.issues]
    assert hits[0].severity == "high"
    assert hits[0].evidence["survival_ratio"] < 0.6


def test_authored_text_survival_passes_when_all_steps_present(tmp_path):
    ir = _make_ir("guide.pdf", "pdf", _prose_blocks(_GUIDE_STEPS))
    result = _run_gate(
        ir,
        _empty_structured_output(),
        "# 指南\n\n" + "\n\n".join(_GUIDE_STEPS),
        tmp_path=tmp_path,
    )
    assert not [issue for issue in result.issues if issue.code == "authored_text_dropped"]


def test_watermark_term_leak_fires_when_configured_term_in_output(tmp_path, monkeypatch):
    from app.pipeline.corpus_rules import CorpusRules

    monkeypatch.setattr(
        "app.pipeline.quality_gate.get_corpus_rules",
        lambda: CorpusRules(document_markers={"watermark_terms": ("示範40",)}),
    )
    ir = _make_ir("guide.pdf", "pdf", _prose_blocks(["操作說明如下，請依序執行。"]))
    result = _run_gate(
        ir,
        _empty_structured_output(),
        "操作說明如下，請依序執行。\n\n示範40",
        tmp_path=tmp_path,
    )
    codes = {issue.code: issue.severity for issue in result.issues}
    assert codes.get("watermark_term_leak") == "warning", codes


def test_figure_semantics_language_mismatch_fires_for_english_only_figure(tmp_path):
    ir = _make_ir("guide.pdf", "pdf", _prose_blocks(["操作說明如下，請依序執行。"]))
    asset = SimpleNamespace(
        type="figure_asset",
        block_id="fig-a",
        asset_id="fig0000",
        page_idx=0,
        title="Figure 1",
        retrieval_text="The rear panel has a rocker power switch below the arrow sticker.",
        needs_review=False,
    )
    result = asyncio.run(
        run_quality_gate(
            document_ir=ir,
            source_md="操作說明如下，請依序執行。",
            assets=[asset],
            structured_output=_empty_structured_output(),
            enrichments={},
            run_path=tmp_path,
            vlm_adapter=None,
            max_vlm_audits=0,
            semantic_output_language="zh-TW",
        )
    )
    codes = {issue.code: issue.severity for issue in result.issues}
    assert codes.get("figure_semantics_language_mismatch") == "warning", codes


def test_enrichment_failure_check_flags_unknown_error(tmp_path):
    """UNKNOWN_ERROR salvages slipped past the JSON_PARSE_FAILED-only filter
    (observed live: a reasoning-dump enrichment with _error=UNKNOWN_ERROR shipped
    into rag.md with a passing gate)."""
    ir = _make_ir("guide.pdf", "pdf", _prose_blocks(["操作說明如下，請依序執行。"]))
    result = _run_gate(
        ir,
        _empty_structured_output(),
        "操作說明如下，請依序執行。",
        enrichments={
            "b000009": {
                "kind": "figure_description",
                "input": {"page_idx": 0},
                "output": {"semantic_caption": "", "_error": "UNKNOWN_ERROR"},
            }
        },
        tmp_path=tmp_path,
    )
    codes = {issue.code: issue.severity for issue in result.issues}
    assert codes.get("vlm_enrichment_parse_failed") == "high", codes


# ---------------------------------------------------------------------------
# VLM audit context must audit the delivered text
# ---------------------------------------------------------------------------

def test_audit_context_falls_back_to_delivered_rag_text_when_structured_empty():
    # For default_chunks documents structured_text is legitimately empty; the
    # deliverable is rag.md. Auditing an "(empty structured semantic output)"
    # placeholder made the VLM (correctly) report everything as missing —
    # confidence-1.0 false alarms worth -0.5 on the gate score.
    from app.pipeline.quality_gate import _audit_context

    document_ir = _make_ir("guide.pdf", "pdf", _prose_blocks(["會議室設備介紹頁。"]))

    context = _audit_context(
        document_ir,
        0,
        source_md="# 會議室設備使用說明\n\n電腦中心提供各會議室設備一覽。",
        structured_text="",
        reasons=["structured_output_empty"],
    )

    assert "(empty structured semantic output)" not in context
    assert "會議室設備一覽" in context


def test_audit_context_window_covers_the_audited_page_in_long_documents():
    # The audited page's slice of the delivered text sits far beyond a
    # head-truncated 9000-char window; the context must centre on it. The
    # neighbour sentence exists only in source_md (not in IR blocks), so the
    # MinerU-evidence section cannot satisfy this assertion.
    from app.pipeline.quality_gate import _audit_context

    page_marker = "第二十頁的獨特內容標記字串"
    neighbour = "緊接著標記的補充敘述甲乙丙"
    long_head = "前段內容。" * 4000  # far beyond the 9000-char head window
    document_ir = _make_ir("guide.pdf", "pdf", _prose_blocks([page_marker]))

    context = _audit_context(
        document_ir,
        0,
        source_md=long_head + page_marker + neighbour + "。",
        structured_text="",
        reasons=["structured_output_empty"],
    )

    assert neighbour in context
