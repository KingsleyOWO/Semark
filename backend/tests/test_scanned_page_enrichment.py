"""
Scanned visual pages must route through the full-page form_asset VLM flow.

Live evidence: fully scanned PDFs (e.g. 10-3.pdf, one full-page scan image
per page) produced ZERO enrichment entries — no page ever reached the VLM.
These tests pin the fix: pages dominated by IMAGE blocks with little machine
text are rendered at full page (200 DPI) and enriched with the same
form_asset entry shape as detected form pages, so package.py needs no change.
"""

from pathlib import Path

import fitz
import pytest

from app.adapters.vlm import EnrichmentOutput
from app.config import EnrichConfig, PipelineConfig, settings
from app.models.document_ir import (
    Block,
    BlockType,
    DocumentIR,
    EngineInfo,
    PageInfo,
    SourceInfo,
)
from app.pipeline.stages.enrich import EnrichStage, is_scanned_visual_page

# Neutral zh-TW OCR text: no form cue terms, no form filename patterns.
NEUTRAL_OCR_TEXT = (
    "本院一一三年度第三次會議紀錄，出席人員如列，會議決議事項摘要如下，"
    "包含年度工作報告與後續追蹤事項，另就下年度重點方向進行討論。"
) * 5  # ~300+ chars


class FakeVLMAdapter:
    def __init__(self):
        self.form_calls = []
        self.figure_calls = []

    async def check_available(self):
        return True, "ok"

    def get_prompt_version(self, kind: str) -> str:
        return f"{kind}:test"

    async def enrich_form(self, image_path, context_text="", **kwargs):
        self.form_calls.append(
            {"image_path": image_path, "context_text": context_text, **kwargs}
        )
        return EnrichmentOutput(
            success=True,
            kind="form_asset",
            output={
                "document_type": "reference_table",
                "title": "掃描文件",
                "summary": "掃描頁測試輸出",
                "structured_content": "| 項目 | 內容 |",
            },
            tokens_used=10,
            duration_seconds=0.1,
        )

    async def enrich_figure(self, image_path, context_text="", **kwargs):
        self.figure_calls.append({"image_path": image_path, **kwargs})
        return EnrichmentOutput(
            success=True,
            kind="figure_caption",
            output={"caption": "圖說"},
            tokens_used=5,
            duration_seconds=0.1,
        )


class FakeCacheManager:
    def __init__(self, cached: dict | None = None):
        self.cached = dict(cached or {})
        self.get_calls = []
        self.set_calls = []

    async def get_enrich_cache(self, doc_id, block_id, vlm_config_hash, prompt_version):
        self.get_calls.append(block_id)
        return self.cached.get(block_id)

    async def set_enrich_cache(self, **kwargs):
        self.set_calls.append(kwargs)


def _make_stage(enrich_config: EnrichConfig, cached: dict | None = None) -> EnrichStage:
    stage = EnrichStage(db=None, config=PipelineConfig(enrich=enrich_config))
    stage.vlm_adapter = FakeVLMAdapter()
    stage.cache_manager = FakeCacheManager(cached)
    return stage


def _scanned_enrich_config(**overrides) -> EnrichConfig:
    params = {
        "enable_vlm": True,
        "vlm_enrich_forms": False,
        "vlm_enrich_figures": False,
        "vlm_enrich_tables": False,
    }
    params.update(overrides)
    return EnrichConfig(**params)


def _scanned_document(
    num_pages: int = 2,
    *,
    ocr_text: str = "",
    doc_id: str = "doc-scan",
    filename: str = "10-3.pdf",
    extra_blocks: list[Block] | None = None,
) -> DocumentIR:
    """Document where each page is one full-page scan IMAGE block (+ OCR text)."""
    pages: list[PageInfo] = []
    blocks: list[Block] = []
    for i in range(num_pages):
        pages.append(PageInfo(page_idx=i))
        blocks.append(
            Block(
                block_id=f"img-{i}",
                type=BlockType.IMAGE,
                page_idx=i,
                bbox_norm=[0, 0, 1000, 1000],
                payload={"img_path": f"images/scan_{i}.jpg"},
            )
        )
        if ocr_text:
            blocks.append(
                Block(
                    block_id=f"ocr-{i}",
                    type=BlockType.TEXT,
                    page_idx=i,
                    bbox_norm=[50, 50, 950, 950],
                    payload={"text": ocr_text},
                )
            )
    blocks.extend(extra_blocks or [])
    return DocumentIR(
        doc_id=doc_id,
        run_id="run-scan",
        source=SourceInfo(path=filename, ext=".pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        pages=pages,
        blocks=blocks,
    )


@pytest.fixture
def fake_workspace(tmp_path, monkeypatch):
    """Point global settings workspace at tmp so _export_form_page finds the PDF."""
    workspace = tmp_path / "ws"
    monkeypatch.setattr(settings, "workspace_path", workspace)
    return workspace


def _write_source_pdf(doc_id: str, num_pages: int) -> Path:
    source_dir = settings.get_doc_path(doc_id) / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open()
    for _ in range(num_pages):
        page = pdf.new_page(width=595, height=842)
        page.insert_text((72, 72), "scan page")
    pdf_path = source_dir / "original.pdf"
    pdf.save(pdf_path)
    pdf.close()
    return pdf_path


# ---------------------------------------------------------------------------
# Pure detection helper
# ---------------------------------------------------------------------------


def test_is_scanned_visual_page_true_for_low_text_image_page():
    doc = _scanned_document(num_pages=1, ocr_text="")
    # Partial-coverage image but almost no text: low-text branch fires.
    doc.blocks[0].bbox_norm = [100, 100, 800, 600]
    assert is_scanned_visual_page(doc, 0) is True


def test_is_scanned_visual_page_true_for_full_page_image_with_ocr_text():
    # OCR text exceeds the low-text threshold, but one IMAGE block covers
    # (almost) the whole page: coverage branch fires.
    doc = _scanned_document(num_pages=1, ocr_text=NEUTRAL_OCR_TEXT)
    assert len(NEUTRAL_OCR_TEXT) > 200
    assert is_scanned_visual_page(doc, 0) is True


def test_is_scanned_visual_page_false_without_image_blocks():
    doc = DocumentIR(
        doc_id="doc-txt",
        run_id="run-txt",
        source=SourceInfo(path="notes.pdf", ext=".pdf", sha256="abc", size_bytes=10),
        engine=EngineInfo(backend="pipeline", method="txt"),
        pages=[PageInfo(page_idx=0)],
        blocks=[
            Block(
                block_id="t0",
                type=BlockType.TEXT,
                page_idx=0,
                bbox_norm=[10, 10, 990, 100],
                payload={"text": "短文字"},
            )
        ],
    )
    assert is_scanned_visual_page(doc, 0) is False


def test_is_scanned_visual_page_false_for_text_dominated_page_with_small_image():
    doc = _scanned_document(num_pages=1, ocr_text=NEUTRAL_OCR_TEXT)
    # Shrink the image to a small logo: neither branch fires.
    doc.blocks[0].bbox_norm = [100, 100, 300, 300]
    assert is_scanned_visual_page(doc, 0) is False


# ---------------------------------------------------------------------------
# Stage routing: scanned pages go through the form_asset full-page flow
# ---------------------------------------------------------------------------


async def test_scanned_pages_route_through_form_asset_flow(tmp_path, fake_workspace):
    document_ir = _scanned_document(num_pages=2, ocr_text=NEUTRAL_OCR_TEXT)
    _write_source_pdf(document_ir.doc_id, num_pages=2)
    stage = _make_stage(_scanned_enrich_config())

    result = await stage.run(
        doc_id=document_ir.doc_id,
        run_id=document_ir.run_id,
        document_ir=document_ir,
        run_path=tmp_path / "run",
        parse_cache_path=tmp_path / "missing-cache",
        use_cache=False,
    )

    assert result.success is True
    assert result.stats["form_pages_detected"] == 0
    assert result.stats["scanned_pages_detected"] == 2
    assert result.stats["scanned_pages_enriched"] == 2
    assert result.stats["vlm_calls_by_kind"] == {"form_asset": 2}

    assert [e.block_id for e in result.enrichments] == [
        "form_page_0000",
        "form_page_0001",
    ]
    for page_idx, entry in enumerate(result.enrichments):
        assert entry.kind == "form_asset"
        assert entry.prompt_version.startswith("form_asset:test")
        assert entry.input["route"] == "scanned_page"
        assert entry.evidence["page_idx"] == page_idx
        assert entry.evidence["bbox"] is None  # full page
        asset_path = Path(entry.evidence["asset_path"])
        assert asset_path.name == f"form_p{page_idx:04d}.png"
        assert asset_path.exists()  # rendered at full page

    # OCR text is fed to the VLM as grounding context.
    assert len(stage.vlm_adapter.form_calls) == 2
    assert "會議紀錄" in stage.vlm_adapter.form_calls[0]["context_text"]


async def test_scanned_page_budget_caps_enrichment(tmp_path, fake_workspace):
    document_ir = _scanned_document(num_pages=3, doc_id="doc-budget")
    _write_source_pdf(document_ir.doc_id, num_pages=3)
    stage = _make_stage(_scanned_enrich_config(scanned_page_vlm_budget=1))

    result = await stage.run(
        doc_id=document_ir.doc_id,
        run_id=document_ir.run_id,
        document_ir=document_ir,
        run_path=tmp_path / "run",
        parse_cache_path=tmp_path / "missing-cache",
        use_cache=False,
    )

    assert result.success is True
    assert result.stats["scanned_pages_detected"] == 3
    assert result.stats["scanned_pages_enriched"] == 1
    assert len(result.enrichments) == 1
    assert result.enrichments[0].block_id == "form_page_0000"
    assert len(stage.vlm_adapter.form_calls) == 1


async def test_scanned_page_budget_zero_disables_routing(tmp_path, fake_workspace):
    document_ir = _scanned_document(num_pages=2, doc_id="doc-off")
    _write_source_pdf(document_ir.doc_id, num_pages=2)
    stage = _make_stage(_scanned_enrich_config(scanned_page_vlm_budget=0))

    result = await stage.run(
        doc_id=document_ir.doc_id,
        run_id=document_ir.run_id,
        document_ir=document_ir,
        run_path=tmp_path / "run",
        parse_cache_path=tmp_path / "missing-cache",
        use_cache=False,
    )

    assert result.success is True
    assert result.stats["scanned_pages_detected"] == 0
    assert result.stats["scanned_pages_enriched"] == 0
    assert result.enrichments == []
    assert stage.vlm_adapter.form_calls == []


async def test_form_detected_pages_do_not_consume_scanned_budget(
    tmp_path, fake_workspace
):
    form_cue_block = Block(
        block_id="form-text",
        type=BlockType.TEXT,
        page_idx=0,
        bbox_norm=[10, 10, 990, 400],
        payload={"text": "申請人：王小明 電話：02-12345678 簽章：＿＿＿ 請勾選 □是 □否"},
    )
    # Page 0: form-cue text page (no image). Page 1: scanned page.
    document_ir = _scanned_document(num_pages=2, doc_id="doc-mixed")
    document_ir.blocks = [b for b in document_ir.blocks if b.page_idx != 0]
    document_ir.blocks.insert(0, form_cue_block)
    _write_source_pdf(document_ir.doc_id, num_pages=2)

    stage = _make_stage(
        _scanned_enrich_config(vlm_enrich_forms=True, scanned_page_vlm_budget=1)
    )

    result = await stage.run(
        doc_id=document_ir.doc_id,
        run_id=document_ir.run_id,
        document_ir=document_ir,
        run_path=tmp_path / "run",
        parse_cache_path=tmp_path / "missing-cache",
        use_cache=False,
    )

    assert result.success is True
    assert result.stats["form_pages_detected"] == 1
    assert result.stats["scanned_pages_detected"] == 1
    # Budget of 1 is fully available to the scanned page: the form page
    # (route a) must not consume the scanned budget.
    assert result.stats["scanned_pages_enriched"] == 1
    block_ids = {e.block_id for e in result.enrichments}
    assert block_ids == {"form_page_0000", "form_page_0001"}
    assert all(e.kind == "form_asset" for e in result.enrichments)
    assert len(stage.vlm_adapter.form_calls) == 2


async def test_scanned_page_cache_hit_reuses_enrichment(tmp_path, fake_workspace):
    document_ir = _scanned_document(num_pages=1, doc_id="doc-cache")
    _write_source_pdf(document_ir.doc_id, num_pages=1)
    cached_output = {
        "document_type": "reference_table",
        "title": "快取掃描頁",
        "structured_content": "| a | b |",
    }
    stage = _make_stage(
        _scanned_enrich_config(), cached={"form_page_0000": cached_output}
    )

    result = await stage.run(
        doc_id=document_ir.doc_id,
        run_id=document_ir.run_id,
        document_ir=document_ir,
        run_path=tmp_path / "run",
        parse_cache_path=tmp_path / "missing-cache",
        use_cache=True,
    )

    assert result.success is True
    assert result.stats["cache_hits"] == 1
    assert result.stats["scanned_pages_enriched"] == 1
    assert stage.vlm_adapter.form_calls == []  # no fresh VLM call
    assert stage.cache_manager.set_calls == []
    entry = result.enrichments[0]
    assert entry.kind == "form_asset"
    assert entry.input["cached"] is True
    assert entry.input["route"] == "scanned_page"
    assert entry.output["title"] == "快取掃描頁"


async def test_scanned_page_skipped_when_page_already_enriched(
    tmp_path, fake_workspace
):
    # Figure route CAN run (crop image exists): the page gets a figure
    # enrichment, so the scanned route must leave it alone.
    document_ir = _scanned_document(num_pages=1, doc_id="doc-fig")
    _write_source_pdf(document_ir.doc_id, num_pages=1)
    parse_cache = tmp_path / "parse-cache"
    (parse_cache / "images").mkdir(parents=True)
    (parse_cache / "images" / "scan_0.jpg").write_bytes(b"fake-jpg")

    stage = _make_stage(_scanned_enrich_config(vlm_enrich_figures=True))

    result = await stage.run(
        doc_id=document_ir.doc_id,
        run_id=document_ir.run_id,
        document_ir=document_ir,
        run_path=tmp_path / "run",
        parse_cache_path=parse_cache,
        use_cache=False,
    )

    assert result.success is True
    assert result.stats["scanned_pages_detected"] == 1
    assert result.stats["scanned_pages_enriched"] == 0
    kinds = [e.kind for e in result.enrichments]
    assert "form_asset" not in kinds
    assert len(stage.vlm_adapter.figure_calls) == 1
    assert stage.vlm_adapter.form_calls == []


async def test_scanned_page_enriched_when_figure_route_lacks_image(
    tmp_path, fake_workspace
):
    # The live 10-3.pdf failure: the IMAGE block is a figure candidate but no
    # crop exists in the parse cache, so the figure route skips it and the
    # page previously got ZERO enrichment. The scanned route must catch it.
    document_ir = _scanned_document(num_pages=1, doc_id="doc-live")
    _write_source_pdf(document_ir.doc_id, num_pages=1)
    empty_cache = tmp_path / "empty-cache"
    empty_cache.mkdir()

    stage = _make_stage(_scanned_enrich_config(vlm_enrich_figures=True))

    result = await stage.run(
        doc_id=document_ir.doc_id,
        run_id=document_ir.run_id,
        document_ir=document_ir,
        run_path=tmp_path / "run",
        parse_cache_path=empty_cache,
        use_cache=False,
    )

    assert result.success is True
    assert result.stats["scanned_pages_detected"] == 1
    assert result.stats["scanned_pages_enriched"] == 1
    assert [e.kind for e in result.enrichments] == ["form_asset"]
    assert result.enrichments[0].block_id == "form_page_0000"
    assert len(stage.vlm_adapter.form_calls) == 1
    assert stage.vlm_adapter.figure_calls == []
