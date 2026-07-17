"""
Cache-poisoning fix for the enrich stage's VLM cache key.

Verified defect: the enrich cache key is (doc_id, block_id, vlm_config_hash,
prompt_version) -- the DB unique constraint at backend/app/db/database.py:87
-- but block_id is POSITIONAL (b{index:06d} assigned in normalize.py, or
yolo_fig_{page_idx}_{det_idx} assigned in the enrich stage's YOLO-figure
loop). Re-parsing the same document under a different MinerU config (method/
backend/lang/...) shifts segmentation, so the same block_id can end up
denoting a DIFFERENT figure/table across parses -- a cache hit then attaches
the previous parse's caption to the wrong block. Separately, prompts vary
with the resolved semantic_output_language, but before this fix only the
form/scanned-page route folded the language into prompt_version; the
block-level loop and the YOLO-figure loop used the bare prompt version, so
switching semantic_output_language and re-running served stale captions in
the old language.

These tests pin, against the REAL CacheManager + a temp SQLite DB (not a
hand-rolled fake that could itself mask a get/set mismatch):

  1. Unchanged MinerU parse config + unchanged language -> second run hits
     cache (proves get/set key composition is symmetric).
  2. Changed MinerU parse config, same doc_id/block_id/language -> cache
     MISS (the positional-block_id defect).
  3. Changed semantic_output_language, block-level loop (table kind) ->
     cache MISS.
  4/5. Same two scenarios for the YOLO-detected-figure loop.
  6. Non-regression: the form/scanned-page route (which already folded
     language into prompt_version before this fix) still hits/misses
     correctly now that vlm_config_hash is composed with parse_config_hash.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import fitz
import pytest

from app.adapters.vlm import EnrichmentOutput
from app.config import (
    EnrichConfig,
    MinerUBackend,
    MinerUConfig,
    MinerUMethod,
    PackageConfig,
    PipelineConfig,
    SemanticOutputLanguage,
    settings,
)
from app.core.cache import compute_config_hash
from app.db.database import Database
from app.db.repositories import DocRepository
from app.models.document_ir import (
    Block,
    BlockType,
    DocumentIR,
    EngineInfo,
    PageInfo,
    SourceInfo,
)
from app.models.entities import DocCreate
from app.pipeline.stages.enrich import EnrichStage

# ---------------------------------------------------------------------------
# Shared fakes / fixtures
# ---------------------------------------------------------------------------


class FakeVLMAdapter:
    """Deterministic VLM stand-in.

    `counter` is a shared mutable box (e.g. [0]) so two adapter instances
    used across two stage runs can prove whether a VLM call actually
    happened (counter advances, output content changes) or the result was
    served from cache (counter does not advance, output is byte-identical
    to the first run).
    """

    def __init__(self, counter: list[int] | None = None):
        self.counter = counter if counter is not None else [0]
        self.table_calls: list[dict] = []
        self.figure_calls: list[dict] = []
        self.form_calls: list[dict] = []

    async def check_available(self):
        return True, "ok"

    def get_prompt_version(self, kind: str) -> str:
        return f"{kind}-v1"

    async def enrich_table(self, image_path, context_text="", **kwargs):
        self.counter[0] += 1
        self.table_calls.append({"image_path": image_path, **kwargs})
        return EnrichmentOutput(
            success=True,
            kind="table_summary",
            output={"table_summary": f"summary-{self.counter[0]}"},
            tokens_used=1,
            duration_seconds=0.01,
        )

    async def enrich_figure(self, image_path, context_text="", **kwargs):
        self.counter[0] += 1
        self.figure_calls.append({"image_path": image_path, **kwargs})
        return EnrichmentOutput(
            success=True,
            kind="figure_caption",
            output={"caption": f"caption-{self.counter[0]}"},
            tokens_used=1,
            duration_seconds=0.01,
        )

    async def enrich_form(self, image_path, context_text="", **kwargs):
        self.counter[0] += 1
        self.form_calls.append({"image_path": image_path, **kwargs})
        return EnrichmentOutput(
            success=True,
            kind="form_asset",
            output={"title": f"title-{self.counter[0]}"},
            tokens_used=1,
            duration_seconds=0.01,
        )


# Two MinerU parse configs that differ only in `method` -- exactly the kind
# of re-parse setting change called out in the defect (method/backend/lang).
MINERU_A = MinerUConfig(method=MinerUMethod.AUTO, backend=MinerUBackend.PIPELINE, lang="chinese_cht")
MINERU_B = MinerUConfig(method=MinerUMethod.OCR, backend=MinerUBackend.PIPELINE, lang="chinese_cht")

TABLE_ENRICH_CONFIG = EnrichConfig(
    enable_vlm=True,
    vlm_enrich_forms=False,
    vlm_enrich_figures=False,
    vlm_enrich_tables=True,
)

FIGURE_ENRICH_CONFIG = EnrichConfig(
    enable_vlm=True,
    vlm_enrich_forms=False,
    vlm_enrich_figures=True,
    vlm_enrich_tables=False,
)

FORM_ENRICH_CONFIG = EnrichConfig(
    enable_vlm=True,
    vlm_enrich_forms=False,
    vlm_enrich_figures=False,
    vlm_enrich_tables=False,
    scanned_page_vlm_budget=1,
)


@asynccontextmanager
async def _open_db(tmp_path: Path):
    db = Database(db_path=tmp_path / "cache.db")
    await db.connect()
    try:
        yield db
    finally:
        await db.disconnect()


async def _seed_doc(db: Database, doc_id: str) -> None:
    """enrich_entries.doc_id has FOREIGN KEY REFERENCES docs(doc_id) (see
    database.py schema) and foreign_keys=ON is enabled on connect; a docs
    row must exist before set_enrich_cache can insert."""
    await DocRepository(db).create(
        doc_id,
        DocCreate(source_path=f"{doc_id}.pdf", sha256="abc", ext=".pdf", size_bytes=100),
    )


def _make_stage(
    db: Database,
    *,
    mineru: MinerUConfig,
    language: SemanticOutputLanguage,
    enrich: EnrichConfig,
    counter: list[int] | None = None,
) -> tuple[EnrichStage, FakeVLMAdapter]:
    config = PipelineConfig(
        mineru=mineru,
        enrich=enrich,
        package=PackageConfig(semantic_output_language=language),
    )
    stage = EnrichStage(db=db, config=config)
    fake_vlm = FakeVLMAdapter(counter=counter)
    stage.vlm_adapter = fake_vlm
    return stage, fake_vlm


@pytest.fixture
def fake_workspace(tmp_path, monkeypatch):
    """Point global settings workspace at tmp so source-PDF lookups work."""
    workspace = tmp_path / "ws"
    monkeypatch.setattr(settings, "workspace_path", workspace)
    return workspace


def _write_source_pdf(doc_id: str) -> Path:
    source_dir = settings.get_doc_path(doc_id) / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open()
    pdf.new_page(width=595, height=842)
    pdf_path = source_dir / "original.pdf"
    pdf.save(pdf_path)
    pdf.close()
    return pdf_path


def _write_yolo_model_json(parse_cache_path: Path) -> None:
    """One page with one ImageBody (category_id=3) YOLO detection."""
    parse_cache_path.mkdir(parents=True, exist_ok=True)
    detections = [
        {
            "layout_dets": [
                {
                    "category_id": 3,  # MinerUCategoryId.ImageBody
                    "score": 0.95,
                    "poly": [50, 50, 300, 50, 300, 300, 50, 300],
                }
            ]
        }
    ]
    (parse_cache_path / "doc_model.json").write_text(
        json.dumps(detections), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Fixtures: documents
# ---------------------------------------------------------------------------


def _table_document(doc_id: str) -> DocumentIR:
    """Single TABLE block with a fixed, POSITIONAL block_id (b000001),
    mirroring normalize.py's b{index:06d} assignment. No crop image is
    needed: table_summary can run text-only."""
    table_body = "\n".join(
        [
            "| 地區 | 日支數額 |",
            "| --- | --- |",
            "| 亞洲 | 120 |",
            "| 歐洲 | 160 |",
        ]
    )
    return DocumentIR(
        doc_id=doc_id,
        run_id="run-cache-test",
        source=SourceInfo(path="差旅費標準表.pdf", ext=".pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TABLE,
                page_idx=0,
                bbox_norm=[10, 10, 900, 400],
                payload={"table_body": table_body},
            )
        ],
    )


def _figure_document(doc_id: str) -> DocumentIR:
    """No IR blocks: the YOLO-figure route is driven entirely by
    parse_cache_path's *_model.json, independent of document_ir.blocks."""
    return DocumentIR(
        doc_id=doc_id,
        run_id="run-cache-test",
        source=SourceInfo(path="figures.pdf", ext=".pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[],
    )


def _scanned_page_document(doc_id: str) -> DocumentIR:
    """One full-page IMAGE block: routes through the form_asset flow via
    the scanned-visual-page detector (see is_scanned_visual_page)."""
    return DocumentIR(
        doc_id=doc_id,
        run_id="run-cache-test",
        source=SourceInfo(path="scan.pdf", ext=".pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        pages=[PageInfo(page_idx=0)],
        blocks=[
            Block(
                block_id="img-0",
                type=BlockType.IMAGE,
                page_idx=0,
                bbox_norm=[0, 0, 1000, 1000],
                payload={"img_path": "images/scan_0.jpg"},
            )
        ],
    )


# ---------------------------------------------------------------------------
# Block-level loop (table_summary kind)
# ---------------------------------------------------------------------------


async def test_block_loop_cache_hit_when_parse_config_and_language_unchanged(tmp_path):
    """Second run with an unchanged MinerU config and language must reuse
    the cached enrichment: get and set must compose the identical key."""
    async with _open_db(tmp_path) as db:
        doc = _table_document("doc-hit")
        await _seed_doc(db, doc.doc_id)

        stage1, vlm1 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW, enrich=TABLE_ENRICH_CONFIG
        )
        result1 = await stage1.run(
            doc_id=doc.doc_id,
            run_id="run-1",
            document_ir=doc,
            run_path=tmp_path / "run1",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
        )
        assert result1.success is True
        assert result1.stats["cache_hits"] == 0
        assert result1.stats["enriched"] == 1
        assert len(vlm1.table_calls) == 1

        stage2, vlm2 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW, enrich=TABLE_ENRICH_CONFIG
        )
        result2 = await stage2.run(
            doc_id=doc.doc_id,
            run_id="run-2",
            document_ir=doc,
            run_path=tmp_path / "run2",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
        )
        assert result2.stats["cache_hits"] == 1
        assert result2.stats["enriched"] == 0
        assert vlm2.table_calls == []  # no fresh VLM call: served from cache
        assert result2.enrichments[0].output == result1.enrichments[0].output
        assert result2.enrichments[0].prompt_version == "table_summary-v1:zh-TW"


async def test_block_loop_cache_miss_when_mineru_parse_config_changes(tmp_path):
    """A different MinerU parse config (segmentation-affecting) must NOT
    reuse a cached caption for the same positional block_id, even though
    doc_id/block_id/language are unchanged. This is the core cache-
    poisoning scenario: re-parsing under different MinerU settings shifts
    segmentation, so a stale hit would attach the wrong caption."""
    assert compute_config_hash(MINERU_A) != compute_config_hash(MINERU_B)  # sanity

    async with _open_db(tmp_path) as db:
        doc = _table_document("doc-parse-miss")
        await _seed_doc(db, doc.doc_id)
        counter = [0]

        stage1, vlm1 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW,
            enrich=TABLE_ENRICH_CONFIG, counter=counter,
        )
        result1 = await stage1.run(
            doc_id=doc.doc_id,
            run_id="run-1",
            document_ir=doc,
            run_path=tmp_path / "run1",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
        )
        assert result1.enrichments[0].output == {"table_summary": "summary-1"}

        stage2, vlm2 = _make_stage(
            db, mineru=MINERU_B, language=SemanticOutputLanguage.ZH_TW,
            enrich=TABLE_ENRICH_CONFIG, counter=counter,
        )
        result2 = await stage2.run(
            doc_id=doc.doc_id,
            run_id="run-2",
            document_ir=doc,
            run_path=tmp_path / "run2",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
        )
        assert result2.stats["cache_hits"] == 0
        assert result2.stats["enriched"] == 1
        assert len(vlm2.table_calls) == 1  # fresh call, not served from stale cache
        assert result2.enrichments[0].output == {"table_summary": "summary-2"}
        assert result2.enrichments[0].output != result1.enrichments[0].output


async def test_block_loop_cache_miss_when_semantic_output_language_changes(tmp_path):
    """Switching semantic_output_language must bust the block-loop
    (table_summary) cache instead of serving a caption in the old
    language."""
    async with _open_db(tmp_path) as db:
        doc = _table_document("doc-lang-miss")
        await _seed_doc(db, doc.doc_id)
        counter = [0]

        stage1, vlm1 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW,
            enrich=TABLE_ENRICH_CONFIG, counter=counter,
        )
        result1 = await stage1.run(
            doc_id=doc.doc_id,
            run_id="run-1",
            document_ir=doc,
            run_path=tmp_path / "run1",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
        )
        assert result1.enrichments[0].prompt_version == "table_summary-v1:zh-TW"

        stage2, vlm2 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.EN,
            enrich=TABLE_ENRICH_CONFIG, counter=counter,
        )
        result2 = await stage2.run(
            doc_id=doc.doc_id,
            run_id="run-2",
            document_ir=doc,
            run_path=tmp_path / "run2",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
        )
        assert result2.stats["cache_hits"] == 0
        assert len(vlm2.table_calls) == 1
        assert result2.enrichments[0].prompt_version == "table_summary-v1:en"
        assert result2.enrichments[0].output != result1.enrichments[0].output


# ---------------------------------------------------------------------------
# YOLO-detected-figure loop (yolo_fig_{page}_{det} positional block_id)
# ---------------------------------------------------------------------------


async def test_yolo_figure_cache_hit_when_unchanged(tmp_path, fake_workspace):
    doc_id = "doc-yolo-hit"
    _write_source_pdf(doc_id)
    parse_cache_path = tmp_path / "parse-cache"
    _write_yolo_model_json(parse_cache_path)

    async with _open_db(tmp_path) as db:
        doc = _figure_document(doc_id)
        await _seed_doc(db, doc_id)

        stage1, vlm1 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW, enrich=FIGURE_ENRICH_CONFIG
        )
        result1 = await stage1.run(
            doc_id=doc_id,
            run_id="run-1",
            document_ir=doc,
            run_path=tmp_path / "run1",
            parse_cache_path=parse_cache_path,
            use_cache=True,
        )
        assert result1.stats["yolo_figures_detected"] == 1
        assert len(vlm1.figure_calls) == 1
        assert result1.enrichments[0].block_id == "yolo_fig_0000_000"

        stage2, vlm2 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW, enrich=FIGURE_ENRICH_CONFIG
        )
        result2 = await stage2.run(
            doc_id=doc_id,
            run_id="run-2",
            document_ir=doc,
            run_path=tmp_path / "run2",
            parse_cache_path=parse_cache_path,
            use_cache=True,
        )
        assert result2.stats["cache_hits"] == 1
        assert vlm2.figure_calls == []
        assert result2.enrichments[0].output == result1.enrichments[0].output


async def test_yolo_figure_cache_miss_when_semantic_output_language_changes(
    tmp_path, fake_workspace
):
    """Mirrors the block-loop language test for the YOLO-detected-figure
    route: switching language must not serve a caption cached under the
    old language for the same positional yolo_fig_{page}_{det} id."""
    doc_id = "doc-yolo-lang-miss"
    _write_source_pdf(doc_id)
    parse_cache_path = tmp_path / "parse-cache"
    _write_yolo_model_json(parse_cache_path)

    async with _open_db(tmp_path) as db:
        doc = _figure_document(doc_id)
        await _seed_doc(db, doc_id)
        counter = [0]

        stage1, vlm1 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW,
            enrich=FIGURE_ENRICH_CONFIG, counter=counter,
        )
        result1 = await stage1.run(
            doc_id=doc_id,
            run_id="run-1",
            document_ir=doc,
            run_path=tmp_path / "run1",
            parse_cache_path=parse_cache_path,
            use_cache=True,
        )
        assert result1.enrichments[0].prompt_version == "figure_description-v1:zh-TW"

        stage2, vlm2 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.EN,
            enrich=FIGURE_ENRICH_CONFIG, counter=counter,
        )
        result2 = await stage2.run(
            doc_id=doc_id,
            run_id="run-2",
            document_ir=doc,
            run_path=tmp_path / "run2",
            parse_cache_path=parse_cache_path,
            use_cache=True,
        )
        assert result2.stats["cache_hits"] == 0
        assert len(vlm2.figure_calls) == 1  # fresh call; not served stale zh-TW caption
        assert result2.enrichments[0].prompt_version == "figure_description-v1:en"
        assert result2.enrichments[0].output != result1.enrichments[0].output


# ---------------------------------------------------------------------------
# Form / scanned-page route: non-regression (already language-aware pre-fix)
# ---------------------------------------------------------------------------


async def test_form_route_cache_hit_when_unchanged_and_miss_on_language_change(
    tmp_path, fake_workspace
):
    """The form/scanned-page route already composed language into
    prompt_version before this fix (it must keep behaving the same way).
    This guards against a partial refactor breaking its get/set symmetry
    while vlm_config_hash becomes a composed value shared by all three
    enrichment routes."""
    doc_id = "doc-form-route"
    _write_source_pdf(doc_id)

    async with _open_db(tmp_path) as db:
        doc = _scanned_page_document(doc_id)
        await _seed_doc(db, doc_id)
        counter = [0]

        stage1, vlm1 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW,
            enrich=FORM_ENRICH_CONFIG, counter=counter,
        )
        result1 = await stage1.run(
            doc_id=doc_id,
            run_id="run-1",
            document_ir=doc,
            run_path=tmp_path / "run1",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
        )
        assert len(vlm1.form_calls) == 1
        assert result1.enrichments[0].prompt_version == "form_asset-v1:zh-TW"
        assert result1.enrichments[0].output == {"title": "title-1"}

        # Unchanged config/language -> hit.
        stage2, vlm2 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW,
            enrich=FORM_ENRICH_CONFIG, counter=counter,
        )
        result2 = await stage2.run(
            doc_id=doc_id,
            run_id="run-2",
            document_ir=doc,
            run_path=tmp_path / "run2",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
        )
        assert result2.stats["cache_hits"] == 1
        assert vlm2.form_calls == []
        assert result2.enrichments[0].output == {"title": "title-1"}

        # Changed language -> miss.
        stage3, vlm3 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.EN,
            enrich=FORM_ENRICH_CONFIG, counter=counter,
        )
        result3 = await stage3.run(
            doc_id=doc_id,
            run_id="run-3",
            document_ir=doc,
            run_path=tmp_path / "run3",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
        )
        assert result3.stats["cache_hits"] == 0
        assert len(vlm3.form_calls) == 1
        assert result3.enrichments[0].prompt_version == "form_asset-v1:en"
        assert result3.enrichments[0].output == {"title": "title-2"}


async def test_block_loop_cache_miss_when_mineru_version_changes(tmp_path):
    """Upgrading the MinerU package re-parses with new segmentation even when
    the MinerU config is unchanged; positional block ids can then denote
    different content, so captions cached under the previous version must not
    be served."""
    async with _open_db(tmp_path) as db:
        doc = _table_document("doc-version-miss")
        await _seed_doc(db, doc.doc_id)
        counter = [0]

        stage1, _vlm1 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW,
            enrich=TABLE_ENRICH_CONFIG, counter=counter,
        )
        result1 = await stage1.run(
            doc_id=doc.doc_id,
            run_id="run-1",
            document_ir=doc,
            run_path=tmp_path / "run1",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
            mineru_version="2.1.0",
        )
        assert result1.stats["enriched"] == 1

        stage2, vlm2 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW,
            enrich=TABLE_ENRICH_CONFIG, counter=counter,
        )
        result2 = await stage2.run(
            doc_id=doc.doc_id,
            run_id="run-2",
            document_ir=doc,
            run_path=tmp_path / "run2",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
            mineru_version="2.5.0",
        )
        assert result2.stats["cache_hits"] == 0
        assert len(vlm2.table_calls) == 1  # fresh call, not stale-version cache


async def test_block_loop_cache_hit_when_mineru_version_unchanged(tmp_path):
    async with _open_db(tmp_path) as db:
        doc = _table_document("doc-version-hit")
        await _seed_doc(db, doc.doc_id)
        counter = [0]

        stage1, _vlm1 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW,
            enrich=TABLE_ENRICH_CONFIG, counter=counter,
        )
        await stage1.run(
            doc_id=doc.doc_id,
            run_id="run-1",
            document_ir=doc,
            run_path=tmp_path / "run1",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
            mineru_version="2.1.0",
        )

        stage2, vlm2 = _make_stage(
            db, mineru=MINERU_A, language=SemanticOutputLanguage.ZH_TW,
            enrich=TABLE_ENRICH_CONFIG, counter=counter,
        )
        result2 = await stage2.run(
            doc_id=doc.doc_id,
            run_id="run-2",
            document_ir=doc,
            run_path=tmp_path / "run2",
            parse_cache_path=tmp_path / "missing-cache",
            use_cache=True,
            mineru_version="2.1.0",
        )
        assert result2.stats["cache_hits"] == 1
        assert vlm2.table_calls == []


async def test_orchestrator_passes_parse_mineru_version_to_enrich(tmp_path, monkeypatch):
    """The orchestrator must thread the parse stage's resolved MinerU version
    (recorded in ctx.stage_stats) into the enrich stage so the cache key can
    include it; absent parse stats (resumed runs) fall back to None."""
    from types import SimpleNamespace

    from app.config import PipelineConfig
    from app.core import orchestrator as orch_mod

    captured: dict = {}

    class _FakeEnrichStage:
        def __init__(self, db, config):
            del db, config

        async def run(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(success=True, stats={"enriched": 0}, error=None)

    monkeypatch.setattr(orch_mod, "EnrichStage", _FakeEnrichStage)
    orch = orch_mod.PipelineOrchestrator(db=object())
    ctx = orch_mod.PipelineContext(
        doc_id="d",
        run_id="r",
        config=PipelineConfig(),
        run_path=tmp_path,
        document_ir=_table_document("d"),
    )
    ctx.stage_stats["parse"] = {"cache_hit": True, "mineru_version": "9.9.9"}

    await orch._run_enrich(ctx, use_cache=True)

    assert captured["mineru_version"] == "9.9.9"
