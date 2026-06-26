from pathlib import Path

import pytest

from app.adapters.vlm import EnrichmentOutput
from app.config import EnrichConfig, PipelineConfig
from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.stages.enrich import EnrichStage


class FakeCacheManager:
    async def set_enrich_cache(self, **kwargs):
        return None


class FakeVLMAdapter:
    def __init__(self):
        self.table_calls = []

    async def check_available(self):
        return True, "ok"

    def get_prompt_version(self, kind: str) -> str:
        return f"{kind}:test"

    async def enrich_table(self, image_path, context_text="", **kwargs):
        self.table_calls.append(
            {
                "image_path": image_path,
                "context_text": context_text,
                **kwargs,
            }
        )
        return EnrichmentOutput(
            success=True,
            kind="table_summary",
            output={
                "table_summary": "差旅費生活費日支數額標準表。",
                "key_columns": ["地區", "日支數額"],
                "key_rows": ["亞洲", "歐洲"],
            },
            tokens_used=12,
            duration_seconds=0.1,
        )


def _document_with_table() -> DocumentIR:
    table_body = "\n".join(
        [
            "| 地區 | 日支數額 |",
            "| --- | --- |",
            "| 亞洲 | 120 |",
            "| 歐洲 | 160 |",
        ]
    )
    return DocumentIR(
        doc_id="doc-test",
        run_id="run-test",
        source=SourceInfo(path="差旅費標準表.pdf", ext=".pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[
            Block(
                block_id="table-1",
                type=BlockType.TABLE,
                page_idx=0,
                bbox_norm=[10, 10, 900, 400],
                payload={"table_body": table_body},
            )
        ],
    )


@pytest.mark.asyncio
async def test_table_without_crop_routes_to_text_vlm(tmp_path: Path):
    config = PipelineConfig(
        enrich=EnrichConfig(
            enable_vlm=True,
            vlm_enrich_forms=False,
            vlm_enrich_figures=False,
            vlm_enrich_tables=True,
        )
    )
    stage = EnrichStage(db=None, config=config)
    fake_vlm = FakeVLMAdapter()
    stage.vlm_adapter = fake_vlm
    stage.cache_manager = FakeCacheManager()

    result = await stage.run(
        doc_id="doc-test",
        run_id="run-test",
        document_ir=_document_with_table(),
        run_path=tmp_path,
        parse_cache_path=tmp_path / "missing-cache",
        use_cache=False,
    )

    assert result.success is True
    assert result.stats["gated_blocks"] == 1
    assert result.stats["enriched"] == 1
    assert result.stats["vlm_calls_by_kind"] == {"table_summary": 1}
    assert fake_vlm.table_calls[0]["image_path"] is None
    assert "亞洲" in fake_vlm.table_calls[0]["table_body"]
    assert result.enrichments[0].input["route"] == "vlm_text_from_mineru_table"
    assert result.enrichments[0].evidence["asset_path"] is None
