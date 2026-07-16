"""
Fidelity tests for the Normalize stage.

Covers MinerU content_list block type coverage (page furniture, chart, code,
list), supplement bbox coordinate unification, and dedup priority.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models.document_ir import Block, BlockType
from app.pipeline.stages.normalize import HAS_PYMUPDF, NormalizeStage

# ---------------------------------------------------------------------------
# Block type coverage
# ---------------------------------------------------------------------------

PAGE_FURNITURE_ITEMS = [
    {"type": "header", "text": "國立編譯館 出版品目錄", "page_idx": 0, "bbox": [100, 10, 900, 40]},
    {"type": "footer", "text": "本文件僅供內部使用", "page_idx": 0, "bbox": [100, 960, 900, 990]},
    {"type": "page_number", "text": "12", "page_idx": 0, "bbox": [480, 960, 520, 990]},
    {"type": "page_footnote", "text": "註：資料截至民國112年", "page_idx": 0, "bbox": [100, 900, 900, 950]},
    {"type": "aside_text", "text": "側欄說明文字", "page_idx": 0, "bbox": [950, 100, 990, 500]},
]


@pytest.mark.parametrize("item", PAGE_FURNITURE_ITEMS, ids=lambda i: i["type"])
def test_page_furniture_kept_as_tagged_text(item):
    block = NormalizeStage()._parse_block(dict(item), 0)

    assert block is not None
    assert block.type == BlockType.TEXT
    assert block.payload["text"] == item["text"]
    assert block.payload["origin"] == "page_furniture"


def test_is_page_furniture_helper():
    from app.pipeline.stages.normalize import is_page_furniture

    stage = NormalizeStage()
    furniture = stage._parse_block({"type": "header", "text": "頁首文字", "page_idx": 0}, 0)
    body = stage._parse_block({"type": "text", "text": "內文段落", "page_idx": 0}, 1)

    assert is_page_furniture(furniture) is True
    assert is_page_furniture(body) is False


def test_chart_maps_to_image_with_content_preserved():
    item = {
        "type": "chart",
        "img_path": "images/chart_01.jpg",
        "content": "2021 15% 2022 20% 2023 25%",
        "chart_caption": ["圖一 歷年成長率"],
        "chart_footnote": ["資料來源：主計總處"],
        "page_idx": 2,
        "bbox": [100, 200, 800, 700],
    }

    block = NormalizeStage()._parse_block(item, 3)

    assert block is not None
    assert block.type == BlockType.IMAGE
    assert block.payload["img_path"] == "images/chart_01.jpg"
    assert block.payload["caption"] == ["圖一 歷年成長率"]
    assert block.payload["footnote"] == ["資料來源：主計總處"]
    assert block.payload["chart_content"] == "2021 15% 2022 20% 2023 25%"
    assert block.payload["origin"] == "chart"


def test_code_block_emits_text_with_code_body():
    item = {
        "type": "code",
        "sub_type": "code",
        "code_body": "def main():\n    return 0",
        "code_caption": ["範例程式"],
        "code_language": "python",
        "page_idx": 1,
        "bbox": [100, 100, 500, 300],
    }

    block = NormalizeStage()._parse_block(item, 5)

    assert block is not None
    assert block.type == BlockType.TEXT
    assert block.payload["text"] == "def main():\n    return 0"
    assert block.payload["code_language"] == "python"
    assert block.get_text() == "def main():\n    return 0"


def test_list_block_emits_text_joining_list_items():
    item = {
        "type": "list",
        "sub_type": "text_list",
        "list_items": [
            "一、申請人應檢附身分證明。",
            "二、費用應檢據覈實報支。",
            "三、逾期不予受理。",
        ],
        "page_idx": 1,
        "bbox": [80, 300, 900, 500],
    }

    block = NormalizeStage()._parse_block(item, 7)

    assert block is not None
    assert block.type == BlockType.TEXT
    assert block.payload["text"] == (
        "一、申請人應檢附身分證明。\n二、費用應檢據覈實報支。\n三、逾期不予受理。"
    )
    assert block.payload["origin"] == "list"


def test_table_payload_preserves_img_path():
    item = {
        "type": "table",
        "table_body": "<table><tr><td>項目</td><td>金額</td></tr></table>",
        "table_caption": ["表一"],
        "table_footnote": [],
        "img_path": "images/table_01.jpg",
        "page_idx": 0,
        "bbox": [100, 100, 900, 500],
    }

    block = NormalizeStage()._parse_block(item, 2)

    assert block.type == BlockType.TABLE
    assert block.payload["img_path"] == "images/table_01.jpg"


def test_table_payload_preserves_footnote():
    # MinerU attaches slide reminders printed under a table to table_footnote.
    # Dropping it silently loses authored content (e.g. 「請先與總務預借」).
    item = {
        "type": "table",
        "table_body": "<table><tr><td>項目</td><td>金額</td></tr></table>",
        "table_caption": ["表一"],
        "table_footnote": [
            "提醒：1.使用麥克風系統請在會前先與示範單位預借。",
            "2.合併借用時請提早一天與資訊中心聯絡。",
        ],
        "page_idx": 0,
        "bbox": [100, 100, 900, 500],
    }

    block = NormalizeStage()._parse_block(item, 2)

    assert block.type == BlockType.TABLE
    assert block.payload["table_footnote"] == [
        "提醒：1.使用麥克風系統請在會前先與示範單位預借。",
        "2.合併借用時請提早一天與資訊中心聯絡。",
    ]


# ---------------------------------------------------------------------------
# Dedup priority
# ---------------------------------------------------------------------------

def _image_block(block_id: str, order: int) -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.IMAGE,
        page_idx=0,
        bbox_norm=[100, 100, 500, 500],
        reading_order=order,
        payload={"img_path": "images/region.jpg"},
    )


def _table_block(block_id: str, order: int) -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.TABLE,
        page_idx=0,
        bbox_norm=[102, 101, 498, 502],
        reading_order=order,
        payload={
            "table_body": "<table><tr><td>甲</td><td>乙</td></tr></table>",
            "img_path": "images/region.jpg",
        },
    )


def test_dedup_keeps_table_over_overlapping_image():
    stage = NormalizeStage()

    kept = stage._dedup_overlapping_blocks([_image_block("b000000", 0), _table_block("b000001", 1)])
    assert [b.type for b in kept] == [BlockType.TABLE]
    assert kept[0].payload["table_body"].startswith("<table>")
    assert kept[0].payload["img_path"] == "images/region.jpg"

    # Order independence: table first, image second
    kept = stage._dedup_overlapping_blocks([_table_block("b000000", 0), _image_block("b000001", 1)])
    assert [b.type for b in kept] == [BlockType.TABLE]


# ---------------------------------------------------------------------------
# Supplement bbox coordinate unification
# ---------------------------------------------------------------------------

class _FakePage:
    def __init__(self, width: float, height: float, text_blocks: list[dict]):
        self.rect = SimpleNamespace(width=width, height=height)
        self._text_blocks = text_blocks

    def get_text(self, kind: str) -> dict:
        return {"blocks": self._text_blocks}


class _FakeDoc:
    def __init__(self, pages: list[_FakePage]):
        self._pages = pages

    def __len__(self) -> int:
        return len(self._pages)

    def __getitem__(self, idx: int) -> _FakePage:
        return self._pages[idx]

    def close(self) -> None:
        pass


def _pdf_text_block(text: str, bbox: tuple[float, float, float, float]) -> dict:
    return {
        "type": 0,
        "bbox": bbox,
        "lines": [{"spans": [{"text": text}]}],
    }


def _patch_pdf(monkeypatch, tmp_path: Path, fake_doc: _FakeDoc) -> None:
    from app.pipeline.stages import normalize as normalize_module

    monkeypatch.setattr(
        NormalizeStage, "_find_pdf_path", lambda self, doc_id, cpath: tmp_path / "fake.pdf"
    )
    monkeypatch.setattr(normalize_module, "fitz", SimpleNamespace(open=lambda path: fake_doc))


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
async def test_supplement_bboxes_normalized_to_thousandths(monkeypatch, tmp_path):
    # Page is 500x1000 points; a text block at (50, 100, 250, 500) points
    # must be stored in MinerU's 0-1000 normalized space.
    page = _FakePage(500.0, 1000.0, [_pdf_text_block("這是一段補充文字", (50.0, 100.0, 250.0, 500.0))])
    _patch_pdf(monkeypatch, tmp_path, _FakeDoc([page]))

    stage = NormalizeStage()
    blocks, count = await stage._supplement_missing_text(
        doc_id="doc-test",
        blocks=[],
        content_list_path=tmp_path / "content_list.json",
    )

    assert count == 1
    supplement = [b for b in blocks if b.payload.get("origin") == "pymupdf_supplement"]
    assert len(supplement) == 1
    assert supplement[0].bbox_norm == [100, 100, 500, 500]


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
async def test_supplement_survives_list_captions(monkeypatch, tmp_path):
    # MinerU emits captions as lists; coverage matching must not crash on them.
    page = _FakePage(500.0, 1000.0, [_pdf_text_block("這是一段補充文字", (50.0, 100.0, 250.0, 500.0))])
    _patch_pdf(monkeypatch, tmp_path, _FakeDoc([page]))

    image_block = Block(
        block_id="b000000",
        type=BlockType.IMAGE,
        page_idx=0,
        bbox_norm=[100, 600, 900, 900],
        reading_order=0,
        payload={"img_path": "images/chart.jpg", "caption": ["圖一 成長率"], "origin": "chart"},
    )

    stage = NormalizeStage()
    blocks, count = await stage._supplement_missing_text(
        doc_id="doc-test",
        blocks=[image_block],
        content_list_path=tmp_path / "content_list.json",
    )

    assert count == 1
    assert len(blocks) == 2


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
async def test_supplement_blocks_sort_in_normalized_space(monkeypatch, tmp_path):
    # MinerU block sits mid-page (normalized y0=500). The PDF supplement is
    # physically below it: raw y0=450pt on a 792pt page (normalized 568).
    # Sorting raw points against normalized coords would mis-order them.
    page = _FakePage(612.0, 792.0, [_pdf_text_block("補充在頁面下方的文字", (72.0, 450.0, 540.0, 700.0))])
    _patch_pdf(monkeypatch, tmp_path, _FakeDoc([page]))

    mineru_block = Block(
        block_id="b000000",
        type=BlockType.TEXT,
        page_idx=0,
        bbox_norm=[100, 500, 900, 550],
        reading_order=0,
        payload={"text": "既有段落內容文字", "text_level": 0},
    )

    stage = NormalizeStage()
    blocks, count = await stage._supplement_missing_text(
        doc_id="doc-test",
        blocks=[mineru_block],
        content_list_path=tmp_path / "content_list.json",
    )

    assert count == 1
    assert [b.block_id for b in blocks] == ["b000000", "s000001"]
    assert [b.reading_order for b in blocks] == [0, 1]


def test_text_payload_rejoins_url_broken_by_ocr_linewrap():
    # MinerU wraps long URLs across lines and re-emits them with a space in
    # the middle (live: com.qnap.qfil e&hl=en) — copying that URL 404s.
    item = {
        "type": "text",
        "text": "Android： https://play.example.com/store/apps/details?id=com.demo.qfil e&hl=en",
        "page_idx": 0,
        "bbox": [0, 0, 100, 10],
    }

    block = NormalizeStage()._parse_block(item, 0)

    assert "qfil e" not in block.payload["text"]
    assert "id=com.demo.qfile&hl=en" in block.payload["text"]


def test_text_payload_keeps_prose_after_complete_url():
    item = {
        "type": "text",
        "text": "see https://demo.example.tw docs folder for details 詳見文件",
        "page_idx": 0,
        "bbox": [0, 0, 100, 10],
    }

    block = NormalizeStage()._parse_block(item, 0)

    assert block.payload["text"] == "see https://demo.example.tw docs folder for details 詳見文件"


def test_ocr_noise_not_promoted_to_heading():
    # Live: a phone push-notification screenshot OCR'd into headings —
    # 「## QNAP QTS: CORP\d32@Drive-1」 and the TOTP code 「## 380 671」 —
    # polluting every chunk's heading_path under them.
    stage = NormalizeStage()

    code_block = stage._parse_block({"type": "text", "text": "380 671", "text_level": 1}, 10)
    assert code_block.payload["text_level"] == 0
    assert code_block.payload["text"] == "380 671"

    notif_block = stage._parse_block(
        {"type": "text", "text": "DEMO QTS: CORP\\d32@Drive-1", "text_level": 1}, 10
    )
    assert notif_block.payload["text_level"] == 0
    assert notif_block.payload["text"] == "DEMO QTS: CORP\\d32@Drive-1"

    real_heading = stage._parse_block(
        {"type": "text", "text": "二、檔案上傳方式", "text_level": 2}, 3
    )
    assert real_heading.payload["text_level"] == 2

    numbered_heading = stage._parse_block(
        {"type": "text", "text": "3.僅建立分享連結", "text_level": 2}, 11
    )
    assert numbered_heading.payload["text_level"] == 2
