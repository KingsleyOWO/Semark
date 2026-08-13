import json
import zipfile

from app.api.routes.converters import md_to_txt
from app.api.routes.download import (
    FileType,
    OutputFormat,
    _get_document_files,
    _get_file_content,
)


def test_document_zip_paths_keep_run_id_out_of_file_names(tmp_path):
    outputs = tmp_path / "outputs"
    documents_dir = outputs / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "main.md").write_text("# Main", encoding="utf-8")
    (documents_dir / "form_0000.md").write_text("# Form", encoding="utf-8")
    (outputs / "documents_index.json").write_text(
        json.dumps(
            [
                {
                    "document_id": "main",
                    "kind": "main",
                    "title": "主文",
                    "file": str(documents_dir / "main.md"),
                },
                {
                    "document_id": "form_0000",
                    "kind": "form",
                    "title": "表單",
                    "file": str(documents_dir / "form_0000.md"),
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    files = _get_document_files(
        outputs,
        source_name="人員管理辦法",
        format=OutputFormat.MD,
        archive_folder_name="人員管理辦法_01KT0FRA7KE1",
    )

    assert [name for name, _content in files] == [
        "人員管理辦法_01KT0FRA7KE1_documents/人員管理辦法_main.md",
        "人員管理辦法_01KT0FRA7KE1_documents/人員管理辦法_form01.md",
    ]


def test_document_zip_paths_respect_selected_document_ids(tmp_path):
    outputs = tmp_path / "outputs"
    documents_dir = outputs / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "main.md").write_text("# Main", encoding="utf-8")
    (documents_dir / "form_0000.md").write_text("# Form", encoding="utf-8")
    (outputs / "documents_index.json").write_text(
        json.dumps(
            [
                {"document_id": "main", "kind": "main", "file": str(documents_dir / "main.md")},
                {
                    "document_id": "form_0000",
                    "kind": "form",
                    "file": str(documents_dir / "form_0000.md"),
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    files = _get_document_files(
        outputs,
        source_name="人員管理辦法",
        format=OutputFormat.TXT,
        document_ids=["form_0000"],
    )

    assert len(files) == 1
    assert files[0][0] == "人員管理辦法_documents/人員管理辦法_form01.txt"
    assert files[0][1].decode("utf-8").strip() == "Form"

def _write_viewer_main_document(outputs):
    """Create the split main document exactly as the viewer serves it."""
    documents_dir = outputs / "documents"
    documents_dir.mkdir(parents=True)
    viewer_main_md = (
        "# 使用說明\n\n"
        "## 一、登入系統\n\n"
        "(1)開啟瀏覽器後進入員工專區。\n\n"
        "圖片顯示登入畫面的介面。\n"
    )
    (documents_dir / "main.md").write_text(viewer_main_md, encoding="utf-8")
    (outputs / "documents_index.json").write_text(
        json.dumps(
            [
                {
                    "document_id": "main",
                    "kind": "main",
                    "title": "使用說明",
                    "file": str(documents_dir / "main.md"),
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return viewer_main_md


def test_main_text_download_serves_viewer_main_document(tmp_path):
    outputs = tmp_path / "outputs"
    viewer_main_md = _write_viewer_main_document(outputs)
    # Legacy divergent render must be ignored when the viewer document exists.
    (outputs / "main_text.md").write_text("# 使用說明\n\n舊的純文字版。\n", encoding="utf-8")

    result = _get_file_content(
        outputs,
        FileType.MAIN_TEXT,
        OutputFormat.MD,
        run_id="01TESTRUN",
        source_name="使用說明",
    )

    assert result is not None
    content, filename = result
    assert content.decode("utf-8") == viewer_main_md
    assert filename == "使用說明_main.md"


def test_main_text_download_converts_viewer_main_document(tmp_path):
    outputs = tmp_path / "outputs"
    viewer_main_md = _write_viewer_main_document(outputs)

    result = _get_file_content(
        outputs,
        FileType.MAIN_TEXT,
        OutputFormat.TXT,
        run_id="01TESTRUN",
        source_name="使用說明",
    )

    assert result is not None
    content, filename = result
    assert content.decode("utf-8") == md_to_txt(viewer_main_md)
    assert filename == "使用說明_main.txt"


def test_main_text_download_falls_back_for_legacy_runs(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "main_text.md").write_text("# 使用說明\n\n舊版主文。\n", encoding="utf-8")

    result = _get_file_content(
        outputs,
        FileType.MAIN_TEXT,
        OutputFormat.MD,
        run_id="01TESTRUN",
        source_name="使用說明",
    )

    assert result is not None
    content, filename = result
    assert content.decode("utf-8") == "# 使用說明\n\n舊版主文。\n"
    assert filename == "使用說明_main_text.md"


def test_batch_zip_names_do_not_collide_across_runs_of_same_source():
    from app.api.routes.download import _dedupe_zip_entry_name

    used: set[str] = set()
    first = _dedupe_zip_entry_name("使用說明_main.md", "01KXMSEA64P5WFHRVPG1Z7F1J4", used)
    second = _dedupe_zip_entry_name("使用說明_main.md", "01KXMXHAJWM176238VVFG5TYWM", used)
    third = _dedupe_zip_entry_name("使用說明_main.md", "01KXMXHAJWM176238VVFG5TYWM", used)

    assert first == "使用說明_main.md"
    assert second == "使用說明_main_01KXMXHAJWM1.md"
    assert third == "使用說明_main_01KXMXHAJWM1_2.md"
    assert len({first, second, third}) == 3


def test_source_name_strips_edge_whitespace():
    from app.api.routes.download import _source_name_for

    assert _source_name_for("/data/uploads/201電子白板 .pdf", "01RUNABCDEFGHIJK") == "201電子白板"
    assert _source_name_for(None, "01RUNABCDEFGHIJK") == "01RUNABCDEFG"


def test_dedupe_by_doc_keeps_newest_run_per_document():
    from app.api.routes.download import _dedupe_runs_by_doc

    class Run:
        def __init__(self, run_id, doc_id):
            self.run_id = run_id
            self.doc_id = doc_id

    old = Run("01KXG051R8EP000000000000AA", "doc-a")
    new = Run("01KXN1ZYYV00000000000000BB", "doc-a")
    other = Run("01KXG051TX00000000000000CC", "doc-b")

    kept = _dedupe_runs_by_doc(
        [(old, "204電子白板"), (new, "204電子白板"), (other, "606電子白板")]
    )

    assert [(run.run_id, name) for run, name in kept] == [
        ("01KXN1ZYYV00000000000000BB", "204電子白板"),
        ("01KXG051TX00000000000000CC", "606電子白板"),
    ]


def _seed_run_outputs(workspace, doc_id, run_id, titles):
    """Write a run's split documents exactly where _build_download_zip reads them."""
    documents_dir = (
        workspace / "store" / "docs" / doc_id / "runs" / run_id / "outputs" / "documents"
    )
    documents_dir.mkdir(parents=True)
    index = []
    for document_id, kind, title in titles:
        filename = f"{document_id}.md"
        (documents_dir / filename).write_text(f"# {title}", encoding="utf-8")
        index.append(
            {
                "document_id": document_id,
                "kind": kind,
                "title": title,
                "file": str(documents_dir / filename),
            }
        )
    (documents_dir.parent / "documents_index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )


def test_flat_document_zip_puts_every_markdown_at_the_archive_root(tmp_path):
    """
    A 300-document export otherwise extracts to 300 folders, each holding the
    markdown that the user actually wants. The file names already carry the
    source name, so the folder is optional structure — not identity.
    """

    outputs = tmp_path / "outputs"
    documents_dir = outputs / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "main.md").write_text("# Main", encoding="utf-8")
    (documents_dir / "table_0000.md").write_text("# Table", encoding="utf-8")
    (outputs / "documents_index.json").write_text(
        json.dumps(
            [
                {"document_id": "main", "kind": "main", "file": str(documents_dir / "main.md")},
                {
                    "document_id": "table_0000",
                    "kind": "table",
                    "file": str(documents_dir / "table_0000.md"),
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    files = _get_document_files(
        outputs,
        source_name="人員管理辦法",
        format=OutputFormat.MD,
        archive_folder_name="人員管理辦法_01KT0FRA7KE1",
        flatten=True,
    )

    names = [name for name, _content in files]
    assert names == ["人員管理辦法_main.md", "人員管理辦法_table01.md"]
    assert all("/" not in name for name in names)


def test_flat_zip_keeps_both_copies_when_two_sources_share_a_name(tmp_path, monkeypatch):
    """
    Dropping the per-source folder removes what used to keep same-named sources
    apart, so the flat path has to dedupe or the second export silently
    overwrites the first on extraction.

    Both runs carry the same title as well as the same source name — a differing
    title would make the names unique on its own and leave the dedupe untested.
    Distinct bodies are what prove neither file was lost.
    """

    from app.api.routes.download import DownloadRequest, _build_download_zip
    from app.config import settings

    monkeypatch.setattr(settings, "workspace_path", tmp_path)
    _seed_run_outputs(
        tmp_path, "doc-a", "01RUNAAAAAAAAAAAAAAAAAAAAA", [("main", "main", "年度營運回顧")]
    )
    _seed_run_outputs(
        tmp_path, "doc-b", "01RUNBBBBBBBBBBBBBBBBBBBBB", [("main", "main", "年度營運回顧")]
    )
    # Make the two files distinguishable by content despite the identical title.
    for doc_id, run_id, body in (
        ("doc-a", "01RUNAAAAAAAAAAAAAAAAAAAAA", "# 第一份"),
        ("doc-b", "01RUNBBBBBBBBBBBBBBBBBBBBB", "# 第二份"),
    ):
        (
            tmp_path
            / "store"
            / "docs"
            / doc_id
            / "runs"
            / run_id
            / "outputs"
            / "documents"
            / "main.md"
        ).write_text(body, encoding="utf-8")

    class Run:
        def __init__(self, run_id, doc_id):
            self.run_id = run_id
            self.doc_id = doc_id

    runs_with_docs = [
        (Run("01RUNAAAAAAAAAAAAAAAAAAAAA", "doc-a"), "年報"),
        (Run("01RUNBBBBBBBBBBBBBBBBBBBBB", "doc-b"), "年報"),
    ]
    request = DownloadRequest(
        run_ids=[run.run_id for run, _name in runs_with_docs],
        file_types=[FileType.DOCUMENTS],
        format=OutputFormat.MD,
        flatten=True,
    )

    buffer = _build_download_zip(runs_with_docs, request)
    with zipfile.ZipFile(buffer) as zf:
        names = zf.namelist()
        bodies = {name: zf.read(name).decode("utf-8") for name in names}

    assert len(names) == 2, names
    assert all("/" not in name for name in names)
    # Same source name, same title — the second entry must have been renamed.
    assert names[0] == "年報_main_年度營運回顧.md"
    assert names[1] == "年報_main_年度營運回顧_01RUNBBBBBBB.md"
    assert sorted(bodies.values()) == ["# 第一份", "# 第二份"]


def test_folder_zip_remains_the_default(tmp_path, monkeypatch):
    """Existing callers keep the per-source folder they package against today."""

    from app.api.routes.download import DownloadRequest, _build_download_zip
    from app.config import settings

    monkeypatch.setattr(settings, "workspace_path", tmp_path)
    _seed_run_outputs(tmp_path, "doc-a", "01RUNAAAAAAAAAAAAAAAAAAAAA", [("main", "main", "第一份")])
    _seed_run_outputs(tmp_path, "doc-b", "01RUNBBBBBBBBBBBBBBBBBBBBB", [("main", "main", "第二份")])

    class Run:
        def __init__(self, run_id, doc_id):
            self.run_id = run_id
            self.doc_id = doc_id

    runs_with_docs = [
        (Run("01RUNAAAAAAAAAAAAAAAAAAAAA", "doc-a"), "年報"),
        (Run("01RUNBBBBBBBBBBBBBBBBBBBBB", "doc-b"), "年報"),
    ]
    request = DownloadRequest(
        run_ids=[run.run_id for run, _name in runs_with_docs],
        file_types=[FileType.DOCUMENTS],
        format=OutputFormat.MD,
    )

    buffer = _build_download_zip(runs_with_docs, request)
    with zipfile.ZipFile(buffer) as zf:
        names = zf.namelist()

    assert names == [
        "年報_01RUNAAAAAAA_documents/年報_main_第一份.md",
        "年報_01RUNBBBBBBB_documents/年報_main_第二份.md",
    ]


def test_main_download_name_carries_the_article_title():
    """
    Every main title in the 2026-08 corpus was a real article title, while the
    source name was a numeric upload stem — so 141 exports all read
    ``1202606095826_main.md`` and had to be opened to be told apart.
    """

    from app.api.routes.download import _document_download_base_name

    base = _document_download_base_name(
        source_name="1202606095826",
        entry={
            "document_id": "main",
            "kind": "main",
            "title": "共存與對抗：示範島在示範峰會的互動關係",
        },
        fallback_stem="main",
        counters={},
    )

    assert base == "1202606095826_main_共存與對抗：示範島在示範峰會的互動關係"


def test_asset_download_name_carries_a_real_caption():
    from app.api.routes.download import _document_download_base_name

    counters: dict[str, int] = {}
    base = _document_download_base_name(
        source_name="1202606095826",
        entry={
            "document_id": "tbl0000",
            "kind": "table_asset",
            "title": "成衣類代表產品全球市占率前五名國家變化",
        },
        fallback_stem="tbl0000",
        counters=counters,
    )

    # The kind prefix stays in front so files still sort and parse by kind.
    assert base == "1202606095826_table01_成衣類代表產品全球市占率前五名國家變化"


def test_asset_download_name_drops_vlm_descriptions_and_placeholders():
    """
    Half the asset titles are the VLM's semantic caption, not a caption from the
    source. Those keep today's name rather than becoming a sentence.
    """

    from app.api.routes.download import _document_download_base_name

    counters: dict[str, int] = {}
    description = _document_download_base_name(
        source_name="示範報告",
        entry={
            "document_id": "fig0000",
            "kind": "figure_asset",
            "title": "圖片顯示「作者」二字，以白色字體呈現於兩個相鄰的灰色圓形背景中，作為文件作者欄位的標題。",
        },
        fallback_stem="fig0000",
        counters=counters,
    )
    placeholder = _document_download_base_name(
        source_name="示範報告",
        entry={"document_id": "fig0001", "kind": "figure_asset", "title": "Figure 9"},
        fallback_stem="fig0001",
        counters=counters,
    )
    overlong = _document_download_base_name(
        source_name="示範報告",
        entry={
            "document_id": "fig0002",
            "kind": "figure_asset",
            "title": "此圖表展示了智慧點餐打造智慧餐飲生活服務流程涵蓋從消費者互動到最終體驗六個階段",
        },
        fallback_stem="fig0002",
        counters=counters,
    )

    assert description == "示範報告_figure01"
    assert placeholder == "示範報告_figure02"
    assert overlong == "示範報告_figure03"


def test_download_name_strips_characters_that_break_extraction():
    """
    A title reaches the file name verbatim, so anything a filesystem treats as
    a separator has to go — otherwise the entry extracts into a stray folder.
    """

    from app.api.routes.download import _document_download_base_name

    base = _document_download_base_name(
        source_name="示範報告",
        entry={
            "document_id": "main",
            "kind": "main",
            "title": '2025/2026 展望：\n「風險*與\t機會」?  ',
        },
        fallback_stem="main",
        counters={},
    )

    assert base == "示範報告_main_2025 2026 展望： 「風險 與 機會」"
    assert not any(ch in base for ch in '\\/:*?"<>|\n\t')
    assert not base.endswith((".", " "))


def test_main_download_name_truncates_a_very_long_title():
    from app.api.routes.download import _MAIN_TITLE_MAX_LEN, _document_download_base_name

    title = "示" * 200
    base = _document_download_base_name(
        source_name="示範報告",
        entry={"document_id": "main", "kind": "main", "title": title},
        fallback_stem="main",
        counters={},
    )

    assert base == f"示範報告_main_{'示' * _MAIN_TITLE_MAX_LEN}"


def test_untitled_documents_keep_the_name_they_have_today():
    from app.api.routes.download import _document_download_base_name

    counters: dict[str, int] = {}
    main = _document_download_base_name(
        source_name="示範報告",
        entry={"document_id": "main", "kind": "main", "title": ""},
        fallback_stem="main",
        counters=counters,
    )
    asset = _document_download_base_name(
        source_name="示範報告",
        entry={"document_id": "tbl0000", "kind": "table_asset"},
        fallback_stem="tbl0000",
        counters=counters,
    )

    assert main == "示範報告_main"
    assert asset == "示範報告_table01"
