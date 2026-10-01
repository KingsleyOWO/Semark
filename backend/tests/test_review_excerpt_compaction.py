"""Compact serialization must preserve source order, owner and untrusted text."""
from copy import deepcopy

from app.pipeline.review_sources import compact_delivery_excerpts, review_scope_digest


def excerpt(line, text, *, file="main.md", owners=None, binding="source_text"):
    return {"file": file, "paragraph": line, "text": text,
            "block_ids": ["source"] if owners is None else owners, "binding": binding}


def test_compaction_is_lossless_and_does_not_mutate_source_records():
    records = [excerpt(10, "軸：功能"), excerpt(11, "品牌 > 左上"),
               excerpt(12, "品牌 > 左上"), excerpt(13, "  保留空格  ")]
    before = deepcopy(records)
    compact = compact_delivery_excerpts(records)
    assert records == before
    assert len(compact) == 1
    assert compact[0]["text"] == "\n".join(r["text"] for r in records)
    assert compact[0]["paragraph"] == 10
    assert compact[0]["paragraph_end"] == 13
    assert compact[0]["text"].count("品牌 > 左上") == 2


def test_compaction_never_crosses_a_gap_owner_file_or_binding():
    records = [excerpt(0, "一"), excerpt(2, "二"),
               excerpt(3, "三", owners=["other"]),
               excerpt(4, "四", owners=["other"], binding="unbound_neighbor_context"),
               excerpt(5, "五", owners=["other"], binding="unbound_neighbor_context", file="figure.md")]
    assert compact_delivery_excerpts(records) == records


def test_compaction_preserves_untrusted_insertions_for_source_audit():
    records = [excerpt(0, "來源文字"),
               excerpt(1, "捏造的投資報酬率", binding="unbound_neighbor_context"),
               excerpt(2, "來源文字")]
    assert compact_delivery_excerpts(records) == records


def test_scoped_digest_ignores_global_offsets_but_binds_range_adjacency_and_text():
    records = compact_delivery_excerpts([excerpt(10, "甲"), excerpt(11, "乙"), excerpt(13, "丙")])
    payload = {"delivered_documents": records}
    shifted = deepcopy(payload)
    for item in shifted["delivered_documents"]:
        item["paragraph"] += 50
        if "paragraph_end" in item:
            item["paragraph_end"] += 50
    baseline = review_scope_digest("test", payload, [])
    assert baseline == review_scope_digest("test", shifted, [])
    shifted["delivered_documents"][1]["paragraph"] -= 1
    assert baseline != review_scope_digest("test", shifted, [])
    changed = deepcopy(payload)
    changed["delivered_documents"][0]["text"] = "甲\n假文字"
    assert baseline != review_scope_digest("test", changed, [])
