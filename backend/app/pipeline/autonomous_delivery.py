"""Terminal automatic delivery with explicit gaps and a source-verified RAG subset."""
from pathlib import Path
from urllib.parse import quote

from app.models.document_ir import BlockType
from app.pipeline.delivery import atomic_write_json, atomic_write_jsonl, atomic_write_text
from app.pipeline.privacy import scrub_transcribed_privacy
from app.pipeline.review_policy import dependency_closed_subset, reviewed_block_usable

# Picture content is delivered whether or not the source review could confirm
# it (policy chosen 2026-10-01: enlarged figures now read correctly, while the
# same model as verifier passed 3 of 18 usable Validation figures). The review
# decides which label the content carries, never whether it ships. The label is
# written only in the reading copy (delivery.md); text a RAG imports stays clean
# and carries the review as metadata (decided 2026-10-07: a 「未核對」 line inside
# retrieved text reads oddly).
IMAGE_LABELS = {
    "zh-TW": {"verified": "〔圖片內容：自動辨識，已核對〕", "unverified": "〔圖片內容：自動辨識，未核對〕"},
    "en": {"verified": "[Image content: automatic reading, verified]",
           "unverified": "[Image content: automatic reading, not verified]"},
}
# Gaps (unreadable regions), invalid evidence and missing context still exclude.
_UNVERIFIED_PICTURE_REASONS = {"content_or_context_unverified", "source_not_mapped"}


def is_picture_content(block):
    return block.type == BlockType.IMAGE or bool(block.payload.get("agentic_image_origin"))


class DeliveredBlockEvidence:
    """Exact text in chunks owned by this block, including recorded bounded parts.

    Whitespace can change during formatting; punctuation, signs, case and values
    cannot. Only a complete, consistently owned split family may be reassembled.
    """
    def __init__(self, chunks):
        from collections import defaultdict

        self.by_block = defaultdict(list)
        parents = defaultdict(list)
        for chunk in chunks:
            self._add(chunk.get("block_ids", []), chunk.get("content", ""))
            parent = chunk.get("metadata", {}).get("parent_chunk_id")
            if parent:
                parents[parent].append(chunk)
        for parts in parents.values():
            parts.sort(key=lambda part: part["metadata"].get("part_index", 0))
            count = len(parts)
            ids = parts[0].get("block_ids", [])
            if any(part["metadata"].get("part_count") != count
                   or part["metadata"].get("part_index") != index
                   or part.get("block_ids", []) != ids
                   for index, part in enumerate(parts, 1)):
                continue
            text = parts[0].get("content", "")
            for previous, part in zip(parts, parts[1:]):
                following = part.get("content", "")
                overlap = min(32, len(previous.get("content", "")) // 4)
                if not overlap or text[-overlap:] != following[:overlap]:
                    break
                text += following[overlap:]
            else:
                self._add(ids, text)

    def _add(self, ids, text):
        compact = "".join(text.split())
        for bid in ids:
            self.by_block[bid].append(compact)

    def covers(self, bid, text):
        compact = "".join(text.split())
        return bool(compact) and any(compact in part for part in self.by_block.get(bid, []))


def build_automatic_delivery(ir, run_path, config, review_summary, ledger, upstream_veto=False, complete_allowed=True):
    from app.pipeline.final_delivery import expected_blocks, read_json, read_jsonl
    from app.pipeline.review_sources import delivery_digest
    from app.pipeline.stages.chunk import ChunkStage, _scrub_chunk_metadata, estimate_tokens

    outputs = run_path / "outputs"
    report = read_json(outputs / "review_coverage.json", {})
    records = {b["block_id"]: b for b in report.get("blocks", [])}
    pages = {p["page_idx"]: p for p in report.get("pages", [])}
    coverage = {b["block_id"]: b for b in ledger}
    rendered = {b.block_id: (text, reason) for b, text, reason in expected_blocks(ir, outputs)}
    valid = (review_summary.get("evidence_valid", False) and review_summary.get("inventory_complete")
             and config.agentic.mode != "off" and not upstream_veto)
    eligible, reasons, dependencies, navigation = set(), {}, {}, {}
    heading_stack = []
    for block in ir.blocks:
        bid = block.block_id
        text, excluded = rendered[bid]
        if excluded:
            continue
        if block.type == BlockType.TEXT and (level := block.payload.get("text_level", 0)) > 0:
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            dependencies[bid] = {h[1] for h in heading_stack}
            heading_stack.append((level, bid))
        else:
            dependencies[bid] = {h[1] for h in heading_stack}
        navigation[bid] = set(dependencies[bid])
        record = records.get(bid, {})
        source_records = [pages.get(index, {}) for index in record.get("source_pages", [])]
        findings = [next((v for v in p.get("review", {}).get("blocks", []) if v["block_id"] == bid), {})
                    for p in source_records]
        if findings and all(v.get("context_complete") for v in findings):
            # Navigation is optional for explicitly self-contained content; units/qualifiers
            # identified by the verifier remain hard dependencies below.
            dependencies[bid] = set()
        dependencies[bid].update(record.get("depends_on_block_ids", []))
        dependencies[bid].update(block.payload.get("agentic_depends_on_block_ids", []))
        source_pages = record.get("source_pages", [])
        good = bool(valid and source_pages and not block.payload.get("agentic_gap"))
        for index in source_pages:
            page = pages.get(index, {})
            observations = [b for b in page.get("review", {}).get("blocks", []) if b["block_id"] == bid]
            good = good and bool(page.get("images") and len(observations) == 1 and
                                reviewed_block_usable(page, observations[0]))
        good = good and coverage[bid]["documents"] and coverage[bid]["chunks"] is not False
        if good:
            eligible.add(bid)
        else:
            reasons[bid] = ("region_not_interpreted" if block.payload.get("agentic_gap", {}).get("reason") == "region_not_interpreted" else
                            "unreadable_source" if block.payload.get("agentic_gap") else
                            "source_evidence_invalid" if not valid else
                            "source_not_mapped" if not source_pages else "content_or_context_unverified")
    pictures = {b.block_id for b in ir.blocks if is_picture_content(b) and not rendered[b.block_id][1]}
    unverified_pictures = {bid for bid in pictures if reasons.get(bid) in _UNVERIFIED_PICTURE_REASONS} if valid else set()
    # Dependants cannot survive missing required context, including chains. Navigation alone is not required:
    # a picture already shipping labelled unverified does not need its section heading verified (a 31-page
    # PDF held back 29 of 46 pictures that way). Units, qualifiers and other recorded dependencies still bind it.
    closure = {bid: deps - navigation.get(bid, set()) if bid in unverified_pictures
               and not (set(records.get(bid, {}).get("depends_on_block_ids", []))
                        | set(ir.get_block(bid).payload.get("agentic_depends_on_block_ids", []))) & navigation.get(bid, set())
               else deps for bid, deps in dependencies.items()}
    retained = dependency_closed_subset(eligible | unverified_pictures, closure)
    reasons.update({bid: "required_context_unverified" for bid in eligible - retained})
    unverified_pictures &= retained
    eligible &= retained

    renderer = ChunkStage(config)
    renderer._asset_text = renderer._load_asset_retrieval_text(outputs)
    asset_records = {item.get("block_id"): item for item in read_jsonl(outputs / "assets_index.jsonl")}
    non_informative = {b.block_id for b in ir.blocks if b.block_id in retained
                       and b.payload.get("agentic_information_status") == "non_informative"}
    rag_eligible = dependency_closed_subset(retained - non_informative, closure)
    # The review binds source blocks to the delivered chunks. Re-grouping verified
    # blocks changes inter-block boundaries, so prove each owned block separately
    # rather than asking a fuzzy global matcher to certify the new concatenation.
    evidence = DeliveredBlockEvidence(read_jsonl(outputs / "chunks.jsonl"))
    absent = {bid for bid in rag_eligible
              if not evidence.covers(bid, scrub_transcribed_privacy(
                  renderer._blocks_to_text([ir.get_block(bid)])))}
    rag_eligible = dependency_closed_subset(rag_eligible - absent, closure)
    # Gaps delimit chunk groups: overlap may never bridge an excluded region.
    groups, current = [], []
    for block in ir.blocks:
        if block.block_id in rag_eligible:
            current.append(block)
        elif current and not rendered[block.block_id][1]:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    chunks = []
    export_reasons = {bid: "non_informative_image" if bid in non_informative else "required_context_not_retrievable"
                      for bid in retained - rag_eligible}
    from app.pipeline.semantic.language import resolve_semantic_output_language
    labels = IMAGE_LABELS[resolve_semantic_output_language(config.package.semantic_output_language.value, ir)]
    export_reasons.update({bid: "rechunk_content_not_verified" for bid in absent})
    if config.package.generate_chunks:
        for group in groups:
            # Supply only verified heading/dependency blocks; never borrow unreviewed metadata.
            required = set().union(*(closure[b.block_id] for b in group))
            selected = {b.block_id for b in group} | required
            subset = ir.model_copy(update={"blocks": [b for b in ir.blocks if b.block_id in selected]})
            generated = renderer._chunk_blocks(subset, "rag", config.package.chunk_max_tokens,
                                               config.package.chunk_overlap_tokens)
            def closed_chunk(ids, heading_path):
                included = set(ids)
                while True:
                    expanded = included | set().union(*(closure[bid] for bid in included))
                    if expanded == included:
                        break
                    included = expanded
                chunk = renderer._create_chunk(
                    chunk_id="pending", doc_id=ir.doc_id, run_id=ir.run_id, view="rag",
                    blocks=[b for b in ir.blocks if b.block_id in included], heading_path=heading_path)
                chunk.content = scrub_transcribed_privacy(chunk.content)
                return chunk

            for base in generated:
                heading_path = base.metadata.get("heading_path", [])
                chunk = closed_chunk(base.block_ids, heading_path)
                candidates = [(chunk, base.block_ids)]
                if estimate_tokens(chunk.content) > config.package.chunk_max_tokens:
                    # Expand dependencies before deciding which semantic unit fits.
                    # A neighbor with a large closure cannot veto an independent table.
                    candidates = [(closed_chunk([bid], heading_path), [bid]) for bid in base.block_ids]
                for chunk, targets in candidates:
                    parts = [(chunk, [chunk.content])]
                    if estimate_tokens(chunk.content) > config.package.chunk_max_tokens:
                        parts = split_verified_table(renderer, ir, chunk, targets, config.package.chunk_max_tokens)
                        if not parts:
                            parts = split_verified_image(renderer, ir, chunk, targets,
                                                         config.package.chunk_max_tokens, asset_records)
                        if not parts:
                            parts = split_picture_lines(renderer, ir, chunk, targets,
                                                        config.package.chunk_max_tokens, asset_records)
                        if not parts:
                            export_reasons.update({bid: "complete_context_exceeds_chunk_limit" for bid in targets})
                            continue
                    for part, _ in parts:
                        if any(c.block_ids == part.block_ids and c.content == part.content for c in chunks):
                            continue
                        _scrub_chunk_metadata(part.metadata)
                        part.chunk_id = f"verified_{len(chunks):06d}"
                        unverified = sorted(bid for bid in part.block_ids if bid not in eligible)
                        part.metadata.update(source_verified=not unverified, unverified_block_ids=unverified,
                                              source_pages=sorted({p for bid in part.block_ids
                                              for p in records[bid]["source_pages"]}),
                                              content_kinds=sorted({ir.get_block(bid).payload.get("agentic_content_kind", "transcription")
                                                                   for bid in part.block_ids}))
                        chunks.append(part)
    exported = {bid for chunk in chunks for bid in chunk.block_ids}
    atomic_write_jsonl(outputs / "rag_chunks.jsonl", (c.to_dict() for c in chunks))
    atomic_write_text(outputs / "rag_verified.md", "\n\n".join(c.content for c in chunks))

    gaps, reading_copy = [], []
    for block in ir.blocks:
        bid = block.block_id
        text, excluded = rendered[bid]
        if excluded:
            continue
        if bid in eligible:
            reading_copy.append(labels["verified"] + "\n" + text if bid in pictures else text)
        elif bid in unverified_pictures:
            reading_copy.append(labels["unverified"] + "\n" + text)
            gaps.append({"block_id": bid, "source_pages": records.get(bid, {}).get("source_pages", []),
                         "reason": reasons.get(bid, "not_verified"), "marker": labels["unverified"],
                         "included": True, "evidence": [], "detail": {}})
        else:
            record = records.get(bid, {})
            indices = record.get("source_pages", [])
            location = "、".join(str(p + 1) for p in indices) if indices else "來源頁碼未定位"
            marker = f"〔內容未確認：原文 {location}，區塊 {bid}〕"
            if block.payload.get("agentic_gap"):
                marker = text
            reading_copy.append(marker)
            evidence = [dict(i, url=f"/api/runs/{quote(ir.run_id, safe='')}/processing-evidence?path={quote(i['path'], safe='')}")
                        for index in indices for i in pages.get(index, {}).get("images", [])]
            if source_image := block.payload.get("agentic_source_image"):
                matching = next((i for i in evidence if i["path"] == source_image), None)
                if matching:
                    reading_copy.append(f"[來源原圖]({matching['url']})")
            gaps.append({"block_id": bid, "source_pages": indices, "reason": reasons.get(bid, "not_verified"),
                         "marker": marker, "evidence": evidence,
                         "detail": block.payload.get("agentic_gap", {})})
    represented_pages = {p for record in records.values() for p in record.get("source_pages", [])}
    for index, page in pages.items():
        if index not in represented_pages and page.get("status") != "verified":
            marker = f"〔內容未確認：原文第 {index + 1} 頁，未取得可核對的內容區塊〕"
            reading_copy.append(marker)
            gaps.append({"block_id": None, "source_pages": [index], "reason": "source_page_missing",
                         "marker": marker, "evidence": page.get("images", [])})
        elif page.get("status") != "verified" and not page.get("review", {}).get("source_complete"):
            # Verified blocks do not conceal additional source content omitted by the parser.
            if not page.get("review", {}).get("declared_gaps_only"):
                marker = f"〔內容可能不完整：原文第 {index + 1} 頁仍有未核對範圍〕"
                reading_copy.append(marker)
                gaps.append({"block_id": None, "source_pages": [index], "reason": "page_coverage_incomplete",
                             "marker": marker, "evidence": page.get("images", [])})
    status = ("complete" if complete_allowed and review_summary.get("complete") and not gaps and not upstream_veto else
              "partial" if eligible else "unverified")
    summary = {"schema_version": 1, "status": status, "requires_human_review": False,
               "verified_blocks": len(eligible), "excluded_blocks": len(reasons), "gap_count": len(gaps),
               "rag_ready": bool(chunks), "rag_chunks": len(chunks), "rag_blocks": len(exported),
               "rag_file": "rag_chunks.jsonl", "reading_file": "delivery.md",
               "delivery_sha256": delivery_digest(run_path), "upstream_veto": upstream_veto}
    atomic_write_text(outputs / "delivery.md", "\n\n".join(reading_copy) + "\n")
    atomic_write_json(outputs / "processing_report.json", {**summary, "gaps": gaps,
                      "blocks": [{"block_id": b.block_id, "verified": b.block_id in eligible,
                                  "rag_eligible": b.block_id in exported, "reason": reasons.get(b.block_id),
                                  "rag_exclusion_reason": None if b.block_id in exported else export_reasons.get(b.block_id, reasons.get(b.block_id)),
                                  "source_pages": records.get(b.block_id, {}).get("source_pages", []),
                                  "image_label": ("verified" if b.block_id in eligible else "unverified"
                                                  if b.block_id in unverified_pictures else None)
                                  if b.block_id in pictures else None}
                                 for b in ir.blocks if not rendered[b.block_id][1]]}, ensure_ascii=False, indent=2)
    index_image_readings(ir, run_path, renderer, pictures, eligible,
                         [b for b in ir.blocks if not rendered[b.block_id][1]])
    return summary


def index_image_readings(ir, run_path, renderer, pictures, eligible, delivered_blocks):
    """Mark in documents_index.json what is an automatic image reading and whether review confirmed it.

    A picture's own document gets content_origin and review_status. Any other document lists
    image_reading_spans: file line ranges whose lines bind to exactly one picture block, so a RAG
    importer can attach the review to chunks without a label in the text. The index is not part
    of the delivery digest, so reviewed records stay valid.
    """
    from app.pipeline.final_delivery import read_json, read_jsonl
    from app.pipeline.review_sources import DeliveryExcerpts

    outputs = run_path / "outputs"
    index = read_json(outputs / "documents_index.json", None)
    if not isinstance(index, list):
        return
    origin = {bid: ir.get_block(bid).payload.get("agentic_image_origin") or bid for bid in pictures}

    def review(ids):
        confirmed = [bid in eligible for bid in ids]
        return "verified" if all(confirmed) else "partially_verified" if any(confirmed) else "unverified"
    assets = {row.get("asset_id"): row.get("block_id") for row in read_jsonl(outputs / "assets_index.jsonl")}
    excerpts = DeliveryExcerpts(run_path, delivered_blocks)
    excerpts._bind(delivered_blocks, renderer)
    exact = {"source_marker", "asset_document", "source_text", "contained_source_text"}
    for entry in index:
        entry.pop("image_reading_spans", None)
        picture = assets.get(entry.get("document_id"))
        if picture in origin.values():
            entry.update(content_origin="image_reading",
                         review_status=review([bid for bid, root in origin.items() if root == picture]))
            continue
        name = "documents/" + Path(str(entry.get("file") or "")).name
        spans = []
        for paragraph in excerpts.paragraphs:
            owners = paragraph["block_ids"]
            if paragraph["file"] != "outputs/" + name:
                continue
            if paragraph["binding"] not in exact or len(owners) != 1 or owners[0] not in origin:
                spans.append(None)  # any other line ends a span
                continue
            status = "verified" if owners[0] in eligible else "unverified"
            last = spans[-1] if spans else None
            if last and last["block_id"] == origin[owners[0]] and last["review_status"] == status:
                last["end_line"] = paragraph["line"]
            else:
                spans.append({"block_id": origin[owners[0]], "review_status": status,
                              "start_line": paragraph["line"], "end_line": paragraph["line"]})
        if spans := [span for span in spans if span]:
            entry["image_reading_spans"] = spans
    atomic_write_json(outputs / "documents_index.json", index, ensure_ascii=False, indent=2)


def split_verified_table(renderer, ir, chunk, targets, max_tokens):
    """Repeat verified headers/context while preserving complete source row groups."""
    import re
    from copy import deepcopy

    from app.pipeline.agentic_content import replacement_text
    from app.pipeline.package_utils import is_table_note_row
    from app.pipeline.stages.chunk import estimate_tokens
    from app.pipeline.table_structure import table_structure

    if len(targets) != 1:
        return []
    block = ir.get_block(targets[0])
    if block.type != BlockType.TABLE or replacement_text(block) or block.payload.get("verified_semantics"):
        return []
    grid = table_structure(block)
    if not grid or grid.row_count < 2:
        return []
    text = scrub_transcribed_privacy(renderer._block_to_text(block))
    if re.search(r"(?m)^- (?:第 \d+ 列：|Row \d+: )", text):
        return _split_coordinate_table(renderer, ir, chunk, block, grid, text, max_tokens)
    if any(c.rowspan != 1 or c.colspan != 1 for c in grid.cells):
        return []
    rows = [[c for c in grid.cells if c.row == i] for i in range(1, grid.row_count + 1)]
    if (any(len(row) != grid.column_count for row in rows) or not all(c.header for c in rows[0])
            or any(c.header for row in rows[1:] for c in row)
            or any(is_table_note_row([c.text for c in row]) for row in rows[1:])
            or block.payload.get("table_footnote")):
        return []
    sections = re.split(r"(?m)(?=^### )", text)
    if len(sections) != grid.row_count:
        return []
    prefix, records = sections[0].strip(), sections[1:]
    context = [scrub_transcribed_privacy(renderer._blocks_to_text([ir.get_block(bid)]))
               for bid in chunk.block_ids if bid != block.block_id]
    parts = []
    for row, record in enumerate(records, 2):
        texts = [t for t in [*context, prefix, record.strip()] if t]
        content = "\n\n".join(texts)
        if estimate_tokens(content) > max_tokens:
            return []  # Export all source rows or retain the whole table as a gap.
        part = deepcopy(chunk)
        part.content = content
        part.metadata.pop("table_html", None)
        part.metadata["table_source_rows"] = {block.block_id: [1, row]}
        parts.append((part, texts))
    return parts


def _split_coordinate_table(renderer, ir, chunk, block, grid, text, max_tokens):
    """Slice canonical row lines without renumbering cells or breaking rowspans."""
    import re
    from copy import deepcopy

    from app.pipeline.package_utils import is_table_note_row
    from app.pipeline.stages.chunk import estimate_tokens

    matches = list(re.finditer(r"(?m)^- (?:第 (\d+) 列：|Row (\d+): ).*$", text))
    rows = {int(m.group(1) or m.group(2)): m.group(0) for m in matches}
    anchors = {c.row for c in grid.cells if c.text.strip()}
    if not matches or set(rows) != anchors or len(rows) != len(matches):
        return []
    if any(text[a.end():b.start()].strip() for a, b in zip(matches, matches[1:])):
        return []
    # Leading units/period rows and every explicit header level are mandatory.
    header_end = max((c.row + c.rowspan - 1 for c in grid.cells if c.header), default=0)
    if not header_end or header_end >= grid.row_count:
        return []
    header_rows = set(range(1, header_end + 1))
    notes = {row for row in rows if is_table_note_row([c.text for c in grid.cells if c.row == row])}
    prefix, suffix = text[:matches[0].start()].strip(), text[matches[-1].end():].strip()
    context = [scrub_transcribed_privacy(renderer._blocks_to_text([ir.get_block(bid)]))
               for bid in chunk.block_ids if bid != block.block_id]
    parts, exported, section_rows = [], set(), set()
    for row in sorted(set(rows) - header_rows - notes):
        # A label-only row qualifies later records until the next section label.
        # Coordinates retain its original extent; it is never invented as a header.
        if len([c for c in grid.cells if c.row == row and c.text.strip()]) == 1:
            section_rows = {row}
        selected = header_rows | notes | section_rows | {row}
        while True:
            expanded = selected | {r for c in grid.cells
                if selected.intersection(range(c.row, c.row + c.rowspan))
                for r in range(c.row, c.row + c.rowspan)}
            if expanded == selected:
                break
            selected = expanded
        if row in exported:
            continue
        texts = [t for t in [*context, prefix, *(rows[r] for r in sorted(selected) if r in rows), suffix] if t]
        content = "\n\n".join(texts)
        if estimate_tokens(content) > max_tokens:
            return []  # Never silently drop one oversized record from a verified table.
        part = deepcopy(chunk)
        part.content = content
        part.metadata.pop("table_html", None)
        part.metadata["table_source_rows"] = {block.block_id: sorted(selected)}
        part.metadata["table_header_rows"] = {block.block_id: sorted(header_rows)}
        part.metadata["table_source_cells"] = {block.block_id: [c.id for c in grid.cells if c.row in selected]}
        parts.append((part, texts))
        exported.update(selected)
    return parts if set(rows).issubset(exported) else []


def split_verified_image(renderer, ir, chunk, targets, max_tokens, asset_records):
    """Split explicit figure facts from its intact transcription, never OCR lines."""
    from copy import deepcopy

    from app.pipeline.stages.chunk import estimate_tokens

    if len(targets) != 1:
        return []
    block = ir.get_block(targets[0])
    if block.type != BlockType.IMAGE:
        return []
    asset = asset_records.get(block.block_id, {})
    title, facts = asset.get("title"), asset.get("facts")
    if not isinstance(title, str) or not title.strip() or not isinstance(facts, list) or not facts:
        return []
    if not all(isinstance(fact, str) and fact.strip() for fact in facts):
        return []
    title = scrub_transcribed_privacy(title)
    facts = [scrub_transcribed_privacy(fact) for fact in facts]
    text = scrub_transcribed_privacy(renderer._blocks_to_text([block]))
    prefix = "\n".join([title, *facts])
    if not text.startswith(prefix + "\n"):
        return []  # Metadata is not authority to introduce text absent from the reviewed body.
    remainder = text[len(prefix):].strip()
    context = [scrub_transcribed_privacy(renderer._blocks_to_text([ir.get_block(bid)]))
               for bid in chunk.block_ids if bid != block.block_id]
    parts = []
    for section, body in (("facts", "\n".join(facts)), ("transcription", remainder)):
        texts = [t for t in [*context, title, body] if t]
        content = "\n\n".join(texts)
        if not body or estimate_tokens(content) > max_tokens:
            return []
        part = deepcopy(chunk)
        part.content = content
        part.metadata["image_content_section"] = {block.block_id: section}
        parts.append((part, texts))
    return parts


def split_picture_lines(renderer, ir, chunk, targets, max_tokens, asset_records):
    """Last resort for picture content over the chunk limit: whole lines per part.

    Dropping the picture loses all of it (the Validation 8-brand table, 4 of 26
    figures). Each part repeats the picture's title and supplied context so it
    stays attributable, and no line is ever cut; a single line over the limit
    keeps the old refusal.
    """
    from copy import deepcopy

    from app.pipeline.stages.chunk import estimate_tokens

    if len(targets) != 1:
        return []
    block = ir.get_block(targets[0])
    if not is_picture_content(block):
        return []
    lines = [line for line in scrub_transcribed_privacy(renderer._blocks_to_text([block])).splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    title = scrub_transcribed_privacy(str(asset_records.get(block.block_id, {}).get("title") or lines[0]))
    body = lines[1:] if lines[0].strip() == title.strip() else lines
    context = [scrub_transcribed_privacy(renderer._blocks_to_text([ir.get_block(bid)]))
               for bid in chunk.block_ids if bid != block.block_id]
    header = "\n\n".join(t for t in [*context, title] if t)
    budget = max_tokens - estimate_tokens(header)
    if budget <= 0 or any(estimate_tokens(line) > budget for line in body):
        return []
    groups, current = [], []
    for line in body:
        if current and estimate_tokens("\n".join([*current, line])) > budget:
            groups.append(current)
            current = []
        current.append(line)
    groups.append(current)
    parts = []
    for index, group in enumerate(groups, 1):
        part = deepcopy(chunk)
        part.content = header + "\n" + "\n".join(group)
        part.metadata["picture_part"] = {"block_id": block.block_id, "index": index, "count": len(groups)}
        parts.append((part, [part.content]))
    return parts


def checked_automatic_artifact(run_path, filename):
    """Fail closed for old, incomplete, edited or stale exports; no raw fallback."""
    import hashlib

    from app.pipeline.final_delivery import read_json
    from app.pipeline.review_sources import delivery_digest

    if filename not in {"rag_chunks.jsonl", "rag_verified.md", "delivery.md", "processing_report.json"}:
        raise ValueError("Unsupported automatic export")
    manifest = read_json(run_path / "outputs/delivery_manifest.json", {})
    report = read_json(run_path / "outputs/processing_report.json", {})
    if not report or report.get("delivery_sha256") != delivery_digest(run_path):
        raise ValueError("Verified export unavailable or stale; create a new processing run")
    # Bind the selected export and its eligibility report to the published revision.
    members = manifest.get("files", {})
    required = {"outputs/" + filename, "outputs/processing_report.json", "outputs/review_coverage.json"}
    required.update(name for name in members if name.startswith("agentic/evidence/") or name == "document_ir.json")
    for name in required:
        path = run_path / name
        expected = members.get(name)
        if (not path.resolve().is_relative_to(run_path.resolve()) or not expected or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError("Automatic export changed after validation")
    return run_path / "outputs" / filename
