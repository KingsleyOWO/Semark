"""Recover one picture as separately auditable visual/text/gap blocks."""
import hashlib
import re
import shutil

from app.models.document_ir import Block, BlockType
from app.pipeline.delivery import atomic_write_json
from app.pipeline.image_reading import (
    read_contract,
    read_image_parts_bounded,
    reconcile_image_parts,
)
from app.pipeline.image_source_review import appears_in
from app.pipeline.review_policy import dependency_closed_subset


def repaired_parts(ir, block_id):
    """The blocks a repaired picture is now delivered as, in order; none while the picture is in the IR."""
    if ir.get_block(block_id) is not None:
        return []
    return [b for b in ir.blocks if b.payload.get("agentic_image_origin") == block_id]


def picture_block(runner, ir, block_id):
    """The picture to repair: its IR block, or for a repaired picture the block its first repair replaced.

    Rerun 2026-10-16: a committed repair turned the picture into text blocks no method could
    reach again, so 13 of 30 repaired pictures kept faults their next audit spelled out (a
    radar chart's completion date, a certificate's company name one character off).
    """
    block = ir.get_block(block_id)
    if block is None and repaired_parts(ir, block_id):
        stored = runner.state.get("repaired_pictures", {}).get(block_id)
        return Block.model_validate(stored) if stored else None
    return block


def picture_part_ids(ir, block_id):
    return [b.block_id for b in repaired_parts(ir, block_id)]


async def repair_image(runner, ir, problem, decision, folder, skill):
    block = picture_block(runner, ir, problem.get("block_id") or decision.block_id)
    page = problem["page_idx"]
    if not block or block.type != BlockType.IMAGE or runner.inventory.mapping.get(block.block_id) != [page]:
        raise ValueError("Image recovery requires one mapped source image on one physical page")
    original = runner.inventory.word_images.get(block.block_id)
    positions = [m["bbox"] for m in runner.inventory.image_positions.get(block.block_id, [])
                 if m["page_idx"] == page]
    region = positions[0] if len(positions) == 1 else [0, 0, 1000, 1000]
    if original is None:
        if not ir.has_physical_pages():
            raise ValueError("Original image evidence unavailable; do not infer it from the caption")
        from app.pipeline.document_recovery import crop_region
        region = block.bbox_norm
        if len(region) != 4 or region[0] >= region[2] or region[1] >= region[3]:
            raise ValueError("Image has no source coordinates")
        original, region = crop_region(runner.inventory, page, region,
                                       runner.inventory.directory / "image-parts", block.block_id)
    sha = hashlib.sha256(original.read_bytes()).hexdigest()
    source = runner.root / "agentic/evidence/image-parts" / (sha + ".png")
    source.parent.mkdir(parents=True, exist_ok=True)
    if source != original:
        shutil.copy2(original, source)
    prior = runner.owner.state.get("pages", {}).get(str(page), {}).get("word_image_readings", {}).get(block.block_id, {})
    part_ids = picture_part_ids(ir, block.block_id)
    method = next_image_method(runner.owner.state, runner.state, page, block.block_id, part_ids)
    if method == "delivery_patch":
        statements = delivery_patch_statements(runner.owner.state, page, block.block_id)
        return build_delivery_patch_candidate(runner, ir, block, page, region, source, statements, folder), [page]
    if method == "review_correction":
        return await build_review_correction_candidate(runner, ir, block, page, region, source, original, folder), [page]
    if method == "targeted_edit":
        return await build_targeted_edit_candidate(runner, ir, block, page, region, source, original, folder), [page]
    from app.pipeline.image_geometry import geometry_eligible, measured_axes, repair_axis_image
    if geometry_eligible(prior) and (axes := measured_axes(source)):
        return await repair_axis_image(runner, ir, block, page, region, source, folder, prior, axes,
                                       model_role=decision.model_role), [page]
    requests = image_read_requests(runner.owner.config, ir, skill, decision.model_role)
    readings = [await read_image_parts_bounded(runner.owner, task, payload, source) for task, payload in requests]
    primary, independent = readings
    prior_text = prior.get("text") if prior.get("kind") == "transcription" else None
    parts = reconcile_image_parts(primary, independent, prior_transcription=prior_text)
    from app.pipeline.image_evidence import resolve_image_evidence
    parts, evidence_ids = await resolve_image_evidence(runner, block, page, source, primary, independent,
                                                       prior, parts, folder, model_role=decision.model_role)
    from app.adapters import ocr
    from app.pipeline.image_edits import known_issues
    drafted = [line.removeprefix(READ_LINE_PREFIX) for line in picture_read_lines(runner, ir, block)]
    lines = await runner.owner._ocr_lines(original, runner.inventory) if drafted and ocr.ocr_python() else None
    parts, carried, named, contested = keep_drafted_read_lines(
        parts, drafted, known_issues(runner.owner.state, runner.state, page, block.block_id, part_ids), lines)
    atomic_write_json(folder / "image-parts.json", {
        "source_block_id": block.block_id, "source_sha256": sha,
        "primary": primary.model_dump(), "independent": independent.model_dump(),
        "prior_source_reading": prior, "candidate": parts,
        "drafted_read_lines": {"carried": carried, "named_wrong": named, "contested": contested},
    }, ensure_ascii=False, indent=2)
    # Audits of such candidates were spent on what was already delivered, on gaps, or on a line
    # the same audits would fault again.
    if not named and not any(not appears_in(line, "\n".join(drafted)) for line in reread_lines(parts)):
        raise ValueError("full_reread_found_nothing_new")
    if contested:
        raise ValueError("full_reread_could_not_settle_a_disputed_line")
    return build_image_parts_candidate(runner, ir, block, page, region, source, parts, evidence_ids, folder), [page]


def reread_lines(parts):
    return [item["text"] for item in parts["text_regions"] if not item.get("carried")] + list(parts["visual_facts"])


# A remark saying quoted wording is not in the picture at all, as opposed to omitted or misplaced.
_NOT_IN_PICTURE = re.compile(r"不存在|並未出現|未出現在|虛構|捏造|無中生有|憑空|"
                             r"not (?:present|visible|shown|found) in|does not (?:appear|exist)|hallucin|fabricat|invent", re.I)


def keep_drafted_read_lines(parts, drafted, issues, ocr_lines=()):
    """A full re-read keeps every drafted line read off the picture that no audit showed wrong.

    Edits already never drop a transcription line. Rerun 2026-10-15: a re-read that came
    back as one whole-picture gap replaced 48 logo names an audit had called omitted, not
    wrong, and others lost a brand name and a seal their issues asked for. Being quoted is
    not being wrong: one audit quoted two correct column headings to say a row sat under
    the wrong one. A drafted line is wrong when an audit spells out a correction pair for
    it and the re-read or PaddleOCR reads the other side and not this one, or quotes it as
    not in the picture while neither reads it. Returns (parts, carried lines, lines shown wrong,
    carried lines an audit disputes that neither settles).
    """
    from app.pipeline.content_coverage import normalized_text
    from app.pipeline.image_edits import _QUOTED, spelled_out_corrections
    from app.pipeline.review_policy import soft_review_remark

    read = "\n".join(reread_lines(parts))
    seen = "\n".join([read, *(ocr_lines or [])])
    wrong = set()
    for a, b in spelled_out_corrections(issues):
        for bad, good in ((a, b), (b, a)):
            if appears_in(good, seen) and not appears_in(bad, seen):
                wrong.add(normalized_text(bad))
    wrong |= {normalized_text(next(g for g in m.groups() if g is not None))
              for remark in issues if _NOT_IN_PICTURE.search(remark) and not soft_review_remark(remark)
              for m in _QUOTED.finditer(remark)
              if not appears_in(next(g for g in m.groups() if g is not None), seen)}
    wrong.discard("")
    disputed = {normalized_text(side) for pair in spelled_out_corrections(issues) for side in pair} - {""}
    from app.pipeline.image_edits import _misreading
    reread = [line for item in reread_lines(parts) for line in item.split("\n") if normalized_text(line)]
    ocr = "\n".join(ocr_lines or [])
    draft = "\n".join(drafted)
    carried, named, contested = [], [], []
    for line in drafted:
        if not normalized_text(line) or appears_in(line, read):
            continue
        # The re-read read the same line better when OCR reads its version and not the drafted one:
        # 「由晴偃恩發佈」 was carried beside the re-read's 「由晴偃思發佈」 (rerun 2026-10-16). A piece
        # of a drafted line, or another line the draft has, is no other version of it: an award wall
        # lost 「THE COMMUNICATOR AWARDS」 to OCR's 「COMMUNICATOR」 and 「GEWINNER」 to the
        # 「BERLINER」 of another logo (test run 2026-10-19).
        if any(q in normalized_text(line) for q in wrong) or ocr and not appears_in(line, ocr) and any(
                _misreading(line, other) and appears_in(other, ocr) and not appears_in(other, draft)
                for other in reread):
            named.append(line)
            continue
        carried.append(line)
        if any(q in normalized_text(line) for q in disputed):
            contested.append(line)
    if carried:
        parts = {**parts, "text_regions": [*parts["text_regions"],
                                           *({"text": line, "bbox": None, "carried": True} for line in carried)]}
    return parts, carried, named, contested


def delivery_patch_statements(state, page, block_id):
    """What a picture's delivery omits, when omission is all that failed; otherwise nothing.

    Two kinds of omission, each seen by two looks at the source: a relationship
    the blind reading listed and the audit found missing from the delivery, and a
    transcription line the review's reading and PaddleOCR both read but the
    delivery lacks. A picture whose text and meaning are right needs those
    sentences, not a fresh reading of every pixel. A delivered line OCR
    contradicts is never rewritten here.
    """
    from app.pipeline.image_source_review import WordPictureItem
    from app.pipeline.zh_text import to_taiwan_traditional

    record = state.get("pages", {}).get(str(page), {})
    raw = record.get("word_image_readings", {}).get(block_id)
    finding = next((b for b in record.get("review", {}).get("blocks", []) if b["block_id"] == block_id), None)
    issues = [i for i in record.get("image_issues", []) if i["block_id"] == block_id]
    kinds = {i["failure_kind"] for i in issues}
    if not raw or not finding or not kinds or not kinds <= {"image_relationship_mismatch", "reading_disagreement"}:
        return []
    reading = WordPictureItem.model_validate(raw)
    if not reading.complete:
        return []
    statements = []
    if "image_relationship_mismatch" in kinds:
        relations = {r.relation_id: r for r in reading.relationships}
        checks = [c for c in finding.get("image_relationship_checks", []) if c["relation_id"] in relations]
        missing = [c for c in checks if c["status"] == "missing"]
        if (not reading.relationships_complete or not missing
                or any(c["status"] not in {"matched", "missing"} for c in checks)
                or any(not relations[c["relation_id"]].certain for c in missing)):
            return []
        statements += ["圖片關係：" + to_taiwan_traditional(relations[c["relation_id"]].source_fact) for c in missing]
    if "reading_disagreement" in kinds:
        verdicts = record.get("line_adjudications", {}).get(block_id) or []
        omitted = [v for v in verdicts if v["verdict"] == "reading" and not v["delivered_line"]]
        # An omitted line OCR cannot settle is left out, not a reason to leave out those it reads: a slide
        # grid's tiny print held back every line of its picture (rerun 2026-10-16).
        if not omitted or any(v["delivered_line"] and v["verdict"] != "delivery" for v in verdicts):
            return []
        statements += ["圖片文字：" + to_taiwan_traditional(v["reading_line"]) for v in omitted]
    return statements


KEPT_PIECE_CHARS = 600  # well under one chunk, so each kept piece is one exact delivered span


def kept_text_pieces(text):
    """Split kept delivery into consecutive pieces, never reordering or dropping a character."""
    pieces, current = [], ""
    for line in text.splitlines():
        while len(line) > KEPT_PIECE_CHARS:
            cut = line.rfind("。", 0, KEPT_PIECE_CHARS) + 1 or KEPT_PIECE_CHARS
            pieces, current = pieces + ([current] if current else []), ""
            pieces.append(line[:cut])
            line = line[cut:]
        if current and len(current) + 1 + len(line) > KEPT_PIECE_CHARS:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)
    return pieces


def method_attempted(recovery_state, block_id, method):
    return any(s.get("action") == "repair_image" and s.get("block_id") == block_id
               and s.get("method") == method for s in recovery_state.get("steps", []))


def delivery_patch_attempted(recovery_state, block_id):
    return method_attempted(recovery_state, block_id, "delivery_patch")


EDITABLE_FAILURES = {"image_relationship_mismatch", "reading_disagreement", "image_content_mismatch"}
# Methods that keep the delivered text and change only what they name.
LOCAL_EDIT_METHODS = {"delivery_patch", "targeted_edit", "review_correction"}
# The last of them, which also carries the review's spelled-out corrections, is accepted when its audit
# finds nothing wrong in the wording it brought in, older faults left as they were; an earlier method
# must leave nothing hard behind. Any of them on a picture already repaired is judged the same way,
# its older faults known when that repair was kept (agentic_improves_repair).
IMPROVING_METHODS = {"targeted_edit"}


def next_image_method(owner_state, recovery_state, page, block_id, part_ids=()):
    """Cheapest untried method first, each once: apply corrections a review spelled out, add
    omissions, fix named spans, then re-read.

    A picture that is only unreadable has nothing an edit could fix. A repaired picture's
    remarks are on the blocks it is delivered as (``part_ids``); any hard one is editable.
    """
    from app.pipeline.image_edits import known_issues, spelled_out_corrections

    if (spelled_out_corrections(known_issues(owner_state, recovery_state, page, block_id, part_ids))
            and not method_attempted(recovery_state, block_id, "review_correction")):
        return "review_correction"
    if (delivery_patch_statements(owner_state, page, block_id)
            and not method_attempted(recovery_state, block_id, "delivery_patch")):
        return "delivery_patch"
    record = owner_state.get("pages", {}).get(str(page), {})
    editable = any(i["block_id"] == block_id and i["failure_kind"] in EDITABLE_FAILURES
                   for i in record.get("image_issues", []))
    if part_ids:
        from app.pipeline.review_policy import picture_problems

        editable = editable or bool(picture_problems(record.get("review", {}), part_ids))
    if editable and not method_attempted(recovery_state, block_id, "targeted_edit"):
        return "targeted_edit"
    return "full"


def picture_read_lines(runner, ir, block):
    """The lines read off the picture as delivered now: its draft's, or for a repaired picture the
    read lines of the blocks it is delivered as, written as they are delivered."""
    from app.pipeline.image_edits import transcription_lines

    parts = repaired_parts(ir, block.block_id)
    if not parts:
        return transcription_lines(runner.root / "outputs", block.block_id)
    lines = []
    for part in parts:
        if part.payload.get("agentic_gap"):
            continue
        if "agentic_read_lines" in part.payload:
            lines += part.payload["agentic_read_lines"]
        elif part.payload.get("agentic_content_kind") == "transcription":
            lines += part.get_text().split("\n")
    return [line for line in lines if line.strip()]


def delivered_picture_text(runner, block, ir=None):
    """The picture's real delivered text; a "[Image]" placeholder would replace content.

    A repaired picture's is the text of the blocks it is delivered as, its gap marker aside:
    the marker stays a marker.
    """
    from app.pipeline.stages.chunk import ChunkStage

    if ir is not None and (parts := repaired_parts(ir, block.block_id)):
        return "\n".join(b.get_text() for b in parts if not b.payload.get("agentic_gap")).strip()
    renderer = ChunkStage(runner.owner.config)
    renderer._asset_text = renderer._load_asset_retrieval_text(runner.root / "outputs")
    return renderer._asset_text.get(block.block_id, "").strip()


def build_delivery_patch_candidate(runner, ir, block, page, region, source, statements, folder):
    """Keep the picture's delivered text verbatim and add what it omits.

    The same per-image and page audits decide; a rejected patch leaves the
    original delivery untouched (no pruning to gaps), and the next attempt is a
    different method, not this again.
    """
    delivered = delivered_picture_text(runner, block, ir)
    if not delivered:
        raise ValueError("delivery_patch_requires_delivered_picture_text")
    atomic_write_json(folder / "delivery-patch.json", {"source_block_id": block.block_id, "statements": statements,
                                                       "kept_delivered_text": delivered}, ensure_ascii=False, indent=2)
    return kept_text_candidate(runner, ir, block, page, region, source, delivered, statements, folder, "delivery_patch",
                               read_lines=picture_read_lines(runner, ir, block),
                               new_wording=[s.split("：", 1)[-1] for s in statements])


async def build_targeted_edit_candidate(runner, ir, block, page, region, source, original, folder):
    """Fix the spans every audit so far named, under rules the system checks (see image_edits)."""
    from app.adapters import ocr
    from app.pipeline.image_edits import (
        EDIT_POLICY,
        ImageEditProposal,
        apply_corrections,
        apply_edits,
        known_issues,
        review_corrections,
    )
    from app.pipeline.image_source_review import WordPictureItem
    from app.pipeline.semantic.language import resolve_semantic_output_language

    delivered = delivered_picture_text(runner, block, ir)
    if not delivered:
        raise ValueError("targeted_edit_requires_delivered_picture_text")
    raw = runner.owner.state.get("pages", {}).get(str(page), {}).get("word_image_readings", {}).get(block.block_id)
    reading = WordPictureItem.model_validate(raw) if raw else None
    lines = await runner.owner._ocr_lines(original, runner.inventory) if ocr.ocr_python() else None
    transcription = picture_read_lines(runner, ir, block)
    issues = known_issues(runner.owner.state, runner.state, page, block.block_id, picture_part_ids(ir, block.block_id))
    language = resolve_semantic_output_language(runner.owner.config.package.semantic_output_language.value, ir)
    payload = {"delivered_text": delivered, "transcription_lines": transcription, "known_issues": issues,
               "independent_reading": {"text": reading.text, "relationships": [r.model_dump() for r in reading.relationships]}
               if reading else None,
               "ocr_lines": lines or [], "output_language": language, "edit_policy": EDIT_POLICY}
    proposal = await read_contract(runner.owner, "propose_image_edits", payload, [source], ImageEditProposal)
    # What a review spelled out and OCR reads goes in as-is; a rejected review_correction is not lost here.
    corrections = review_corrections(issues, delivered, lines or [])
    corrected = apply_corrections(delivered, corrections)
    transcription = [apply_corrections(line, corrections) for line in transcription]
    # Other pictures of the document PaddleOCR has read: the same printed name, legible elsewhere.
    document_ocr = [entry["lines"] for entry in runner.owner.state.get("ocr_lines", {}).values() if entry.get("lines")]
    text, statements, applied, rejected = apply_edits(corrected, proposal.edits, transcription=transcription,
                                                      ocr_lines=lines, reading=reading, document_ocr=document_ocr)
    applied = [{"op": "replace", "quote": wrong, "replacement": right, "reason": "review_correction"}
               for wrong, right in corrections.items()] + applied
    record = {"source_block_id": block.block_id, "known_issues": issues, "ocr_lines": lines,
              "proposal": proposal.model_dump(), "applied": applied, "rejected": rejected,
              "delivered_text": delivered, "edited_text": text, "statements": statements}
    atomic_write_json(folder / "targeted-edit.json", record, ensure_ascii=False, indent=2)
    if not applied:
        raise ValueError("targeted_edit_found_no_supported_edit")
    edit_evidence = runner.evidence.add(provider="targeted_image_edit", kind="image_edit_record", page=page,
                                        block_id=block.block_id, source=source, content=record)
    from app.pipeline.content_coverage import normalized_text as _normalized
    from app.pipeline.image_edits import apply_corrections
    # A drafted line an accepted edit corrected is still a line read off the picture: rerun 2026-10-16,
    # an unverified radar chart delivered neither 「品牌統御力」 nor 3.33 after both were fixed.
    replaced = {e["quote"]: e["replacement"] for e in applied if e["op"] == "replace"}
    read = transcription + [apply_corrections(line, replaced) for line in transcription] + [
        line for line in text.split("\n") if lines and _normalized(line) and appears_in(line, "\n".join(lines))]
    wording = [e["replacement"] for e in applied if e["op"] == "replace"] + [s.split("：", 1)[-1] for s in statements]
    return kept_text_candidate(runner, ir, block, page, region, source, text, statements, folder, "targeted_edit",
                               evidence_ids=[edit_evidence], read_lines=read, new_wording=wording)


async def build_review_correction_candidate(runner, ir, block, page, region, source, original, folder):
    """Apply the corrections a review spelled out and PaddleOCR confirms; no model call writes text."""
    from app.adapters import ocr
    from app.pipeline.image_edits import apply_corrections, known_issues, review_corrections

    delivered = delivered_picture_text(runner, block, ir)
    if not delivered:
        raise ValueError("review_correction_requires_delivered_picture_text")
    lines = await runner.owner._ocr_lines(original, runner.inventory) if ocr.ocr_python() else None
    issues = known_issues(runner.owner.state, runner.state, page, block.block_id, picture_part_ids(ir, block.block_id))
    corrections = review_corrections(issues, delivered, lines or [])
    if not corrections:
        raise ValueError("review_correction_found_no_confirmed_edit")
    text = apply_corrections(delivered, corrections)
    record = {"source_block_id": block.block_id, "known_issues": issues, "ocr_lines": lines,
              "corrections": corrections, "delivered_text": delivered, "corrected_text": text}
    atomic_write_json(folder / "review-correction.json", record, ensure_ascii=False, indent=2)
    evidence = runner.evidence.add(provider="review_correction", kind="image_edit_record", page=page,
                                   block_id=block.block_id, source=source, content=record)
    read = [apply_corrections(line, corrections) for line in picture_read_lines(runner, ir, block)]
    return kept_text_candidate(runner, ir, block, page, region, source, text, [], folder, "review_correction",
                               evidence_ids=[evidence], read_lines=read, new_wording=list(corrections.values()))


def kept_text_candidate(runner, ir, block, page, region, source, kept, statements, folder, method, *, evidence_ids=(),
                        read_lines=(), new_wording=()):
    """Replace the picture with its kept text in chunk-sized pieces plus any added statements.

    Each piece records which of its lines were read off the picture (an unverified picture
    keeps only those) and every piece the wording the method brought in, which is what its
    audit judges.
    """
    from app.pipeline.content_coverage import normalized_text
    from app.pipeline.image_source_review import WordPictureItem

    common = {**runner.source_payload(page), "agentic_source_region": region,
              "agentic_source_image": str(source.relative_to(runner.root)),
              "agentic_recovery_tool": "repair_image", "agentic_image_repair_method": method,
              "agentic_image_origin": block.block_id, "image_source_context": block.payload.get("word_source", {}),
              "agentic_evidence_ids": list(evidence_ids), "agentic_content_kind": "visual_summary",
              **picture_identity(block)}
    # A PDF picture has no blind Word reading; a Word picture's relationships must stay audited.
    if raw := runner.owner.state["pages"][str(page)].get("word_image_readings", {}).get(block.block_id):
        relationship_evidence = runner.evidence.add(
            provider="blind_word_image_reading", kind="image_relationship_reading", page=page, block_id=block.block_id,
            source=source, content=WordPictureItem.model_validate(raw).model_dump())
        common["agentic_evidence_ids"].insert(0, relationship_evidence)
        common["agentic_relationship_evidence_id"] = relationship_evidence
    read = {normalized_text(line) for line in read_lines if normalized_text(line)}
    common["agentic_new_wording"] = [w for w in dict.fromkeys(new_wording) if w.strip()]
    if repaired_parts(ir, block.block_id):
        # The faults an earlier repair left were known when it was kept; this edit is judged on its own change.
        common["agentic_improves_repair"] = True

    def piece_payload(text):
        lines = [line for line in text.split("\n")
                 if normalized_text(line) in read or line.startswith("圖片文字：")]
        return {**common, "text": text, "agentic_read_lines": lines}
    replacements = [Block(block_id=f"recovery_{folder.name}_image_k{index}", type=BlockType.TEXT,
                          page_idx=block.page_idx, bbox_norm=region, payload=piece_payload(piece))
                    for index, piece in enumerate(kept_text_pieces(kept))]
    if statements:
        replacements.append(Block(block_id=f"recovery_{folder.name}_image_r0", type=BlockType.TEXT,
                                  page_idx=block.page_idx, bbox_norm=region,
                                  payload=piece_payload("\n".join(statements))))
    return replace_picture(ir, block.block_id, replacements, keep_gaps=True)


DEPENDENCY_FIELDS = ("agentic_depends_on_block_ids", "agentic_source_depends_on_block_ids",
                     "agentic_review_depends_on_block_ids")


def replace_picture(ir, block_id, replacements, *, keep_gaps=False):
    """A copy of the IR with the picture replaced, or a repaired picture's blocks.

    An edit of a repaired picture keeps its gap marker as it was, since an edit leaves what
    it does not name alone; a re-read replaces the marker as well. A kept marker's links to
    replaced blocks point at the first replacement.
    """
    candidate = ir.model_copy(deep=True)
    removed = ({block_id} if candidate.get_block(block_id) is not None else
               {b.block_id for b in candidate.blocks if b.payload.get("agentic_image_origin") == block_id
                and not (keep_gaps and b.payload.get("agentic_gap"))})
    if not removed:
        raise ValueError("picture_has_no_delivered_block_to_replace")
    index = next(i for i, b in enumerate(candidate.blocks) if b.block_id in removed)
    candidate.blocks = [*candidate.blocks[:index], *replacements,
                        *(b for b in candidate.blocks[index:] if b.block_id not in removed)]
    first = replacements[0].block_id if replacements else None
    added = {b.block_id for b in replacements}
    for item in candidate.blocks:
        if item.payload.get("agentic_image_origin") != block_id or item.block_id in added:
            continue
        for field in DEPENDENCY_FIELDS:
            if set(item.payload.get(field) or []) & removed:
                item.payload[field] = sorted({first if bid in removed else bid for bid in item.payload[field]}
                                             - {None, item.block_id})
    for order, item in enumerate(candidate.blocks):
        item.reading_order = order
    return candidate


def picture_identity(block):
    """The replaced picture's file, which keeps its own figure document and link (spot check
    2026-10-16: 16 repaired pictures lost both). Its caption is not carried: a repair can be the
    very thing that found a drafted caption wrong."""
    return {"agentic_image_path": block.payload.get("img_path", "")}


PENDING_GAP_REASONS = {"image_evidence_budget_deferred", "image_evidence_response_contract_failed"}
READ_LINE_PREFIX = "圖片文字："
DISPUTED_LINE_PREFIX = "圖片文字（有疑義，未核對）："
GAP_KINDS = {"pending": "尚未完成證據核對的文字或必要內容", "illegible": "無法可靠辨識的文字或必要內容",
             "omitted": "未納入文字解讀的區域（保留來源圖）"}


def _gap_kind(reason):
    return ("omitted" if reason == "region_not_interpreted" else
            "pending" if reason in PENDING_GAP_REASONS else "illegible")


def gap_marker(page, regions):
    """One line for all of a picture's undelivered regions, by kind and count.

    A line per region named the picture by its internal ID: one meeting photo shipped
    eleven of them, and each held its page back on its own.
    """
    from collections import Counter

    counts = Counter(_gap_kind(region["reason"]) for region in regions)
    marker = (f"〔原文第 {page + 1} 頁圖片："
              + "、".join(f"{GAP_KINDS[kind]} {counts[kind]} 處" for kind in GAP_KINDS if counts[kind]) + "〕")
    readings = list(dict.fromkeys(region["reading"] for region in regions if region.get("reading")))
    return "\n".join([marker, *(DISPUTED_LINE_PREFIX + reading for reading in readings)])


def picture_gap(origin, regions):
    """The agentic_gap of a picture's one gap block; every region keeps its box and reasons."""
    boxes = [region["image_bbox"] for region in regions]
    union = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    reasons = [region["reason"] for region in regions]
    gap = {"reason": next((r for r in reasons if r != "region_not_interpreted"), reasons[0]), "source_id": origin,
           "image_bbox": union, "regions": regions}
    if sources := sorted({r for region in regions for r in region.get("source_reasons", [region["reason"]])}
                         if any("source_reasons" in region for region in regions) else []):
        gap["source_reasons"] = sources
    return gap


def merge_picture_gaps(blocks, origin, page):
    """Fold every gap block of one picture into its first, in place of separate markers.

    Pruning and region revision mark parts one at a time; the delivery shows one marker.
    Returns the blocks with the others removed and dependencies on them moved to the first.
    """
    gaps = [b for b in blocks if b.payload.get("agentic_image_origin") == origin and b.payload.get("agentic_gap")]
    if len(gaps) < 2 and not (gaps and "regions" not in gaps[0].payload["agentic_gap"]):
        return blocks
    first = gaps[0]
    regions = [region for b in gaps for region in b.payload["agentic_gap"].get("regions") or [{
        "reason": b.payload["agentic_gap"]["reason"], "image_bbox": b.payload["agentic_gap"].get(
            "image_bbox", b.payload.get("agentic_image_region", [0, 0, 1000, 1000])),
        **({"source_reasons": b.payload["agentic_gap"]["source_reasons"]}
           if "source_reasons" in b.payload["agentic_gap"] else {})}]]
    first.payload["agentic_gap"] = picture_gap(origin, regions)
    first.payload["agentic_image_region"] = first.payload["agentic_gap"]["image_bbox"]
    first.payload["text"] = gap_marker(page, regions)
    removed = {b.block_id for b in gaps[1:]}
    fields = ("agentic_depends_on_block_ids", "agentic_source_depends_on_block_ids",
              "agentic_review_depends_on_block_ids")
    for field in fields:
        first.payload[field] = sorted(set().union(*(b.payload.get(field, []) for b in gaps)) - {first.block_id} - removed)
    kept = [b for b in blocks if b.block_id not in removed]
    for b in kept:
        for field in fields:
            if field in b.payload:
                b.payload[field] = sorted({first.block_id if bid in removed else bid
                                           for bid in b.payload[field]} - {b.block_id})
    return kept


def build_image_parts_candidate(runner, ir, block, page, region, source, parts, evidence_ids, folder,
                                *, relationship_reading=None, relationship_parents=()):
    """Materialize source observations for the same ordinary scoped delivery audit.

    A re-read of a repaired picture keeps the relationship evidence its blocks carry: an edit attaches
    the blind reading even when it lists no relationship, and a re-read that dropped it was refused as
    having lost the picture's provenance (test run 2026-10-19).
    """
    inherited = any(b.payload.get("agentic_relationship_evidence_id") for b in repaired_parts(ir, block.block_id))
    replacements = build_image_parts_blocks(runner, block, page, region, source, parts, evidence_ids, folder,
        relationship_reading=relationship_reading, relationship_parents=relationship_parents,
        keep_relationship_evidence=inherited)
    return replace_picture(ir, block.block_id, replacements)


def distinct_text_regions(regions):
    """Text regions whose wording no other region already delivers in full.

    Readers return a block and then its lines, or a line and a phrase inside it: one
    table shipped 「Customer\\nCost…」 and each word again, a photo ten nested regions.
    A region whose text another one contains (an equal one: a later one) adds nothing.
    """
    from app.pipeline.content_coverage import normalized_text
    from app.pipeline.zh_text import to_taiwan_traditional

    texts = [normalized_text(to_taiwan_traditional(item["text"])) for item in regions]
    return [item for index, item in enumerate(regions)
            if texts[index] and not any(other != index and texts[index] in texts[other]
                                        and (len(texts[other]) > len(texts[index]) or other < index)
                                        for other in range(len(regions)))]


def build_image_parts_blocks(runner, block, page, region, source, parts, evidence_ids, folder,
                             *, relationship_reading=None, relationship_parents=(), keep_relationship_evidence=False):
    """Share scene/text/gap materialization between image and native-page recovery."""
    replacements = []
    prior = (relationship_reading if relationship_reading is not None else runner.owner.state.get(
        "pages", {}).get(str(page), {}).get("word_image_readings", {}).get(block.block_id, {}))
    relationship_evidence = None
    if "relationships_complete" in prior and (prior.get("relationships") or not prior["relationships_complete"]
                                              or keep_relationship_evidence):
        from app.pipeline.image_source_review import WordPictureItem
        prior = WordPictureItem.model_validate(prior).model_dump()
        relationship_evidence = runner.evidence.add(
            provider="geometry_image_repair" if relationship_reading is not None else "blind_word_image_reading",
            kind="image_relationship_reading", page=page, block_id=block.block_id, source=source, content=prior,
            parents=relationship_parents)
        evidence_ids = list(dict.fromkeys([*evidence_ids, relationship_evidence]))
    common = {**runner.source_payload(page), "agentic_source_region": region,
              "agentic_source_image": str(source.relative_to(runner.root)),
              "agentic_recovery_tool": "repair_image", "agentic_image_origin": block.block_id,
              "image_source_context": block.payload.get("word_source", {}),
              "agentic_evidence_ids": evidence_ids, **picture_identity(block)}
    if relationship_evidence:
        common["agentic_relationship_evidence_id"] = relationship_evidence

    def add(text, kind, suffix, gap=None, image_region=None):
        payload = {**common, "text": text, "agentic_content_kind": kind}
        if image_region:
            payload["agentic_image_region"] = image_region
        if gap:
            payload["agentic_gap"] = gap
        replacements.append(Block(block_id=f"recovery_{folder.name}_image_{suffix}", type=BlockType.TEXT,
                                  page_idx=block.page_idx, bbox_norm=region, payload=payload))

    if parts.get("information_status") == "non_informative" and not parts["gaps"]:
        add("圖片內容：僅空白或裝飾性圖形，沒有可提取的實質資訊。", "visual_summary", "information")
        replacements[-1].payload["agentic_information_status"] = "non_informative"
    # Model text gets the zh-TW repair enrichment text gets in render_vlm_text;
    # a repaired photo otherwise shipped 「画面…」/「晴诺米…」 labelled verified.
    from app.pipeline.zh_text import to_taiwan_traditional

    for index, text in enumerate(parts["visual_facts"] if parts.get("information_status") != "non_informative" else []):
        add("圖片可確認內容：" + to_taiwan_traditional(text), "visual_summary", f"v{index}")
    for index, item in enumerate(distinct_text_regions(parts["text_regions"])):
        add("圖片文字：" + to_taiwan_traditional(item["text"]), "transcription", f"t{index}", image_region=item["bbox"])
        if item.get("carried"):
            replacements[-1].payload["agentic_drafted_line"] = True
    gaps = parts["gaps"]
    if not replacements:
        # One whole-image gap avoids nested placeholders, but must not turn a
        # protocol/budget deferral into a claim that the source pixels are illegible.
        reasons = sorted({gap["reason"] for gap in gaps})
        reason = next((reason for reason in reasons if reason in PENDING_GAP_REASONS),
                      "image_essential_content_unverified")
        gaps = [{"bbox": [0, 0, 1000, 1000], "reason": reason, "source_reasons": reasons}]
    if gaps:
        regions = [{"reason": gap["reason"], "image_bbox": gap["bbox"],
                    **({"source_reasons": gap["source_reasons"]} if "source_reasons" in gap else {})} for gap in gaps]
        gap = picture_gap(block.block_id, regions)
        add(gap_marker(page, regions), "transcription", "g0", gap=gap,
            image_region=gap["image_bbox"])
    if parts["text_dependent"]:
        for item in replacements:
            item.payload["agentic_depends_on_block_ids"] = [other.block_id for other in replacements
                                                           if other.block_id != item.block_id]
    return replacements


async def validate_image_parts(runner, candidate, pages, folder, origin, *, allow_pruning=True):
    """Verify one replacement image without requiring unrelated bad images to pass."""
    from app.pipeline.document_review import PageReview, target_block_reviews
    from app.pipeline.stages.chunk import ChunkStage

    old = picture_block(runner, runner.inventory.ir, origin)
    selected = [b for b in candidate.blocks if b.payload.get("agentic_image_origin") == origin]
    if not old or old.type != BlockType.IMAGE or not selected or pages != runner.inventory.mapping.get(origin):
        raise ValueError("Image candidate does not match its source scope")
    # The picture is its block, or for a repaired one the blocks it is delivered as now.
    outside = [b for b in runner.inventory.ir.blocks
               if b.block_id != origin and b.payload.get("agentic_image_origin") != origin]
    before = {b.block_id: b.model_dump(exclude={"reading_order"}) for b in outside}
    after = {b.block_id: b.model_dump(exclude={"reading_order"}) for b in candidate.blocks if b not in selected}
    remaining_order = [b.block_id for b in candidate.blocks if b not in selected]
    if (before != after or remaining_order != [b.block_id for b in outside]
            or len({b.block_id for b in candidate.blocks}) != len(candidate.blocks)):
        raise ValueError("Image candidate changed unrelated source content/order or duplicated block IDs")
    paths = {b.payload.get("agentic_source_image") for b in selected}
    if len(paths) != 1 or None in paths:
        raise ValueError("Image candidate lacks one common source image")
    image = folder / paths.pop()
    if not image.is_file():
        raise ValueError("Candidate original image evidence missing")
    renderer = ChunkStage(runner.owner.config)
    renderer._asset_text = renderer._load_asset_retrieval_text(folder / "outputs")
    docs, chunks = image_delivery_excerpts(selected, folder, renderer,
                                          chunks_required=runner.owner.config.package.generate_chunks)
    payload = {"block_id": origin, "required_block_ids": [b.block_id for b in selected],
               "blocks": [{"block_id": b.block_id, "kind": b.payload.get("agentic_content_kind"),
                           "image_region": b.payload.get("agentic_image_region"),
                           "information_status": b.payload.get("agentic_information_status", "uncertain"),
                           "untrusted_extracted_text": renderer._block_to_text(b)} for b in selected],
               "declared_gaps": [{"block_id": b.block_id, **b.payload["agentic_gap"]}
                                 for b in selected if b.payload.get("agentic_gap")],
               "delivered_documents": docs, "delivered_chunks": chunks,
               "chunks_required": runner.owner.config.package.generate_chunks}
    from app.pipeline.image_source_review import (
        RELATION_AUDIT_INSTRUCTIONS,
        apply_group_relationship_review,
        inherited_image_readings,
    )
    relation_readings = inherited_image_readings(selected, folder)
    prior = runner.owner.state.get("pages", {}).get(str(pages[0]), {}).get("word_image_readings", {}).get(origin, {})
    if ("relationships_complete" in prior and (prior.get("relationships") or not prior["relationships_complete"])
            and origin not in relation_readings):
        raise ValueError("Image candidate dropped its existing source relationship evidence")
    if relation_readings:
        payload["independent_image_readings"] = {key: value.model_dump() for key, value in relation_readings.items()}
        payload["image_reading_scopes"] = {origin: payload["required_block_ids"]}
        payload["image_relationship_policy"] = RELATION_AUDIT_INSTRUCTIONS
    from app.pipeline.image_evidence import image_detail_views, source_region_views
    refs = sorted({eid for b in selected for eid in b.payload.get("agentic_evidence_ids", [])})
    views, regions = image_detail_views(runner.evidence, image, refs)
    # Show every one of the four permitted local revision targets; OCR context
    # must not consume the old three-crop limit and hide later targets.
    target_regions = [b.payload["agentic_image_region"] for b in selected if b.payload.get("agentic_image_region")]
    views, regions = source_region_views(image, target_regions + regions, max_regions=6)
    payload.update(source_detail_regions=regions, image_order=["original image"] + ["source crop" for _ in regions],
        response_instruction="Return every required block ID exactly once. Keep each block observation under "
                             "300 characters and the overall observation under 600 characters; use concise "
                             "source findings instead of repeating the candidate's full transcription.")
    from app.pipeline.review_policy import SOURCE_COMPLETENESS_POLICY
    payload["source_completeness_policy"] = SOURCE_COMPLETENESS_POLICY
    if len(str(payload)) > 42000:
        raise ValueError("Image candidate exceeds verification context limit")
    review = await read_contract(runner.owner, "audit_image_parts", payload, views, PageReview,
                                required_block_ids=payload["required_block_ids"], max_attempts=2)
    review = target_block_reviews(review, payload["required_block_ids"])
    apply_group_relationship_review(review, relation_readings, {origin: payload["required_block_ids"]},
        {b.block_id: renderer._block_to_text(b) for b in selected}, documents=docs,
        chunks=chunks if payload["chunks_required"] else None)
    atomic_write_json(folder / "validation-image.json", {
        "scope": "one original image only; not page certification", "source_block_id": origin,
        "source_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "source_views": [{"path": str(p.relative_to(folder)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in views],
        "source_detail_regions": regions, "review": review.model_dump(),
    }, ensure_ascii=False, indent=2)
    findings = {b.block_id: b for b in review.blocks}
    for block in selected:
        replace_image_review_dependencies(block, findings[block.block_id])
    accepted = image_review_accepted(selected, findings, review)
    if any(b.payload.get("agentic_image_repair_method") in {"axis_geometry", *LOCAL_EDIT_METHODS} for b in selected):
        # One measured proposal / one verbatim patch or edit: no fallback to guessing or pruning content into gaps.
        return accepted
    if accepted and not any(b.payload.get("agentic_gap") for b in selected):
        return True
    if not allow_pruning:
        return accepted
    from app.pipeline.agentic import BudgetExhausted
    from app.pipeline.image_reading import ReadingFormatError
    from app.pipeline.image_refinement import revise_image_regions, revision_budget
    try:
        revised = await revise_image_regions(runner, candidate, selected, review, image, folder, origin, pages[0])
        if revised is not None and revision_budget(runner.owner, propose=False):
            if await validate_image_variant(runner, candidate, revised, pages, folder, origin, "region-revision"):
                return True
    except (ReadingFormatError, BudgetExhausted, TimeoutError) as exc:
        atomic_write_json(folder / "image-region-revision-error.json", {"reason": str(exc)[:1000]},
                          ensure_ascii=False, indent=2)
    if accepted:
        return True  # Failed local improvement must preserve the already audited partial result.
    # A failed proposal never replaces the original candidate. Independently check
    # the conservative subset, including the exact omission markers in delivery.
    subset = candidate.model_copy(deep=True)
    if revision_budget(runner.owner, propose=False) and prune_candidate(subset, findings, review, origin, pages[0]):
        return await validate_image_variant(runner, candidate, subset, pages, folder, origin, "verified-subset")
    return False


def replace_image_review_dependencies(block, finding):
    """Source requirements persist; a new source audit replaces its own old findings.

    Appending every review forever made corrected independent scene facts depend
    on a gap rejected by a previous review. Never clear real chart/label links.
    """
    payload = block.payload
    source = payload.setdefault("agentic_source_depends_on_block_ids", sorted(
        set(payload.get("agentic_depends_on_block_ids", []))
        - set(payload.get("agentic_review_depends_on_block_ids", []))))
    current = sorted(set(finding.depends_on_block_ids))
    payload["agentic_review_depends_on_block_ids"] = current
    payload["agentic_depends_on_block_ids"] = sorted(set(source) | set(current))


def image_review_accepted(blocks, findings, review):
    """A repaired picture passes when its audit finds nothing wrong in what the repair delivered.

    Decided 2026-10-13: an audit had to find nothing at all, so a relationship left
    unstated, illegible print, or an older fault a local edit never touched failed a
    repair whose own change was right. Now remarks that only miss a relationship or meet
    illegible print do not count, and a local edit fails only on a remark quoting the
    wording it brought in. Declared gaps stay gaps; no readable part may depend on one.
    """
    from app.pipeline.review_policy import partial_review_accepted

    gaps = [b.block_id for b in blocks if b.payload.get("agentic_gap")]
    readable = {b.block_id for b in blocks if not b.payload.get("agentic_gap")}
    dependencies = {b.block_id: set(b.payload.get("agentic_depends_on_block_ids", [])) for b in blocks}
    if not review.reading_order_correct or dependency_closed_subset(readable, dependencies) != readable:
        return False
    if gaps and partial_review_accepted(review.model_dump(), gaps, [b.block_id for b in blocks]):
        return True
    return bool(readable) and not repaired_picture_problems(review.model_dump(), blocks)


def repaired_picture_problems(review, blocks, *, unattributed=True):
    """Hard remarks a review makes about a repaired picture's readable parts; for the improving edit,
    only those quoting the wording it brought in (an edit that only deletes is judged in full)."""
    from app.pipeline.review_policy import picture_problems

    readable = {b.block_id for b in blocks if not b.payload.get("agentic_gap")}
    improving = any(b.payload.get("agentic_image_repair_method") in IMPROVING_METHODS
                    or b.payload.get("agentic_improves_repair") for b in blocks)
    wording = [w for b in blocks for w in b.payload.get("agentic_new_wording", [])] if improving else []
    return picture_problems(review, readable, new_wording=wording, unattributed=unattributed)


async def validate_image_variant(runner, candidate, revised, pages, folder, origin, name):
    child = folder / name
    if not await runner.validate(revised, pages, child, image_origin=origin, image_pruned=True):
        return False
    # Publish only artifacts actually checked in this isolated candidate folder.
    shutil.copy2(folder / "validation-image.json", folder / ("validation-image-before-" + name + ".json"))
    shutil.rmtree(folder / "outputs")
    shutil.copytree(child / "outputs", folder / "outputs")
    shutil.copytree(child / "agentic/evidence", folder / "agentic/evidence", dirs_exist_ok=True)
    shutil.copy2(child / "validation-image.json", folder / "validation-image.json")
    candidate.blocks = revised.blocks
    return True


def image_delivery_excerpts(blocks, folder, renderer, *, chunks_required):
    """Return actual delivered spans for the target image, not adjacent shared paragraphs.

    These candidates are plain TEXT blocks. Exact spans must exist on each required
    output surface; IR text alone is never substituted for missing delivery.
    """
    from app.pipeline.final_delivery import delivered_documents, read_jsonl
    documents = [(p.name, p.read_text()) for p in delivered_documents(folder / "outputs")]
    chunks = read_jsonl(folder / "outputs/chunks.jsonl")
    docs, excerpts = [], []
    for block in blocks:
        text = renderer._block_to_text(block).strip()
        if not text:
            raise ValueError("Empty image candidate has no deliverable span")
        found_doc, found_chunk = False, False
        for filename, body in documents:
            start = body.find(text)
            if start >= 0:
                docs.append({"block_id": block.block_id, "file": filename, "start": start,
                             "text": body[start:start + len(text)]})
                found_doc = True
                break
        for chunk in chunks:
            body = chunk.get("content", "")
            start = body.find(text)
            if block.block_id in chunk.get("block_ids", []) and start >= 0:
                excerpts.append({"block_id": block.block_id, "chunk_id": chunk.get("chunk_id"),
                                 "start": start, "content": body[start:start + len(text)]})
                found_chunk = True
                break
        if not found_doc or chunks_required and not found_chunk:
            raise ValueError("Image candidate is missing an exact span in its actual documents/chunks")
    return docs, excerpts


def prune_unverified_parts(blocks, findings, review, origin, page):
    """Keep a verified independent subset; convert rejected parts to explicit gaps."""
    if not review.reading_order_correct:
        return False
    from app.pipeline.review_policy import soft_review_remark

    readable = [b for b in blocks if not b.payload.get("agentic_gap")]
    good = {b.block_id for b in readable if findings[b.block_id].status == "verified"
            or soft_review_remark(findings[b.block_id].observation)}
    dependencies = {b.block_id: set(findings[b.block_id].depends_on_block_ids)
                    | set(b.payload.get("agentic_depends_on_block_ids", [])) for b in readable}
    good = dependency_closed_subset(good, dependencies)
    rejected = [b for b in blocks if b.block_id not in good]
    if not good or not rejected:
        return False
    for block in readable:
        block.payload["agentic_depends_on_block_ids"] = sorted(
            set(block.payload.get("agentic_depends_on_block_ids", []))
            | set(findings[block.block_id].depends_on_block_ids))
    from app.pipeline.image_refinement import mark_region_omitted
    for block in rejected:
        mark_region_omitted(block, origin, page)
    return True


def prune_candidate(candidate, findings, review, origin, page):
    """prune_unverified_parts on a candidate, its gap markers folded into one."""
    blocks = [b for b in candidate.blocks if b.payload.get("agentic_image_origin") == origin]
    if not prune_unverified_parts(blocks, findings, review, origin, page):
        return False
    candidate.blocks = merge_picture_gaps(candidate.blocks, origin, page)
    for order, item in enumerate(candidate.blocks):
        item.reading_order = order
    return True


def image_read_requests(config, ir, skill, model_role="primary"):
    """Execution and budget estimation must use the same request identity."""
    from app.pipeline.semantic.language import resolve_semantic_output_language
    language = resolve_semantic_output_language(config.package.semantic_output_language.value, ir)
    return [("read_image_parts_alternate" if model_role == "review" else "read_image_parts",
             {"skill": skill, "source_id": "original-image", "output_language": language}),
            ("verify_image_parts", {"source_id": "original-image", "output_language": language})]
