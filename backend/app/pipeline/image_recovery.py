"""Recover one picture as separately auditable visual/text/gap blocks."""
import hashlib
import shutil

from app.models.document_ir import Block, BlockType
from app.pipeline.delivery import atomic_write_json
from app.pipeline.image_reading import (
    read_contract,
    read_image_parts_bounded,
    reconcile_image_parts,
)
from app.pipeline.review_policy import dependency_closed_subset, partial_review_accepted


async def repair_image(runner, ir, problem, decision, folder, skill):
    block = ir.get_block(problem.get("block_id") or decision.block_id)
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
    atomic_write_json(folder / "image-parts.json", {
        "source_block_id": block.block_id, "source_sha256": sha,
        "primary": primary.model_dump(), "independent": independent.model_dump(),
        "prior_source_reading": prior, "candidate": parts,
    }, ensure_ascii=False, indent=2)
    return build_image_parts_candidate(runner, ir, block, page, region, source, parts, evidence_ids, folder), [page]


def build_image_parts_candidate(runner, ir, block, page, region, source, parts, evidence_ids, folder,
                                *, relationship_reading=None, relationship_parents=()):
    """Materialize source observations for the same ordinary scoped delivery audit."""
    replacements = build_image_parts_blocks(runner, block, page, region, source, parts, evidence_ids, folder,
        relationship_reading=relationship_reading, relationship_parents=relationship_parents)
    candidate = ir.model_copy(deep=True)
    index = next(i for i, old in enumerate(candidate.blocks) if old.block_id == block.block_id)
    candidate.blocks[index:index + 1] = replacements
    for order, item in enumerate(candidate.blocks):
        item.reading_order = order
    return candidate


def build_image_parts_blocks(runner, block, page, region, source, parts, evidence_ids, folder,
                             *, relationship_reading=None, relationship_parents=()):
    """Share scene/text/gap materialization between image and native-page recovery."""
    replacements = []
    prior = (relationship_reading if relationship_reading is not None else runner.owner.state.get(
        "pages", {}).get(str(page), {}).get("word_image_readings", {}).get(block.block_id, {}))
    relationship_evidence = None
    if "relationships_complete" in prior and (prior.get("relationships") or not prior["relationships_complete"]):
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
              "agentic_evidence_ids": evidence_ids}
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
    for index, text in enumerate(parts["visual_facts"] if parts.get("information_status") != "non_informative" else []):
        add("圖片可確認內容：" + text, "visual_summary", f"v{index}")
    for index, item in enumerate(parts["text_regions"]):
        add("圖片文字：" + item["text"], "transcription", f"t{index}", image_region=item["bbox"])
    gaps = parts["gaps"]
    pending_reasons = {"image_evidence_budget_deferred", "image_evidence_response_contract_failed"}
    if not replacements:
        # One whole-image gap avoids nested placeholders, but must not turn a
        # protocol/budget deferral into a claim that the source pixels are illegible.
        reasons = sorted({gap["reason"] for gap in gaps})
        reason = next((reason for reason in reasons if reason in pending_reasons),
                      "image_essential_content_unverified")
        gaps = [{"bbox": [0, 0, 1000, 1000], "reason": reason, "source_reasons": reasons}]
    for index, gap in enumerate(gaps):
        limitation = "尚未完成證據核對" if gap["reason"] in pending_reasons else "無法可靠辨識"
        add(f"〔{limitation}：原文第 {page + 1} 頁，圖片 {block.block_id}，區域 {index + 1} 的文字或必要內容〕",
            "transcription", f"g{index}", gap={"reason": gap["reason"], "source_id": block.block_id,
                                             "image_bbox": gap["bbox"],
                                             **({"source_reasons": gap["source_reasons"]} if "source_reasons" in gap else {})},
            image_region=gap["bbox"])
    if parts["text_dependent"]:
        for item in replacements:
            item.payload["agentic_depends_on_block_ids"] = [other.block_id for other in replacements
                                                           if other.block_id != item.block_id]
    return replacements


async def validate_image_parts(runner, candidate, pages, folder, origin, *, allow_pruning=True):
    """Verify one replacement image without requiring unrelated bad images to pass."""
    from app.pipeline.document_review import PageReview, target_block_reviews
    from app.pipeline.stages.chunk import ChunkStage

    old = runner.inventory.ir.get_block(origin)
    selected = [b for b in candidate.blocks if b.payload.get("agentic_image_origin") == origin]
    if not old or old.type != BlockType.IMAGE or not selected or pages != runner.inventory.mapping.get(origin):
        raise ValueError("Image candidate does not match its source scope")
    before = {b.block_id: b.model_dump(exclude={"reading_order"}) for b in runner.inventory.ir.blocks if b.block_id != origin}
    after = {b.block_id: b.model_dump(exclude={"reading_order"}) for b in candidate.blocks if b not in selected}
    remaining_order = [b.block_id for b in candidate.blocks if b not in selected]
    if (before != after or remaining_order != [b.block_id for b in runner.inventory.ir.blocks if b.block_id != origin]
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
    if any(b.payload.get("agentic_image_repair_method") == "axis_geometry" for b in selected):
        return accepted  # One measured proposal; no fallback to free-form region guessing.
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
    blocks = [b for b in subset.blocks if b.payload.get("agentic_image_origin") == origin]
    if revision_budget(runner.owner, propose=False) and prune_unverified_parts(blocks, findings, review, origin, pages[0]):
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
    gaps = [b.block_id for b in blocks if b.payload.get("agentic_gap")]
    readable = {b.block_id for b in blocks if not b.payload.get("agentic_gap")}
    good = {bid for bid in readable if findings[bid].status == "verified" and findings[bid].context_complete}
    dependencies = {b.block_id: set(b.payload.get("agentic_depends_on_block_ids", [])) for b in blocks}
    if not review.reading_order_correct or dependency_closed_subset(good, dependencies) != readable:
        return False
    if gaps:
        return partial_review_accepted(review.model_dump(), gaps, [b.block_id for b in blocks])
    return bool(readable and review.source_complete and not review.issues)


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
    readable = [b for b in blocks if not b.payload.get("agentic_gap")]
    good = {b.block_id for b in readable if findings[b.block_id].status == "verified"
            and findings[b.block_id].context_complete}
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


def image_read_requests(config, ir, skill, model_role="primary"):
    """Execution and budget estimation must use the same request identity."""
    from app.pipeline.semantic.language import resolve_semantic_output_language
    language = resolve_semantic_output_language(config.package.semantic_output_language.value, ir)
    return [("read_image_parts_alternate" if model_role == "review" else "read_image_parts",
             {"skill": skill, "source_id": "original-image", "output_language": language}),
            ("verify_image_parts", {"source_id": "original-image", "output_language": language})]
