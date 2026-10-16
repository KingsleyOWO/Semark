"""One source-bound local proposal after image audit, never self-certification."""
from app.adapters.vlm import VLMAdapter
from app.pipeline.delivery import atomic_write_json
from app.pipeline.image_evidence import source_region_views
from app.pipeline.image_reading import (
    ReadingFormatError,
    image_validation_tasks,
    overlap,
    read_contract,
)
from app.pipeline.recovery_models import ImageRegionPatch
from app.pipeline.review_budget import call_seconds
from app.pipeline.semantic.language import resolve_semantic_output_language


def revision_budget(owner, *, propose=True):
    tasks = (["revise_image_regions"] * 2 if propose else []) + image_validation_tasks(owner)
    calls, seconds = owner.available_budget()
    return calls >= len(tasks) and seconds >= sum(call_seconds(owner.state, t) for t in tasks)


def mark_region_omitted(block, origin, page):
    """State an omission without asserting the source pixels are illegible."""
    from app.pipeline.image_recovery import gap_marker, picture_gap

    regions = [{"reason": "region_not_interpreted",
                "image_bbox": block.payload.get("agentic_image_region", [0, 0, 1000, 1000])}]
    block.payload.update(text=gap_marker(page, regions),
                         agentic_information_status="uncertain", agentic_region_kind="uncertain")
    block.payload["agentic_gap"] = picture_gap(origin, regions)


async def revise_image_regions(runner, candidate, selected, review, source, folder, origin, page):
    findings = {b.block_id: b for b in review.blocks}
    targets = [b for b in selected if b.payload.get("agentic_gap") or
               findings[b.block_id].status != "verified" or not findings[b.block_id].context_complete]
    if not targets or len(targets) > 4 or not revision_budget(runner.owner):
        return None
    specs = []
    for block in targets:
        region = block.payload.get("agentic_image_region", [0, 0, 1000, 1000])
        eid = runner.evidence.add(provider="source", kind="image_region", page=page, block_id=origin,
                                  source=source, bbox=region, content={"target_block_id": block.block_id})
        specs.append({"block_id": block.block_id, "bbox": region, "source_evidence_id": eid,
                      "untrusted_candidate": block.get_text(),
                      "audit_observation": findings[block.block_id].observation})
    views, regions = source_region_views(source, [s["bbox"] for s in specs], max_regions=4)
    language = resolve_semantic_output_language(runner.owner.config.package.semantic_output_language.value, candidate)
    proposal = await read_contract(runner.owner, "revise_image_regions", {
        "source_id": origin, "targets": specs, "output_language": language,
        "coordinate_system": "0..1000 original image", "source_detail_regions": regions,
        "image_order": ["original image"] + ["original-source crop" for _ in regions],
        "audit_issues": review.issues[:4],
        "retained_context": [{"block_id": b.block_id, "text": b.get_text()[:600],
                              "bbox": b.payload.get("agentic_image_region", [0, 0, 1000, 1000])} for b in selected
                              if b not in targets][:24],
    }, views, ImageRegionPatch, max_attempts=2)
    ids = [item.block_id for item in proposal.regions]
    allowed = {s["block_id"]: s["source_evidence_id"] for s in specs}
    if len(ids) != len(set(ids)) or set(ids) != set(allowed):
        raise ReadingFormatError("image_region_patch_outside_target_scope")
    if any(set(item.evidence_ids) != {allowed[item.block_id]} for item in proposal.regions):
        raise ReadingFormatError("image_region_patch_unowned_source_evidence")
    retained = {b.block_id: b for b in selected if b not in targets}
    target_boxes = {s["block_id"]: s["bbox"] for s in specs}
    covered = {}
    for item in proposal.regions:
        if not item.covered_by:
            continue
        other = retained.get(item.covered_by)
        if (other is None or not overlap(target_boxes[item.block_id],
                other.payload.get("agentic_image_region", [0, 0, 1000, 1000]))):
            raise ReadingFormatError("image_region_patch_invalid_coverage_link")
        covered[item.block_id] = item.covered_by
    revised = candidate.model_copy(deep=True)
    for item in proposal.regions:
        block = revised.get_block(item.block_id)
        block.payload.pop("agentic_gap", None)
        if item.covered_by:
            pass  # Only the independently audited variant may remove this duplicate.
        elif item.region_kind == "uncertain":
            mark_region_omitted(block, origin, page)
        else:
            from app.pipeline.zh_text import to_taiwan_traditional

            text = ("圖片文字：" + to_taiwan_traditional(item.text) if item.text else "")
            if item.description:
                text += ("\n" if text else "") + "圖片可確認內容：" + to_taiwan_traditional(item.description)
            if item.region_kind == "no_information":
                text = "圖片區域：無可提取的實質資訊。"
            block.payload.update(text=text, agentic_region_kind=item.region_kind,
                agentic_content_kind="transcription" if item.text else "visual_summary",
                agentic_information_status="non_informative" if item.region_kind == "no_information" else "informative")
        eid = runner.evidence.add(provider="vlm_region_revision", kind="image_region_revision", page=page,
            block_id=origin, source=source, bbox=block.payload.get("agentic_image_region", [0, 0, 1000, 1000]),
            content=item.model_dump(), parents=item.evidence_ids,
            metadata={"model_config_sha256": VLMAdapter.compute_config_hash(runner.owner.config.vlm)})
        original_refs = set(block.payload.get("agentic_evidence_ids", []))
        if item.covered_by:
            block = revised.get_block(item.covered_by)
        block.payload["agentic_evidence_ids"] = sorted(original_refs | set(block.payload.get("agentic_evidence_ids", []))
                                                      | set(item.evidence_ids) | {eid})
    dependency_fields = ("agentic_depends_on_block_ids", "agentic_source_depends_on_block_ids",
                         "agentic_review_depends_on_block_ids")
    for removed, retained_id in covered.items():
        target = revised.get_block(retained_id)
        for field in dependency_fields:
            target.payload[field] = sorted(set(target.payload.get(field, [])) |
                                          set(revised.get_block(removed).payload.get(field, [])))
    revised.blocks = [b for b in revised.blocks if b.block_id not in covered]
    for block in revised.blocks:
        if block.payload.get("agentic_image_origin") == origin:
            for field in dependency_fields:
                dependencies = {covered.get(bid, bid) for bid in block.payload.get(field, [])}
                block.payload[field] = sorted(dependencies - {block.block_id})
    from app.pipeline.image_recovery import merge_picture_gaps
    revised.blocks = merge_picture_gaps(revised.blocks, origin, page)
    for order, block in enumerate(revised.blocks):
        block.reading_order = order
    atomic_write_json(folder / "image-region-patch.json", {"status": "candidate_requires_source_audit",
        "targets": specs, "proposal": proposal.model_dump(), "source_detail_regions": regions}, ensure_ascii=False, indent=2)
    return revised
