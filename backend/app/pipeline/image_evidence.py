"""Adjudicate image observations against source pixels, then require ordinary delivery audit."""
from app.adapters.vlm import VLMAdapter
from app.pipeline.content_coverage import normalized_text
from app.pipeline.delivery import atomic_write_json
from app.pipeline.image_reading import (
    ReadingFormatError,
    image_validation_tasks,
    overlap,
    read_contract,
    reading_request_key,
)
from app.pipeline.recovery_models import ImageEvidenceBatchResolution, ImageEvidenceResolution


async def resolve_image_evidence(runner, block, page, source, primary, independent, prior, parts, folder, *, model_role="primary"):
    store = runner.evidence
    candidates, evidence_ids = [], []
    for provider, reading, config in [("vlm_primary", primary, runner.owner.config.review_vlm if model_role == "review" else runner.owner.config.vlm),
                                      ("vlm_independent", independent, runner.owner.config.review_vlm)]:
        eid = store.add(provider=provider, kind="image_reading", page=page, block_id=block.block_id,
                        source=source, content=reading.model_dump(),
                        metadata={"model_config_sha256": VLMAdapter.compute_config_hash(config)})
        evidence_ids.append(eid)
        candidates.extend({"source_ref": eid + ":" + r.region_id, "evidence_id": eid, "text": r.text,
                           "bbox": r.bbox, "provider": provider, "readable": r.readable} for r in reading.text_regions)
    if prior:
        eid = store.add(provider="prior_vlm", kind="image_reading", page=page, block_id=block.block_id,
                        source=source, content=prior)
        evidence_ids.append(eid)
    for observation in one_ocr_observation(store.for_target(page, block.block_id, provider="paddleocr"),
                                           evidence_source_sha(source)):
        evidence_ids.append(observation["evidence_id"])
        candidates.extend({"source_ref": observation["evidence_id"] + ":" + item["line_id"],
                           "evidence_id": observation["evidence_id"], "text": item["text"],
                           "bbox": item["bbox"], "provider": "paddleocr", "score": item["score"]}
                          for item in observation["content"]["lines"] if item["text"].strip())
    # Existing agreement is still only a candidate and still goes through source audit.
    # A conflict gets another source-grounded decision; no automatic majority vote.
    if not candidates or (not parts["gaps"] and not any(c["provider"] == "paddleocr" for c in candidates)):
        return parts, evidence_ids
    if len(candidates) > 128 or sum(len(c["text"]) for c in candidates) > 18000:
        raise ReadingFormatError("image_evidence_capacity_exceeded; select a smaller OCR/source region")
    resolution, protocol_gaps, batches = await resolve_image_evidence_bounded(
        runner.owner, block.block_id, source, candidates)
    ids = [s.source_ref for s in resolution.selections]
    available = {c["source_ref"]: c for c in candidates}
    if len(ids) != len(set(ids)) or set(ids) != set(available):
        raise ReadingFormatError("image_resolution_missing_duplicated_or_invented_evidence")
    selected = [available[s.source_ref] for s in resolution.selections if s.decision == "supported"]
    if any(not c["text"].strip() for c in selected):
        raise ReadingFormatError("image_resolution_selected_empty_transcription")
    supported = {c["source_ref"] for c in selected}
    for selection in resolution.selections:
        if selection.covered_by and (selection.decision == "supported" or selection.covered_by not in supported
                or not overlap(available[selection.source_ref]["bbox"], available[selection.covered_by]["bbox"])):
            raise ReadingFormatError("image_resolution_invalid_region_link")
    decisions = {s.source_ref: s for s in resolution.selections}
    retained, disputed = select_agreeing_readings(selected)
    # Replace only gaps for source regions that have now been directly resolved.
    gaps = protocol_gaps + [g for g in parts["gaps"] if g["reason"] not in {
        "image_text_unreadable_or_disputed", "image_text_not_accounted_for_by_both_readers",
        "image_text_conflicts_with_prior_source_reading", "image_visual_depends_on_unverified_content"}]
    gaps += [{"bbox": bbox, "reason": "image_text_readings_conflict"} for bbox in disputed]
    regions = [c for c in candidates if c["provider"] != "paddleocr"]
    for region in regions:
        decision = decisions[region["source_ref"]]
        if decision.decision == "context_only" or decision.covered_by:
            continue
        if not any(overlap(region["bbox"], item["bbox"]) >= 0.5 for item in retained):
            if not any(overlap(region["bbox"], gap["bbox"]) >= 0.5 for gap in gaps):
                gaps.append({"bbox": region["bbox"], "reason": "image_text_unresolved_after_evidence_review"})
    for selection in resolution.selections:
        item = available[selection.source_ref]
        if selection.decision == "uncertain" and not selection.covered_by and not any(overlap(item["bbox"], s["bbox"]) >= 0.5 for s in retained):
            if not any(overlap(item["bbox"], gap["bbox"]) >= 0.5 for gap in gaps):
                gaps.append({"bbox": item["bbox"], "reason": "image_text_unresolved_after_evidence_review"})
    if not resolution.accounted_for and not protocol_gaps:
        gaps.append({"bbox": [0, 0, 1000, 1000], "reason": "image_evidence_coverage_incomplete"})
    text_dependent = parts["text_dependent"]
    # Readings both readers and the adjudication support stay beside the gaps of the regions that
    # are not: a chart or table used to lose every reading over one disputed cell, and its re-read
    # became a placeholder its audit always failed (rerun 2026-10-15, 3-6 minutes each). The pieces
    # of a text-dependent picture depend on one another, gaps included, so none is certified alone.
    text = [{"text": item["text"], "bbox": item["bbox"], "evidence_id": item["evidence_id"]} for item in retained]
    visual = parts["visual_facts"]
    if not gaps and primary.visual_facts and independent.visual_facts:
        visual = primary.visual_facts
    elif text_dependent:
        visual = []
    evidence = store.add(provider="evidence_adjudicator", kind="candidate_selection", page=page,
        block_id=block.block_id, source=source, content=resolution.model_dump(), parents=evidence_ids,
        metadata={"model_config_sha256": VLMAdapter.compute_config_hash(runner.owner.config.vlm),
                  "batches": batches})
    evidence_ids.append(evidence)
    result = {**parts, "text_regions": text, "visual_facts": visual, "gaps": gaps, "partial": bool(gaps)}
    atomic_write_json(folder / "evidence-resolution.json",
        {"resolution": resolution.model_dump(), "candidate": result, "evidence_ids": evidence_ids,
         "batches": batches,
         "status": "candidate_requires_source_audit"}, ensure_ascii=False, indent=2)
    return result, evidence_ids



def one_ocr_observation(observations, source_sha):
    """One PaddleOCR observation of these pixels: the widest region, at its largest scale.

    A planned OCR step and the repair's own OCR both read a picture, at scale 1 and 2; counting
    both doubled every line, so a 64-line chart came to 155 candidates and was refused as over
    capacity (rerun 2026-10-15). Evidence for another revision of the source never votes.
    """
    def area(observation):
        region = observation["content"].get("region") or observation.get("bbox") or [0, 0, 1000, 1000]
        return (region[2] - region[0]) * (region[3] - region[1])

    current = [o for o in observations if o["source"]["sha256"] == source_sha]
    if not current:
        return []
    return [max(current, key=lambda o: (area(o), o["content"].get("scale", 1), len(o["content"].get("lines", []))))]


def _folded(text):
    from app.pipeline.zh_text import to_taiwan_traditional
    return normalized_text(to_taiwan_traditional(text))


def select_agreeing_readings(selected):
    """The supported readings to deliver, and the regions where two of them disagree.

    Readings of one region agree when, in either script, they are equal or one holds the other;
    the longer is kept, so OCR's fragment 「No.」 does not cut the 「1」 off 「No.1」 and its
    simplified 「输」 does not count against 「輪」. A real disagreement holds back that region
    only, as a gap; it used to fail the whole picture. Rerun 2026-10-15: 11 re-reads (49 min)
    failed so, on 31 pairs of which 24 were containment and 4 only script.
    """
    retained, disputed = [], []
    for item in selected:
        if any(overlap(item["bbox"], bbox) >= 0.5 for bbox in disputed):
            continue
        for index, other in enumerate(retained):
            if overlap(item["bbox"], other["bbox"]) >= 0.5:
                mine, theirs = _folded(item["text"]), _folded(other["text"])
                if mine in theirs:
                    pass
                elif theirs in mine:
                    retained[index] = item
                else:
                    retained.pop(index)
                    disputed.append([min(item["bbox"][0], other["bbox"][0]), min(item["bbox"][1], other["bbox"][1]),
                                     max(item["bbox"][2], other["bbox"][2]), max(item["bbox"][3], other["bbox"][3])])
                break
        else:
            retained.append(item)
    return retained, disputed


IMAGE_RESOLUTION_BATCH_SIZE = 24
MAX_IMAGE_RESOLUTION_CALLS = 6


def estimate_image_resolution_calls(candidate_count):
    """Ordinary batches; admission additionally reserves the six-call retry cap."""
    return min(MAX_IMAGE_RESOLUTION_CALLS,
               (candidate_count + IMAGE_RESOLUTION_BATCH_SIZE - 1) // IMAGE_RESOLUTION_BATCH_SIZE)


def candidate_bounds(candidates):
    return [min(c["bbox"][0] for c in candidates), min(c["bbox"][1] for c in candidates),
            max(c["bbox"][2] for c in candidates), max(c["bbox"][3] for c in candidates)]


async def resolve_image_evidence_bounded(owner, source_id, source, candidates):
    """Source-scoped short aliases, regional batches and a shared six-call ceiling.

    Protocol/budget failures remain unexamined regions, never evidence that pixels
    are illegible. Every candidate has exactly one decision or an explicit deferral.
    The ordinary delivery audit still judges the assembled candidate against source.
    """
    from app.pipeline.agentic import BudgetExhausted
    from app.pipeline.review_budget import call_seconds

    ordered = sorted(candidates, key=lambda c: (c["bbox"][1], c["bbox"][0], c["bbox"][3], c["bbox"][2], c["source_ref"]))
    aliases = {c["source_ref"]: f"c{i:03d}" for i, c in enumerate(ordered)}
    full_refs = {alias: ref for ref, alias in aliases.items()}
    by_ref = {c["source_ref"]: c for c in ordered}
    pending, current = [], []
    for item in ordered:
        # Input and output capacity both matter. No transcription is truncated.
        if current and (len(current) == IMAGE_RESOLUTION_BATCH_SIZE or
                        sum(len(c["text"]) for c in current) + len(item["text"]) > 6000):
            pending.append(current)
            current = []
        current.append(item)
    if current:
        pending.append(current)
    selections, gaps, batches, errors = [], [], [], []
    calls, completed = 0, 0

    def defer(batch, reason):
        gaps.append({"bbox": candidate_bounds(batch), "reason": reason})
        selections.extend({"source_ref": c["source_ref"], "decision": "uncertain",
                           "reason": reason} for c in batch)
        batches.append({"source_refs": [c["source_ref"] for c in batch], "status": reason})

    while pending:
        batch = pending.pop(0)
        bounds = candidate_bounds(batch)
        views, regions = source_region_views(source, [bounds])
        refs = [aliases[c["source_ref"]] for c in batch]
        # A long line and its detector fragments can straddle batch boundaries.
        # Supply only overlapping prior selections as fallible source context,
        # never repeat their IDs in required responses or treat them as certified.
        context = [by_ref[s["source_ref"]] for s in selections if s["decision"] == "supported"
                   and any(overlap(by_ref[s["source_ref"]]["bbox"], c["bbox"]) for c in batch)][:8]
        context_refs = [aliases[c["source_ref"]] for c in context]
        payload = {"source_id": source_id, "coordinate_system": "0..1000 original image",
            "source_detail_regions": regions, "image_order": ["original image"] + ["original-source crop" for _ in regions],
            "response_scope": "Only the listed candidate aliases and their visible source regions are in scope. "
                "Return one decision per alias. Accounted_for applies to these regions only. "
                "Do not use unlisted candidates or neighboring lettering. A covered_by link must point to a "
                "supported alias in this response or one supplied context_candidate for the SAME visible text, "
                "not merely an overlapping rectangle. Context candidates are fallible prior selections; "
                "check their pixels again before using a link. Return no selection rows for context_candidates.",
            "context_candidates": [{"source_ref": aliases[c["source_ref"]], "text": c["text"],
                                    "bbox": c["bbox"]} for c in context],
            "candidates": [{**{k: v for k, v in c.items() if k not in {"source_ref", "evidence_id"}},
                            "source_ref": aliases[c["source_ref"]]} for c in batch],
            "required_source_refs": refs,
            "instruction": "Judge source pixels, never votes, scores or prior captions. Return supported only for "
                "literal visible text. Do not support empty text. Use context_only only for a false text hypothesis "
                "or non-text noise, never unreadable real text. Do not create text or concatenate candidates. "
                "Keep reasons short (maximum 120 characters)."}
        key = reading_request_key(owner, "resolve_image_evidence", payload, views, ImageEvidenceBatchResolution)
        spent = owner.state.get("source_reading_attempts", {}).get(key, {}).get("calls", 0)
        cached = key in owner.state.get("source_reading_cache", {}) or spent >= 1
        available_calls, available_seconds = owner.available_budget()
        audit_tasks = image_validation_tasks(owner)
        reserve = sum(call_seconds(owner.state, t) for t in audit_tasks)
        if not cached and (calls >= MAX_IMAGE_RESOLUTION_CALLS or available_calls < len(audit_tasks) + 1 or
                available_seconds < reserve + call_seconds(owner.state, "resolve_image_evidence")):
            defer(batch, "image_evidence_budget_deferred")
            continue
        before = owner.state.get("usage", {}).get("model_calls", 0)
        try:
            result = await read_contract(owner, "resolve_image_evidence", payload, views,
                ImageEvidenceBatchResolution, max_attempts=1, required_source_refs=refs)
        except ReadingFormatError as exc:
            errors.append(str(exc))
            batches.append({"source_refs": [c["source_ref"] for c in batch],
                            "status": "response_contract_failed", "error": str(exc)[:600]})
            if len(batch) > 1:
                middle = len(batch) // 2
                pending[:0] = [batch[:middle], batch[middle:]]
            else:
                defer(batch, "image_evidence_response_contract_failed")
            continue
        except (BudgetExhausted, TimeoutError):
            defer(batch, "image_evidence_budget_deferred")
            continue
        finally:
            # Resuming a failed repair cannot reset its six-call ceiling. Cached
            # parent failures still split so already completed child reads survive.
            calls += spent + owner.state.get("usage", {}).get("model_calls", 0) - before
        completed += 1
        for item in result.selections:
            if item.covered_by and item.covered_by not in refs + context_refs:
                raise ReadingFormatError("image_resolution_invalid_region_link")
            selections.append({**item.model_dump(), "source_ref": full_refs[item.source_ref],
                               "covered_by": full_refs.get(item.covered_by)})
        if not result.accounted_for:
            gaps.append({"bbox": bounds, "reason": "image_evidence_coverage_incomplete"})
        batches.append({"source_refs": [c["source_ref"] for c in batch], "aliases": refs,
            "context_source_refs": [c["source_ref"] for c in context], "status": "candidate_requires_source_audit", "resolution": result.model_dump(),
            "source_detail_regions": regions, "view_sha256": [evidence_source_sha(p) for p in views]})
    if not completed and errors:
        raise ReadingFormatError(errors[-1])
    resolution = ImageEvidenceResolution(selections=selections, accounted_for=not gaps,
        observation="Bounded source-region evidence selections; deferred regions remain explicit gaps; not certified.")
    return resolution, gaps, batches


def evidence_source_sha(source):
    import hashlib
    return hashlib.sha256(source.read_bytes()).hexdigest()


def image_detail_views(store, source, evidence_ids):
    """Up to two bounded original-pixel crops, shared by adjudication and final audit."""
    sha = evidence_source_sha(source)
    regions = []
    for eid in evidence_ids:
        record = store.get(eid)
        if record["provider"] != "paddleocr" or record["source"]["sha256"] != sha:
            continue
        lines = record["content"].get("lines", [])
        if not lines:
            continue
        region = [max(0, min(line["bbox"][0] for line in lines) - 40),
                  max(0, min(line["bbox"][1] for line in lines) - 40),
                  min(1000, max(line["bbox"][2] for line in lines) + 40),
                  min(1000, max(line["bbox"][3] for line in lines) + 40)]
        if region not in regions:
            regions.append(region)
    return source_region_views(source, regions[:2])


def source_region_views(source, regions, *, max_regions=3):
    from PIL import Image
    sha = evidence_source_sha(source)
    regions = [list(r) for r in dict.fromkeys(tuple(r) for r in regions) if list(r) != [0, 0, 1000, 1000]][:max_regions]
    if any(len(r) != 4 or not 0 <= r[0] < r[2] <= 1000 or not 0 <= r[1] < r[3] <= 1000 for r in regions):
        raise ValueError("Invalid original-image detail region")
    paths = [source]
    for region in regions:
        with Image.open(source) as picture:
            width, height = picture.size
            box = (region[0] * width // 1000, region[1] * height // 1000,
                   (region[2] * width + 999) // 1000, (region[3] * height + 999) // 1000)
            crop = picture.crop(box)
            scale = min(4, 2000 / max(crop.size))
            path = source.parent / ("detail-" + sha[:16] + "-" + "-".join(map(str, region)) + ".png")
            crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                        Image.Resampling.LANCZOS).save(path)
            paths.append(path)
    return paths, regions
