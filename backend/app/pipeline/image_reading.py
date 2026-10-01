"""Bounded source reading contracts and conservative image-part reconciliation."""
import hashlib
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from app.adapters.vlm import VLMAdapter
from app.pipeline.agentic import REVIEW_TASKS, BudgetExhausted, ModelOutputError
from app.pipeline.content_coverage import normalized_text
from app.pipeline.recovery_models import Contract
from app.pipeline.review_sources import digest

READING_VERSION = "source-contracts-20260929.3-image-information"


def image_validation_tasks(owner):
    """Final page audit is reserved either by the transaction or by this local tool."""
    protected = getattr(owner, "_protected_budget", None) or (0, 0)
    return ["audit_image_parts"] + ([] if protected[0] else ["audit_page"])


class ReadingFormatError(ValueError):
    """The response contract failed; this is not evidence of unreadable source content."""


def reading_request_key(owner, task, payload, images, schema, *, source_ids=None):
    paths = images if isinstance(images, list) else [images]
    config = owner.config.review_vlm if task in REVIEW_TASKS else owner.config.vlm
    return digest([READING_VERSION, task, payload, schema.model_json_schema(), source_ids,
                  [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
                  VLMAdapter.compute_config_hash(config)])


def remaining_read_calls(owner, task, payload, images, schema, *, source_ids=None):
    """Estimate only unspent attempts for the exact request used by read_contract."""
    key = reading_request_key(owner, task, payload, images, schema, source_ids=source_ids)
    if key in owner.state.get("source_reading_cache", {}):
        return 0
    return max(0, 2 - owner.state.get("source_reading_attempts", {}).get(key, {}).get("calls", 0))


async def read_contract(owner, task, payload, images, schema, *, source_ids=None, max_attempts=2,
                        required_block_ids=None, context_block_ids=(), required_source_refs=None):
    """One initial attempt and one format repair, persisted across candidates/resume.

    Only validated responses are cached. Readability=false is a valid source
    observation; changing tools/models/evidence creates a different request.
    """
    key = reading_request_key(owner, task, payload, images, schema, source_ids=source_ids)
    cache = owner.state.setdefault("source_reading_cache", {})
    attempts = owner.state.setdefault("source_reading_attempts", {})
    if key in cache:
        return schema.model_validate(cache[key])
    request = attempts.setdefault(key, {"calls": 0, "errors": []})
    while request["calls"] < max_attempts:
        if not all(owner.available_budget()):
            if request["errors"]:
                raise ReadingFormatError("response_contract_retry_budget_exhausted; source readability remains unknown: "
                                         + request["errors"][-1][:300])
            raise BudgetExhausted("No budget for source reading or its contract repair")
        request["calls"] += 1
        owner._save()
        current = dict(payload)
        if request["errors"]:
            current["response_contract_error"] = (
                "Previous response did not satisfy the schema/IDs. Read the same source and return valid JSON. "
                "Do not put deliverable content only in observation. complete=true requires nonempty content. "
                "Return exactly the supplied source IDs in order. " + request["errors"][-1][:500])
        try:
            result = schema.model_validate(await owner._call(task, current, images, schema))
            if source_ids is not None and [item.source_id for item in result.items] != source_ids:
                raise ReadingFormatError("source_ids_omitted_or_invented_or_reordered")
            if required_source_refs is not None:
                refs = [item.source_ref for item in result.selections]
                if len(refs) != len(set(refs)) or set(refs) != set(required_source_refs):
                    raise ReadingFormatError("image_resolution_missing_duplicated_or_invented_evidence")
            if required_block_ids is not None:
                ids = [item.block_id for item in result.blocks]
                required = set(required_block_ids)
                if (len(ids) != len(set(ids)) or not required.issubset(ids)
                        or not set(ids).issubset(required | set(context_block_ids))):
                    raise ReadingFormatError("review_block_ids_missing_duplicated_or_invented")
        except (ValidationError, ModelOutputError, ReadingFormatError) as exc:
            request["errors"].append(str(exc)[:1000])
            owner._save()
            continue
        cache[key] = result.model_dump()
        owner._save()
        return result
    raise ReadingFormatError(f"reading_format_error: {max_attempts} bounded attempts exhausted; source readability remains unknown; "
                             + (request["errors"][-1][:350] if request["errors"] else "previous attempt interrupted"))


async def read_native_batch(owner, task, payload, images, *, source_ids):
    """Shrink failed grouped responses; preserve each image ID and all context.

    A valid incomplete reading is never retried as a formatting failure.
    Children use the same persisted request cache and document-wide budget.
    """
    from app.pipeline.recovery_models import NativeReading
    try:
        return await read_contract(owner, task, payload, images, NativeReading,
                                   source_ids=source_ids, max_attempts=1 if len(source_ids) > 1 else 2)
    except ReadingFormatError:
        if len(source_ids) <= 1:
            raise
    midpoint = len(source_ids) // 2
    items = []
    for lo, hi in ((0, midpoint), (midpoint, len(source_ids))):
        subset = source_ids[lo:hi]
        child = {**payload, "sources": payload["sources"][lo:hi],
                 "image_order": subset + payload.get("image_order", [])[len(source_ids):],
                 "response_scope": "Only these source IDs; previous grouped output failed formatting."}
        reading = await read_native_batch(owner, task, child,
            images[lo:hi] + images[len(source_ids):], source_ids=subset)
        items.extend(reading.items)
    return NativeReading(items=items)


class ImageText(Contract):
    region_id: str = Field(min_length=1, max_length=40)
    bbox: list[int] = Field(min_length=4, max_length=4)
    readable: bool
    text: str = Field(max_length=1600)

    @model_validator(mode="after")
    def valid_region(self):
        x0, y0, x1, y1 = self.bbox
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError("bbox must be a nonempty rectangle in original-image 0..1000 coordinates")
        if self.readable != bool(self.text.strip()):
            raise ValueError("Readable text needs a transcription; unreadable text must be empty")
        return self


class ImageParts(Contract):
    information_status: Literal["informative", "non_informative", "uncertain"] = "uncertain"
    information_reason: str = Field(default="", max_length=300)
    image_type: str = Field(pattern="^(photo|logo|diagram|chart|table|document|other)$")
    visual_facts: list[str] = Field(max_length=6)
    visual_independent_of_text: bool
    text_regions: list[ImageText] = Field(max_length=16)
    complete: bool
    observation: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def valid_parts(self):
        if any(not text.strip() or len(text) > 500 for text in self.visual_facts):
            raise ValueError("Each visual fact must contain 1..500 characters")
        ids = [r.region_id for r in self.text_regions]
        if len(ids) != len(set(ids)):
            raise ValueError("Text region IDs must be unique")
        if self.information_status == "non_informative" and (not self.information_reason.strip() or self.text_regions):
            raise ValueError("Non-informative classification needs a reason and cannot conceal text regions")
        if self.complete and not (self.visual_facts or self.text_regions) and self.information_status != "non_informative":
            raise ValueError("A complete reading must put visible content in visual_facts or text_regions")
        return self


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if intersection else 0


def reconcile_image_parts(primary, independent, *, prior_transcription=None):
    """Agreement only admits a candidate for source audit, never certifies it.

    Any disputed/illegible/omitted region remains an explicit gap. Photo scene
    facts can survive only when both readers say they do not rely on lettering.
    Text-dependent diagrams/logos cannot hide a missing label behind a summary.
    """
    information = (primary.information_status if primary.information_status == independent.information_status
                   and primary.complete and independent.complete else "uncertain")
    text, gaps, used = [], [], set()
    for region in primary.text_regions:
        candidates = [(i, other) for i, other in enumerate(independent.text_regions)
                      if overlap(region.bbox, other.bbox) >= 0.5]
        if len(candidates) == 1 and candidates[0][0] not in used:
            index, other = candidates[0]
            used.add(index)
            if (region.readable and other.readable and normalized_text(region.text)
                    and normalized_text(region.text) == normalized_text(other.text)):
                text.append({"text": region.text, "bbox": region.bbox})
                continue
        gaps.append({"bbox": region.bbox, "reason": "image_text_unreadable_or_disputed"})
    for index, region in enumerate(independent.text_regions):
        if index not in used:
            gaps.append({"bbox": region.bbox, "reason": "image_text_not_accounted_for_by_both_readers"})
    # Repeating the same model cannot erase a conflicting blind reading of this
    # exact source image. Keep the conflict until another evidence source/tool exists.
    if prior_transcription and text and normalized_text(prior_transcription) != normalized_text(
            "".join(item["text"] for item in text)):
        gaps.extend({"bbox": item["bbox"], "reason": "image_text_conflicts_with_prior_source_reading"} for item in text)
        text = []
    if not primary.complete or not independent.complete:
        gaps.append({"bbox": [0, 0, 1000, 1000], "reason": "image_reading_incomplete"})
    independent_visual = (primary.image_type == independent.image_type == "photo"
                          and primary.visual_independent_of_text and independent.visual_independent_of_text)
    visual = (primary.visual_facts if primary.visual_facts and independent.visual_facts
              and (not gaps or independent_visual) else [])
    accepted_text = {normalized_text(item["text"]) for item in text}
    disputed = {normalized_text(region.text) for region in primary.text_regions + independent.text_regions
                if region.text.strip() and normalized_text(region.text) not in accepted_text}
    visual = [fact for fact in visual if not any(value and value in normalized_text(fact) for value in disputed)]
    if primary.visual_facts and not visual:
        gaps.append({"bbox": [0, 0, 1000, 1000], "reason": "image_visual_depends_on_unverified_content"})
    # Without all material labels, isolated diagram values have no safe interpretation.
    if gaps and not independent_visual:
        text = []
    return {"visual_facts": visual, "text_regions": text, "gaps": gaps,
            "image_type": primary.image_type, "text_dependent": not independent_visual, "partial": bool(gaps),
            "information_status": information}


class CombinedImageParts(ImageParts):
    """Internal aggregate only; each model response still has the 16-region limit."""
    visual_facts: list[str] = Field(max_length=24)
    text_regions: list[ImageText] = Field(max_length=64)


async def read_image_parts_bounded(owner, task, payload, source, *, max_depth=2):
    """Shrink contract failures to at most four disjoint crops (seven reads).

    The original remains context; all boxes return to original coordinates.
    Valid incomplete readings are observations and never trigger format retries.
    Every call shares the document budget and the exact-request resume cache.
    """
    if max_depth not in {1, 2}:
        raise ValueError("Image contract split depth must be one or two")
    from PIL import Image
    with Image.open(source) as picture:
        width, height = picture.size
    directory = source.parent / ("contract-parts-" + hashlib.sha256(source.read_bytes()).hexdigest()[:16])

    async def read(box, depth):
        left, top, right, bottom = box
        if depth == 0:
            current, images = payload, source
        else:
            directory.mkdir(exist_ok=True)
            path = directory / ("part-" + "-".join(str(v) for v in box) + ".png")
            # Recreate from source, never trust a leftover crop after interruption.
            with Image.open(source) as picture:
                picture.crop(box).save(path)
            current = {**payload, "source_core": [left * 1000 / width, top * 1000 / height,
                                                   right * 1000 / width, bottom * 1000 / height],
                       "response_scope": "Image 1 is the owned crop; image 2 is whole-image context. "
                           "Read only content whose top-left starts in the crop; finish cut lines from context. "
                           "Return bbox relative to image 1. Do not repeat content starting outside the crop."}
            images = [path, source]
        try:
            result = await read_contract(owner, task, current, images, ImageParts, max_attempts=1)
            return [(result, box)]
        except ReadingFormatError:
            if depth >= max_depth or bottom - top < 64:
                raise
        middle = (top + bottom) // 2
        return (await read((left, top, right, middle), depth + 1)
                + await read((left, middle, right, bottom), depth + 1))

    readings = await read((0, 0, width, height), 0)
    if len(readings) == 1:
        return readings[0][0]
    regions, results = [], []
    for index, (result, (left, top, right, bottom)) in enumerate(readings):
        results.append(result)
        for part in result.text_regions:
            x0, y0, x1, y1 = part.bbox
            box = [round((left + x0 * (right - left) / 1000) * 1000 / width),
                   round((top + y0 * (bottom - top) / 1000) * 1000 / height),
                   round((left + x1 * (right - left) / 1000) * 1000 / width),
                   round((top + y1 * (bottom - top) / 1000) * 1000 / height)]
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ReadingFormatError("Mapped image part has no source area")
            regions.append(part.model_copy(update={"region_id": f"part{index}-{len(regions)}", "bbox": box}))
    kinds = {r.image_type for r in results}
    information = {r.information_status for r in results}
    return CombinedImageParts(information_status=next(iter(information)) if len(information) == 1 else 'uncertain',
        information_reason='Source crops classified independently; whole-image audit still required.', image_type=results[0].image_type if len(kinds) == 1 else 'other',
        visual_facts=list(dict.fromkeys(f for r in results for f in r.visual_facts)),
        visual_independent_of_text=all(r.visual_independent_of_text for r in results),
        text_regions=regions, complete=all(r.complete for r in results),
        observation="Bounded source-image partition after response format failure; still requires source audit.")
