"""Always-planned source review, bounded repairs, and review of the final delivery."""
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.adapters.vlm import VLMAdapter
from app.config import settings
from app.models.document_ir import BlockType
from app.pipeline.agentic import AgenticRepair, BudgetExhausted, RegionModel, _crop, visual_issues
from app.pipeline.delivery import atomic_write_json
from app.pipeline.final_delivery import read_json
from app.pipeline.image_source_review import ImageRelationshipCheck
from app.pipeline.review_policy import reviewed_block_usable
from app.pipeline.review_sources import (
    DeliveryExcerpts,
    SourceInventory,
    compact_delivery_excerpts,
    content_blocks,
    coverage_summary,
    delivery_digest,
    digest,
    input_identity,
    review_scope_digest,
)

NATIVE_TEXT_PROOF = "Word-rendered PDF text layer for physical page {}"
OCR_VOTE_POLICY = (
    "independent_ocr_votes: PaddleOCR, a text-recognition engine sharing nothing with you or the blind reader, "
    "read the original picture wherever the blind reading and the delivery disagree. verdict=delivery means OCR "
    "reads what the delivery says; ocr_reads is what it read. Source lettering is delivered as printed, including "
    "the author's typos and variant spellings: never 'correct' it, and do not fail a block for a spelling OCR "
    "confirms. verdict=delivery settles that line as delivered, even where your own reading of it differs. "
    "OCR is evidence, not certification: still check every other part of the picture against the pixels.")
REVIEW_VERSION = "document-review-20261016.2-table-cells-match"
SCHEDULER_VERSION = "tool-scheduler-20260930.1-complete-tool-budget"


UNEXPLAINED_FINDING = "The reviewer marked this block needs_review and gave no observation."


class BlockReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: str
    status: Literal["verified", "needs_review"]
    # A verified block is not explained (2026-10-15): those sentences were 13% of every audit's output.
    observation: str = Field(default="", max_length=700)
    context_complete: bool = False
    depends_on_block_ids: list[str] = Field(default_factory=list, max_length=20)
    image_relationship_checks: list[ImageRelationshipCheck] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def failure_is_explained(self):
        # An empty remark would read as "nothing found wrong" wherever remarks are weighed.
        if self.status == "needs_review" and not self.observation.strip():
            self.observation = UNEXPLAINED_FINDING
        return self


class PageReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blocks: list[BlockReview]
    source_complete: bool
    reading_order_correct: bool
    observation: str = Field(min_length=1, max_length=1000)
    issues: list[str] = Field(max_length=30)
    declared_gaps_only: bool = False


# Largest request, in characters of its JSON, the review model is given at once.
REGION_PAYLOAD_LIMIT = 65000
LAYOUT_PAYLOAD_LIMIT = 42000
# A band is halved no further than this, in either direction (page side is 1000).
MIN_REGION_HEIGHT = 30
LAYOUT_PART_NOTE = ("The delivered boundaries are split into consecutive parts, each checked on its own; this part "
                    "starts with the last boundary of the previous part. Judge only the order of the boundaries "
                    "shown here. Page content absent from this part is in the other parts: never report it as "
                    "missing, and never report a gap in paragraph numbers.")


def layout_boundaries(documents, chunks):
    """The delivered boundaries the global-order check sees, as (kind, boundary).

    A picture's own document is one unit of the page order; the order of its lines is
    the picture's, already checked crop by crop, so it is one boundary from its first
    line to its last. On a logo wall this halves the list, and fewer parts leave the
    model fewer chances to call another part's content missing.
    """
    def cut(text):
        return text[:220] + " … " + text[-120:]

    folded, heads = [], {}
    for item in documents:
        last = folded[-1][1] if folded else None
        if item["binding"] == "asset_document" and last and last["file"] == item["file"] \
                and last["binding"] == "asset_document":
            last.update(text=heads[id(last)] + " … " + item["text"][-120:], last_paragraph=item["paragraph"])
            continue
        boundary = {**item, "text": cut(item["text"])}
        heads[id(boundary)] = item["text"][:220]
        folded.append(("document", boundary))
    return folded + [("chunk", {**c, "content": cut(c["content"])}) for c in chunks]


def merge_part_reviews(reviews):
    """A band checked in several crops passes only if every crop passed."""
    if len(reviews) == 1:
        return reviews[0]
    return PageReview(blocks=[b for r in reviews for b in r.blocks],
                      source_complete=all(r.source_complete for r in reviews),
                      reading_order_correct=all(r.reading_order_correct for r in reviews),
                      observation=" | ".join(r.observation for r in reviews)[:1000],
                      issues=[i for r in reviews for i in r.issues][:30])


def layout_parts(page, boundaries, cores, limit=LAYOUT_PAYLOAD_LIMIT):
    """The global-order check as consecutive slices of the delivered boundaries, each one request.

    Order holds when every adjacent pair is in order, so each slice repeats the previous
    slice's last boundary and every pair is still judged in some request. 2026-10-08: five
    image-dense pages were never checked, their boundaries 55-109k characters long.
    """
    def payload(items, part=None):
        result = {"page_idx": page,
                  "document_boundaries": [item for kind, item in items if kind == "document"],
                  "chunk_boundaries": [item for kind, item in items if kind == "chunk"],
                  "region_cores": cores}
        return {**result, "boundary_part": part} if part else result

    if len(str(payload(boundaries))) <= limit:
        return [payload(boundaries)]
    slices, current = [], []
    for item in boundaries:
        if len(current) > (1 if slices else 0) and len(str(payload(current + [item], LAYOUT_PART_NOTE))) > limit:
            slices.append(current)
            current = current[-1:]
        current.append(item)
    slices.append(current)
    return [payload(items, {"part": number, "of": len(slices), "note": LAYOUT_PART_NOTE})
            for number, items in enumerate(slices, 1)]


def reading_parts(payload, limit=None):
    """A Word page whose pictures' independent readings overflow one request, judged in parts.

    Each part carries the readings of some pictures and answers for their blocks only;
    every other block stays as context and the whole page and delivery stay in view, so
    completeness and order are still judged on the entire page. Blocks no picture owns
    are answered in the first part. None when one picture's part alone overflows.
    2026-10-08: a Word page of 8 blocks was left unreviewed this way.
    """
    limit = limit or REGION_PAYLOAD_LIMIT
    readings, scopes = payload.get("independent_image_readings") or {}, payload.get("image_reading_scopes") or {}
    if not readings:
        return None
    votes = payload.get("independent_ocr_votes") or {}
    owned = {bid for ids in scopes.values() for bid in ids}
    free = [bid for bid in payload["required_block_ids"] if bid not in owned]
    base = {key: value for key, value in payload.items()
            if key not in {"independent_image_readings", "image_reading_scopes", "independent_ocr_votes"}}

    def part(origins, first):
        ids = set(free if first else []) | {bid for origin in origins for bid in scopes.get(origin, [])}
        required = [bid for bid in payload["required_block_ids"] if bid in ids]
        result = {**base, "required_block_ids": required,
                  "blocks": [b for b in payload["blocks"] if b["block_id"] in ids],
                  "context_blocks": payload.get("context_blocks", []) + [
                      {"block_id": b["block_id"], "untrusted_extracted_text": b.get("untrusted_extracted_text", "")}
                      for b in payload["blocks"] if b["block_id"] not in ids],
                  "independent_image_readings": {o: readings[o] for o in origins},
                  "image_reading_scopes": {o: scopes[o] for o in origins if o in scopes},
                  "response_scope": "The pictures' independent readings come in several parts. Return block "
                                    "reviews only for required_block_ids; all other blocks are context. Still "
                                    "check ENTIRE page completeness/order."}
        if part_votes := {bid: vote for bid, vote in votes.items() if bid in ids}:
            result["independent_ocr_votes"] = part_votes
        return result

    groups, current = [], []
    for origin in readings:
        if current and len(str(part(current + [origin], not groups))) > limit:
            groups.append(current)
            current = []
        current.append(origin)
    groups.append(current)
    parts = [part(origins, index == 0) for index, origins in enumerate(groups)]
    return None if any(len(str(p)) > limit for p in parts) else parts


def target_block_reviews(review, required_ids, context_ids=()):
    """Keep required evidence strict; observations of supplied neighbors certify no target."""
    required, context = set(required_ids), set(context_ids)
    ids = [b.block_id for b in review.blocks]
    if (len(ids) != len(set(ids)) or not required.issubset(ids)
            or not set(ids).issubset(required | context)):
        raise ValueError("Page review omitted, duplicated, or invented source block IDs")
    return review.model_copy(update={"blocks": [b for b in review.blocks if b.block_id in required]})


# A space beside a non-Latin character separates table cells; Latin words stay one phrase.
_CELL_BREAK = re.compile(r"(?<=[^\x00-\x7f])\s+|\s+(?=[^\x00-\x7f])")


def _line_delivered(line, delivered):
    """A line as written, or each of its cells: a table read across its rows is delivered down its columns.

    2026-10-16: a 4P/4C table delivered every cell but stayed unverified because the review read
    「構成要素 產品（Product） 顧客（Customer）」 as one line; so did three other tables.
    """
    from app.pipeline.content_coverage import normalized_text
    from app.pipeline.image_source_review import appears_in
    if appears_in(line, delivered):
        return True
    cells = [cell for cell in _CELL_BREAK.split(line) if normalized_text(cell)]
    return len(cells) > 1 and all(appears_in(cell, delivered) for cell in cells)


def word_picture_issue(reading, delivered):
    """Blind source reading can reject a caption, never certify it on its own."""
    from app.pipeline.content_coverage import normalized_text
    if not reading.complete:
        return "Independent Word image reading is incomplete: " + reading.observation
    if not reading.text.strip():
        return "Independent Word image reading supplied no content; verification is unavailable."
    if reading.kind == "transcription":
        lines = [line for line in reading.text.splitlines() if normalized_text(line)]
        if not lines or not all(_line_delivered(line, delivered) for line in lines):
            return "Independent original image wording is absent or different in delivery: " + reading.text[:400]
    return ""


class SharedModel:
    """A local repair spends the same persisted model/time budget as page review."""
    def __init__(self, controller):
        self.controller = controller

    async def query(self, task, payload, image, schema):
        if task == "repair":
            if self.controller.state["usage"]["actions"] >= self.controller.limits.max_actions:
                raise BudgetExhausted("repair action budget exhausted")
            self.controller.state["usage"]["actions"] += 1
            self.controller._save()
        return await self.controller._call(task, payload, image, schema), 0


class DocumentReview(AgenticRepair):
    def _record(self, record, status, reason):
        record.update(status=status, reason=reason)
        self._save()

    def available_budget(self):
        reserve = (getattr(self, "_reserve", None) if getattr(self, "_phase", "") == "audit"
                   else getattr(self, "_protected_budget", None)) or (0, 0)
        return (max(0, self.limits.max_model_calls - self.state["usage"]["model_calls"] - reserve[0]),
                max(0, self._remaining() - reserve[1]))

    async def _call(self, task, payload, image, schema):
        calls, seconds = self.available_budget()
        if calls <= 0 or seconds <= 0:
            raise BudgetExhausted("reserved_for_other_work")
        if task == "recovery_plan":
            from app.pipeline.review_budget import call_seconds
            seconds = min(seconds, max(120, call_seconds(self.state, task) * 2))
        # Enforce the reservation on the running call, not only before it starts.
        return await super()._call(task, payload, image, schema, timeout_seconds=seconds)

    def _remaining(self):
        return self.limits.max_seconds - self.state["usage"]["seconds"]

    async def _word_picture_readings(self, blocks, inventory):
        from app.pipeline.image_source_review import (
            WORD_PICTURE_BATCH_SIZE,
            WordPictureItem,
            read_word_picture_batch,
        )
        pictures = getattr(inventory, "word_images", {})
        cache = self.state.setdefault("word_image_readings", {})
        pending, selected = {}, {}
        for block in blocks:
            if path := pictures.get(block.block_id):
                key = digest([REVIEW_VERSION, hashlib.sha256(path.read_bytes()).hexdigest(),
                              VLMAdapter.compute_config_hash(self.config.review_vlm)])
                selected[block.block_id] = key
                if key not in cache:
                    pending[key] = path
        pending = list(pending.items())
        from app.pipeline.semantic.language import resolve_semantic_output_language
        # Relationship sentences can be delivered verbatim by a delivery patch, so they
        # are written in the document's language; printed lettering is still copied as printed.
        language = resolve_semantic_output_language(self.config.package.semantic_output_language.value, inventory.ir)
        for start in range(0, len(pending), WORD_PICTURE_BATCH_SIZE):
            batch = pending[start:start + WORD_PICTURE_BATCH_SIZE]
            ids = [f"image{i}" for i in range(len(batch))]
            # No captions, neighboring prose or previous outputs enter this blind call.
            reading = await read_word_picture_batch(self, {
                "sources": [{"source_id": sid} for sid in ids], "image_order": ids,
                "relationship_language": language,
            }, [path for _, path in batch], source_ids=ids)
            for (key, _), item in zip(batch, reading.items, strict=True):
                cache[key] = item.model_dump()
            self._save()
        from app.pipeline.image_source_review import inherited_image_readings
        readings = inherited_image_readings(blocks, inventory.run_path)
        readings.update({bid: WordPictureItem.model_validate(cache[key]) for bid, key in selected.items()})
        return readings

    async def _audit_page_contract(self, payload, images, depth=0):
        from app.pipeline.image_reading import ReadingFormatError, read_contract
        ids = payload["required_block_ids"]
        try:
            review = await read_contract(self, "audit_page", payload, images, PageReview,
                                         max_attempts=2 if len(ids) <= 1 else 1, required_block_ids=ids,
                                         context_block_ids=[b["block_id"] for b in payload.get("context_blocks", [])])
            try:
                return target_block_reviews(review, ids, [b["block_id"] for b in payload.get("context_blocks", [])])
            except ValueError as exc:
                raise ReadingFormatError(str(exc)) from exc
        except ReadingFormatError:
            if len(ids) <= 1 or depth >= 2:
                raise
        # All source images and actual delivery excerpts remain visible in every
        # batch: reduce the response scope without weakening whole-page coverage.
        results = []
        midpoint = len(ids) // 2
        for subset in (ids[:midpoint], ids[midpoint:]):
            child = {**payload, "required_block_ids": subset,
                     "blocks": [b for b in payload["blocks"] if b["block_id"] in subset],
                     "context_blocks": payload.get("context_blocks", []) + [
                         b for b in payload["blocks"] if b["block_id"] not in subset],
                     "response_scope": "Return block reviews only for required_block_ids. "
                                       "All other blocks are context. Still check ENTIRE page completeness/order."}
            results.append(await self._audit_page_contract(child, images, depth + 1))
        return PageReview(blocks=[b for result in results for b in result.blocks],
            source_complete=all(result.source_complete for result in results),
            reading_order_correct=all(result.reading_order_correct for result in results),
            declared_gaps_only=all(result.declared_gaps_only for result in results),
            issues=list(dict.fromkeys(issue for result in results for issue in result.issues))[:30],
            observation="Source and actual delivery audited with smaller response scopes after format failure.")

    async def _inspect(self, page, ir, inventory, run_path):
        before = self.state["usage"]["seconds"]
        started = time.monotonic()
        try:
            record = await self._inspect_page(page, ir, inventory, run_path)
        except Exception as exc:
            record = {"page_idx": page, "status": "needs_review", "reason": "source_review_error: " + str(exc)[:500]}
            self.state["pages"][str(page)] = record
        try:
            return self._certify_native_text_blocks(page, ir, inventory, run_path, record)
        finally:
            counted = self.state["usage"]["seconds"] - before
            self.state["usage"]["seconds"] += max(0, time.monotonic() - started - counted)
            self._save()

    def build_page_review_input(self, page, ir, inventory, run_path):
        """Build the exact audit scope without a model call or review-state mutation.

        Candidate admission and live inspection use the same source images, actual
        delivery spans, neighboring context, and chunk membership/order.
        """
        from app.pipeline.review_policy import SOURCE_COMPLETENESS_POLICY
        from app.pipeline.stages.chunk import ChunkStage

        bundle = delivery_digest(run_path)
        blocks = inventory.blocks(page, ir)
        image = inventory.image(page)
        if image is None:
            raise ValueError("source_image_unavailable")
        key = (str(run_path.resolve()), bundle, digest(input_identity(ir)))
        if getattr(self, "_delivery_cache_key", None) != key:
            renderer = ChunkStage(self.config)
            renderer._asset_text = renderer._load_asset_retrieval_text(run_path / "outputs")
            self._delivery_cache = (renderer, DeliveryExcerpts(run_path, content_blocks(ir)))
            self._delivery_cache_key = key
        renderer, excerpts = self._delivery_cache
        context_blocks = []
        current_ids = {b.block_id for b in blocks}
        for neighbor in (page - 1, page + 1):
            candidates = inventory.blocks(neighbor, ir) if neighbor in inventory.pages else []
            # Continuations may exist even when the parser assigned the entire table
            # to one page. Supply boundary context without treating it as audited here.
            boundary = candidates[-2:] if neighbor < page else candidates[:2]
            for block in boundary:
                if block.block_id not in current_ids:
                    context_blocks.append(block)
        documents, chunks = excerpts.for_blocks(blocks + context_blocks, renderer)
        documents = compact_delivery_excerpts(documents)
        payload = {
            "page_idx": page, "block_id": f"page:{page}",
            "required_block_ids": [b.block_id for b in blocks],
            "declared_gaps": [{"block_id": b.block_id, **b.payload["agentic_gap"]}
                              for b in blocks if b.payload.get("agentic_gap")],
            "blocks": [{"block_id": b.block_id, "type": b.type.value,
                        "bbox_norm": b.bbox_norm if ir.has_physical_pages() else [],
                        "reading_order": b.reading_order,
                        "original_image_source": b.payload.get("agentic_source_image"),
                        "image_origin": b.payload.get("agentic_image_origin"),
                        **({"relationship_evidence_id": b.payload["agentic_relationship_evidence_id"]}
                           if b.payload.get("agentic_relationship_evidence_id") else {}),
                        "image_region": b.payload.get("agentic_image_region"),
                        "content_kind": b.payload.get("agentic_content_kind"),
                        "information_status": b.payload.get("agentic_information_status", "uncertain"),
                        "untrusted_extracted_text": renderer._block_to_text(b)} for b in blocks],
            "context_blocks": [{"block_id": b.block_id, "untrusted_extracted_text": renderer._block_to_text(b)}
                               for b in context_blocks],
            "delivered_documents": documents, "delivered_chunks": chunks,
            "chunks_required": self.config.package.generate_chunks,
            "policy": ("Ignore page numbers, repetitive furniture and privacy masks only; preserve all substantive content. "
                       + SOURCE_COMPLETENESS_POLICY),
        }
        # One target page plus neighbors catches continuation even with zero risk signals.
        images = [image]
        labels = [f"Target source page {page + 1}"]
        spanning = any(len(inventory.mapping.get(b.block_id, [])) > 1 for b in blocks)
        for neighbor in (page - 1, page + 1):
            if neighbor in inventory.pages:
                path = inventory.image(neighbor) if spanning else inventory.context_image(neighbor)
                if path:
                    images.append(path)
                    labels.append(f"Context source page {neighbor + 1}" +
                                  ("; full detail for a cross-page block" if spanning else "; overview only") +
                                  "; audit the target page only; do not guess unclear context")
        # Retain fine table/image detail without using risk signals as a filter.
        if ir.has_physical_pages():
            for block in blocks:
                if block.type in {BlockType.TABLE, BlockType.IMAGE}:
                    crop = _crop(image, block, run_path / "agentic")
                    if crop != image:
                        images.append(crop)
                        labels.append("Detail crop of " + block.block_id)
        for block in blocks:
            if original := getattr(inventory, "word_images", {}).get(block.block_id):
                images.append(original)
                labels.append(f"Original Word embedded image for {block.block_id}; physical page {page + 1}; "
                              "read its small text directly, not from the untrusted delivered description")
            if declared := block.payload.get("agentic_source_image"):
                path = run_path / declared
                if path.resolve().is_relative_to((run_path / "agentic/evidence").resolve()) and path.is_file():
                    if path not in images:
                        images.append(path)
                        labels.append(f"Original embedded source for {block.block_id}; physical page {page + 1}, "
                                      f"region {block.payload.get('agentic_source_region')}")
        payload["evidence_images"] = labels
        evidence = [{"path": str(path.relative_to(run_path)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "label": label} for label, path in zip(labels, images, strict=True)]
        fingerprint = digest([REVIEW_VERSION, payload, evidence])
        scoped_fingerprint = review_scope_digest(REVIEW_VERSION, payload, evidence)
        return {"payload": payload, "images": images, "evidence": evidence,
                "fingerprint": fingerprint, "scoped_fingerprint": scoped_fingerprint,
                "delivery_sha256": bundle}

    def review_scope_fingerprints(self, ir, inventory, run_path, pages=None):
        """Return model-free page identities for candidate transaction planning."""
        return {page: self.build_page_review_input(page, ir, inventory, run_path)["scoped_fingerprint"]
                for page in (inventory.pages if pages is None else pages)}

    async def _ocr_lines(self, source, inventory):
        """PaddleOCR lines of one original picture, cached per picture and OCR runtime; None if unreadable.

        Local OCR spends no model call, but its time counts against the review's limit.
        """
        from PIL import Image

        from app.adapters import ocr

        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        key = digest([sha, ocr.ocr_runtime_identity()])
        cache = self.state.setdefault("ocr_lines", {})
        if key not in cache:
            remaining = self._remaining()
            if remaining < 60:
                return None
            started = time.monotonic()
            try:
                with Image.open(source) as image:
                    scale = 2 if image.width * image.height < 1_500_000 else 1
                folder = inventory.directory / "ocr" / sha[:16]
                try:
                    regions = [[0, 0, 1000, 1000]]
                    value = await ocr.read_ocr(source, regions[0], scale, folder, timeout=min(90, remaining))
                    lines = [line.get("text", "") for line in value.get("lines", [])]
                except ValueError as exc:
                    if not str(exc).startswith("ocr_output_capacity_exceeded"):
                        raise
                    # More lines than one response holds: overlapping bands, so no line is cut at a border.
                    lines = []
                    for top in (0, 250, 500, 750):
                        value = await ocr.read_ocr(source, [0, top, 1000, min(1000, top + 300)], scale, folder,
                                                   timeout=min(90, max(1, self._remaining())))
                        lines += [line.get("text", "") for line in value.get("lines", [])]
                cache[key] = {"lines": lines}
            except (ValueError, TimeoutError, OSError) as exc:
                cache[key] = {"error": str(exc)[:200]}
            finally:
                elapsed = time.monotonic() - started
                cache.setdefault(key, {})["seconds"] = round(elapsed, 2)
                self.state["usage"]["seconds"] += elapsed
                self._save()
        return cache[key].get("lines")

    async def _ocr_votes(self, blocks, readings, inventory, record):
        """PaddleOCR votes on each picture line the blind reading cannot find in the delivery.

        Only pictures whose readings disagree with their delivery wait for OCR.
        """
        from app.adapters import ocr
        from app.pipeline.content_coverage import normalized_text
        from app.pipeline.line_adjudication import adjudicate

        if not ocr.ocr_python():
            return {}
        votes = {}
        for block in blocks:
            reading = readings.get(block.block_id)
            source = getattr(inventory, "word_images", {}).get(block.block_id)
            if not reading or source is None or not reading.complete or reading.kind != "transcription":
                continue
            delivered = self._delivery_cache[0]._block_to_text(block)
            disputed = [line for line in reading.text.splitlines()
                        if normalized_text(line) and not _line_delivered(line, delivered)]
            if disputed and (lines := await self._ocr_lines(source, inventory)):
                votes[block.block_id] = adjudicate(disputed, delivered, lines)
        if votes:
            record.setdefault("line_adjudications", {}).update(votes)
        return votes

    async def _settle_with_ocr(self, block_id, reading, delivered, inventory, run_path, record):
        """Let PaddleOCR vote on the lines the review's own reading disagrees with the delivery on.

        True only when OCR backs the delivery on every disputed line; the votes are
        kept as hashed evidence. Anything else leaves the picture unverified.
        """
        from app.adapters import ocr
        from app.pipeline.content_coverage import normalized_text
        from app.pipeline.line_adjudication import adjudicate

        source = getattr(inventory, "word_images", {}).get(block_id)
        disputed = [line for line in reading.text.splitlines() if normalized_text(line) and not _line_delivered(line, delivered)]
        if source is None or not disputed or not ocr.ocr_python():
            return False
        lines = await self._ocr_lines(source, inventory)
        if not lines:
            return False
        verdicts = adjudicate(disputed, delivered, lines)
        record.setdefault("line_adjudications", {})[block_id] = verdicts
        if any(v["verdict"] != "delivery" for v in verdicts):
            return False
        content = json.dumps({"method": "paddleocr_line_adjudication", "block_id": block_id,
                              "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                              "ocr_runtime": ocr.ocr_runtime_identity(), "ocr_lines": lines,
                              "adjudications": verdicts}, ensure_ascii=False, indent=2).encode()
        sha = hashlib.sha256(content).hexdigest()
        proof = inventory.directory / f"ocr-adjudication-{block_id}-{sha[:16]}.json"
        if not proof.is_file():
            proof.write_bytes(content)
        label = f"PaddleOCR line adjudication for {block_id}"
        record["images"] = [i for i in record["images"] if i.get("label") != label] + [
            {"path": str(proof.relative_to(run_path)), "sha256": sha, "label": label}]
        return True

    def _native_text_evidence(self, inventory):
        from app.pipeline.native_page_check import NativePageText

        if getattr(inventory, "_native_text_pdf", None) != inventory.pdf:
            inventory._native_text = NativePageText(inventory.pdf)
            inventory._native_text_pdf = inventory.pdf
        return inventory._native_text

    def _native_text_proof(self, page, blocks, inventory, run_path, evidence):
        """Content-addressed, so a later proof of the same page never invalidates a recorded one."""
        content = json.dumps({
            "method": "native_word_text_layer", "page_idx": page,
            "pdf_sha256": hashlib.sha256(inventory.pdf.read_bytes()).hexdigest(),
            "block_ids": [b.block_id for b in blocks], "printed_lines": evidence.lines[page],
        }, ensure_ascii=False, indent=2).encode()
        sha = hashlib.sha256(content).hexdigest()
        proof = inventory.directory / f"native-text-p{page:04d}-{sha[:16]}.json"
        if not proof.is_file():
            proof.write_bytes(content)
        return {"path": str(proof.relative_to(run_path)), "sha256": sha, "label": NATIVE_TEXT_PROOF.format(page + 1)}

    def _certify_native_text_blocks(self, page, ir, inventory, run_path, record):
        """After a picture page's audit, certify its native text from the text layer.

        One picture's failed reading left a whole Validation page of paragraphs
        unreviewed and out of RAG. The page stays needs_review and the pictures
        keep whatever the audit said; only text and table blocks the text layer
        proves become verified with complete context. A model finding that the
        reading order is wrong is not overridden.
        """
        from app.pipeline.native_page_check import document_source_text
        from app.pipeline.review_sources import content_blocks

        if (self.limits.mode == "off" or record.get("status") == "verified" or not inventory.pdf
                or not inventory.complete or ir.has_physical_pages()):
            return record
        blocks = inventory.blocks(page, ir)
        pictures = [b for b in blocks if b.type == BlockType.IMAGE or b.payload.get("agentic_image_origin")]
        text_blocks = [b for b in blocks if b not in pictures]
        previous = record.get("review") or {}
        if not pictures or not text_blocks or (previous and not previous.get("reading_order_correct")):
            return record
        try:
            scope = self.build_page_review_input(page, ir, inventory, run_path)
        except ValueError:
            return record
        evidence = self._native_text_evidence(inventory)
        nearby = [b for neighbor in (page - 1, page + 1) if neighbor in inventory.pages
                  for b in inventory.blocks(neighbor, ir) if b not in blocks and b.type != BlockType.IMAGE
                  and not b.payload.get("agentic_image_origin")]
        payload = scope["payload"]
        reason = evidence.check(page, text_blocks, inventory.mapping, documents=payload["delivered_documents"],
                                chunks=payload["delivered_chunks"], chunks_required=payload["chunks_required"],
                                nearby=nearby, document_text=document_source_text(content_blocks(ir)),
                                expected_pictures=len({b.payload.get("agentic_image_origin") or b.block_id
                                                       for b in pictures}))
        self.state.setdefault("native_text_checks", {})[str(page)] = "text blocks: " + (reason or "pass")[:290]
        if reason:
            return record
        observation = "Native Word text matches the Word-rendered page text layer and its delivered documents and chunks."
        findings = {f["block_id"]: f for f in previous.get("blocks", [])}
        for block in text_blocks:
            findings[block.block_id] = BlockReview(block_id=block.block_id, status="verified",
                                                   observation=observation, context_complete=True).model_dump()
        for block in pictures:
            findings.setdefault(block.block_id, BlockReview(
                block_id=block.block_id, status="needs_review",
                observation="Picture content is not covered by the text layer and was not confirmed.").model_dump())
        record["review"] = PageReview(
            blocks=[BlockReview.model_validate(findings[b.block_id]) for b in blocks],
            source_complete=False, reading_order_correct=True,
            observation=previous.get("observation") or "Text and tables matched the text layer; pictures unconfirmed.",
            issues=(previous.get("issues") or ["Picture content was not confirmed against the source."])[:30],
            declared_gaps_only=False).model_dump()
        proven = NATIVE_TEXT_PROOF.format(page + 1)
        record["images"] = [i for i in record.get("images") or scope["evidence"] if i.get("label") != proven] + [
            self._native_text_proof(page, text_blocks, inventory, run_path, evidence)]
        record.setdefault("block_ids", [b.block_id for b in blocks])
        record.setdefault("delivery_sha256", scope["delivery_sha256"])
        record["native_text_blocks_verified"] = [b.block_id for b in text_blocks]
        self._save()
        return record

    def _native_text_review(self, page, ir, inventory, run_path, record, bundle):
        """Certify a plain native-Word page from the Word-rendered PDF text layer.

        No model call and no budget: a page holding only native Word text and
        tables is proven by string comparison, or handed back (None) to the
        model audit with the reason kept in review_state.
        """
        from app.pipeline.native_page_check import document_source_text
        from app.pipeline.review_sources import content_blocks

        if not inventory.pdf or not inventory.complete or ir.has_physical_pages():
            return None
        blocks = inventory.blocks(page, ir)
        if not blocks or any(b.type == BlockType.IMAGE or "word_source" not in b.payload for b in blocks):
            return None
        try:
            scope = self.build_page_review_input(page, ir, inventory, run_path)
        except ValueError:
            return None
        evidence = self._native_text_evidence(inventory)
        nearby = [b for neighbor in (page - 1, page + 1) if neighbor in inventory.pages
                  for b in inventory.blocks(neighbor, ir) if b not in blocks and b.type != BlockType.IMAGE]
        payload = scope["payload"]
        reason = evidence.check(page, blocks, inventory.mapping, documents=payload["delivered_documents"],
                                chunks=payload["delivered_chunks"], chunks_required=payload["chunks_required"],
                                nearby=nearby, document_text=document_source_text(content_blocks(ir)))
        if reason:
            self.state.setdefault("native_text_checks", {})[str(page)] = reason[:300]
            return None
        self.state.setdefault("native_text_checks", {})[str(page)] = "pass"
        images = scope["evidence"] + [self._native_text_proof(page, blocks, inventory, run_path, evidence)]
        observation = "Native Word text matches the Word-rendered page text layer and its delivered documents and chunks."
        review = PageReview(
            blocks=[BlockReview(block_id=b.block_id, status="verified", observation=observation, context_complete=True)
                    for b in blocks],
            source_complete=True, reading_order_correct=True, issues=[],
            observation="Every printed line and every block was matched by string comparison; no model call.")
        record.clear()
        record.update(page_idx=page, fingerprint=scope["fingerprint"], scoped_fingerprint=scope["scoped_fingerprint"],
                      images=images, delivery_sha256=bundle, reviewed_delivery_sha256=bundle,
                      block_ids=[b.block_id for b in blocks], declared_gap_ids=[], review=review.model_dump())
        self._record(record, "verified", "native_word_text_layer_match")
        return record

    async def _inspect_page(self, page, ir, inventory, run_path):
        record = self.state["pages"].setdefault(str(page), {"page_idx": page, "status": "not_reviewed"})
        bundle = delivery_digest(run_path)
        if self.limits.mode != "off" and (native := self._native_text_review(page, ir, inventory, run_path,
                                                                             record, bundle)):
            return native
        exhausted = self.state["usage"]["model_calls"] >= self.limits.max_model_calls or self._remaining() <= 0
        if exhausted and record.get("reason") and not record.get("review") and record.get("delivery_sha256") == bundle:
            return record
        if self.limits.mode == "off" or (exhausted and not record.get("review")):
            record.clear()
            record.update(page_idx=page)
            self._record(record, "not_reviewed", "review_disabled" if self.limits.mode == "off" else "review_budget_exhausted")
            return record
        if not record.get("review") and not all(self.available_budget()):
            self._record(record, "not_reviewed", "reserved_for_other_work")
            return record
        try:
            scope = self.build_page_review_input(page, ir, inventory, run_path)
        except ValueError as exc:
            if str(exc) != "source_image_unavailable":
                raise
            record.clear()
            record.update(page_idx=page)
            self._record(record, "not_reviewed", "source_image_unavailable")
            return record
        payload, images, evidence = scope["payload"], scope["images"], scope["evidence"]
        fingerprint, scoped_fingerprint = scope["fingerprint"], scope["scoped_fingerprint"]
        blocks = inventory.blocks(page, ir)
        context_blocks = payload["context_blocks"]
        if ((record.get("fingerprint") == fingerprint or record.get("scoped_fingerprint") == scoped_fingerprint)
                and record.get("review") and record.get("status") != "not_reviewed"):
            record.setdefault("reviewed_delivery_sha256", record.get("delivery_sha256"))
            record["scoped_fingerprint"] = scoped_fingerprint
            record["delivery_sha256"] = bundle
            if record.get("fingerprint") != fingerprint:
                record["reuse_reason"] = "unchanged_scoped_source_content_order_and_adjacency"
            return record
        if exhausted or self.limits.mode == "off":
            record.clear()
            record.update(page_idx=page)
            self._record(record, "not_reviewed", "review_disabled" if self.limits.mode == "off" else "review_budget_exhausted")
            return record
        record.clear()
        record.update(page_idx=page, status="not_reviewed", fingerprint=fingerprint, scoped_fingerprint=scoped_fingerprint,
                      images=evidence, delivery_sha256=bundle, reviewed_delivery_sha256=bundle,
                      block_ids=[b.block_id for b in blocks],
                      declared_gap_ids=[b.block_id for b in blocks if b.payload.get("agentic_gap")])
        if len(blocks) > 45 or len(str(payload)) > 65000 or len(images) > 15:
            if any(b.block_id in getattr(inventory, "word_images", {})
                   or b.payload.get("agentic_relationship_evidence_id") for b in blocks):
                self._record(record, "not_reviewed", "page_exceeds_native_image_relationship_review_scope")
                return record
            if (self.limits.recovery_enabled and inventory.pdf
                    and all(ir.has_physical_pages() or b.payload.get("agentic_source_region") for b in blocks)):
                return await self._inspect_segments(page, ir, inventory, run_path, record, blocks)
            self._record(record, "not_reviewed", "page_exceeds_review_context_limit")
            return record
        if self.limits.mode == "off":
            self._record(record, "not_reviewed", "review_disabled")
            return record
        try:
            readings = await self._word_picture_readings(blocks, inventory)
            record["word_image_readings"] = {bid: item.model_dump() for bid, item in readings.items()}
            if readings:
                from app.pipeline.image_source_review import RELATION_AUDIT_INSTRUCTIONS
                payload["independent_image_readings"] = record["word_image_readings"]
                payload["image_reading_scopes"] = {
                    origin: [b.block_id for b in blocks if b.block_id == origin
                             or b.payload.get("agentic_image_origin") == origin]
                    for origin in readings}
                payload["image_relationship_policy"] = RELATION_AUDIT_INSTRUCTIONS
                votes = await self._ocr_votes(blocks, readings, inventory, record)
                if votes:
                    # The audit model shares the drafting model's weights and its habit of
                    # "correcting" printed typos; it judges with OCR's reading in hand.
                    payload["independent_ocr_votes"] = votes
                    payload["ocr_vote_policy"] = OCR_VOTE_POLICY
            parts = reading_parts(payload) if len(str(payload)) > REGION_PAYLOAD_LIMIT else [payload]
            if not parts:
                self._record(record, "not_reviewed", "page_exceeds_image_relationship_context_limit")
                return record
            reviews = [await self._audit_page_contract(part, images) for part in parts]
            review = reviews[0] if len(reviews) == 1 else PageReview(
                blocks=[b for result in reviews for b in result.blocks],
                source_complete=all(result.source_complete for result in reviews),
                reading_order_correct=all(result.reading_order_correct for result in reviews),
                declared_gaps_only=all(result.declared_gaps_only for result in reviews),
                issues=list(dict.fromkeys(issue for result in reviews for issue in result.issues))[:30],
                observation=" | ".join(result.observation for result in reviews)[:1000])
            review = target_block_reviews(review, [b.block_id for b in blocks],
                                          [b["block_id"] for b in context_blocks])
            record["image_only_issues"] = (review.source_complete and review.reading_order_correct
                                           and not review.issues and all(b.status == "verified" for b in review.blocks))
            record["image_issues"] = []
            from app.pipeline.image_source_review import apply_group_relationship_review
            grouped = {origin: ids for origin, ids in payload.get("image_reading_scopes", {}).items()
                       if ids != [origin]}
            group_issues = apply_group_relationship_review(review, readings, grouped,
                {b.block_id: self._delivery_cache[0]._block_to_text(b) for b in blocks},
                documents=payload["delivered_documents"],
                chunks=payload["delivered_chunks"] if payload["chunks_required"] else None)
            record["image_issues"].extend({"block_id": bid, "failure_kind": "image_relationship_mismatch", "detail": issue[:500]}
                                         for bid, issue in group_issues.items())
            for finding in review.blocks:
                if finding.block_id in readings:
                    block = next(b for b in blocks if b.block_id == finding.block_id)
                    from app.pipeline.image_source_review import relationship_issue
                    reading = readings[finding.block_id]
                    delivered = self._delivery_cache[0]._block_to_text(block)
                    relation_issue = relationship_issue(reading, finding, delivered,
                        documents=payload["delivered_documents"], chunks=payload["delivered_chunks"]
                        if payload["chunks_required"] else None)
                    issue = relation_issue or word_picture_issue(reading, delivered)
                    if (issue and not relation_issue and reading.complete and reading.kind == "transcription"
                            and await self._settle_with_ocr(block.block_id, reading, delivered, inventory, run_path, record)):
                        issue = ""
                    if issue:
                        finding.status = "needs_review"
                        finding.context_complete = False
                        finding.observation = issue[:700]
                        review.source_complete = False
                        review.issues = (review.issues + [finding.block_id + ": " + issue])[:30]
                        record["image_issues"].append({"block_id": finding.block_id,
                            "failure_kind": "image_relationship_mismatch" if relation_issue else
                                            "unreadable_text" if not reading.complete else "reading_disagreement",
                            "detail": issue[:500]})
            existing = {item["block_id"] for item in record["image_issues"]}
            for finding in review.blocks:
                block = next(b for b in blocks if b.block_id == finding.block_id)
                if finding.status == "needs_review" and block.type == BlockType.IMAGE and finding.block_id not in existing:
                    record["image_issues"].append({"block_id": finding.block_id,
                        "failure_kind": "image_content_mismatch", "detail": finding.observation})
            record["review"] = review.model_dump()
            passed = (review.source_complete and review.reading_order_correct and not review.issues
                      and all(b.status == "verified" for b in review.blocks)
                      and not any(b.payload.get("agentic_gap") for b in blocks))
            self._record(record, "verified" if passed else "needs_review",
                         "source_and_delivery_checked" if passed else "source_delivery_mismatch")
        except BudgetExhausted:
            self._record(record, "not_reviewed", "review_budget_exhausted")
        except Exception as exc:
            from app.pipeline.image_reading import ReadingFormatError
            prefix = "reading_format_error: " if isinstance(exc, ReadingFormatError) else "review_failed: "
            self._record(record, "needs_review", prefix + str(exc)[:500])
        return record


    async def _inspect_segments(self, page, ir, inventory, run_path, record, blocks):
        from app.pipeline.document_recovery import regions
        from app.pipeline.stages.chunk import ChunkStage

        renderer = ChunkStage(self.config)
        renderer._asset_text = renderer._load_asset_retrieval_text(run_path / "outputs")
        excerpts = DeliveryExcerpts(run_path, content_blocks(ir))
        cores = regions("horizontal4")
        record["regions"] = []
        observations = {}
        try:
            for index, core in enumerate(cores):
                parts = await self._review_region(page, core, str(index), blocks, excerpts, renderer,
                                                  inventory, run_path, record)
                region = {"core": core, "review": merge_part_reviews([review for _, review in parts]).model_dump()}
                if len(parts) > 1:
                    region["parts"] = [{"core": part, "review": review.model_dump()} for part, review in parts]
                record["regions"].append(region)
                for _, review in parts:
                    for item in review.blocks:
                        observations.setdefault(item.block_id, []).append(item)
                self._save()
            if set(observations) != {b.block_id for b in blocks}:
                raise ValueError("Segments failed to cover all source blocks")
            from app.pipeline.recovery_models import LayoutVerification
            actual_docs, actual_chunks = excerpts.for_blocks(blocks, renderer)
            boundaries = layout_boundaries(actual_docs, actual_chunks)
            layouts = [LayoutVerification.model_validate(await self._call(
                "verify_layout", payload, inventory.image(page), LayoutVerification))
                for payload in layout_parts(page, boundaries, cores)]
            layout = layouts[0] if len(layouts) == 1 else LayoutVerification(
                reading_order_correct=all(part.reading_order_correct for part in layouts),
                observation=" | ".join(part.observation for part in layouts)[:1000],
                issues=[issue for part in layouts for issue in part.issues][:10])
            record["layout_review"] = layout.model_dump()
            if len(layouts) > 1:
                record["layout_review"]["parts"] = len(layouts)
            combined = {
                "blocks": [{"block_id": bid,
                            "status": "verified" if all(v.status == "verified" for v in values) else "needs_review",
                            "observation": " | ".join(v.observation for v in values)[:700]}
                           for bid, values in observations.items()],
                "source_complete": all(r["review"]["source_complete"] for r in record["regions"]),
                "reading_order_correct": layout.reading_order_correct and all(r["review"]["reading_order_correct"] for r in record["regions"]),
                "observation": "Entire source page covered by four contiguous source regions.",
                "issues": (layout.issues + [issue for r in record["regions"] for issue in r["review"]["issues"]])[:30],
            }
            record["review"] = combined
            passed = (combined["source_complete"] and combined["reading_order_correct"] and not combined["issues"]
                      and all(b["status"] == "verified" for b in combined["blocks"]))
            self._record(record, "verified" if passed else "needs_review",
                         "source_regions_and_delivery_checked" if passed else "source_delivery_mismatch")
        except BudgetExhausted:
            self._record(record, "not_reviewed", "region_review_budget_exhausted")
        except Exception as exc:
            self._record(record, "not_reviewed", "region_review_failed: " + str(exc)[:400])
        return record

    async def _review_region(self, page, core, name, blocks, excerpts, renderer, inventory, run_path, record):
        """Audit one source band; a band too large for one request is halved until each half fits.

        Each half is its own crop, so the model still sees everything printed there and can
        report content the parser missed, which a list of blocks split across requests could not.
        """
        from app.pipeline.document_recovery import crop_region

        selected = []
        for block in blocks:
            box = block.payload.get("agentic_source_region") or block.bbox_norm
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("region_source_coordinates_unavailable")
            if box[0] < core[2] and box[2] > core[0] and box[1] < core[3] and box[3] > core[1]:
                selected.append(block)
        docs, chunks = excerpts.for_blocks(selected, renderer)
        payload = {
            "page_idx": page, "target_region": core,
            "required_block_ids": [b.block_id for b in selected],
            "blocks": [{"block_id": b.block_id, "untrusted_extracted_text": renderer._block_to_text(b)}
                       for b in selected],
            "delivered_documents": docs, "delivered_chunks": chunks,
            "chunks_required": self.config.package.generate_chunks,
            "policy": "Verify only content inside the core region; repeated overlap is context, not duplicated content.",
        }
        # Halve top/bottom; a strip already thin is halved left/right (a logo wall: 120
        # logos within a quarter of one page, 2026-10-08).
        if core[3] - core[1] >= 2 * MIN_REGION_HEIGHT:
            middle = (core[1] + core[3]) // 2
            halves = ([core[0], core[1], core[2], middle], [core[0], middle, core[2], core[3]])
        elif core[2] - core[0] >= 2 * MIN_REGION_HEIGHT:
            middle = (core[0] + core[2]) // 2
            halves = ([core[0], core[1], middle, core[3]], [middle, core[1], core[2], core[3]])
        else:
            halves = None

        async def in_halves():
            return [*await self._review_region(page, halves[0], name + "a", blocks,
                                               excerpts, renderer, inventory, run_path, record),
                    *await self._review_region(page, halves[1], name + "b", blocks,
                                               excerpts, renderer, inventory, run_path, record)]

        if len(selected) > 45 or len(str(payload)) > REGION_PAYLOAD_LIMIT:
            if not halves:
                raise ValueError("region_exceeds_review_context_limit")
            return await in_halves()
        image, crop = crop_region(inventory, page, core, inventory.directory / f"segments-{page}", name)
        payload["crop_bbox"] = crop
        try:
            review = PageReview.model_validate(await self._call("audit_page", payload, image, PageReview))
        except BudgetExhausted:
            raise
        except Exception as exc:
            # A request that fits can still need a longer answer than the model may give
            # (40 logos, one observation each): fewer blocks per crop.
            if not str(exc).startswith("output_truncated") or not halves or len(selected) < 2:
                raise
            return await in_halves()
        ids = [b.block_id for b in review.blocks]
        if len(ids) != len(set(ids)) or set(ids) != {b.block_id for b in selected}:
            raise ValueError("Region review omitted/duplicated/invented IDs")
        record["images"].append({"path": str(image.relative_to(run_path)),
                                 "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                                 "label": f"Core region {core}"})
        return [(core, review)]

    async def _repair(self, block, issue, ir, run_path, cache, source_path, rebuild, inventory):
        block_id = block.block_id
        # Never repeat a completed failed repair just because the run resumed.
        if block_id in self.state["repair_completed"]:
            return ir
        remaining_calls = self.limits.max_model_calls - self.state["usage"]["model_calls"]
        remaining_actions = self.limits.max_actions - self.state["usage"]["actions"]
        # Three local model calls plus an independent final page recheck.
        if remaining_calls < 4 or remaining_actions <= 0 or self._remaining() <= 0:
            return ir
        cfg = self.config.model_copy(deep=True)
        specs = self.state.setdefault("repair_specs", {})
        if block_id in specs:
            cfg.agentic = cfg.agentic.model_validate(specs[block_id])
        else:
            cfg.agentic.max_model_calls = remaining_calls - 1
            cfg.agentic.max_seconds = self._remaining()
            cfg.agentic.max_actions = min(remaining_actions, self.limits.max_attempts_per_block)
            specs[block_id] = cfg.agentic.model_dump()
        directory = run_path / "agentic/repairs" / hashlib.sha256(block_id.encode()).hexdigest()[:16]
        self.state["inflight_repair"] = {"block_id": block_id, "seconds": self._remaining()}
        self._save()
        before_seconds = self.state["usage"]["seconds"]
        started = time.monotonic()
        result = None
        try:
            proxy = SharedModel(self)
            repair_ir = ir.model_copy(deep=True)
            if repair_ir.has_physical_pages():
                for page in repair_ir.pages:
                    if page.page_idx in inventory.pages and (image := inventory.image(page.page_idx)):
                        page.page_image_path = str(image.relative_to(run_path))
            repaired, result = await asyncio.wait_for(
                AgenticRepair(cfg, proxy, proxy).run(
                    repair_ir, run_path, parse_cache_path=cache, source_path=source_path,
                    issues_override=[issue], state_directory=directory,
                ), min(cfg.agentic.max_seconds, self._remaining()),
            )
            self.state["repair_completed"].append(block_id)
            self.state["repair_reports"][block_id] = result
            if cfg.agentic.mode == "apply" and (result["accepted"] or result["kept"]):
                payload = repaired.get_block(block_id).payload
                self.state["repairs"][block_id] = {
                    k: v for k, v in payload.items() if k.startswith("agentic_") or k == "verified_semantics"
                }
                self._save()
                ir.get_block(block_id).payload.update(self.state["repairs"][block_id])
                if result["accepted"]:
                    await rebuild(ir)
        except TimeoutError:
            self.state["repair_completed"].append(block_id)
            self.state["events"].append({"block_id": block_id, "status": "repair_timeout"})
        finally:
            # Shared model calls are already charged; charge conversion/render overhead once.
            counted = self.state["usage"]["seconds"] - before_seconds
            self.state["usage"]["seconds"] += max(0, time.monotonic() - started - counted)
            self.state.pop("inflight_repair", None)
            self._save()
        return ir

    async def run(self, ir, run_path, parse_cache_path=None, source_path=None, rebuild=None, progress=None,
                  elapsed_seconds=0.0):
        from app.pipeline.table_structure import materialize_table_structures

        if self.config.agentic.mode == "apply":
            ir = ir.model_copy(deep=True)
            materialize_table_structures(ir)
        # Keep configured hard ceilings intact; effective limits belong to this run.
        self.limits = self.config.agentic.model_copy(deep=True)
        directory = run_path / "agentic"
        directory.mkdir(parents=True, exist_ok=True)
        self.state_path = directory / "review_state.json"
        if source_path is None:
            source_path = next((settings.get_doc_path(ir.doc_id) / "source").glob("original.*"), None)
            if source_path is None and Path(ir.source.path).is_file():
                source_path = Path(ir.source.path)
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path and source_path.is_file() else None
        from app.pipeline.document_recovery import SKILLS
        skill_hash = digest([(str(p.relative_to(SKILLS)), p.read_text()) for p in sorted(SKILLS.glob("*/SKILL.md"))])
        fingerprint = digest([REVIEW_VERSION, skill_hash, input_identity(ir), source_sha, self.limits.model_dump(),
                              VLMAdapter.compute_config_hash(self.config.vlm),
                              VLMAdapter.compute_config_hash(self.config.review_vlm)])
        saved = read_json(self.state_path, {})
        if saved.get("fingerprint") == fingerprint:
            self.state = saved
            if pending := self.state.pop("inflight_repair", None):
                self.state["usage"]["seconds"] += pending["seconds"]
                self.state["events"].append({"block_id": pending["block_id"], "status": "interrupted_repair_reserved"})
        else:
            if saved:
                atomic_write_json(directory / ("previous-review-" + saved["fingerprint"][:16] + ".json"), saved)
            self.state = {"fingerprint": fingerprint, "version": REVIEW_VERSION, "mode": self.limits.mode,
                          "usage": {"model_calls": 0, "seconds": 0.0, "tokens": 0, "actions": 0},
                          "calls": [], "events": [], "pages": {}, "repairs": {},
                          "repair_completed": [], "repair_reports": {}}
        self._save()
        working = ir.model_copy(deep=True)
        changed = False
        for block_id, payload in self.state["repairs"].items():
            block = working.get_block(block_id)
            if block and any(block.payload.get(k) != v for k, v in payload.items()):
                block.payload.update(payload)
                changed = True
        from app.pipeline.document_recovery import RecoveryController, restore_recovery
        working, recovered = restore_recovery(self, working, run_path)
        changed = changed or recovered
        if changed and rebuild:
            await rebuild(working)
        inventory = SourceInventory(working, run_path, source_path)
        if self.limits.mode == "off":
            await inventory.prepare(allow_conversion=False)
        else:
            before = self.state["usage"]["seconds"]
            can_convert = self._remaining() > 0
            timeout = min(120, max(1, self._remaining()))
            self.state["usage"]["seconds"] += timeout
            self._save()
            started = time.monotonic()
            try:
                await asyncio.wait_for(inventory.prepare(allow_conversion=can_convert), timeout)
            except Exception as exc:
                inventory.error = "Source inventory failed: " + str(exc)[:300]
                inventory.fallback()
            finally:
                self.state["usage"]["seconds"] = before + time.monotonic() - started
                self._save()
        if not inventory.complete:
            # An exhausted-budget resume must never shrink a previously observed
            # source inventory down to only the pages the parser happened to keep.
            inventory.pages = sorted(set(inventory.pages) | {int(p) for p in self.state["pages"]})
        from app.pipeline.review_budget import document_budget, timing_summary
        budget = document_budget(self.config.agentic, len(inventory.pages), elapsed_seconds=elapsed_seconds)
        self.limits.max_model_calls = budget["max_model_calls"]
        self.limits.max_seconds = budget["max_seconds"]
        self.state["budget"] = budget
        # Planner-only tuning keeps previously verified evidence and spent usage valid.
        self.state["scheduler_version"] = SCHEDULER_VERSION
        self.model = self.model or RegionModel(VLMAdapter(self.config.vlm), self.limits.max_output_tokens)
        self.reviewer = self.reviewer or RegionModel(VLMAdapter(self.config.review_vlm), self.limits.max_output_tokens)
        risks = {i["block_id"]: i for i in visual_issues(working)}
        priorities = {p: max((risks.get(b.block_id, {}).get("priority", 0)
                             for b in inventory.blocks(p, working)), default=0) for p in inventory.pages}
        order = sorted(inventory.pages, key=lambda p: (-priorities[p], p))
        # Every source page is planned before any call or early exit.
        for page in inventory.pages:
            self.state["pages"].setdefault(str(page), {"page_idx": page, "status": "not_reviewed", "reason": "queued"})
        self._save()
        recovery_runner = (RecoveryController(self, inventory, run_path, rebuild, parse_cache_path)
                           if self.limits.recovery_enabled and rebuild and self.limits.mode != "off" else None)
        self._reserve = (0, self.limits.max_seconds * self.limits.recovery_reserve_share) if recovery_runner else None
        self._protected_budget = (0, 0)
        self._phase = "audit"
        for completed, page in enumerate(order):
            if progress:
                await progress({"phase": "source_review", "total": len(order), "completed": completed,
                                "current": {"page_idx": page}, "message": "Reviewing source against final documents and chunks"})
            record = await self._inspect(page, working, inventory, run_path)
            if not rebuild or self.limits.mode == "off":
                continue
            if self.limits.recovery_enabled:
                # Queue recovery only after every source page has had its initial inspection.
                continue
            if self.limits.recovery_enabled or record["status"] != "needs_review":
                continue
            findings = [r for r in record.get("review", {}).get("blocks", []) if r["status"] == "needs_review"]
            for finding in findings:
                block = working.get_block(finding["block_id"])
                if block.type not in {BlockType.TABLE, BlockType.IMAGE}:
                    continue
                issue = {"block_id": block.block_id, "page_idx": block.page_idx,
                         "risk": finding["observation"], "priority": priorities[page]}
                working = await self._repair(block, issue, working, run_path, parse_cache_path, source_path, rebuild, inventory)
            # Repair success is not final delivery approval.
            await self._inspect(page, working, inventory, run_path)
        self._phase = "recovery"
        if self.limits.recovery_enabled and rebuild and self.limits.mode != "off":
            if progress:
                await progress({"phase": "recovery", "message": "Agent selecting source recovery tools"})
            working = await recovery_runner.run(working)
        # Recovery can introduce raw HTML tables. Bind the final audit to the
        # normalized snapshot that will actually be persisted and published.
        if self.limits.mode == "apply":
            materialize_table_structures(working)
        self._phase = "final_review"
        # Changed paragraphs can affect earlier pages; invalidate/recheck their exact inputs too.
        final_order = sorted(order, key=lambda page: (bool(self.state["pages"].get(str(page), {}).get("review")),
                                                     order.index(page)))
        for completed, page in enumerate(final_order):
            if progress:
                await progress({"phase": "final_review", "total": len(order), "completed": completed,
                                "current": {"page_idx": page}, "message": "Verifying final delivery after recovery"})
            await self._inspect(page, working, inventory, run_path)
        recovery = self.state.get("recovery", {})
        for issue in recovery.get("issues", {}).values():
            if issue["status"] == "resolved" and "page_idx" in issue:
                if self.state["pages"].get(str(issue["page_idx"]), {}).get("status") != "verified":
                    issue["status"] = "applied_partial" if any(b.payload.get("agentic_gap") for b in working.blocks
                                                                    if issue["page_idx"] in inventory.mapping.get(b.block_id, [])) else "applied_unverified"
            if issue["status"] == "located":
                indices = inventory.mapping.get(issue.get("block_id") or issue.get("located_block_id"), [])
                issue["status"] = ("resolved" if indices and all(
                    self.state["pages"].get(str(p), {}).get("status") == "verified" for p in indices
                ) else "located_unverified")
        if recovery:
            atomic_write_json(run_path / "outputs/recovery_report.json", recovery, ensure_ascii=False, indent=2)
        pages = [self.state["pages"][str(p)] for p in inventory.pages]
        blocks = []
        for block in content_blocks(working):
            indices = inventory.mapping.get(block.block_id, [])
            observations = []
            usable = []
            dependencies = set(block.payload.get("agentic_depends_on_block_ids", []))
            for page in indices:
                record = self.state["pages"][str(page)]
                found = next((b for b in record.get("review", {}).get("blocks", [])
                              if b["block_id"] == block.block_id), None)
                if found:
                    dependencies.update(found.get("depends_on_block_ids", []))
                usable.append(reviewed_block_usable(record, found))
                # A page-level discrepancy must not leave its constituent blocks
                # certified by contradictory per-block model labels.
                observations.append("needs_review" if record["status"] == "needs_review" else
                                    found["status"] if found else "not_reviewed")
            status = ("needs_review" if "needs_review" in observations else "verified"
                      if observations and all(v == "verified" for v in observations) else "not_reviewed")
            blocks.append({"block_id": block.block_id, "source_pages": indices, "status": status,
                           "reason": "unmapped_source_block" if not indices else "page_observations",
                           "rag_eligible": bool(indices and all(usable) and not block.payload.get("agentic_gap")),
                           "depends_on_block_ids": sorted(dependencies)})
            if self.limits.mode == "apply":
                block.payload["agentic_needs_review"] = status != "verified"
        report = {"schema_version": 1, "version": REVIEW_VERSION, "mode": self.limits.mode,
                  "inventory_complete": inventory.complete, "inventory_error": inventory.error,
                  "source_sha256": source_sha, "pages": pages, "blocks": blocks,
                  "excluded_blocks": [b.block_id for b in working.blocks if b not in content_blocks(working)],
                  "delivery_sha256": delivery_digest(run_path), "ir_sha256": digest(input_identity(working)),
                  "input_fingerprint": fingerprint,
                  "usage": self.state["usage"], "budget": self.state["budget"],
                  "scheduler_version": SCHEDULER_VERSION,
                  "timing": timing_summary(self.state["calls"]), "repairs": self.state["repair_reports"],
                  "recovery": self.state.get("recovery", {})}
        report["summary"] = coverage_summary(report)
        atomic_write_json(run_path / "outputs/review_coverage.json", report, ensure_ascii=False, indent=2)
        atomic_write_json(run_path / "outputs/agentic_report.json", {
            "mode": self.limits.mode, "version": REVIEW_VERSION, "review": report["summary"],
            "usage": report["usage"], "budget": report["budget"], "timing": report["timing"],
            "repairs": report["repairs"], "recovery": report["recovery"],
        }, ensure_ascii=False, indent=2)
        self._save()
        if progress:
            await progress({"phase": "source_review", "total": len(order), "completed": len(order),
                            "message": "Automatic processing finished; unresolved content is recorded as gaps"})
        return (working if self.limits.mode == "apply" else ir), report

