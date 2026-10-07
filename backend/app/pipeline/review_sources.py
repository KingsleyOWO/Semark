"""Source-page inventory and excerpts from the actual delivery, never risk-filtered."""
import copy
import hashlib
import json
import re
from pathlib import Path

import fitz

from app.pipeline.agentic_evidence import page_image
from app.pipeline.content_coverage import normalized_text, plain_text
from app.pipeline.final_delivery import delivered_documents, read_json, read_jsonl
from app.pipeline.review_policy import partial_review_accepted
from app.pipeline.stages.normalize import is_non_content
from app.pipeline.word_mapping import locate_word_blocks, word_image_matches
from app.pipeline.word_mapping import word_block_pages as word_block_pages


def needs_content_recovery(record):
    """A failed reviewer response is not evidence that document content needs rewriting."""
    reason = record.get("reason", "")
    review = record.get("review", {})
    gaps = set(record.get("declared_gap_ids", []))
    if partial_review_accepted(review, gaps, record.get("block_ids")):
        return False  # Explicitly verified partial completion is terminal for this evidence.
    return (reason == "page_exceeds_review_context_limit" or
            record.get("status") == "needs_review" and
            not reason.startswith(("review_failed:", "source_review_error:", "reading_format_error:")))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def compact_delivery_excerpts(excerpts):
    """Join adjacent lines with identical provenance, retaining every source byte.

    Repeating the same file/owner/binding for each line can dominate the model
    input. A range carries the same evidence without reclassifying or deduping
    content. Ownership, untrusted bindings and discontinuities stay separate.
    """
    result = []
    for excerpt in excerpts:
        current = copy.deepcopy(excerpt)
        previous = result[-1] if result else None
        if previous is not None:
            ignored = {"text", "paragraph", "paragraph_end"}
            same_provenance = ({k: v for k, v in current.items() if k not in ignored}
                               == {k: v for k, v in previous.items() if k not in ignored})
            previous_end = previous.get("paragraph_end", previous.get("paragraph"))
            if (same_provenance and isinstance(previous_end, int)
                    and current.get("paragraph") == previous_end + 1):
                previous["text"] += "\n" + current["text"]
                previous["paragraph_end"] = current.get("paragraph_end", current["paragraph"])
                continue
        result.append(current)
    return result



def review_scope_digest(version, payload, evidence):
    """Reuse exact scoped content/order despite unrelated global numbering shifts."""
    scoped = copy.deepcopy(payload)
    orders = [b["reading_order"] for b in scoped.get("blocks", []) if "reading_order" in b]
    if orders:
        for block in scoped["blocks"]:
            if "reading_order" in block:
                block["reading_order"] -= min(orders)
    positions = {}
    for excerpt in scoped.get("delivered_documents", []):
        previous = positions.get(excerpt["file"])
        original = excerpt["paragraph"]
        end = excerpt.get("paragraph_end", original)
        # Preserve adjacency and ordering; counts of omitted, unrelated paragraphs
        # do not change the evidence actually supplied to the reviewer.
        relative = 0 if previous is None else previous[1] + (1 if original == previous[0] + 1 else 2)
        relative_end = relative + (end - original)
        positions[excerpt["file"]] = (end, relative_end)
        excerpt["paragraph"] = relative
        if "paragraph_end" in excerpt:
            excerpt["paragraph_end"] = relative_end
    for chunk in scoped.get("delivered_chunks", []):
        chunk.pop("chunk_id", None)  # content, source membership and list order remain binding
    # Evidence can be copied into an isolated candidate directory. Pixels and
    # their audit roles bind identity; a storage path does not change evidence.
    scoped_evidence = [{k: v for k, v in item.items() if k != "path" or not item.get("sha256")}
                       for item in evidence]
    return digest([version, scoped, scoped_evidence])


def delivery_digest(run_path):
    outputs = run_path / "outputs"
    files = delivered_documents(outputs)
    files += [p for p in (outputs / "chunks.jsonl", outputs / "assets_index.jsonl") if p.is_file()]
    return digest([(str(p.relative_to(run_path)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])


def content_blocks(ir):
    return [b for b in ir.blocks if not is_non_content(b)]


def input_identity(ir):
    value = ir.to_dict()
    value.pop("run_id", None)
    for block in value["blocks"]:
        block["payload"] = {k: v for k, v in block["payload"].items()
                            if (not k.startswith("agentic_") and k != "verified_semantics")
                            or k in {"agentic_source_pages", "agentic_source_sha256", "agentic_source_region",
                                       "agentic_information_status", "agentic_region_kind", "agentic_gap", "agentic_source_image", "agentic_content_kind", "agentic_image_origin", "agentic_image_region", "agentic_depends_on_block_ids", "agentic_relationship_evidence_id"}}
    return value


class SourceInventory:
    def __init__(self, ir, run_path, source_path):
        self.ir, self.run_path, self.source_path = ir, run_path, source_path
        self.pdf = None
        self.directory = run_path / "agentic/evidence"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pages = []
        self.mapping = {}
        self.mapping_details = {}
        self.word_images = {}
        self.image_positions = {}
        self.complete = False
        self.error = ""

    async def prepare(self, *, allow_conversion=True):
        source = self.source_path
        if source and source.is_file():
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            if source.suffix.lower() in {".doc", ".docx"}:
                from app.pipeline.stages.parse import ParseStage
                folder = self.directory / ("word-" + sha)
                pdf = folder / (source.stem + ".pdf")
                if not pdf.is_file():
                    if not allow_conversion:
                        self.error = "Word conversion unavailable within the remaining review budget"
                        self.fallback()
                        return
                    pdf, error = await ParseStage._convert_office_to_pdf(source, folder)
                    if error:
                        self.error = error
                        self.fallback()
                        return
                self.pdf = pdf
            elif source.suffix.lower() == ".pdf":
                self.pdf = source
            else:
                # PyMuPDF also opens supported original image files.
                try:
                    with fitz.open(source) as original:
                        if not original.is_pdf:
                            folder = self.directory / ("source-" + sha)
                            folder.mkdir(exist_ok=True)
                            target = folder / "source.pdf"
                            if not target.is_file():
                                target.write_bytes(original.convert_to_pdf())
                            self.pdf = target
                except Exception:
                    self.error = "Source format has no reliable physical-page renderer"
            if self.pdf:
                try:
                    with fitz.open(self.pdf) as document:
                        self.pages = list(range(len(document)))
                        texts = [normalized_text(p.get_text()) for p in document] if not self.ir.has_physical_pages() else []
                    self.complete = bool(self.pages)
                    self.directory = self.directory / ("review-" + sha + "-" + hashlib.sha256(self.pdf.read_bytes()).hexdigest()[:16])
                    self.directory.mkdir(exist_ok=True)
                    pdf_sha = hashlib.sha256(self.pdf.read_bytes()).hexdigest()
                    blocks = content_blocks(self.ir)
                    overrides = {}
                    for block in blocks:
                        override = block.payload.get("agentic_source_pages")
                        if (block.payload.get("agentic_source_sha256") == pdf_sha and
                                isinstance(override, list) and override and
                                all(type(p) is int and p in self.pages for p in override)):
                            overrides[block.block_id] = sorted(set(override))
                    if self.ir.has_physical_pages():
                        self.mapping = {b.block_id: overrides.get(b.block_id,
                                        [b.page_idx] if b.page_idx in self.pages else []) for b in blocks}
                    else:
                        matches = word_image_matches(blocks, self.source_path, self.pdf, evidence_dir=self.directory)
                        self.image_positions = matches
                        self.word_images = {bid: self.directory / values[0]["source_image"]
                                            for bid, values in matches.items() if values and values[0].get("source_image")}
                        self.mapping, self.mapping_details = locate_word_blocks(
                            blocks, texts, image_matches=matches, overrides=overrides)
                        from app.pipeline.delivery import atomic_write_json
                        atomic_write_json(self.directory / "word-page-mapping.json", {
                            "version": "word-layout-v1", "source_sha256": sha, "pdf_sha256": pdf_sha,
                            "blocks": self.mapping_details, "image_matches": matches,
                        }, ensure_ascii=False, indent=2)
                    return
                except Exception as exc:
                    self.error = "Source cannot be opened: " + str(exc)[:300]
        self.fallback()

    def fallback(self):
        # Parsed page images cannot prove the source has no omitted pages.
        self.pages = sorted({p.page_idx for p in self.ir.pages} | {b.page_idx for b in self.ir.blocks}) or [0]
        self.mapping = {b.block_id: [b.page_idx] if self.ir.has_physical_pages() else []
                        for b in content_blocks(self.ir)}
        self.error = self.error or "Original source-page inventory unavailable"
        self.complete = False

    def image(self, index):
        if self.pdf:
            target = self.directory / f"p{index:04d}.png"
            if not target.is_file():
                with fitz.open(self.pdf) as doc:
                    doc[index].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(target)
            return target
        return page_image(self.ir, index, self.run_path)

    def context_image(self, index):
        """Lower-cost neighboring overview; target and cross-page evidence stay full-size."""
        from PIL import Image
        source = self.image(index)
        if not source:
            return None
        target = self.directory / f"context-{index:04d}.png"
        if not target.is_file():
            with Image.open(source) as picture:
                picture.thumbnail((480, 680))
                picture.save(target)
        return target

    def blocks(self, index, ir):
        return sorted((b for b in content_blocks(ir) if index in self.mapping.get(b.block_id, [])),
                      key=lambda b: (b.reading_order, b.block_id))


class DeliveryExcerpts:
    """Read stable actual-output lines and bind them to source occurrences.

    A size/line-count window is not a source boundary: one image expansion would
    shift every later window. Lines retain the bytes/order actually delivered.
    Exact source markers win; repeated exact lines are assigned in source order
    only when occurrence counts agree. Ambiguity remains explicit, never an
    arbitrary top-three match or a substitute for the delivered text.
    """
    def __init__(self, run_path, source_blocks=None):
        outputs = run_path / "outputs"
        self.source_blocks = source_blocks
        self.paragraphs = []
        self._bound = False
        assets = {row.get("asset_id"): row.get("block_id") for row in read_jsonl(outputs / "assets_index.jsonl")
                  if row.get("block_id")}
        # Standalone asset documents already declare their source through the
        # document and asset indexes. Repeated company names inside a picture
        # must not be attributed to matching prose on unrelated source pages.
        document_owners = {Path(row["file"]).name: assets[row["document_id"]]
                           for row in read_json(outputs / "documents_index.json", [])
                           if row.get("file") and row.get("document_id") in assets}
        for path in delivered_documents(outputs):
            file_owner = document_owners.get(path.name)
            owner, index = None, 0
            for number, line in enumerate(path.read_text().splitlines(keepends=True), 1):
                if not line.strip():
                    owner = None
                    continue
                marker = re.fullmatch(r"\s*<!-- source-block:([^\s>]+) -->\s*", line)
                if marker:
                    owner = marker[1]
                    continue
                self.paragraphs.append({
                    "file": str(path.relative_to(run_path)), "paragraph": index, "line": number,
                    "text": line, "normalized": normalized_text(plain_text(line)),
                    "block_ids": [owner or file_owner] if owner or file_owner else [],
                    "binding": "source_marker" if owner else "asset_document" if file_owner else "unmatched",
                })
                index += 1
        self.chunks = read_jsonl(outputs / "chunks.jsonl")

    def _bind(self, blocks, renderer):
        if self._bound:
            return
        sources = sorted(self.source_blocks if self.source_blocks is not None else blocks,
                         key=lambda block: (block.reading_order, block.block_id))
        atoms, full_text = {}, {}
        for block in sources:
            text = renderer._block_to_text(block)
            full_text[block.block_id] = normalized_text(plain_text(text))
            for line in text.splitlines():
                if atom := normalized_text(plain_text(line)):
                    atoms.setdefault(atom, []).append(block.block_id)
        # A marker for a removed/unknown block cannot make output disappear.
        known_ids = {block.block_id for block in sources}
        for paragraph in self.paragraphs:
            if paragraph["binding"] in {"source_marker", "asset_document"} and not set(paragraph["block_ids"]) <= known_ids:
                paragraph.update(block_ids=[], binding="unmatched")
        # Each delivered file can contain the same source (main + child document).
        groups = {}
        for paragraph in self.paragraphs:
            if paragraph["binding"] == "unmatched" and paragraph["normalized"]:
                groups.setdefault((paragraph["file"], paragraph["normalized"]), []).append(paragraph)
        for (_, text), paragraphs in groups.items():
            candidates = atoms.get(text, [])
            if candidates:
                if len(candidates) == len(paragraphs):
                    for paragraph, bid in zip(paragraphs, candidates, strict=True):
                        paragraph.update(block_ids=[bid], binding="source_text")
                else:
                    owners = sorted(set(candidates))
                    for paragraph in paragraphs:
                        paragraph.update(block_ids=owners, binding="source_text" if len(owners) == 1
                                         else "ambiguous_source_occurrence")
                continue
            # Rendering may add a label/bullet or flatten a table row. Prefer a
            # complete exact span; fuzzy matches are only a fallback, with ties
            # retained as ambiguity rather than ranked by a global line number.
            exact = [bid for bid, source in full_text.items() if source and (
                len(text) >= 12 and text in source or len(source) >= 8 and source in text)]
            if exact:
                owners, binding = exact, "contained_source_text"
            else:
                grams = {text[i:i + 5] for i in range(0, len(text) - 4, 3)}
                scores = []
                for bid, source in full_text.items():
                    hits = sum(gram in source for gram in grams)
                    density = hits / min(len(grams), max(1, len(source) // 3)) if grams else 0
                    if hits >= 3 and density >= 0.4:
                        scores.append((density, bid))
                best = max((score for score, _ in scores), default=0)
                owners = [bid for score, bid in scores if score == best]
                binding = "fuzzy_source_text"
            for paragraph in paragraphs:
                paragraph.update(block_ids=sorted(owners), binding=binding if len(owners) <= 1
                                 else "ambiguous_" + binding)
        # Resolve repeated labels using surrounding uniquely owned output, not
        # every occurrence of the same company name throughout the document.
        # A short label can be absent from one figure's rendered main content;
        # global occurrence counts alone would then make all later labels ambiguous.
        source_order = {b.block_id: i for i, b in enumerate(sources)}
        per_file = {}
        for paragraph in self.paragraphs:
            per_file.setdefault(paragraph["file"], []).append(paragraph)
        for paragraphs in per_file.values():
            following, next_order = [], len(sources)
            for paragraph in reversed(paragraphs):
                following.append(next_order)
                if len(paragraph["block_ids"]) == 1:
                    next_order = source_order[paragraph["block_ids"][0]]
            previous_order = -1
            for paragraph, next_order in zip(paragraphs, reversed(following), strict=True):
                if len(paragraph["block_ids"]) == 1:
                    previous_order = source_order[paragraph["block_ids"][0]]
                elif paragraph["binding"] == "ambiguous_source_occurrence" and previous_order <= next_order:
                    bounded = [bid for bid in paragraph["block_ids"]
                               if previous_order <= source_order[bid] <= next_order]
                    if len(bounded) == 1:
                        paragraph.update(block_ids=bounded, binding="source_text")
        # Never hide output that failed source matching. Bind it as untrusted
        # context to its actual output neighbors, not as transcribed source text.
        # This catches an inserted hallucinated line without invalidating every
        # unrelated page merely because its absolute line number moved.
        files = {}
        # Callers without a full source inventory request a text lookup only;
        # they cannot establish that unmatched output belongs to those blocks.
        if self.source_blocks is not None:
            for paragraph in self.paragraphs:
                files.setdefault(paragraph["file"], []).append(paragraph)
        for paragraphs in files.values():
            next_owners, following = [], []
            for paragraph in reversed(paragraphs):
                following.append(next_owners)
                if paragraph["block_ids"]:
                    next_owners = paragraph["block_ids"]
            previous = []
            for paragraph, following_owners in zip(paragraphs, reversed(following), strict=True):
                if paragraph["block_ids"]:
                    previous = paragraph["block_ids"]
                    continue
                owners = sorted(set(previous + following_owners))
                paragraph.update(block_ids=owners or [b.block_id for b in sources],
                                 binding="unbound_neighbor_context" if owners else "unbound_document")
        self._bound = True

    def for_blocks(self, blocks, renderer):
        self._bind(blocks, renderer)
        block_ids = {b.block_id for b in blocks}
        # The file line number only locates spans for the documents index; the audit never sees it.
        documents = [{k: v for k, v in paragraph.items() if k not in {"normalized", "line"}}
                     for paragraph in self.paragraphs if block_ids.intersection(paragraph["block_ids"])]
        chunks = [{"chunk_id": c.get("chunk_id"), "content": c.get("content", ""),
                   "block_ids": c.get("block_ids", [])}
                  for c in self.chunks if block_ids.intersection(c.get("block_ids", []))]
        return documents, chunks


def coverage_summary(report):
    pages = report.get("pages", [])
    blocks = report.get("blocks", [])
    result = {"mode": report.get("mode"), "total_pages": len(pages), "total_blocks": len(blocks),
              "inventory_complete": report.get("inventory_complete", False)}
    for status in ("verified", "needs_review", "not_reviewed"):
        result[status + "_pages"] = sum(p.get("status") == status for p in pages)
        result[status + "_blocks"] = sum(b.get("status") == status for b in blocks)
    result["complete"] = bool(
        result["inventory_complete"] and pages
        and result["verified_pages"] == len(pages)
        and result["verified_blocks"] == len(blocks)
    )
    return result



def validate_review_coverage(ir, run_path, config):
    """Validate evidence and the exact delivered bundle, independent of risk rules."""
    from app.pipeline.final_delivery import read_json
    report = read_json(run_path / "outputs/review_coverage.json", {})
    if not report:
        return {
            "mode": config.agentic.mode, "complete": False, "inventory_complete": False,
            "total_pages": len(ir.pages), "verified_pages": 0, "needs_review_pages": 0,
            "not_reviewed_pages": len(ir.pages), "total_blocks": len(content_blocks(ir)),
            "verified_blocks": 0, "needs_review_blocks": 0,
            "not_reviewed_blocks": len(content_blocks(ir)), "reason": "source_review_missing",
        }
    summary = coverage_summary(report)
    valid = report.get("mode") == config.agentic.mode
    valid = valid and report.get("delivery_sha256") == delivery_digest(run_path)
    valid = valid and report.get("ir_sha256") == digest(input_identity(ir))
    expected = {b.block_id for b in content_blocks(ir)}
    ids = [b.get("block_id") for b in report.get("blocks", [])]
    valid = valid and len(set(ids)) == len(ids) and set(ids) == expected
    pages = report.get("pages", [])
    indices = [p.get("page_idx") for p in pages]
    valid = valid and indices == list(range(len(pages)))
    for page in pages:
        for image in page.get("images", []):
            path = run_path / image.get("path", "")
            valid = valid and (path.resolve().is_relative_to(run_path.resolve()) and path.is_file()
                               and hashlib.sha256(path.read_bytes()).hexdigest() == image.get("sha256"))
        if page.get("status") != "verified":
            continue
        review = page.get("review", {})
        valid = valid and bool(page.get("images")) and bool(review.get("source_complete"))
        valid = valid and bool(review.get("reading_order_correct")) and not review.get("issues")
        if regions := page.get("regions"):
            # Four contiguous bands must cover the entire page; an omitted segment cannot pass.
            valid = valid and [r.get("core") for r in regions] == [
                [0, 0, 1000, 250], [0, 250, 1000, 500],
                [0, 500, 1000, 750], [0, 750, 1000, 1000],
            ]
            valid = valid and all(
                r.get("review", {}).get("source_complete") and r["review"].get("reading_order_correct")
                and not r["review"].get("issues") for r in regions
            )
            valid = valid and page.get("layout_review", {}).get("reading_order_correct")
            valid = valid and not page.get("layout_review", {}).get("issues")
        for image in page.get("images", []):
            path = run_path / image.get("path", "")
            valid = valid and (path.resolve().is_relative_to(run_path.resolve()) and path.is_file()
                               and hashlib.sha256(path.read_bytes()).hexdigest() == image.get("sha256"))
    summary["evidence_valid"] = bool(valid)
    summary["complete"] = summary["complete"] and bool(valid)
    summary["reason"] = ("source_review_complete" if summary["complete"] else
                         "source_review_stale_or_invalid" if not valid else "source_review_incomplete")
    return summary
