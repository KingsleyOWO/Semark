"""Model-directed recovery with bounded evidence, isolated candidates and replayable commits."""
import asyncio
import copy
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

import fitz

from app.models.document_ir import Block, BlockType, DocumentIR, PageInfo
from app.pipeline.agentic import BudgetExhausted, ModelOutputError, visual_issues
from app.pipeline.content_coverage import TextCoverage, normalized_text, plain_text
from app.pipeline.delivery import atomic_write_json
from app.pipeline.final_delivery import delivered_documents, expected_blocks, read_json, read_jsonl
from app.pipeline.image_reading import ReadingFormatError
from app.pipeline.recovery_models import (
    DeliveryPatch,
    MatchVerification,
    RecoveryDecision,
    SourceMatch,
)
from app.pipeline.review_policy import (
    _BLOCK_ID,
    _names,
    page_review_problems,
    partial_review_accepted,
    picture_content_confirmed,
    picture_problems,
    soft_review_remark,
)
from app.pipeline.review_sources import (
    SourceInventory,
    content_blocks,
    digest,
    needs_content_recovery,
)

RECOVERY_VERSION = "document-tools-v15-keep-read-lines-honest-attempts"
SKILLS = Path(__file__).with_name("recovery_skills")
TOOL_DESCRIPTIONS = {
    "ocr_image": "Only for small disputed lettering: PaddleOCR on ONE smaller original-image 0..1000 region of one mapped image at scale 2/4, around a line bbox an earlier OCR returned. repair_image already reads the whole picture with PaddleOCR itself, so a whole-image OCR request is run as repair_image. Returns source-bound line IDs, coordinates and scores. No content is published.",
    "inspect_layout": "List source-bound MinerU page regions. Reuse parse artifacts; otherwise parse only this page. Read-only, not verification. Then select up to three region_ids for read_regions detail crops or one Word image ID for repair_image.",
    "resolve_source": "Locate one unmapped block using candidate source pages; does not rewrite content.",
    "read_source": "Keep native page text; independently read original images as separate scene, text and gap blocks. Final source audit is required.",
    "read_regions": "Read whole-page source-owned rows/bands; request bounded context when needed, then assemble verbatim. Optionally magnify up to three inspected region_ids; verify actual delivery.",
    "reparse_pdf": "Run MinerU on one physical PDF page, including a Word-converted source; compare a candidate.",
    "rebuild_delivery": "Repair scoped block text/order or exact duplicates; rebuild documents and chunks.",
    "repair_image": "Repair one listed image_targets block_id (or one inspected Word region_id) while preserving every table and other block. Reads the whole picture with PaddleOCR first. Separate scene facts, literal text and explicit gaps; image verification does not certify the page.",
    "defer": "Record why this issue cannot be resolved with current evidence/tools.",
}


def regions(layout):
    """Core rectangles partition the whole page exactly; crops add context later."""
    if layout == "grid4":
        return [[0, 0, 500, 500], [500, 0, 1000, 500],
                [0, 500, 500, 1000], [500, 500, 1000, 1000]]
    count = 4 if layout == "horizontal4" else 2
    return [[0, i * 1000 // count, 1000, (i + 1) * 1000 // count] for i in range(count)]


def crop_region(inventory, page, core, folder, index):
    """Use source PDF coordinates; never model-supplied filesystem paths."""
    folder.mkdir(parents=True, exist_ok=True)
    box = [max(0, core[0] - 20), max(0, core[1] - 20),
           min(1000, core[2] + 20), min(1000, core[3] + 20)]
    target = folder / f"region-{index}.png"
    with fitz.open(inventory.pdf) as pdf:
        rect = pdf[page].rect
        clip = fitz.Rect(box[0] * rect.width / 1000, box[1] * rect.height / 1000,
                         box[2] * rect.width / 1000, box[3] * rect.height / 1000)
        pdf[page].get_pixmap(matrix=fitz.Matrix(min(4, 2400 / max(clip.width, clip.height)),
                                                       min(4, 2400 / max(clip.width, clip.height))), clip=clip).save(target)
    return target, box


async def render_candidate(config, ir, folder, original, parse_cache):
    """Only the candidate directory is writable; copy assets, never writable symlinks."""
    from app.pipeline.stages.chunk import ChunkStage
    from app.pipeline.stages.package import PackageStage
    folder.mkdir(parents=True, exist_ok=True)
    if (original / "assets").exists():
        await asyncio.to_thread(shutil.copytree, original / "assets", folder / "assets", dirs_exist_ok=True)
    (folder / "outputs").mkdir(exist_ok=True)
    enrichments = original / "outputs/enrichments.jsonl"
    if enrichments.is_file():
        shutil.copy2(enrichments, folder / "outputs/enrichments.jsonl")

    package_config = config.model_copy(deep=True)
    package_config.enrich.enable_vlm = False
    package_config.package.enable_semantic_repair = False
    result = await PackageStage(package_config).run(ir.doc_id, ir.run_id, ir, folder, parse_cache_path=parse_cache)
    if not result.success:
        raise ValueError("candidate packaging failed: " + str(result.error))
    result = await ChunkStage(config).run(ir.doc_id, ir.run_id, ir, folder)
    if not result.success:
        raise ValueError("candidate chunking failed: " + str(result.error))


def restore_recovery(controller, ir, run_path):
    """Replay only a hash-checked accepted snapshot belonging to this review fingerprint."""
    recovery = controller.state.get("recovery", {})
    candidate_records = recovery.pop("candidate_review_original_pages", None)
    if candidate_records is not None:
        controller.state["pages"] = candidate_records
        controller._save()
    pending = recovery.pop("pending_commit", None)
    if pending:
        backup = run_path / pending["backup"]
        if not backup.resolve().is_relative_to((run_path / "agentic/recovery").resolve()):
            raise ValueError("Invalid recovery rollback path")
        restore_publication(run_path, backup)
        controller.state["pages"] = pending["pages"]
        controller._save()
    accepted = recovery.get("accepted")
    if not accepted or controller.limits.mode != "apply":
        return ir, False
    path = run_path / accepted["file"]
    if not path.resolve().is_relative_to((run_path / "agentic/recovery").resolve()):
        raise ValueError("Invalid recovery snapshot path")
    if hashlib.sha256(path.read_bytes()).hexdigest() != accepted["sha256"]:
        raise ValueError("Recovery snapshot changed")
    restored = DocumentIR.model_validate_json(path.read_text())
    if (restored.doc_id, restored.run_id, restored.source) != (ir.doc_id, ir.run_id, ir.source):
        raise ValueError("Recovery snapshot source mismatch")
    return restored, restored.to_dict() != ir.to_dict()


def snapshot_publication(run_path, backup):
    """Save delivery plus the root indexes that deterministic packaging rewrites."""
    shutil.copytree(run_path / "outputs", backup / "outputs", dirs_exist_ok=True)
    names = ["manifest.json", "source_map.json"]
    for name in names:
        if (run_path / name).is_file():
            shutil.copy2(run_path / name, backup / name)
    atomic_write_json(backup / "root-files.json", names)


def restore_publication(run_path, backup):
    """Rebuilds are unpublished until commit; interrupted writes restore the last delivery."""
    if not (backup / "outputs").is_dir():
        raise ValueError("Recovery rollback snapshot is missing")
    output = (run_path / "outputs").resolve()
    if not output.is_relative_to(run_path.resolve()):
        raise ValueError("Recovery output path escapes the run")
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(backup / "outputs", output)
    for name in read_json(backup / "root-files.json", []):
        if name not in {"manifest.json", "source_map.json"}:
            raise ValueError("Invalid recovery publication index")
        target = (run_path / name).resolve()
        if not target.is_relative_to(run_path.resolve()):
            raise ValueError("Recovery index path escapes the run")
        if (backup / name).is_file():
            shutil.copy2(backup / name, target)
        elif target.exists():
            target.unlink()


def adopt_candidate_evidence(records, pages, folder, root):
    """Bring the files that committed page records cite from the candidate folder into the live run.

    A record written while auditing a candidate cites paths relative to that
    folder; delivery re-hashes them in the live run, where a missing file
    invalidates the whole review. Live evidence is never replaced.
    """
    for page in pages:
        for image in records.get(str(page), {}).get("images", []):
            live, staged = root / image["path"], folder / image["path"]
            if live.is_file() and hashlib.sha256(live.read_bytes()).hexdigest() == image["sha256"]:
                continue
            if (live.exists() or not live.resolve().is_relative_to(root.resolve()) or not staged.is_file()
                    or hashlib.sha256(staged.read_bytes()).hexdigest() != image["sha256"]):
                raise ValueError("candidate_evidence_missing_or_conflicting: " + image["path"])
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, live)


OCR_MAX_PIXELS = 20_000_000  # read_ocr refuses a larger picture or scaled region


def ocr_scale(source, region, scale):
    """The requested OCR scale, halved until the scaled region fits read_ocr's pixel limit.

    Rerun 2026-10-15: scale 4 on a 3446x1714 picture failed outright where scale 1 reads it.
    """
    from PIL import Image

    with Image.open(source) as image:
        width, height = image.size
    box = (math.floor(region[0] * width / 1000), math.floor(region[1] * height / 1000),
           math.ceil(region[2] * width / 1000), math.ceil(region[3] * height / 1000))
    while scale > 1 and (box[2] - box[0]) * scale * (box[3] - box[1]) * scale > OCR_MAX_PIXELS:
        scale //= 2
    return scale


# Wording an audit quotes: 「…」, “…”, ‘…’, "…" or a single-quoted span not inside a word (image's).
_QUOTED = re.compile(r"「([^」\n]+)」|“([^”\n]+)”|‘([^’\n]+)’|\"([^\"\n]+)\"|(?<![A-Za-z])'([^'\n]+)'(?![A-Za-z])")


def edit_disputes_issue(folder):
    """Whether a targeted edit came back empty: the edit model found nothing wrong where an audit did.

    Rerun 2026-10-15: 8 empty proposals, each followed at once by a 150-580 s full re-read; in
    one the audit was wrong, in three the edit model (shown a transparent picture on black).
    """
    record = read_json(folder / "targeted-edit.json", {})
    return bool(record) and not (record.get("proposal") or {}).get("edits")


class RebuildMismatch(ValueError):
    """The live rebuild did not reproduce the verified candidate; nothing of it was published."""


class InvalidPlan(ValueError):
    """The planner chose a tool or target outside this issue's scope; it is told why and plans again."""


class RecoveryCommitError(RuntimeError):
    """Abort the run without publishing a partly rebuilt accepted candidate."""


class RecoveryController:
    def __init__(self, controller, inventory, run_path, rebuild, parse_cache=None, renderer=None):
        self.owner, self.inventory, self.root = controller, inventory, run_path
        self.rebuild, self.parse_cache = rebuild, parse_cache
        self.renderer = renderer or render_candidate
        self.state = controller.state.setdefault("recovery", {
            "version": RECOVERY_VERSION, "actions": 0, "issues": {}, "steps": [],
        })
        for step in self.state["steps"]:
            if step.get("status") == "running":
                controller.state["usage"]["seconds"] += step.pop("reserved_seconds", 0)
                step.update(status="interrupted", result="Previous process stopped during tool execution")
        self.folder = run_path / "agentic/recovery"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.source_sha = hashlib.sha256(inventory.pdf.read_bytes()).hexdigest() if inventory.pdf else None
        self._layout_store = None
        self.state.setdefault("layout_pages", {})
        self.state.setdefault("queries", 0)
        from app.pipeline.evidence import EvidenceStore
        self.evidence = EvidenceStore(self.root, self.source_sha)

    @property
    def layout_store(self):
        from app.pipeline.layout_evidence import LayoutEvidence
        if self._layout_store is None or self._layout_store.source_sha != self.source_sha:
            self._layout_store = LayoutEvidence(self.inventory, self.owner.config.mineru, self.root,
                                                self.source_sha, self.parse_cache)
        return self._layout_store

    def save(self):
        self.owner._save()
        atomic_write_json(self.root / "outputs/recovery_report.json", self.state, ensure_ascii=False, indent=2)

    def picture_texts(self, ir):
        """Each picture block's delivered text: a figure's retrieval text, a recovered part's own text."""
        from app.pipeline.autonomous_delivery import is_picture_content

        assets = {row.get("block_id"): row.get("retrieval_text") or ""
                  for row in read_jsonl(self.root / "outputs/assets_index.jsonl")}
        return {b.block_id: assets.get(b.block_id, "") if b.type == BlockType.IMAGE else b.get_text()
                for b in ir.blocks if is_picture_content(b)}

    def settled_pictures(self, ir, page, record, texts=None):
        """Pictures of a page whose review found nothing wrong in their wording; repair has nothing to fix there.

        Decided 2026-10-13: 647 relationship issues were queued and 9 resolved; a remark that
        a relationship is not spelled out, or that print is illegible, is not repaired.
        """
        texts = self.picture_texts(ir) if texts is None else texts
        settled = set()
        for block in self.inventory.blocks(page, ir):
            if block.block_id in texts:
                names = {block.block_id, block.payload.get("agentic_image_origin") or block.block_id}
                if picture_content_confirmed(record, block.block_id, names, texts[block.block_id]):
                    settled.add(block.block_id)
        return settled

    def problems(self, ir):
        result = []
        texts = self.picture_texts(ir)
        for page in self.inventory.pages:
            record = self.owner.state["pages"].get(str(page), {})
            reason = record.get("reason", "")
            settled = self.settled_pictures(ir, page, record, texts) if record.get("review") else set()
            for finding in record.get("image_issues", []):
                block = ir.get_block(finding["block_id"])
                if block and block.type == BlockType.IMAGE and block.block_id not in settled:
                    result.append({"id": "image:" + block.block_id, "block_id": block.block_id,
                                   "page_idx": page, "kind": "image", "failure_kind": finding["failure_kind"],
                                   "issues": [finding["detail"]]})
            pictures = {b.block_id for b in self.inventory.blocks(page, ir) if b.block_id in texts}
            if (needs_content_recovery(record) and not record.get("image_only_issues")
                    and (not record.get("review") or page_review_problems(record["review"], pictures))):
                result.append({"id": f"page:{page}", "page_idx": page,
                               "reason": reason, "issues": record.get("review", {}).get("issues", [])[:5]})
        for block in content_blocks(ir):
            if not self.inventory.mapping.get(block.block_id):
                result.append({"id": "block:" + block.block_id, "block_id": block.block_id,
                               "reason": "unmapped_source_block", "text": block.get_text()[:1600]})
        priorities = {item["block_id"]: item["priority"] for item in visual_issues(ir)}
        def priority(problem):
            blocks = ([ir.get_block(problem["block_id"])] if problem.get("block_id") else
                      self.inventory.blocks(problem["page_idx"], ir))
            return max((priorities.get(b.block_id, 0) for b in blocks if b), default=0)
        # A picture a cheap verbatim patch or targeted edit can fix goes first: a full
        # re-read costs 8-20 calls and can use up the budget before any patch runs.
        def patch_first(problem):
            cheap = {"review_correction": 0, "delivery_patch": 0, "targeted_edit": 1}
            return cheap.get(self.repair_method(problem["block_id"]), 2) if problem.get("kind") == "image" else 2
        # An issue the edit model found nothing wrong for waits until every other issue had its
        # turn: a full re-read of it (300-600 s) is spent only when no undisputed issue is waiting.
        def disputed(problem):
            return bool(self.state["issues"].get(problem["id"], {}).get("disputed"))
        return sorted(result, key=lambda problem: (disputed(problem), patch_first(problem), -priority(problem)))

    @staticmethod
    def out_of_budget(issue, **details):
        """Time or calls ran out for this issue: budget_exhausted only when it was never tried.

        Rerun 2026-10-15: 57 issues ended budget_exhausted, 55 of them after 1-3 failed tries,
        so the report read as if repairs had never run. A tried issue stays unresolved with its
        attempts and last result.
        """
        issue.update(details, stopped_by_budget=True)
        issue["status"] = "unresolved" if issue.get("attempts") else "budget_exhausted"

    @staticmethod
    def record_result(issue, step):
        issue["last_result"] = {"action": step["action"], "method": step.get("method", "full"),
                                "status": step.get("status"), "result": str(step.get("result", ""))[:300]}

    def available(self):
        return (self.state["actions"] < self.owner.limits.max_recovery_actions
                and self.owner.state["usage"]["model_calls"] < self.owner.limits.max_model_calls
                and self.owner._remaining() > 0)

    async def call(self, task, payload, images, schema):
        # All calls use the same persisted document-wide budget.
        from pydantic import ValidationError

        for attempt in range(2 if task == "recovery_plan" else 1):
            if len(json.dumps(payload, ensure_ascii=False)) > 42000:
                raise ValueError("recovery_context_limit")
            try:
                return schema.model_validate(await self.owner._call(task, payload, images, schema))
            except (ModelOutputError, ValidationError) as exc:
                if task != "recovery_plan" or attempt:
                    raise
                # A protocol error is not source unreadability or a spent repair action.
                # The retry still consumes the shared call/time budget.
                payload = {**payload, "format_feedback": str(exc)[:500],
                           "format_instruction": "Return one allowed action and only its supported arguments. "
                                                 "Only ocr_image accepts region/scale; all other tools use scale=1."}

    def alternate_available(self):
        return (bool(self.owner.config.review_vlm.model.strip()) and
                (self.owner.config.vlm.base_url, self.owner.config.vlm.model) !=
                (self.owner.config.review_vlm.base_url, self.owner.config.review_vlm.model))

    def image_evidence_revision(self, page, block_id):
        ids = [r["evidence_id"] for r in self.evidence.for_target(page, block_id, provider="paddleocr")]
        return digest(ids) if ids else ""

    def select_strategy(self, decision, recent, allowed):
        # Layout changes do not earn another attempt at the same failed method.
        def strategy(action, role, regions, block, method=None):
            return (action, role, bool(regions) if action != "repair_image" else False,
                    block if action == "repair_image" else None, method if action == "repair_image" else None)
        # Each attempt is a different method: new OCR evidence does not reopen a failed full repair
        # (rerun after an OCR observation it failed the same way every time in validation).
        attempted = {strategy(s["action"], s.get("model_role", "primary"), s.get("region_ids"), s.get("block_id"),
                              s.get("method", "full"))
                     for s in recent}
        method = self.repair_method(decision.block_id) if decision.action == "repair_image" else None
        if decision.model_role == "review" and not self.alternate_available():
            decision.model_role = "primary"
        if decision.action not in {"read_source", "read_regions", "repair_image"}:
            decision.model_role = "primary"
        if decision.action == "defer" or decision.action == "ocr_image" and decision.action in allowed:
            return decision
        if decision.action in allowed and strategy(decision.action, decision.model_role, decision.region_ids,
                                                   decision.block_id, method) not in attempted:
            return decision
        strategies = [("read_source", "primary"), ("reparse_pdf", "primary"), ("read_regions", "primary")]
        if self.alternate_available():
            strategies += [("read_source", "review"), ("read_regions", "review")]
        if decision.action == "repair_image":
            strategies = [("repair_image", "review")] if self.alternate_available() else []
        for action, role in strategies:
            if action == "repair_image" and not decision.block_id:
                continue
            if action in allowed and strategy(action, role, [], decision.block_id, method) not in attempted:
                return decision.model_copy(update={"action": action, "model_role": role, "region_ids": [],
                    "reason": "Escalated to a different method after the proposed strategy was already attempted."})
        return decision.model_copy(update={"action": "defer", "region_ids": [], "reason": "No untried permitted method remains."})

    def repair_method(self, block_id):
        """Which method repair_image will use next for this picture."""
        from app.pipeline.image_recovery import next_image_method
        pages = getattr(getattr(self, "inventory", None), "mapping", {}).get(block_id) or []
        if len(pages) != 1:
            return "full"
        return next_image_method(getattr(self.owner, "state", {}), getattr(self, "state", {}), pages[0], block_id)

    def image_targets(self, ir, page):
        """Only existing images with unambiguous source ownership can be patched."""
        tried = {}
        for step in self.state["steps"]:
            if step["action"] == "repair_image" and step.get("method") not in {"delivery_patch", "targeted_edit"}:
                tried.setdefault(step.get("block_id"), []).append(step.get("model_role", "primary"))
        roles = {"primary", "review"} if self.alternate_available() else {"primary"}
        return [b for b in self.inventory.blocks(page, ir)
                if len(tried.get(b.block_id, [])) < self.owner.limits.max_attempts_per_block
                and not roles.issubset(tried.get(b.block_id, []))
                if b.type == BlockType.IMAGE and self.inventory.mapping.get(b.block_id) == [page]
                and (b.block_id in self.inventory.word_images or ir.has_physical_pages()
                     and len(b.bbox_norm) == 4 and b.bbox_norm[0] < b.bbox_norm[2]
                     and b.bbox_norm[1] < b.bbox_norm[3])]

    def unaccepted_pictures(self, ir, record):
        """Pictures a hard remark of the page's last review names; a re-read picture is reviewed as its parts.

        Rerun 2026-10-15: two pictures were re-read (910 s) for a page held back by remarks on
        other blocks, while their own findings said only that small print was illegible. A
        picture is re-read for its own finding, a page issue or an image issue naming it, and
        only when that remark says something is wrong (review_policy.soft_review_remark).
        """
        review = record.get("review", {})
        pictures = set()
        settled = self.settled_pictures(ir, record.get("page_idx"), record) if "page_idx" in record else set()
        for finding in review.get("blocks", []):
            if finding.get("status") == "verified" or finding.get("block_id") in settled:
                continue
            block = ir.get_block(finding.get("block_id"))
            if block is None:
                continue
            origin = block.payload.get("agentic_image_origin")
            if not (origin or block.type == BlockType.IMAGE):
                continue
            names = {block.block_id, origin or block.block_id}
            own = finding.get("observation") or ""
            remarks = [own, *(i.get("detail", "") for i in record.get("image_issues") or [] if i.get("block_id") in names),
                       *(i for i in review.get("issues") or [] if _names(i, names))]
            # A finding left unexplained says nothing is right; it counts as a hard remark.
            if not own.strip() or any(r.strip() and not soft_review_remark(r) for r in remarks):
                pictures.add(origin or block.block_id)
        return pictures

    @staticmethod
    def unattributed_page_problem(record):
        """A hard page remark naming no block: it may concern any picture of the page."""
        return any(not _BLOCK_ID.search(i) and not soft_review_remark(i)
                   for i in record.get("review", {}).get("issues") or [])

    def picture_only_problems(self, ir, page, record):
        """Whether every remaining problem of the page's review concerns its pictures alone.

        Order, a block that is not a picture, or a hard remark naming no block or another
        block is a page problem a page tool may still fix.
        """
        from app.pipeline.autonomous_delivery import is_picture_content

        review = record.get("review", {})
        pictures = set()
        for block in self.inventory.blocks(page, ir):
            if is_picture_content(block):
                pictures |= {block.block_id, block.payload.get("agentic_image_origin") or block.block_id}
        if not pictures or not review.get("reading_order_correct", False):
            return False
        if any(f.get("status") != "verified" and f.get("block_id") not in pictures for f in review.get("blocks", [])):
            return False
        hard = [i for i in review.get("issues") or [] if not soft_review_remark(i)]
        return all(set(_BLOCK_ID.findall(i)) and set(_BLOCK_ID.findall(i)) <= pictures for i in hard)

    async def quoted_wording_missing(self, problem, origin, ir):
        """Wording the issue quotes that PaddleOCR reads in the picture and its repair still lacks.

        Rerun 2026-10-15: a repair was marked resolved while the two panel headings its issue
        quoted, which OCR reads once the picture is on white, were still not delivered. Only
        quotes naming this picture, or no block, count. Never fails a committed repair.
        """
        from app.adapters.ocr import ocr_python
        from app.pipeline.image_source_review import appears_in

        names = {origin} | {b.block_id for b in ir.blocks if b.payload.get("agentic_image_origin") == origin}
        texts = [t for t in problem.get("issues") or [] if isinstance(t, str)
                 and (problem.get("block_id") == origin or not _BLOCK_ID.search(t) or _names(t, names))]
        quotes = list(dict.fromkeys(q for t in texts for match in _QUOTED.finditer(t)
                                    for q in match.groups() if q and normalized_text(q)))
        source = self.inventory.word_images.get(origin)
        if not quotes or source is None or not ocr_python():
            return []
        try:
            lines = await self.owner._ocr_lines(source, self.inventory)
        except Exception:
            return []
        if not lines:
            return []
        ocr = "\n".join(lines)
        delivered = "\n".join(b.get_text() for b in ir.blocks if b.payload.get("agentic_image_origin") == origin)
        return [q for q in quotes if appears_in(q, ocr) and not appears_in(q, delivered)]

    def reconcile_image_findings(self, ir, page, record):
        """A later source audit can invalidate an earlier scoped success.

        An improvement was kept knowing older faults remain, so only a success is reopened.
        """
        findings = {item["block_id"]: item for item in record.get("review", {}).get("blocks", [])}
        for block in ir.blocks:
            origin = block.payload.get("agentic_image_origin")
            issue = self.state["issues"].get("image:" + origin) if origin else None
            finding = findings.get(block.block_id)
            if (issue and issue.get("page_idx") == page and finding and not block.payload.get("agentic_gap")
                    and issue.get("status") != "improved"
                    and finding.get("status") != "verified" and not soft_review_remark(finding.get("observation", ""))):
                issue.update(status="needs_review", reason="final_page_review_disagrees")
                rejected = issue.setdefault("unverified_block_ids", [])
                if block.block_id not in rejected:
                    rejected.append(block.block_id)

    def source_features(self, problem, blocks):
        features = {"block_count": len(blocks),
                    "page_replacement_safe": "page_idx" in problem and all(
                        self.inventory.mapping.get(b.block_id) == [problem['page_idx']] for b in blocks),
                    "table_blocks": sum(b.type == BlockType.TABLE for b in blocks),
                    "cross_page_blocks": any(len(self.inventory.mapping.get(b.block_id, [])) > 1 for b in blocks),
                    "embedded_images": 0, "native_text_chars": 0,
                    "word_images": sum(b.block_id in self.inventory.word_images for b in blocks)}
        if "page_idx" in problem and self.inventory.pdf:
            with fitz.open(self.inventory.pdf) as pdf:
                page = pdf[problem["page_idx"]]
                features["native_text_chars"] = len(page.get_text())
                features["embedded_images"] = sum(
                    min(i["width"], i["height"]) >= 24 and fitz.Rect(i["bbox"]).get_area() >= 100
                    for i in page.get_image_info())
        return features

    def tool_constraints(self, problem, blocks):
        page = problem.get("page_idx")
        safe = page is not None and all(self.inventory.mapping.get(b.block_id) == [page] for b in blocks)
        constraints = {action: {"available": True} for action in TOOL_DESCRIPTIONS}
        from app.adapters.ocr import ocr_python
        constraints["ocr_image"] = {"available": bool(ocr_python()) and any(
            b.type == BlockType.IMAGE and b.block_id in self.inventory.word_images
            and self.inventory.mapping.get(b.block_id) == [page] for b in blocks),
            "max_lines": 128, "max_pixels": 20000000, "scope": "one original image; observations only"}
        for action in ("read_source", "read_regions", "reparse_pdf"):
            constraints[action] = {"available": safe, "requires": "all replaced blocks belong only to this page"}
        constraints["rebuild_delivery"] = {
            "available": bool(blocks) and len(blocks) <= 24 and safe,
            "max_blocks": 24, "actual_blocks": len(blocks), "requires": "one-page source ownership"}
        constraints["read_regions"].update(accepts_scale=False, accepts_region=False, max_rows_per_response=40, max_context_reads=2,
                                            max_format_splits=2, output="source row IDs and typed cells")
        constraints["read_source"].update(max_images_per_response=1, max_reads_per_image=6,
            format_fallback="one binary source-image split per reader; format failure never implies unreadability",
            available=safe and not any(b.type == BlockType.TABLE for b in blocks),
            requires="single-page blocks without tables; tables require read_regions or reparse_pdf")
        from app.pipeline.image_source_review import required_relationship_origins
        if required_relationship_origins(self.owner, blocks, [page] if page is not None else None):
            for action in ("read_source", "read_regions", "reparse_pdf"):
                constraints[action].update(available=False,
                    requires="preserve typed image relationship evidence; use repair_image before whole-page replacement")
        constraints["rebuild_delivery"]["text_edit_block_ids"] = [b.block_id for b in blocks
            if b.type not in {BlockType.TABLE, BlockType.IMAGE}]
        constraints["repair_image"].update(max_region_revisions=1, max_revision_targets=4, max_candidate_audits=3, max_text_regions_per_response=16, format_fallback="split source image, at most 4 crops / 7 reads per reader", max_aggregate_text_regions=64, max_adjudication_candidates=128, max_evidence_text_chars=18000)
        return constraints

    def estimate_tool(self, action, features, ir, problem, *, layout="horizontal2", model_role="primary", region_count=None):
        from app.pipeline.review_budget import tool_cost
        reads = (3, 3)
        geometry = False
        if action == "repair_image" and (original := self.inventory.word_images.get(problem.get("block_id"))):
            from app.pipeline.image_geometry import geometry_eligible, measured_axes
            prior = self.owner.state.get("pages", {}).get(str(problem.get("page_idx")), {}).get(
                "word_image_readings", {}).get(problem.get("block_id"), {})
            geometry = bool(geometry_eligible(prior) and measured_axes(original))
            from app.pipeline.image_reading import ImageParts, remaining_read_calls
            from app.pipeline.image_recovery import image_read_requests
            skill = (SKILLS / action / "SKILL.md").read_text()
            reads = tuple(0 if remaining_read_calls(self.owner, task, payload, original, ImageParts) == 0 else 3
                          for task, payload in image_read_requests(self.owner.config, ir, skill, model_role))
        page = problem.get("page_idx")
        layout_cached = bool(action == "inspect_layout" and self.layout_store.cached(page))
        if region_count is None:
            region_count = 3 if str(page) in self.state["layout_pages"] else 0
        method = self.repair_method(problem.get("block_id")) if action == "repair_image" else "full"
        patch, edit = method == "delivery_patch", method == "targeted_edit"
        correction = method == "review_correction"
        geometry = geometry and not patch and not edit and not correction
        cost = tool_cost(self.owner.state, action, pictures=features["embedded_images"], layout=layout,
                         word_pictures=features["word_images"], image_read_calls=reads,
                         layout_cached=layout_cached, detail_regions=region_count, image_geometry=geometry,
                         delivery_patch=patch, targeted_edit=edit, review_correction=correction)
        if (action == "repair_image" and method == "full" and self.inventory.word_images.get(problem.get("block_id"))
                and not any(item["content"].get("region") == [0, 0, 1000, 1000] for item in self.evidence.for_target(
                    page, problem.get("block_id"), provider="paddleocr"))):
            cost["seconds"] = round(cost["seconds"] + tool_cost(self.owner.state, "ocr_image")["seconds"], 1)
        if correction:
            cost["method"] = ("review_correction: replace only wording an audit spelled out as wrong and "
                              "PaddleOCR reads as written, then audit")
        if patch:
            cost["method"] = "delivery_patch: keep delivered text, add the omitted relationships/lines, then audit"
        if edit:
            cost["method"] = ("targeted_edit: fix the exact spans every audit named, using only OCR or "
                              "independent-reading wording, then audit")
        if geometry:
            cost["method"] = "source_axis_measurement_and_located_relationships"
        from app.pipeline.review_budget import call_seconds
        if action == "inspect_layout":
            followups = []
            if features["page_replacement_safe"]:
                followups.append(("read_regions", self.estimate_tool("read_regions", features, ir, problem, region_count=3)))
            for target in self.image_targets(ir, page):
                followups.append(("repair_image", self.estimate_tool("repair_image", features, ir,
                                                                    {**problem, "block_id": target.block_id})))
            if not followups:
                return {**cost, "model_calls": self.owner.limits.max_model_calls + 1,
                        "seconds": self.owner.limits.max_seconds + 1}
            followup_action, followup = min(followups, key=lambda item: item[1]["model_calls"])
            query_seconds = 15 if layout_cached else 180
            cost = {**cost, "model_calls": followup["model_calls"] + 1,
                    "seconds": query_seconds + followup["seconds"] + call_seconds(self.owner.state, "recovery_plan"),
                    "includes_followup": followup_action, "followup_seconds": followup["seconds"]}
        elif action == "ocr_image":
            targets = ([ir.get_block(problem["block_id"])] if problem.get("block_id") else self.image_targets(ir, page))
            followups = [self.estimate_tool("repair_image", features, ir, {**problem, "block_id": b.block_id})
                         for b in targets if b]
            if not followups:
                return {**cost, "model_calls": self.owner.limits.max_model_calls + 1,
                        "seconds": self.owner.limits.max_seconds + 1}
            followup = min(followups, key=lambda item: item["model_calls"])
            cost = {**cost, "model_calls": cost["model_calls"] + followup["model_calls"] + 1,
                    "seconds": cost["seconds"] + followup["seconds"] + call_seconds(self.owner.state, "recovery_plan"),
                    "includes_followup": "repair_image and final affected-scope audits"}
        elif action not in {"inspect_layout", "defer"}:
            # Boundary context can invalidate neighboring pages even after stable excerpt anchoring.
            affected = {p for p in (page - 1, page, page + 1) if p in self.inventory.pages} if page is not None else set()
            extra = max(0, len(affected) - 1)
            cost = {**cost, "model_calls": cost["model_calls"] + extra,
                    "seconds": cost["seconds"] + extra * call_seconds(self.owner.state, "audit_page"),
                    "potential_review_pages": sorted(affected)}
        return cost

    async def run(self, ir, *, issue_ids=None, max_new_actions=None):
        if not self.inventory.complete or not self.inventory.pdf:
            self.state["stop_reason"] = "source_inventory_unavailable"
            self.save()
            return ir
        from app.pipeline.review_budget import call_seconds
        initial_actions = self.state["actions"]

        def scheduled():
            # Breadth first: every problem gets its next method before any problem a later one,
            # so a few pictures cannot spend the whole action allowance on their full re-reads.
            # Each round takes an action or ends the schedule, so the action allowance bounds it.
            first = True
            while True:
                actions = self.state["actions"]
                for item in self.problems(ir):
                    status = self.state["issues"].get(item["id"], {}).get("status", "pending")
                    if first or status in {"pending", "unresolved"}:
                        yield item
                first = False
                if self.state["actions"] == actions or not self.available():
                    return
        for problem in scheduled():
            if issue_ids is not None and problem["id"] not in issue_ids:
                continue
            issue = self.state["issues"].setdefault(problem["id"], {**problem, "status": "pending"})
            if issue["status"] in {"resolved", "deferred", "applied_partial", "response_format_error"}:
                continue
            round_actions = self.state["actions"]
            while self.available():
                if max_new_actions is not None and self.state["actions"] - initial_actions >= max_new_actions:
                    self.save()
                    return ir
                if self.state["actions"] > round_actions:
                    break  # this problem resumes with its next method in the next round
                page_scope = "page_idx" in problem and "block_id" not in problem
                current = self.owner.state["pages"].get(str(problem.get("page_idx")), {})
                if page_scope and not needs_content_recovery(current):
                    issue["status"] = ("resolved" if current.get("status") == "verified" else
                                       "applied_partial" if partial_review_accepted(current.get("review", {}),
                                           set(current.get("declared_gap_ids", [])), current.get("block_ids")) else "unverified")
                    break
                recent = [s for s in self.state["steps"] if s["issue_id"] == problem["id"] or
                          (s["action"] == "repair_image" and s.get("page_idx") == problem.get("page_idx")
                           and (page_scope or s.get("block_id") == problem.get("block_id")))]
                # A rejected plan ran no tool, so it is not an attempt; it has its own bound.
                if sum(s["action"] not in {"inspect_layout", "ocr_image", "invalid_plan"}
                       and not (page_scope and s["action"] == "repair_image")
                       for s in recent) >= self.owner.limits.max_recovery_attempts_per_issue:
                    issue["status"] = "attempt_limit"
                    break
                if sum(s["action"] == "invalid_plan" and s["issue_id"] == problem["id"]
                       for s in recent) >= self.owner.limits.max_recovery_attempts_per_issue:
                    issue.update(status="deferred", reason="The planner kept choosing a tool or target outside "
                                                           "this issue's scope")
                    break
                allowed = (["ocr_image", "repair_image", "defer"] if problem.get("kind") == "image" else
                           ["resolve_source", "defer"] if "block_id" in problem else
                           ["resolve_source", "read_source", "read_regions", "reparse_pdf", "rebuild_delivery", "defer"])
                if ("page_idx" in problem and "block_id" not in problem
                        and not any(s["action"] == "inspect_layout" for s in recent)):
                    allowed.insert(0, "inspect_layout")
                try:
                    scoped_blocks = ([ir.get_block(problem["block_id"])] if "block_id" in problem else
                                     self.inventory.blocks(problem["page_idx"], ir))
                    if any(block is None for block in scoped_blocks):
                        issue["status"] = "superseded"
                        break
                    if "block_id" in problem and problem.get("kind") != "image" and scoped_blocks[0].type == BlockType.IMAGE:
                        # resolve_source, the one tool for an unmapped block, cannot place a picture
                        # (rerun 2026-10-15: 11 planned calls refused this way in two reports).
                        issue.update(status="deferred", reason="unmapped_picture: no rendered page matched it and its "
                                                               "page cannot be inferred; it is delivered unverified")
                        break
                    image_targets = self.image_targets(ir, problem["page_idx"]) if page_scope else []
                    unaccepted = self.unaccepted_pictures(ir, current) if page_scope else set()
                    if page_scope and current.get("review") and (unaccepted or not self.unattributed_page_problem(current)):
                        # Re-reading a picture no hard remark names cannot fix the page; a hard remark
                        # naming no block may concern any picture, so all stay targets then.
                        image_targets = [b for b in image_targets if b.block_id in unaccepted]
                    if (page_scope and not image_targets and current.get("review")
                            and self.picture_only_problems(ir, problem["page_idx"], current)):
                        # Rerun 2026-10-15: page issues kept planning re-reads of pictures already
                        # repaired (no target left) until their attempts ran out.
                        issue.update(status="deferred", reason="Only pictures remain wrong on this page and none of "
                                                               "them can be re-read again")
                        break
                    if image_targets:
                        allowed[0:0] = ["ocr_image", "repair_image"]
                    allowed = [a for a in allowed if a != "ocr_image" or sum(
                        s["action"] == "ocr_image" for s in recent) < 2]
                    features = self.source_features(problem, scoped_blocks)
                    constraints = self.tool_constraints(problem, scoped_blocks)
                    allowed = [a for a in allowed if constraints[a]["available"]]
                    if not any(a in allowed for a in ("read_regions", "repair_image")):
                        allowed = [a for a in allowed if a != "inspect_layout"]
                    if allowed == ["defer"]:
                        issue.update(status="deferred", reason="No applicable tool for the current source ownership/size",
                                     tool_constraints=constraints)
                        break
                    estimates = {action: self.estimate_tool(action, features, ir, problem)
                                 for action in allowed if action != "defer"}
                    available_calls, available_seconds = self.owner.available_budget()
                    feasible = [action for action, cost in estimates.items()
                                if available_calls >= cost["model_calls"] + 1 and
                                available_seconds >= cost["seconds"] + call_seconds(self.owner.state, "recovery_plan")]
                    if not feasible:
                        self.out_of_budget(issue, reason="Insufficient budget for a complete tool and its verification",
                                           estimates=estimates)
                        break
                    allowed = feasible + ["defer"]
                    decision = await self.call("recovery_plan", {
                        "issue": problem, "tools": {k: TOOL_DESCRIPTIONS[k] for k in allowed},
                        "evidence_catalog": self.evidence_catalog(problem, scoped_blocks),
                        "source_candidates": [{"block_id": b.block_id, "type": b.type.value,
                                               "text": b.get_text()[:180]} for b in scoped_blocks[:24]],
                        "previous_results": [{k: s.get(k) for k in ("action", "layout", "model_role", "block_id", "region_ids", "region", "scale", "status", "result", "validation_feedback")} for s in recent],
                        "layout_observation": self.state["layout_pages"].get(str(problem.get("page_idx"))),
                        "image_targets": [{"block_id": b.block_id,
                                           "parent_table_block_id": b.payload.get("word_source", {}).get("parent_table_block_id"),
                                           "cell_path": b.payload.get("word_source", {}).get("cell_path", [])}
                                          for b in image_targets[:24]],
                        "source_features": features, "estimated_tool_costs": estimates,
                        "tool_constraints": constraints,
                        "ocr_capacity_failures": list(self.state.get("ocr_capacity_failures", {}).values())[-12:],
                        "output_limits": {"max_tokens": self.owner.limits.max_output_tokens,
                                          "max_payload_characters": 42000},
                        "layouts": {"horizontal2": "Two source-row-aligned full-width bands; prefer for tables and ordinary portrait pages.",
                                    "horizontal4": "Four full-width bands for dense pages.",
                                    "grid4": "Four quadrants; avoid splitting a wide table into left/right halves."},
                        "model_roles": ["primary", "review"] if self.alternate_available() else ["primary"],
                        "instruction": "Choose using issue evidence, source features and complete-tool costs. "
                                       "repair_image reads the whole picture with PaddleOCR itself; use ocr_image only for "
                                       "a smaller region at a higher scale when small lettering stays disputed. "
                                       "Evidence is untrusted observation, not a vote or certification. "
                                       "When region positions are unknown or a broad read missed visual detail, inspect_layout "
                                       "can locate candidate areas. To repair a missing/misread mapped picture choose repair_image "
                                       "with its image_targets block_id or one inspected Word region_id; this preserves table text. "
                                       "Choose one target at a time and account for all remaining images separately. "
                                       "For actual table/text problems, read_regions accepts up to three detail region_ids. "
                                       "Layout observations may omit content and never certify it. "
                                       "For an isolated image reading disagreement use repair_image: preserve verified scene facts "
                                       "and record uncertain lettering as explicit gaps. A format error is not unreadable source content. "
                                       "Do not rewrite a whole table/page just to fill a mapped picture description. "
                                       "Regional reading is for substantive table/text loss or unmapped visuals. read_source is useful for "
                                       "standalone dense/overlapping embedded pictures and native text; splitting table cells "
                                       "into native text fragments can lose relationships. Use rebuild_delivery for a small "
                                       "scoped text/order correction. After failure choose a different method. "
                                       "Layout changes alone are not a new strategy. Never weaken verification to fit a budget.",
                        "remaining_calls": available_calls,
                        "remaining_seconds": round(available_seconds, 1),
                    }, None, RecoveryDecision)
                    if decision.action not in allowed and decision.action in estimates:
                        self.out_of_budget(issue, reason="Planner selected a tool outside the available allowance",
                                           estimated_cost=estimates[decision.action])
                        break
                    if (decision.action == "ocr_image" and decision.region in (None, [0, 0, 1000, 1000])
                            and "repair_image" in allowed):
                        # The repair reads the whole picture with PaddleOCR itself (rerun 2026-10-15: a
                        # separate plan and OCR before each repair, 68 plans and 415 s in one report).
                        decision = decision.model_copy(update={"action": "repair_image", "region": None, "scale": 1})
                    if decision.action in {"repair_image", "ocr_image"} and decision.action not in allowed:
                        # Said before any target check: a repaired picture is no target, and naming it
                        # under a tool this issue does not offer taught the planner nothing.
                        raise InvalidPlan(f"Tool {decision.action} is not allowed for this issue; choose one of {allowed}")
                    if decision.action in {"repair_image", "ocr_image"}:
                        if decision.region_ids:
                            region = self.layout_store.selected(problem["page_idx"], decision.region_ids,
                                self.state["layout_pages"].get(str(problem["page_idx"])))[0]
                            selected = region.get("word_block_id")
                            if not selected or decision.block_id not in {None, selected}:
                                raise InvalidPlan("Selected layout region has no matching source image block")
                            decision.block_id = selected
                        decision.block_id = problem.get("block_id") or decision.block_id
                        permitted = {b.block_id for b in image_targets} if page_scope else {problem.get("block_id")}
                        if decision.block_id not in permitted:
                            raise InvalidPlan(
                                f"Image repair requires an in-scope mapped image target: {decision.block_id} is not "
                                f"one of {sorted(p for p in permitted if p)}. A picture missing from image_targets was "
                                "accepted by the page review or has used both full re-reads; choose a listed target "
                                "or another tool.")
                    decision = self.select_strategy(decision, recent, allowed)
                    if decision.action not in allowed:
                        raise InvalidPlan(f"Tool {decision.action} is not allowed for this issue; choose one of {allowed}")
                    if decision.action == "defer":
                        issue.update(status="deferred", reason=decision.reason)
                        break
                    if decision.action not in {"resolve_source", "repair_image", "ocr_image"}:
                        decision.block_id = None
                    scope = [ir.get_block(decision.block_id)] if decision.action in {"repair_image", "ocr_image"} else scoped_blocks
                    if decision.action == "resolve_source":
                        selected_id = problem.get("block_id") or decision.block_id
                        if selected_id not in {b.block_id for b in scoped_blocks}:
                            raise InvalidPlan("Source resolution requires an in-scope block_id from source_candidates")
                        issue["located_block_id"] = selected_id
                    # The method is part of the key: after a rejected patch the next method is new work.
                    method = self.repair_method(decision.block_id) if decision.action == "repair_image" else "full"
                    key = digest([problem["id"], decision.action, decision.block_id, decision.layout, decision.model_role, sorted(decision.region_ids),
                                  decision.region, decision.scale, self.image_evidence_revision(problem.get("page_idx"), decision.block_id),
                                  [b.model_dump(exclude={"payload": {"agentic_needs_review"}}) for b in scope], self.source_sha]
                                 + ([method] if method != "full" else []))
                    if any(s["key"] == key for s in self.state["steps"]):
                        issue.update(status="no_progress", reason="Same tool and evidence already attempted")
                        break
                    step = {"key": key, "issue_id": problem["id"], "action": decision.action,
                            "reason": decision.reason, "layout": decision.layout, "model_role": decision.model_role,
                            "block_id": decision.block_id, "region_ids": decision.region_ids,
                            "region": decision.region, "scale": decision.scale, "status": "running"}
                    tool_problem = ({**problem, "block_id": decision.block_id}
                                    if decision.action in {"repair_image", "ocr_image"} else problem)
                    cost = self.estimate_tool(decision.action, features, ir, tool_problem,
                                              layout=decision.layout, model_role=decision.model_role, region_count=len(decision.region_ids))
                    available_calls, available_seconds = self.owner.available_budget()
                    if available_calls < cost["model_calls"] or available_seconds < cost["seconds"]:
                        self.out_of_budget(issue, reason="Insufficient budget after planning for tool and verification",
                                           estimated_cost=cost)
                        break
                    step["page_idx"] = problem.get("page_idx")
                    step["evidence_revision"] = self.image_evidence_revision(problem.get("page_idx"), decision.block_id)
                    step["estimated_cost"] = cost
                    self.state["steps"].append(step)
                    self.state["queries" if decision.action in {"inspect_layout", "ocr_image"} else "actions"] += 1
                    if decision.action not in {"inspect_layout", "ocr_image"}:
                        issue["attempts"] = issue.get("attempts", 0) + 1
                    step["reserved_seconds"] = available_seconds
                    self.save()
                    before = self.owner.state["usage"]["seconds"]
                    started = time.monotonic()
                    deadline = started + available_seconds
                    committed = False
                    previous_protection = getattr(self.owner, "_protected_budget", (0, 0))
                    try:
                        folder = self.folder / key[:16]
                        folder.mkdir(exist_ok=True)
                        atomic_write_json(folder / "before_ir.json", ir.to_dict())
                        skill = (SKILLS / decision.action / "SKILL.md").read_text()
                        operation = getattr(self, decision.action)
                        if decision.action == "ocr_image":
                            observation = await asyncio.wait_for(operation(ir, tool_problem, decision, folder, skill),
                                max(0, deadline - time.monotonic()))
                            step.update(status="observed", result=observation)
                            continue
                        if decision.action == "inspect_layout":
                            query_seconds = max(0, deadline - time.monotonic() - cost["followup_seconds"]
                                                - call_seconds(self.owner.state, "recovery_plan"))
                            observation = await asyncio.wait_for(
                                operation(ir, problem, decision, folder, skill), query_seconds)
                            self.state["layout_pages"][str(problem["page_idx"])] = observation
                            step.update(status="observed", result={k: v for k, v in observation.items() if k != "regions"})
                            # A query is evidence for the next plan, never a repair or a page certification.
                            continue
                        review_pages = cost.get("potential_review_pages", [problem.get("page_idx")])
                        reserve = len([p for p in review_pages if p is not None])
                        self.owner._protected_budget = (
                            previous_protection[0] + reserve,
                            previous_protection[1] + reserve * call_seconds(self.owner.state, "audit_page"))
                        try:
                            candidate, pages = await asyncio.wait_for(
                                operation(ir, problem, decision, folder, skill), max(0, deadline - time.monotonic()))
                        finally:
                            # Recorded even when the method fails, so it is never chosen again for this picture.
                            if method != "full":
                                step["method"] = method
                        self.record_candidate(ir, candidate, pages, folder, decision.action)
                        if decision.action == "resolve_source":
                            await self.renderer(self.owner.config, candidate, folder, self.root, self.parse_cache)
                            accepted = True  # Full source/delivery audit remains mandatory before commit.
                        else:
                            accepted = await asyncio.wait_for(
                                self.validate(candidate, pages, folder,
                                              image_origin=(problem.get("block_id") or decision.block_id)
                                              if decision.action == "repair_image" else None),
                                max(0, deadline - time.monotonic()))
                        if not accepted:
                            step["validation_feedback"] = self.validation_feedback(folder)
                            raise ValueError("candidate_failed_source_or_delivery_verification")
                        # Source audit may prune unsupported parts; record the actual resulting patch.
                        self.record_candidate(ir, candidate, pages, folder, decision.action,
                            status="source_mapping_checked" if decision.action == "resolve_source" else "candidate_source_checked")
                        atomic_write_json(folder / "candidate_ir.json", candidate.to_dict())
                        if self.owner.limits.mode == "apply":
                            # Release only this tool's reserved final review; hard document limits persist.
                            self.owner._protected_budget = previous_protection
                            # Validate every changed review input before touching live delivery.
                            checked = await asyncio.wait_for(
                                self.prepare_commit(ir, candidate, pages, folder,
                                    image_origin=decision.block_id if decision.action == "repair_image" else None),
                                max(0, deadline - time.monotonic()))
                            snapshot = folder / "accepted_ir.json"
                            atomic_write_json(snapshot, candidate.to_dict())
                            backup = folder / "publication-before"
                            snapshot_publication(self.root, backup)
                            previous_records = copy.deepcopy(self.owner.state["pages"])
                            self.state["pending_commit"] = {
                                "backup": str(backup.relative_to(self.root)), "pages": previous_records,
                                "candidate": str(snapshot.relative_to(self.root)),
                            }
                            self.save()
                            try:
                                await asyncio.wait_for(self.rebuild(candidate), max(0, deadline - time.monotonic()))
                                live_inventory = await self.candidate_inventory(candidate, self.root)
                                current = self.owner.review_scope_fingerprints(candidate, live_inventory, self.root)
                                if current != checked["fingerprints"]:
                                    differing = sorted(page for page in set(current) | set(checked["fingerprints"])
                                                       if current.get(page) != checked["fingerprints"].get(page))
                                    raise RebuildMismatch(f"rebuilt_delivery_differs_from_verified_candidate: pages {differing}")
                                adopt_candidate_evidence(checked["records"], checked["affected_pages"], folder, self.root)
                                from app.pipeline.review_sources import delivery_digest
                                for page, record in checked["records"].items():
                                    record["delivery_sha256"] = delivery_digest(self.root)
                                    record["scoped_fingerprint"] = current[int(page)]
                                self.owner.state["pages"] = checked["records"]
                                self.state["accepted"] = {
                                    "file": str(snapshot.relative_to(self.root)),
                                    "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                                }
                                self.state.pop("pending_commit", None)
                                self.save()
                                committed = True
                            except BaseException as exc:
                                restore_publication(self.root, backup)
                                self.owner.state["pages"] = previous_records
                                self.state.pop("pending_commit", None)
                                self.save()
                                if isinstance(exc, asyncio.CancelledError):
                                    raise
                                if isinstance(exc, RebuildMismatch):
                                    # The previous publication is intact again: reject this candidate and go on.
                                    # An infrastructure failure (disk, rebuild crash) still stops the run for resume.
                                    raise
                                raise RecoveryCommitError("Candidate rebuild failed; previous publication restored") from exc
                            ir = candidate
                            self.inventory.ir = ir
                            self.inventory.mapping = live_inventory.mapping
                            step["status"] = "applied"
                        else:
                            step["status"] = "candidate_only"
                        partial = any(b.payload.get("agentic_gap") and set(pages).intersection(b.payload.get("agentic_source_pages", []))
                                      and (decision.action != "repair_image"
                                           or b.payload.get("agentic_image_origin") == decision.block_id)
                                      for b in candidate.blocks)
                        remaining = []
                        if decision.action == "repair_image" and self.owner.limits.mode == "apply":
                            parts = {b.block_id for b in ir.blocks
                                     if b.payload.get("agentic_image_origin") == decision.block_id}
                            committed_review = self.owner.state["pages"].get(str(problem["page_idx"]), {}).get("review", {})
                            remaining = picture_problems(committed_review, parts) if committed_review else []
                        issue["status"] = ("candidate_only" if self.owner.limits.mode == "shadow" else
                                           "located" if decision.action == "resolve_source" else
                                           "applied_partial" if partial else "improved" if remaining else "resolved")
                        missing = []
                        if decision.action == "repair_image" and self.owner.limits.mode == "apply":
                            missing = await self.quoted_wording_missing(problem, decision.block_id, ir)
                        if missing:
                            # The audit quoted lettering PaddleOCR reads and the repair still lacks it:
                            # kept, but the issue is open for the next method.
                            issue.update(status="unresolved", reason="quoted_wording_still_missing",
                                         missing_wording=missing)
                        step["result"] = {"affected_pages": pages, "source_verified": True, "source_complete": not partial and decision.action != "resolve_source",
                                          "scope": "image" if decision.action == "repair_image" else "page"}
                        if not page_scope and decision.action == "repair_image" and self.owner.limits.mode == "apply":
                            self.reconcile_image_findings(ir, problem["page_idx"],
                                self.owner.state["pages"].get(str(problem["page_idx"]), {}))
                        if page_scope and decision.action == "repair_image":
                            self.state["issues"]["image:" + decision.block_id] = {
                                "id": "image:" + decision.block_id, "page_idx": problem["page_idx"],
                                "block_id": decision.block_id, "kind": "image", "status": issue["status"]}
                            current = self.owner.state["pages"].get(str(problem["page_idx"]), {})
                            if self.owner.limits.mode == "apply":
                                self.reconcile_image_findings(ir, problem["page_idx"], current)
                            if self.owner.limits.mode == "apply" and needs_content_recovery(current):
                                issue["status"] = "unresolved"
                                continue
                            if self.owner.limits.mode == "apply":
                                issue["status"] = ("unresolved" if missing else
                                    "resolved" if current.get("status") == "verified" else
                                    "applied_partial" if partial_review_accepted(current.get("review", {}),
                                        set(current.get("declared_gap_ids", [])), current.get("block_ids")) else "unverified")
                        break
                    except asyncio.CancelledError:
                        step.update(status="interrupted", result="Operation interrupted; reserved action is retained")
                        raise
                    except RecoveryCommitError:
                        step.update(status="commit_rolled_back", result="Previous publication and reviews restored")
                        raise
                    except ReadingFormatError as exc:
                        step.update(status="response_format_error", result=str(exc)[:1000])
                        issue["status"] = "unresolved"
                        continue
                    except BudgetExhausted as exc:
                        step.update(status="budget_exhausted", result=str(exc))
                        self.out_of_budget(issue)
                        break
                    except Exception as exc:
                        if committed:
                            step.update(status="commit_interrupted", result=str(exc)[:600])
                            raise RecoveryCommitError("Accepted candidate rebuild failed; resume required") from exc
                        step.update(status="rejected", result=(type(exc).__name__ + ": " + str(exc))[:1000])
                        issue["status"] = "unresolved"
                        if method == "targeted_edit" and edit_disputes_issue(folder):
                            issue["disputed"] = True
                    finally:
                        self.owner._protected_budget = previous_protection
                        step.pop("reserved_seconds", None)
                        if decision.action not in {"inspect_layout", "ocr_image"}:
                            self.record_result(issue, step)
                        step["duration_seconds"] = round(time.monotonic() - started, 3)
                        counted = self.owner.state["usage"]["seconds"] - before
                        self.owner.state["usage"]["seconds"] += max(0, time.monotonic() - started - counted)
                        self.save()
                except RecoveryCommitError:
                    raise
                except BudgetExhausted:
                    self.out_of_budget(issue)
                    break
                except InvalidPlan as exc:
                    # Spot check 2026-10-07: nine page issues ended here after one plan named a picture that had
                    # used both re-reads. The rejection is the next plan's evidence; it ran no tool, so it counts
                    # toward its own bound, not the issue's attempts.
                    self.state["steps"].append({
                        "key": digest([problem["id"], "invalid_plan", len(self.state["steps"])]),
                        "issue_id": problem["id"], "action": "invalid_plan", "planned_action": decision.action,
                        "block_id": decision.block_id, "page_idx": problem.get("page_idx"),
                        "status": "rejected", "result": str(exc)[:600]})
                    issue["status"] = "unresolved"
                    self.save()
                    continue
                except Exception as exc:
                    issue.update(status="planner_failed", reason=(type(exc).__name__ + ": " + str(exc))[:600])
                    break
            if not self.available() and issue["status"] in {"pending", "unresolved"}:
                self.out_of_budget(issue)
            self.save()
        self.state["stop_reason"] = ("budget_exhausted" if not self.available() else
                                     "insufficient_tool_budget" if any(
                                         i["status"] == "budget_exhausted"
                                         or i["status"] == "unresolved" and i.get("stopped_by_budget")
                                         for i in self.state["issues"].values()) else "completed")
        self.save()
        return ir

    async def candidate_inventory(self, candidate, folder):
        inventory = SourceInventory(candidate, folder, self.inventory.pdf)
        await inventory.prepare(allow_conversion=False)
        # Preserve existing Word mappings; override only verified candidate source references.
        inventory.mapping = {
            b.block_id: b.payload.get("agentic_source_pages", self.inventory.mapping.get(b.block_id, []))
            for b in candidate.blocks
        }
        # Candidate validation must retain the original Word picture evidence.
        # Opening the converted PDF alone cannot reconstruct Word relationship parts.
        for block in candidate.blocks:
            original = self.inventory.ir.get_block(block.block_id)
            image = self.inventory.word_images.get(block.block_id)
            if (image and original and block.type == BlockType.IMAGE
                    and block.payload.get("word_source") == original.payload.get("word_source")):
                target = folder / image.relative_to(self.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                if image.resolve() != target.resolve():
                    shutil.copy2(image, target)
                inventory.word_images[block.block_id] = target
        return inventory

    async def prepare_commit(self, before, candidate, pages, folder, *, image_origin=None):
        """Audit isolated final artifacts, preserving successful records and spent usage on failure."""
        from app.pipeline.review_budget import call_seconds
        inventory = await self.candidate_inventory(candidate, folder)
        old_scopes = self.owner.review_scope_fingerprints(before, self.inventory, self.root)
        new_scopes = self.owner.review_scope_fingerprints(candidate, inventory, folder)
        changed = {page for page in new_scopes if old_scopes.get(page) != new_scopes[page]}
        changed.update(pages)
        old_records = copy.deepcopy(self.owner.state["pages"])
        # A fresh before/after comparison cannot renew an already stale certificate.
        changed.update(page for page, scope in new_scopes.items()
                       if old_records.get(str(page), {}).get("review")
                       and old_records[str(page)].get("scoped_fingerprint") != scope)
        records = copy.deepcopy(old_records)
        for page in changed:
            checked = read_json(folder / f"validation-page-{page}.json", {})
            if checked.get("scoped_fingerprint") == new_scopes[page]:
                records[str(page)] = checked
            else:
                records.pop(str(page), None)
        pending = [p for p in changed if str(p) not in records]
        calls, seconds = self.owner.available_budget()
        if calls < len(pending) or seconds < len(pending) * call_seconds(self.owner.state, "audit_page"):
            raise BudgetExhausted("Insufficient budget to audit all changed delivery scopes before commit")
        self.state["candidate_review_original_pages"] = old_records
        self.owner._save()
        try:
            self.owner.state["pages"] = records
            for page in sorted(changed):
                record = await self.owner._inspect(page, candidate, inventory, folder)
                atomic_write_json(folder / f"commit-validation-page-{page}.json", record, ensure_ascii=False, indent=2)
                review = record.get("review", {})
                if not review or not review.get("reading_order_correct", False):
                    raise ValueError("candidate_final_review_incomplete")
                findings = {b["block_id"]: b for b in review.get("blocks", [])}
                from app.pipeline.autonomous_delivery import is_picture_content
                from app.pipeline.image_recovery import repaired_picture_problems
                page_blocks = inventory.blocks(page, candidate)
                pictures = {b.block_id for b in page_blocks if is_picture_content(b)}
                previously_verified = {b["block_id"] for b in old_records.get(str(page), {}).get("review", {}).get("blocks", [])
                                       if b.get("status") == "verified" and b.get("context_complete", True)}
                unchanged_ids = {b.block_id for b in candidate.blocks if (old := before.get_block(b.block_id))
                                 and old.model_dump(exclude={"reading_order"}) == b.model_dump(exclude={"reading_order"})}
                # A picture verified before regresses only on a hard remark; other content must stay verified.
                if any(bid not in findings or findings[bid].get("status") != "verified"
                       and (bid not in pictures or not soft_review_remark(findings[bid].get("observation", "")))
                       for bid in previously_verified & unchanged_ids):
                    raise ValueError("candidate_final_review_regressed_verified_content")
                if image_origin and repaired_picture_problems(
                        review, [b for b in page_blocks if b.payload.get("agentic_image_origin") == image_origin],
                        unattributed=False):
                    raise ValueError("candidate_final_review_found_problems_in_the_repair")
                if (old_records.get(str(page), {}).get("status") == "verified" and record.get("status") != "verified"
                        and page_review_problems(review, pictures)):
                    raise ValueError("candidate_final_review_regressed_verified_page")
                if not image_origin and page in pages and record.get("status") != "verified":
                    if (not partial_review_accepted(review, record.get("declared_gap_ids", []), record.get("block_ids"))
                            and page_review_problems(review, pictures)):
                        raise ValueError("candidate_failed_final_source_or_delivery_verification")
            return {"records": copy.deepcopy(self.owner.state["pages"]), "fingerprints": new_scopes,
                    "affected_pages": sorted(changed)}
        finally:
            self.owner.state["pages"] = old_records
            self.state.pop("candidate_review_original_pages", None)
            self.owner._save()

    async def validate(self, candidate, pages, folder, *, image_origin=None, image_pruned=False):
        from app.pipeline.image_source_review import validate_relationship_preservation
        validate_relationship_preservation(self.owner, self.inventory.ir, candidate, self.root, pages)
        self.evidence.validate_candidate(candidate)
        self.evidence.copy_to(folder)
        await self.renderer(self.owner.config, candidate, folder, self.root, self.parse_cache)
        for block in candidate.blocks:
            if image := block.payload.get("agentic_source_image"):
                source = self.root / image
                target = folder / image
                if not source.resolve().is_relative_to((self.root / "agentic/evidence").resolve()):
                    raise ValueError("Native source image escapes source evidence")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        docs = TextCoverage(p.read_text() for p in delivered_documents(folder / "outputs"))
        chunks = TextCoverage(c.get("content", "") for c in read_jsonl(folder / "outputs/chunks.jsonl"))
        for _, text, reason in expected_blocks(candidate, folder / "outputs"):
            if not reason and (not docs.covers(text) or
                               self.owner.config.package.generate_chunks and not chunks.covers(text)):
                return False
        if image_origin:
            from app.pipeline.image_recovery import validate_image_parts
            return await validate_image_parts(self, candidate, pages, folder, image_origin, allow_pruning=not image_pruned)
        inventory = await self.candidate_inventory(candidate, folder)
        original_records = copy.deepcopy(self.owner.state["pages"])
        self.state["candidate_review_original_pages"] = original_records
        self.owner._save()
        try:
            for page in pages:
                self.owner.state["pages"].pop(str(page), None)
                record = await self.owner._inspect(page, candidate, inventory, folder)
                atomic_write_json(folder / f"validation-page-{page}.json", record, ensure_ascii=False, indent=2)
                if record["status"] != "verified":
                    from app.pipeline.autonomous_delivery import is_picture_content
                    gaps = [b for b in inventory.blocks(page, candidate) if b.payload.get("agentic_gap")]
                    review = record.get("review", {})
                    page_blocks = inventory.blocks(page, candidate)
                    readable = [b for b in page_blocks if not b.payload.get("agentic_gap")]
                    if not (readable and (partial_review_accepted(
                            review, [b.block_id for b in gaps], [b.block_id for b in page_blocks])
                            or not page_review_problems(review, {b.block_id for b in page_blocks
                                                                  if is_picture_content(b)}))):
                        return False
            return True
        finally:
            self.owner.state["pages"] = original_records
            self.state.pop("candidate_review_original_pages", None)
            self.owner._save()

    def replace_page(self, ir, page, blocks):
        old = self.inventory.blocks(page, ir)
        if any(self.inventory.mapping.get(b.block_id) != [page] for b in old):
            raise ValueError("Cross-page blocks require source resolution before page replacement")
        # Exact source text matches retain their structural role after a regional repair.
        for block in blocks:
            if block.type != BlockType.TEXT:
                continue
            matches = [b for b in old if b.type == BlockType.TEXT and b.payload.get("text_level")
                       and normalized_text(b.get_text()) == normalized_text(block.get_text())]
            if len(matches) == 1:
                block.payload["text_level"] = matches[0].payload["text_level"]
        candidate = ir.model_copy(deep=True)
        ids = {b.block_id for b in old}
        insert = next((i for i, b in enumerate(candidate.blocks) if b.block_id in ids), None)
        if insert is None:
            # Place omitted pages relative to physical evidence, never at arbitrary document end.
            insert = next((i for i, b in enumerate(candidate.blocks)
                           if self.inventory.mapping.get(b.block_id)
                           and min(self.inventory.mapping[b.block_id]) > page), len(candidate.blocks))
        if not ir.has_physical_pages():
            # Native Word uses logical pages. Physical evidence lives in agentic_source_pages;
            # changing the storage page here would move the replacement to another document section.
            logical_page = (old[0].page_idx if old else
                            candidate.blocks[min(insert, len(candidate.blocks) - 1)].page_idx if candidate.blocks else 0)
            for block in blocks:
                block.page_idx = logical_page
        elif page not in {p.page_idx for p in candidate.pages}:
            candidate.pages.append(PageInfo(page_idx=page))
            candidate.pages.sort(key=lambda p: p.page_idx)
        candidate.blocks[insert:insert] = blocks
        candidate.blocks = [b for b in candidate.blocks if b.block_id not in ids]
        for index, block in enumerate(candidate.blocks):
            block.reading_order = index
        return candidate


    @staticmethod
    def validation_feedback(folder):
        """Expose bounded source-review reasons, not a generic rejected-candidate message."""
        feedback = []
        paths = [folder / "validation-image.json"] + sorted(folder.glob("validation-page-*.json"))
        for path in paths:
            if not path.is_file():
                continue
            record = read_json(path, {})
            review = record.get("review", {})
            feedback.append({"artifact": path.name, "source_complete": review.get("source_complete"),
                "declared_gaps_only": review.get("declared_gaps_only"),
                "issues": [str(v)[:240] for v in review.get("issues", [])[:4]],
                "blocks": [{"block_id": b["block_id"], "status": b.get("status"),
                            "observation": str(b.get("observation", ""))[:240]}
                           for b in review.get("blocks", []) if b.get("status") != "verified"][:4]})
        return feedback[:2]

    def evidence_catalog(self, problem, blocks):
        items = []
        for block in blocks:
            for item in self.evidence.for_target(problem.get("page_idx"), block.block_id):
                summary = {"evidence_id": item["evidence_id"], "block_id": block.block_id,
                           "provider": item["provider"], "kind": item["kind"],
                           "summary": str(item["content"])[:250]}
                if item["provider"] == "paddleocr":
                    lines = item["content"]["lines"]
                    summary.update(region=item["content"]["region"], scale=item["content"]["scale"],
                        line_count=len(lines), lines=[{**line, "text": line["text"][:160]} for line in lines[:8]],
                        lines_truncated=len(lines) > 8)
                items.append(summary)
        return items[-16:]

    async def ocr_image(self, ir, problem, decision, folder, skill):
        from app.adapters.ocr import ocr_runtime_identity, read_ocr
        bid = decision.block_id or problem.get("block_id")
        block = ir.get_block(bid)
        page = problem["page_idx"]
        source = self.inventory.word_images.get(bid)
        if (not block or block.type != BlockType.IMAGE or source is None
                or self.inventory.mapping.get(bid) != [page]):
            raise ValueError("ocr_requires_one_mapped_original_image")
        runtime = ocr_runtime_identity()
        region = decision.region or [0, 0, 1000, 1000]
        scale = ocr_scale(source, region, decision.scale)
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        for old in self.evidence.for_target(page, bid, provider="paddleocr"):
            value = old["content"]
            if (old["metadata"].get("runtime_identity") == runtime and value["source_sha256"] == sha
                    and value["region"] == region and value["scale"] == scale):
                return {"evidence_id": old["evidence_id"], "status": value["status"],
                        "line_count": len(value["lines"]), "cache_hit": True}
        failures = self.state.setdefault("ocr_capacity_failures", {})
        identity = digest([runtime, sha, region, scale])
        if identity in failures:
            raise ValueError(failures[identity]["reason"] + "; cached scope failure; select a smaller region")
        try:
            value = await read_ocr(source, region, scale, folder,
                                   timeout=min(90, self.owner.available_budget()[1]))
        except ValueError as exc:
            if str(exc).startswith(("ocr_output_capacity_exceeded", "ocr_pixel_limit")):
                failures[identity] = {"block_id": bid, "page_idx": page, "source_sha256": sha,
                                      "region": region, "scale": scale, "reason": str(exc)[:300]}
                self.save()
            raise
        eid = self.evidence.add(provider="paddleocr", kind="transcription", page=page, block_id=bid,
            source=source, bbox=region, content=value,
            metadata={"runtime_identity": runtime, "version": value["version"], "detector": value["detector"],
                      "recognizer": value["recognizer"], "device": value["device"]})
        return {"evidence_id": eid, "status": value["status"], "line_count": len(value["lines"]),
                "cache_hit": False, "scope": "observation only; repair_image and source audit required"}

    def record_candidate(self, before, candidate, pages, folder, action, *, status="candidate_not_certified"):
        """Reject unrelated edits and bind every change to the tool's actual source artifacts."""
        old = {b.block_id: b for b in before.blocks}
        new = {b.block_id: b for b in candidate.blocks}
        if len(new) != len(candidate.blocks):
            raise ValueError("candidate_duplicate_block_ids")
        def same(b):
            return b.model_dump(exclude={"reading_order"})
        changed = {bid for bid in old.keys() | new.keys()
                   if bid not in old or bid not in new or same(old[bid]) != same(new[bid])}
        owned = {bid for bid in old if self.inventory.mapping.get(bid)
                 and set(self.inventory.mapping[bid]) <= set(pages)}
        if action == "resolve_source":
            if (len(changed) != 1 or old.keys() != new.keys()
                    or any(old[bid].get_text() != new[bid].get_text() or old[bid].type != new[bid].type for bid in changed)):
                raise ValueError("source_resolution_changed_content")
            owned.update(changed)
        if (changed & old.keys()) - owned:
            raise ValueError("candidate_changed_unrelated_or_cross_page_content")
        if ([b.block_id for b in before.blocks if b.block_id not in owned]
                != [b.block_id for b in candidate.blocks if b.block_id in old and b.block_id not in owned]):
            raise ValueError("candidate_reordered_unrelated_content")
        sources = [self.evidence.add(provider="source", kind="page_image", page=page,
                    source=self.inventory.image(page)) for page in pages]
        observation_ids = {}
        artifacts = {"read_regions": "region-observations.json", "read_source": "native-checks.json",
                     "reparse_pdf": "parser-observations.json", "resolve_source": "source-match.json"}
        artifact = folder / artifacts[action] if action in artifacts else None
        if artifact is not None and artifact.is_file():
            for page, parent in zip(pages, sources, strict=True):
                observation_ids[page] = self.evidence.add(provider=action, kind="tool_observations", page=page,
                    source=self.inventory.image(page), content=read_json(artifact, {}), parents=[parent])
        for bid in changed & new.keys():
            block = new[bid]
            block_pages = block.payload.get("agentic_source_pages", self.inventory.mapping.get(bid, pages))
            if not block_pages or not set(block_pages) <= set(pages):
                raise ValueError("candidate_source_scope_mismatch")
            block.payload.setdefault("agentic_source_pages", block_pages)
            block.payload.setdefault("agentic_source_sha256", self.source_sha)
            refs = set(block.payload.get("agentic_evidence_ids", []))
            refs.update(eid for eid, page in zip(sources, pages, strict=True) if page in block_pages)
            refs.update(observation_ids[p] for p in block_pages if p in observation_ids)
            block.payload["agentic_evidence_ids"] = sorted(refs)
        self.evidence.validate_candidate(candidate)
        patch = {"action": action, "source_sha256": self.source_sha, "pages": pages,
                 "evidence_ids": sources, "changes": [
                     {"block_id": bid, "before_sha256": digest(same(old[bid])) if bid in old else None,
                      "after_sha256": digest(same(new[bid])) if bid in new else None}
                     for bid in sorted(changed)], "status": status}
        atomic_write_json(folder / "candidate-provenance.json", patch, ensure_ascii=False, indent=2)

    def source_payload(self, page):
        return {"agentic_source_pages": [page], "agentic_source_sha256": self.source_sha}

    async def repair_image(self, ir, problem, decision, folder, skill):
        from app.pipeline.image_recovery import repair_image
        block_id = problem.get("block_id") or decision.block_id
        if self.repair_method(block_id) == "full":
            await self.picture_ocr(ir, problem, block_id, folder)
        return await repair_image(self, ir, problem, decision, folder, skill)

    async def picture_ocr(self, ir, problem, block_id, folder):
        """PaddleOCR of the whole picture as repair evidence, unless an earlier observation has it.

        Only a full re-read resolves its reading against OCR observations; the patch and edit
        methods read OCR lines themselves. A picture OCR cannot take is repaired without them.
        """
        from app.adapters.ocr import ocr_python

        page = problem["page_idx"]
        source = self.inventory.word_images.get(block_id)
        if source is None or not ocr_python() or any(
                item["content"].get("region") == [0, 0, 1000, 1000]
                for item in self.evidence.for_target(page, block_id, provider="paddleocr")):
            return
        from PIL import Image

        with Image.open(source) as image:
            scale = 2 if image.width * image.height < 1_500_000 else 1  # as the review's OCR reads it
        decision = RecoveryDecision(action="ocr_image", reason="Whole-picture OCR for the repair",
                                    block_id=block_id, scale=scale)
        try:
            await self.ocr_image(ir, problem, decision, folder / "picture-ocr", None)
        except (ValueError, TimeoutError) as exc:
            atomic_write_json(folder / "picture-ocr-error.json", {"reason": str(exc)[:600]}, ensure_ascii=False)

    async def read_source(self, ir, problem, decision, folder, skill):
        blocks = self.inventory.blocks(problem["page_idx"], ir)
        if not self.tool_constraints(problem, blocks)["read_source"]["available"]:
            raise ValueError("tool_precondition_failed: read_source requires eligible single-page source blocks")
        from app.pipeline.image_reading import read_image_parts_bounded, reconcile_image_parts
        from app.pipeline.image_recovery import build_image_parts_blocks, image_read_requests
        from app.pipeline.recovery_evidence import native_materials
        folder.mkdir(parents=True, exist_ok=True)
        page = problem["page_idx"]
        materials = native_materials(self.inventory, page, self.inventory.directory / f"native-{page}")
        if not materials:
            raise ValueError("native_materials_unavailable: use rendered regions or PDF OCR")
        pictures = [m for m in materials if m["image"]]
        # Two blind readers, each with one bounded binary format split. Keep both
        # delivery audits affordable even when called outside normal tool admission.
        protected = getattr(self.owner, "_protected_budget", None) or (0, 0)
        audit_calls = 1 if protected[0] else 2
        if self.owner.available_budget()[0] < len(pictures) * 6 + audit_calls:
            raise BudgetExhausted("Insufficient calls for image-part reading and final source verification")
        requests = image_read_requests(self.owner.config, ir, skill, decision.model_role)
        # The page's pictures by their Word part: a material whose Word bytes match one replaces it.
        pictures_here = {b.payload.get("word_source", {}).get("image_sha256"): b
                         for b in self.inventory.blocks(page, ir) if b.type == BlockType.IMAGE}
        blocks, checks = [], []
        for material in materials:
            if material["image"]:
                primary, independent = [await read_image_parts_bounded(
                    self.owner, task, payload, material["image"], max_depth=1) for task, payload in requests]
                parts = reconcile_image_parts(primary, independent)
                replaced = pictures_here.get(material.get("word_sha256")) if material.get("word_sha256") else None
                carried = []
                if replaced is not None:
                    # A page re-read keeps the picture's drafted lines no audit showed wrong, as a picture
                    # re-read does (rerun 2026-10-15: re-reads that came back as gaps dropped read names).
                    from app.pipeline.image_edits import known_issues, transcription_lines
                    from app.pipeline.image_recovery import keep_drafted_read_lines
                    parts, carried, _, _ = keep_drafted_read_lines(
                        parts, transcription_lines(self.root / "outputs", replaced.block_id),
                        known_issues(self.owner.state, self.state, page, replaced.block_id))
                checks.append({"source_id": material["id"], "source_sha256": hashlib.sha256(
                    material["image"].read_bytes()).hexdigest(), "primary": primary.model_dump(),
                    "independent": independent.model_dump(), "candidate": parts,
                    "replaced_picture": replaced.block_id if replaced is not None else None,
                    "carried_drafted_lines": carried})
                origin = Block(block_id=material["id"], type=BlockType.IMAGE, page_idx=page,
                               bbox_norm=material["bbox"], payload={})
                replacements = build_image_parts_blocks(self, origin, page, material["bbox"],
                    material["image"], parts, [], folder / (folder.name + "-" + material["id"]))
                for block in replacements:
                    # The native material ID is source-owned, not an original IR
                    # image block ID. Never manufacture an image ownership link; the
                    # picture it replaces, matched by its Word bytes, keeps its own
                    # document and file in the delivery.
                    block.payload.pop("agentic_image_origin", None)
                    block.payload.update(agentic_recovery_tool="read_source",
                                         agentic_native_source_id=material["id"])
                    if replaced is not None:
                        block.payload.update(agentic_replaced_picture=replaced.block_id,
                                             agentic_image_path=replaced.payload.get("img_path", ""),
                                             image_source_context=replaced.payload.get("word_source", {}))
                blocks.extend(replacements)
            else:
                blocks.append(Block(block_id=f"recovery_{folder.name}_{material['id']}", type=BlockType.TEXT,
                    page_idx=page, bbox_norm=material["bbox"], payload={"text": material["text"],
                        **self.source_payload(page), "agentic_source_region": material["bbox"],
                        "agentic_recovery_tool": "read_source", "agentic_content_kind": "transcription",
                        "agentic_native_source_id": material["id"]}))
        atomic_write_json(folder / "native-checks.json", checks, ensure_ascii=False, indent=2)
        # Native material order is stable, with page geometry resolving painting order.
        blocks.sort(key=lambda b: (b.bbox_norm[1], b.bbox_norm[0]))
        atomic_write_json(folder / "native-readings.json", [b.model_dump() for b in blocks], ensure_ascii=False)
        return self.replace_page(ir, page, blocks), [page]

    async def inspect_layout(self, ir, problem, decision, folder, skill):
        result = await self.layout_store.inspect(problem["page_idx"])
        atomic_write_json(folder / "layout-observation.json", result, ensure_ascii=False, indent=2)
        return result

    async def read_regions(self, ir, problem, decision, folder, skill):
        blocks = self.inventory.blocks(problem["page_idx"], ir)
        if not self.tool_constraints(problem, blocks)["read_regions"]["available"]:
            raise ValueError("tool_precondition_failed: read_regions requires eligible single-page source blocks")
        from app.pipeline.region_recovery import read_regions
        return await read_regions(self, ir, problem, decision, folder, skill)

    async def reparse_pdf(self, ir, problem, decision, folder, skill):
        blocks = self.inventory.blocks(problem["page_idx"], ir)
        if not self.tool_constraints(problem, blocks)["reparse_pdf"]["available"]:
            raise ValueError("tool_precondition_failed: reparse_pdf requires eligible single-page source blocks")
        from app.pipeline.stages.normalize import NormalizeStage
        page = problem["page_idx"]
        content_list = await self.layout_store.parse_page(page)
        parse_root = self.layout_store.page_root(page) / "parse"
        items = read_json(content_list, [])
        if not isinstance(items, list) or not items:
            raise ValueError("PDF parser returned no blocks")
        folder.mkdir(parents=True, exist_ok=True)
        atomic_write_json(folder / "parser-observations.json", items, ensure_ascii=False, indent=2)
        normalizer = NormalizeStage(self.owner.config)
        blocks = []
        for index, item in enumerate(items):
            block = normalizer._parse_block(item, index)
            if not block:
                continue
            block.block_id = f"recovery_{folder.name}_{index}"
            block.page_idx = page
            block.payload.update(self.source_payload(page))
            block.payload["agentic_recovery_tool"] = "reparse_pdf"
            # A parser-created asset must stay under its isolated parse directory.
            if image := block.payload.get("img_path"):
                asset = content_list.parent / image
                if not asset.resolve().is_relative_to(parse_root.resolve()) or not asset.is_file():
                    raise ValueError("PDF parser asset missing or outside candidate")
                dest = folder / "assets/recovery" / folder.name / asset.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(asset, dest)
                block.payload["img_path"] = str(dest)
            blocks.append(block)
        if not blocks:
            raise ValueError("PDF parser produced no usable blocks")
        return self.replace_page(ir, page, blocks), [page]

    async def resolve_source(self, ir, problem, decision, folder, skill):
        block = ir.get_block(problem.get("block_id") or decision.block_id)
        if block.type == BlockType.IMAGE:
            raise ValueError("Image caption alone cannot locate its visual source; recover the page instead")
        needle = normalized_text(plain_text(block.get_text()))
        if not needle:
            raise ValueError("No searchable source text; recover the source page instead")
        grams = {needle[i:i + 4] for i in range(max(1, len(needle) - 3))}
        with fitz.open(self.inventory.pdf) as pdf:
            scores = [(sum(g in normalized_text(p.get_text()) for g in grams), i) for i, p in enumerate(pdf)]
        selected = [i for score, i in sorted(scores, reverse=True)[:3] if score > 0]
        if not selected:
            raise ValueError("No candidate source page")
        images = [self.inventory.image(i) for i in selected]
        match = await self.call("match_source", {
            "skill": skill, "block_id": block.block_id, "text": block.get_text()[:12000],
            "candidate_pages": selected, "image_order": selected,
        }, images, SourceMatch)
        if match.ambiguous or match.page_idx not in selected:
            raise ValueError("source_match_ambiguous")
        check = await self.call("verify_match", {
            "skill": skill, "block_id": block.block_id, "text": block.get_text()[:12000],
            "page_idx": match.page_idx, "proposed_evidence": match.evidence,
        }, self.inventory.image(match.page_idx), MatchVerification)
        if not check.supported or check.ambiguous:
            raise ValueError("source_match_not_verified")
        candidate = ir.model_copy(deep=True)
        candidate.get_block(block.block_id).payload.update(self.source_payload(match.page_idx))
        atomic_write_json(folder / "source-match.json", {
            "match": match.model_dump(), "verification": check.model_dump(), "source_sha256": self.source_sha,
        }, ensure_ascii=False, indent=2)
        return candidate, sorted(set(self.inventory.mapping.get(block.block_id, []) + [match.page_idx]))

    async def rebuild_delivery(self, ir, problem, decision, folder, skill):
        blocks = self.inventory.blocks(problem["page_idx"], ir)
        if not self.tool_constraints(problem, blocks)["rebuild_delivery"]["available"]:
            raise ValueError("tool_precondition_failed: rebuild_delivery requires eligible single-page source blocks")
        page = problem["page_idx"]
        blocks = self.inventory.blocks(page, ir)
        if not blocks or len(blocks) > 24:
            raise ValueError("Use regional reading or PDF reparse for missing/dense pages")
        source_evidence = self.evidence.add(provider="source", kind="page_image", page=page,
                                            source=self.inventory.image(page))
        patch = await self.call("repair_delivery", {
            "source_evidence_ids": [source_evidence],
            "skill": skill, "page_idx": page, "issues": problem.get("issues", []),
            "blocks": [{"block_id": b.block_id, "type": b.type.value, "text": b.get_text()} for b in blocks],
            "text_edit_block_ids": self.tool_constraints(problem, blocks)["rebuild_delivery"]["text_edit_block_ids"],
            "constraint": "Every edit must cite source_evidence_ids in evidence_ids. Preserve headings and units. "
                          "Table/image content requires its structured repair tool; never flatten it into a text edit.",
        }, self.inventory.image(page), DeliveryPatch)
        if not patch.complete or not patch.edits:
            raise ValueError("No complete scoped patch")
        candidate = ir.model_copy(deep=True)
        allowed = {b.block_id: b for b in blocks}
        edited = set()
        removed = set()
        for edit in patch.edits:
            if edit.block_id not in allowed or edit.block_id in edited:
                raise ValueError("Patch targets missing/repeated/out-of-scope block")
            if not edit.evidence_ids or any(eid != source_evidence for eid in edit.evidence_ids):
                raise ValueError("patch_requires_in_scope_source_evidence")
            for eid in edit.evidence_ids:
                self.evidence.get(eid, page=page)
            edited.add(edit.block_id)
            block = candidate.get_block(edit.block_id)
            if self.inventory.mapping[block.block_id] != [page]:
                raise ValueError("Cannot change a cross-page block with single-page evidence")
            if edit.duplicate_of:
                retained = allowed.get(edit.duplicate_of)
                if (not retained or retained.block_id == block.block_id or
                        normalized_text(plain_text(retained.get_text())) != normalized_text(plain_text(block.get_text()))
                        or not block.get_text().strip()):
                    raise ValueError("Duplicate removal must retain identical source content")
                removed.add(block.block_id)
            elif edit.text.strip():
                if block.type in {BlockType.TABLE, BlockType.IMAGE}:
                    raise ValueError("tool_precondition_failed: structured blocks cannot become free text")
                # Preserve the source block in before_ir; candidate is independently verified.
                block.type = BlockType.TEXT
                block.payload = {**block.payload, "text": edit.text, **self.source_payload(page),
                                 "agentic_recovery_tool": "rebuild_delivery",
                                 "agentic_evidence_ids": edit.evidence_ids}
            else:
                raise ValueError("Empty replacement is not allowed")
        if any(edit.duplicate_of in edited for edit in patch.edits if edit.duplicate_of):
            raise ValueError("Retained duplicate must remain unchanged")
        candidate.blocks = [b for b in candidate.blocks if b.block_id not in removed]
        atomic_write_json(folder / "patch.json", patch.model_dump(), ensure_ascii=False, indent=2)
        return candidate, [page]
