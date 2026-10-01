"""Source-bound MinerU layout observations; geometry is evidence, never acceptance."""
import hashlib
import json
import math
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import fitz

from app.pipeline.delivery import atomic_write_json
from app.pipeline.review_sources import digest

LAYOUT_VERSION = "mineru-layout-v1"
MAX_REGIONS = 64
MAX_JSON_BYTES = 32 * 1024 * 1024
CATEGORY_NAMES = {0: "title", 1: "text", 2: "discarded", 3: "image", 4: "image_caption",
                  5: "table", 6: "table_caption", 7: "table_footnote", 8: "formula"}


def normalized_bbox(value, size):
    """Convert declared coordinates, never infer their unit from small values."""
    if not isinstance(value, (list, tuple)) or len(value) not in {4, 8}:
        return None
    if not all(type(v) in {int, float} and math.isfinite(v) for v in [*value, *size]):
        return None
    if min(size) <= 0:
        return None
    if len(value) == 8:
        value = [min(value[::2]), min(value[1::2]), max(value[::2]), max(value[1::2])]
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        return None
    box = [round(v * 1000 / size[i % 2]) for i, v in enumerate(value)]
    return box if box[0] < box[2] and box[1] < box[3] else None


def layout_items(data, kind, page, page_size):
    """Yield (block, declared coordinate size) for supported MinerU 3.x outputs."""
    if kind == "middle" and isinstance(data, dict):
        for record in data.get("pdf_info", []):
            if not isinstance(record, dict) or record.get("page_idx") != page:
                continue
            size = record.get("page_size", [])
            if (len(size) != 2 or not all(type(v) in {int, float} and math.isfinite(v) and v > 0 for v in size)
                    or abs(size[0] / size[1] / (page_size[0] / page_size[1]) - 1) > 0.02):
                continue
            for block in record.get("preproc_blocks", record.get("para_blocks", [])):
                yield block, size
    elif kind == "model" and isinstance(data, list):
        for index, record in enumerate(data):
            if isinstance(record, dict) and "layout_dets" in record:
                info = record.get("page_info", {})
                if info.get("page_no", index) == page:
                    size = [info.get("width", 0), info.get("height", 0)]
                    if (all(type(v) in {int, float} and math.isfinite(v) and v > 0 for v in size)
                            and abs(size[0] / size[1] / (page_size[0] / page_size[1]) - 1) <= 0.02):
                        for block in record["layout_dets"]:
                            yield block, size
            elif isinstance(record, list) and index == page:
                # Hybrid/VLM raw model bboxes are fractions, not pixels or 0..1000.
                for block in record:
                    yield block, [1, 1]
    elif kind == "content" and isinstance(data, list):
        for block in data:
            if isinstance(block, dict) and block.get("page_idx") == page:
                yield block, [1000, 1000]


def read_layout(root, page, page_size, source_sha, *, output_page=None):
    """Prefer the common middle schema; do not duplicate model/content observations."""
    root = Path(root)
    for kind, pattern in [("middle", "*_middle.json"), ("model", "*_model.json"),
                          ("content", "*_content_list.json")]:
        paths = sorted(root.rglob(pattern))
        if kind == "content" and not paths:
            paths = sorted(root.rglob("content_list.json"))
        # Multiple parse results in a root have no unambiguous page identity.
        if len(paths) != 1:
            continue
        path = paths[0]
        if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > MAX_JSON_BYTES:
            continue
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
        except (OSError, ValueError):
            continue
        file_sha = hashlib.sha256(raw).hexdigest()
        regions = []
        rejected = 0
        actual_page = page if output_page is None else output_page
        for index, (block, size) in enumerate(layout_items(data, kind, actual_page, page_size)):
            if not isinstance(block, dict):
                rejected += 1
                continue
            box = normalized_bbox(block.get("poly", block.get("bbox")), size)
            if box is None:
                rejected += 1
                continue
            label = block.get("type") or block.get("label") or CATEGORY_NAMES.get(block.get("category_id"), "unknown")
            if not isinstance(label, str):
                rejected += 1
                continue
            score = block.get("score")
            score = score if type(score) in {int, float} and math.isfinite(score) and 0 <= score <= 1 else None
            regions.append({"region_id": "layout_" + digest([LAYOUT_VERSION, source_sha, page, file_sha, index])[:20],
                            "type": label[:60], "bbox": box, "score": score, "provider": "mineru"})
        if not regions:
            continue
        # Keep visual regions visible in dense pages; report every truncation explicitly.
        regions.sort(key=lambda r: (r["type"] not in {"image", "image_body", "table", "chart", "figure"},
                                    r["bbox"][1], r["bbox"][0], r["region_id"]))
        return {"version": LAYOUT_VERSION, "source_sha256": source_sha, "page_idx": page,
                "coordinate_system": "0..1000 relative to the rendered source page",
                "artifact_kind": kind, "artifact_sha256": file_sha,
                "regions": regions[:MAX_REGIONS], "total_regions": len(regions),
                "omitted_regions": max(0, len(regions) - MAX_REGIONS), "rejected_regions": rejected,
                "exhaustive": False, "status": "observed"}
    return None


class LayoutEvidence:
    """Reuse this run's parse artifacts or parse one physical page once, on demand."""
    def __init__(self, inventory, config, root, source_sha, parse_cache=None):
        self.inventory, self.config, self.source_sha = inventory, config, source_sha
        self.parse_cache = parse_cache
        try:
            parser_version = version("mineru")
        except PackageNotFoundError:
            parser_version = "unavailable"
        self.key = digest([LAYOUT_VERSION, parser_version, source_sha, config.model_dump(mode="json")])
        self.root = root / "agentic/layout" / self.key

    def page_root(self, page):
        if type(page) is not int or page not in self.inventory.pages:
            raise ValueError("Layout requires an existing physical source page")
        return self.root / str(page)

    def parsed_page(self, page):
        folder = self.page_root(page)
        manifest = folder / "parse.json"
        try:
            data = json.loads(manifest.read_text())
            if data["key"] != self.key or data["page_idx"] != page:
                return None
            for name, sha in data["files"].items():
                path = folder / "parse" / name
                if (not path.resolve().is_relative_to((folder / "parse").resolve())
                        or hashlib.sha256(path.read_bytes()).hexdigest() != sha):
                    return None
            name = data["content_list"]
            if name not in data["files"]:
                return None
            return folder / "parse" / name
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def cached(self, page):
        self.page_root(page)
        if hashlib.sha256(self.inventory.pdf.read_bytes()).hexdigest() != self.source_sha:
            raise ValueError("Layout source PDF changed during this run")
        with fitz.open(self.inventory.pdf) as pdf:
            size = [pdf[page].rect.width, pdf[page].rect.height]
        # Native Word logical pages are not physical PDF coordinates. Sliced primary
        # parses also have ambiguous offsets; a single-page parse below binds them.
        if (self.parse_cache and self.inventory.ir.has_physical_pages()
                and self.config.start_page is None and self.config.end_page is None):
            result = read_layout(self.parse_cache, page, size, self.source_sha)
            if result:
                return self.with_word_images(page, {**result, "cache_hit": True, "origin": "primary_parse"})
        if self.parsed_page(page):
            result = read_layout(self.page_root(page) / "parse", page, size, self.source_sha, output_page=0)
            if result:
                return self.with_word_images(page, {**result, "cache_hit": True, "origin": "single_page_parse"})
        return None

    def with_word_images(self, page, result):
        # Native Word evidence is independently labelled, never attributed to MinerU.
        # A layout model often detects an outer table without its embedded pictures.
        pictures = []
        for block_id, image in self.inventory.word_images.items():
            if not image.is_file() or not image.resolve().is_relative_to(self.inventory.directory.resolve()):
                continue
            for position in self.inventory.image_positions.get(block_id, []):
                if position.get("page_idx") != page:
                    continue
                box = normalized_bbox(position.get("bbox"), [1000, 1000])
                if box is None:
                    continue
                image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
                pictures.append({"region_id": "word_" + digest([self.source_sha, page, block_id, image_sha, box])[:20],
                                 "type": "image", "bbox": box, "score": None, "provider": "word_native",
                                 "word_block_id": block_id, "image_sha256": image_sha})
        combined = pictures + result["regions"]
        return {**result, "regions": combined[:MAX_REGIONS],
                "total_regions": result["total_regions"] + len(pictures),
                "omitted_regions": result["omitted_regions"] + max(0, len(combined) - MAX_REGIONS)}

    async def parse_page(self, page):
        from app.adapters.mineru import MinerUAdapter
        if hashlib.sha256(self.inventory.pdf.read_bytes()).hexdigest() != self.source_sha:
            raise ValueError("Layout source PDF changed during this run")
        if path := self.parsed_page(page):
            return path
        folder = self.page_root(page)
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / "source-page.pdf"
        with fitz.open(self.inventory.pdf) as original, fitz.open() as single:
            single.insert_pdf(original, from_page=page, to_page=page)
            single.save(source)
        cfg = self.config.model_copy(update={"start_page": None, "end_page": None})
        # A failed/interrupted attempt must not mix old artifacts with a new parse.
        shutil.rmtree(folder / "parse", ignore_errors=True)
        result = await MinerUAdapter(cfg).parse(source, folder / "parse", cfg)
        if not result.success or not result.content_list_path:
            raise ValueError("pdf_reparse_failed: " + str(result.error))
        parse_root = (folder / "parse").resolve()
        if not result.content_list_path.resolve().is_relative_to(parse_root):
            raise ValueError("Parser content list escapes its output directory")
        files = {}
        for path in (folder / "parse").rglob("*"):
            if path.is_file():
                if not path.resolve().is_relative_to(parse_root):
                    raise ValueError("Parser artifact escapes its output directory")
                files[str(path.relative_to(folder / "parse"))] = hashlib.sha256(path.read_bytes()).hexdigest()
        name = str(result.content_list_path.relative_to(folder / "parse"))
        if name not in files:
            raise ValueError("Parser content list is missing")
        atomic_write_json(folder / "parse.json", {"key": self.key, "page_idx": page, "files": files, "content_list": name})
        return result.content_list_path

    async def inspect(self, page):
        if result := self.cached(page):
            return result
        already_parsed = self.parsed_page(page) is not None
        await self.parse_page(page)
        result = self.cached(page)
        if result is None:
            return {"version": LAYOUT_VERSION, "source_sha256": self.source_sha, "page_idx": page,
                    "status": "unavailable", "regions": [], "exhaustive": False,
                    "reason": "No supported layout geometry; use source images/native assets instead."}
        result["cache_hit"] = already_parsed
        return result

    def selected(self, page, ids, observation):
        if not ids:
            return []
        current = self.cached(page)
        if (not current or not observation or observation.get("source_sha256") != self.source_sha
                or observation.get("page_idx") != page
                or current.get("artifact_sha256") != observation.get("artifact_sha256")):
            raise ValueError("Layout selection requires a current, source-bound inspection")
        known = {r["region_id"]: r for r in current["regions"]}
        if len(ids) > 3 or len(ids) != len(set(ids)) or any(i not in known for i in ids):
            raise ValueError("Unknown, duplicate or out-of-scope layout region ID")
        return [known[i] for i in ids]
