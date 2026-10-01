"""Bounded source evidence for physical pages, Word tables and embedded images."""
import hashlib
import re
from pathlib import Path

import fitz
from PIL import Image

from app.models.document_ir import BlockType
from app.pipeline.content_coverage import normalized_text


def table_cells(block):
    from app.pipeline.table_structure import table_structure
    structure = table_structure(block)
    return [{"id": cell.id, **cell.model_dump(exclude={"source_bbox"})}
            for cell in structure.cells] if structure else []


def header_cells(block):
    """Recognize explicit headers and month-axis rows; never guess data rows."""
    cells = table_cells(block)
    if not cells:
        return []
    first = [c for c in cells if c["row"] == cells[0]["row"]]
    month_labels = sum(bool(re.fullmatch(r"(?:1[0-2]|[1-9])\s*月", c["text"])) for c in first)
    if (all(c["header"] for c in first) or month_labels >= 4
            or (len(first) == 1 and int(first[0]["colspan"]) > 1)):
        return [c for c in first if c["text"]]
    return []


def required_cells(block):
    # Even title/header text must be accounted for. Blank month cells carry no
    # textual value; their visual relations are checked for every named row.
    return {c["id"] for c in table_cells(block) if c["text"]}


def page_image(ir, index, run_path):
    page = next((p for p in ir.pages if p.page_idx == index), None)
    candidates = []
    if page and page.page_image_path:
        declared = Path(page.page_image_path)
        candidates += [run_path / declared, run_path / "assets/pages" / declared.name]
    candidates.append(run_path / "assets/pages" / f"p{index:04d}.png")
    for path in candidates:
        if path.is_file() and path.resolve().is_relative_to(run_path.resolve()):
            return path
    return None


def _embedded_source(block, cache):
    declared = Path(str(block.payload.get("img_path") or ""))
    if not cache or not declared.name or declared.is_absolute() or ".." in declared.parts:
        return None
    matches = [p for p in cache.rglob(declared.name)
               if p.is_file() and p.resolve().is_relative_to(cache.resolve())
               and p.as_posix().endswith(declared.as_posix())]
    if len(matches) != 1:
        return None
    return matches[0]


def _embedded_image(block, cache, directory):
    source = _embedded_source(block, cache)
    if source is None:
        return None
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    target = directory / ("raster-" + sha + ".png")
    try:
        # A legacy Word exporter renamed EMF/WMF bytes to .png. Decode before
        # sending to a VLM; a filename is not proof of an image's actual format.
        with Image.open(source) as image:
            image.load()
            if not target.exists():
                image.convert("RGB").save(target)
        return target
    except (OSError, ValueError):
        return None


def _isolate_word_picture(source, image_sha, output):
    from docx import Document
    from docx.oxml.ns import qn

    document = Document(source)
    selected = None
    for child in document.element.body:
        for node in child.iter():
            if node.tag == qn("a:blip"):
                part = document.part.related_parts.get(node.get(qn("r:embed")))
                if part and hashlib.sha256(part.blob).hexdigest() == image_sha:
                    selected = child
                    break
        if selected is not None:
            break
    if selected is None:
        return False
    for child in list(document.element.body):
        if child is not selected and child.tag != qn("w:sectPr"):
            document.element.body.remove(child)
    # Page furniture is not part of the selected embedded picture.
    for section in document.sections:
        for node in list(section._sectPr):
            if node.tag in {qn("w:headerReference"), qn("w:footerReference")}:
                section._sectPr.remove(node)
    document.save(output)
    return True


def match_word_pages(block, texts):
    """Require a distinctive contiguous window, not a guessed logical page 0."""
    labels = list(dict.fromkeys(
        normalized_text(c["text"]) for c in table_cells(block)
        if len(normalized_text(c["text"])) >= 4
    ))
    if len(labels) < 2:
        return []
    # Repeated project names occur in surrounding prose. Distinctive labels
    # (for example the schedule corner header) carry stronger location evidence.
    weights = {label: 1 / max(1, sum(label in text for text in texts)) for label in labels}
    total = sum(weights.values())
    scored = []
    for start in range(len(texts)):
        for length in range(1, min(3, len(texts) - start) + 1):
            combined = "".join(texts[start:start + length])
            hits = sum(weights[label] for label in labels if label in combined)
            score = hits / total
            scored.append((score, -length, start))
    best = max(scored, default=(0, 0, 0))
    score, neg_length, start = best
    selected = set(range(start, start - neg_length))
    outside = max((v[0] for v in scored
                   if selected.isdisjoint(range(v[2], v[2] - v[1]))), default=0)
    if score < 0.65 or score - outside < 0.15:
        return []
    return sorted(selected)


class EvidenceSource:
    def __init__(self, ir, run_path, cache=None):
        self.ir = ir
        self.run_path = run_path
        self.cache = cache
        self.directory = run_path / "agentic/evidence"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.word_pdf = None
        self.word_texts = []
        self.word_images = {}
        self.warning = None

    async def prepare_word(self, source, blocks=None):
        if not source or not source.is_file() or source.suffix.lower() not in {".doc", ".docx"}:
            return
        from app.pipeline.stages.parse import ParseStage

        blocks = self.ir.blocks if blocks is None else blocks
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if any(b.type == BlockType.TABLE for b in blocks):
            folder = self.directory / ("word-" + sha)
            pdf = folder / (source.stem + ".pdf")
            if not pdf.is_file():
                pdf, error = await ParseStage._convert_office_to_pdf(source, folder)
                if error:
                    self.warning = error
                    return
            try:
                with fitz.open(pdf) as doc:
                    self.word_texts = [normalized_text(p.get_text()) for p in doc]
                self.word_pdf = pdf
            except Exception as exc:
                self.warning = f"Word PDF evidence unreadable: {exc}"
        for block in blocks:
            if block.type != BlockType.IMAGE or _embedded_image(block, self.cache, self.directory):
                continue
            original = _embedded_source(block, self.cache)
            if original is None or source.suffix.lower() != ".docx":
                continue
            image_sha = hashlib.sha256(original.read_bytes()).hexdigest()
            folder = self.directory / ("picture-" + sha[:16] + "-" + image_sha[:16])
            folder.mkdir(parents=True, exist_ok=True)
            fragment, pdf = folder / "picture.docx", folder / "picture.pdf"
            try:
                if not pdf.is_file():
                    if not _isolate_word_picture(source, image_sha, fragment):
                        continue
                    pdf, error = await ParseStage._convert_office_to_pdf(fragment, folder)
                    if error:
                        self.warning = error
                        continue
                rendered = []
                with fitz.open(pdf) as doc:
                    if len(doc) > 3:
                        self.warning = "Embedded picture spans more than three rendered pages"
                        continue
                    for index, page in enumerate(doc):
                        boxes = [fitz.Rect(entry[1]) for entry in page.get_bboxlog()]
                        if not boxes:
                            continue
                        clip = boxes[0]
                        for box in boxes[1:]:
                            clip |= box
                        clip = (clip + (-8, -8, 8, 8)) & page.rect
                        if clip.is_empty:
                            continue
                        target = folder / f"p{index:04d}.png"
                        page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip).save(target)
                        rendered.append((f"Original Word embedded vector, rendered part {index + 1}", target))
                self.word_images[block.block_id] = rendered
            except (OSError, ValueError) as exc:
                self.warning = f"Word embedded image rendering failed: {exc}"

    def for_block(self, block):
        if self.ir.has_physical_pages():
            indices = [block.page_idx]
            # A table near a page boundary may continue in either direction.
            if block.type == BlockType.TABLE:
                if block.bbox_norm and block.bbox_norm[1] < 150 and block.page_idx > 0:
                    indices.append(block.page_idx - 1)
                if (not block.bbox_norm or block.bbox_norm[3] >= 750
                        or len({c["row"] for c in table_cells(block)}) >= 8):
                    indices.append(block.page_idx + 1)
            return [(f"PDF page {i + 1}", p) for i in indices
                    if (p := page_image(self.ir, i, self.run_path))]
        if block.type == BlockType.IMAGE:
            image = _embedded_image(block, self.cache, self.directory)
            return [("Original Word embedded image", image)] if image else self.word_images.get(block.block_id, [])
        if block.type == BlockType.TABLE and self.word_pdf:
            indices = match_word_pages(block, self.word_texts)
            results = []
            with fitz.open(self.word_pdf) as doc:
                for i in indices:
                    target = self.word_pdf.parent / f"p{i:04d}.png"
                    if not target.exists():
                        doc[i].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(target)
                    results.append((f"Word converted PDF page {i + 1}", target))
            return results
        return []
