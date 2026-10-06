"""Conservative Word page candidates from source text, order and original pixels."""

import hashlib
import math

import fitz
from PIL import Image, ImageChops, ImageOps, ImageStat

from app.adapters.metafile import open_picture
from app.models.document_ir import BlockType
from app.pipeline.agentic_evidence import match_word_pages
from app.pipeline.content_coverage import normalized_text, plain_text


def word_block_pages(block, texts):
    """A single ambiguous label cannot identify a physical page."""
    if block.type == BlockType.TABLE:
        return match_word_pages(block, texts)
    if block.type == BlockType.IMAGE:
        # Nearby prose and generated captions are not image location evidence.
        return []
    text = normalized_text(plain_text(block.get_text()))
    if not text:
        return []
    exact = [i for i, page in enumerate(texts) if text in page]
    if exact:
        return exact if len(exact) == 1 else []
    if len(text) < 16:
        return []
    grams = {text[i : i + 8] for i in range(0, len(text) - 7, 4)}
    candidates = [(i, {g for g in grams if g in page}) for i, page in enumerate(texts)]
    candidates = [(i, hits) for i, hits in candidates if len(hits) >= max(2, len(grams) * 0.15)]
    if not candidates or len(candidates) > 3:
        return []
    indices = [i for i, _ in candidates]
    covered = set().union(*(hits for _, hits in candidates))
    if indices[-1] - indices[0] + 1 != len(indices) or len(covered) / len(grams) < 0.65:
        return []
    return indices


def _image_data(blob):
    with open_picture(blob) as image:
        if image.width * image.height > 20_000_000:
            return None
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
        signature = (rgba.size, hashlib.sha256(rgba.tobytes()).hexdigest())
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        rgb = background.convert("RGB")
        small = rgb.resize((64, 64), Image.Resampling.LANCZOS)
        return {
            "signature": signature,
            "aspect": rgb.width / rgb.height,
            "small": small,
            "detail": rgb.resize((256, 256), Image.Resampling.LANCZOS),
            "variation": max(ImageStat.Stat(small).stddev),
        }


def _image_distance(a, b):
    return math.sqrt(sum(v * v for v in ImageStat.Stat(ImageChops.difference(a, b)).rms) / 3)


def word_image_matches(blocks, source, pdf_path, *, evidence_dir=None):
    """Source-bound exact pixels or conservative resampling candidates, never captions.

    Office recompresses/resizes pictures. A nonexact candidate must pass both
    64px and 256px comparisons and have a clear margin from competing images.
    These locate evidence for the reviewer; they do not certify picture text.
    """
    eligible = [
        b
        for b in blocks
        if b.type == BlockType.IMAGE
        and b.payload.get("word_source", {}).get("rel_id")
        and b.payload["word_source"].get("image_sha256")
    ]
    if not eligible or not source or source.suffix.lower() != ".docx":
        return {}
    from docx import Document

    document = Document(source)
    sources, by_rel, evidence = {}, {}, {}
    for block in eligible:
        meta = block.payload["word_source"]
        rel = meta["rel_id"]
        part = document.part.related_parts.get(rel)
        if not part or not hasattr(part, "blob"):
            continue
        blob = part.blob
        if len(blob) > 20_000_000 or hashlib.sha256(blob).hexdigest() != meta["image_sha256"]:
            continue
        if rel not in by_rel:
            try:
                by_rel[rel] = _image_data(blob)
            except (OSError, ValueError):
                by_rel[rel] = None
        if by_rel[rel] is not None:
            sources[block.block_id] = by_rel[rel]
            if evidence_dir is not None:
                # Export only the original relationship part whose bytes match the IR.
                # PNG normalizes format/EXIF without reducing the original resolution.
                name = "word-image-" + meta["image_sha256"] + ".png"
                path = evidence_dir / name
                with open_picture(blob) as original:
                    ImageOps.exif_transpose(original).convert("RGBA").save(path)
                evidence[block.block_id] = name
    result, decoded, painted = {}, {}, []
    if not sources:
        return result
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            for image in page.get_image_info(xrefs=True):
                xref = image.get("xref", 0)
                rect = fitz.Rect(image["bbox"]) & page.rect
                if not xref or rect.is_empty or image["width"] * image["height"] > 20_000_000:
                    continue
                if xref not in decoded:
                    try:
                        extracted = pdf.extract_image(xref)
                        if extracted.get("smask"):
                            pix = fitz.Pixmap(
                                fitz.Pixmap(pdf, xref), fitz.Pixmap(pdf, extracted["smask"])
                            )
                            blob = pix.tobytes("png")
                        else:
                            blob = extracted["image"]
                        decoded[xref] = _image_data(blob)
                    except (OSError, ValueError):
                        decoded[xref] = None
                if decoded[xref] is not None:
                    painted.append(
                        (
                            decoded[xref],
                            {
                                "page_idx": page.number,
                                "bbox": [
                                    round(rect.x0 / page.rect.width * 1000),
                                    round(rect.y0 / page.rect.height * 1000),
                                    round(rect.x1 / page.rect.width * 1000),
                                    round(rect.y1 / page.rect.height * 1000),
                                ],
                            },
                        )
                    )
    for bid, original in sources.items():
        exact = [
            dict(position, method="exact_pixels")
            for target, position in painted
            if original["signature"] == target["signature"]
        ]
        if exact:
            result[bid] = exact
            continue
        if original["variation"] < 8:
            continue  # Low-information icons/blank fields cannot locate a page by similarity.
        scores = [
            (_image_distance(original["small"], target["small"]), target, position)
            for target, position in painted
            if abs(original["aspect"] / target["aspect"] - 1) <= 0.03
        ]
        matches, rejected = [], []
        for score, target, position in scores:
            detail = (
                _image_distance(original["detail"], target["detail"])
                if score <= 6
                else float("inf")
            )
            if score <= 6 and detail <= 10:
                matches.append(
                    dict(
                        position,
                        method="resampled_pixels",
                        rmse64=round(score, 3),
                        rmse256=round(detail, 3),
                    )
                )
            else:
                rejected.append(score)
        # A near tie with an image that narrowly fails a threshold is ambiguous.
        if matches and (not rejected or min(rejected) - min(m["rmse64"] for m in matches) >= 12):
            result[bid] = matches
    for bid, matches in result.items():
        if bid in evidence:
            for match in matches:
                match["source_image"] = evidence[bid]
    return result


def locate_word_blocks(blocks, texts, *, image_matches=None, overrides=None):
    """Propose pages only when source candidates fit an unambiguous ordered interval.

    No model certification is implied. Conflicting anchors and repeated content
    with multiple remaining pages are deliberately left unresolved.
    """
    blocks = sorted(blocks, key=lambda b: (b.reading_order, b.block_id))
    image_matches, overrides = image_matches or {}, overrides or {}
    mapping, details, candidates = {}, {}, {}
    for block in blocks:
        bid = block.block_id
        if block.type == BlockType.IMAGE:
            options = sorted({m["page_idx"] for m in image_matches.get(bid, [])})
            pages = options if len(options) == 1 else []
            method = (
                "image_similarity_candidate"
                if any(m.get("method") == "resampled_pixels" for m in image_matches.get(bid, []))
                else "exact_image_pixels"
            )
        else:
            text = normalized_text(plain_text(block.get_text()))
            options = [i for i, page in enumerate(texts) if text and text in page]
            pages = word_block_pages(block, texts)
            method = "source_text"
        candidates[bid] = options
        if bid in overrides:
            pages, method = overrides[bid], "verified_source_override"
        mapping[bid] = pages
        details[bid] = {
            "pages": pages,
            "method": method if pages else "unresolved",
            "candidates": options,
        }

    # Only original, independently matched anchors constrain order; inferred
    # labels must not snowball into proof for more inferred labels.
    anchors = {
        b.block_id: mapping[b.block_id]
        for b in blocks
        if mapping[b.block_id] and b.type != BlockType.IMAGE
    }
    for index, block in enumerate(blocks):
        bid = block.block_id
        if mapping[bid] or not candidates[bid]:
            continue
        parent = block.payload.get("word_source", {}).get("parent_table_block_id")
        if block.type == BlockType.IMAGE and parent in anchors:
            lower, upper = anchors[parent][0], anchors[parent][-1]
            bounds = [parent, parent]
        else:
            before = next((b for b in reversed(blocks[:index]) if b.block_id in anchors), None)
            after = next((b for b in blocks[index + 1 :] if b.block_id in anchors), None)
            if before is None and after is None:
                continue
            lower = anchors[before.block_id][-1] if before else 0
            upper = anchors[after.block_id][0] if after else len(texts) - 1
            bounds = [before.block_id if before else None, after.block_id if after else None]
        selected = [p for p in candidates[bid] if lower <= p <= upper]
        details[bid].update(anchor_block_ids=bounds, page_interval=[lower, upper])
        if len(selected) == 1:
            mapping[bid] = selected
            details[bid].update(
                pages=selected,
                method="ordered_image_pixels"
                if block.type == BlockType.IMAGE
                else "ordered_source_text",
            )
    return mapping, details
