"""Original PDF/Word materials for bounded reading, without guessing page mappings."""
import hashlib
import io
import re
import zipfile

import fitz
from PIL import Image

from app.pipeline.delivery import atomic_write_json


def _pixels(blob):
    with Image.open(io.BytesIO(blob)) as image:
        if image.width * image.height > 20_000_000:
            return None
        return (image.size, hashlib.sha256(image.convert('RGB').tobytes()).hexdigest())


def native_materials(inventory, page, folder):
    """Extract only images actually painted on this page, not its unused resources.

    The conversion can overlap/downscale Word pictures. An exact pixel match to
    the original Word media is preferred when available; never match by caption.
    """
    folder.mkdir(parents=True, exist_ok=True)
    materials = []
    with fitz.open(inventory.pdf) as pdf:
        source = pdf[page]
        text = source.get_text(sort=True).strip()
        if len(text) > 12000:
            raise ValueError('native_text_exceeds_context_limit')
        material_rect = None
        for entry in source.get_text("blocks", sort=True):
            text = entry[4].strip()
            rect = fitz.Rect(entry[:4])
            if not text or (re.fullmatch(r"\d+", text) and (rect.y0 > source.rect.height * .92 or rect.y1 < source.rect.height * .06)):
                continue
            if (materials and len(text) <= 4 and material_rect is not None
                    and abs(rect.x0 - material_rect.x0) < 8 and 0 <= rect.y0 - material_rect.y1 < 40):
                materials[-1]['text'] += '\n' + text
                materials[-1]['bbox'][3] = round(rect.y1 / source.rect.height * 1000)
                material_rect = rect
                continue
            material_rect = rect
            materials.append({'id': f'text-{len(materials)}', 'text': text,
                              'bbox': [round(rect.x0 / source.rect.width * 1000),
                                       round(rect.y0 / source.rect.height * 1000),
                                       round(rect.x1 / source.rect.width * 1000),
                                       round(rect.y1 / source.rect.height * 1000)],
                              'kind': 'native_text', 'image': None})
        seen = set()
        for info in source.get_image_info(xrefs=True):
            xref = info.get('xref', 0)
            rect = fitz.Rect(info['bbox']) & source.rect
            if not xref or rect.is_empty:
                continue
            key = (xref, tuple(rect))
            if key in seen:
                continue
            seen.add(key)
            # Tiny decorative rules remain visible in the full-page context.
            if info['width'] * info['height'] > 20_000_000:
                raise ValueError('native_image_exceeds_pixel_limit')
            if min(info['width'], info['height']) < 24 or rect.width * rect.height < 100:
                continue
            if sum('blob' in m for m in materials) >= 12 or len(materials) >= 24:
                raise ValueError('native_images_exceed_context_limit')
            blob = pdf.extract_image(xref)['image']
            materials.append({'id': f'image-{len(materials)}', 'text': '', 'kind': 'pdf_embedded_image',
                              'bbox': [round(rect.x0 / source.rect.width * 1000),
                                       round(rect.y0 / source.rect.height * 1000),
                                       round(rect.x1 / source.rect.width * 1000),
                                       round(rect.y1 / source.rect.height * 1000)],
                              'blob': blob})
    source_path = inventory.source_path
    if source_path and source_path.suffix.lower() == '.docx':
        signatures = {signature: m for m in materials if 'blob' in m
                      if (signature := _pixels(m['blob'])) is not None}
        with zipfile.ZipFile(source_path) as archive:
            total = 0
            for info in archive.infolist()[:1000]:
                if not info.filename.startswith('word/media/') or info.file_size > 20_000_000:
                    continue
                total += info.file_size
                if total > 80_000_000:
                    break
                blob = archive.read(info)
                try:
                    match = signatures.get(_pixels(blob))
                except (OSError, ValueError):
                    continue
                if match:
                    match.update(blob=blob, kind='word_embedded_image')
    materials.sort(key=lambda m: (m['bbox'][1], m['bbox'][0]))
    for material in materials:
        if 'blob' in material:
            blob = material.pop('blob')
            image_path = folder / (material['id'] + '.png')
            with Image.open(io.BytesIO(blob)) as image:
                image.convert('RGB').save(image_path)
            material['image'] = image_path
            material['sha256'] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    atomic_write_json(folder / 'sources.json', [dict(m, image=str(m['image']) if m['image'] else None)
                                               for m in materials], ensure_ascii=False, indent=2)
    return materials
