"""The bitmap a Word metafile holds, the way Office stores a pasted slide, map or photo.

Spot check 2026-10-06: of the 106 Word metafiles in the stored reports, 88 draw
nothing but one bitmap through EMF+ (GDI+) comment records and 6 are an EMF that
stretches one bitmap. Pillow opens neither and LibreOffice draws the EMF+ ones
blank, so the drafting model was sent bytes it could not load and the review saw an
empty page area. Only a metafile that draws exactly one bitmap and nothing else is
decoded; any other metafile keeps its current handling.
"""
import io
import shutil
import struct
import zipfile
from pathlib import Path

from PIL import Image

EMR_HEADER, EMR_COMMENT, EMR_EOF, EMR_STRETCHDIBITS = 1, 70, 14, 81
SRCCOPY, DIB_RGB_COLORS = 0x00CC0020, 0
PLUS_OBJECT, PLUS_DRAW_IMAGE, PLUS_DRAW_IMAGE_POINTS = 0x4008, 0x401A, 0x401B
OBJECT_IMAGE, IMAGE_BITMAP = 5, 1
CONTINUED = 0x8000
# Records that only set state, clear the canvas or define objects; any other EMF+
# record (lines, fills, text...) draws something the bitmap alone would not show.
PLUS_NON_DRAWING = {0x4001, 0x4002, 0x4003, 0x4004, PLUS_OBJECT, 0x4009} | set(range(0x401D, 0x4036))
# 32 bits per pixel, stored B, G, R, A: premultiplied alpha, straight alpha, no alpha.
RAW_MODES = {0x000E200B: ("RGBa", "BGRa"), 0x0026200A: ("RGBA", "BGRA"), 0x00022009: ("RGBA", "BGRX")}
MAX_PIXELS = 20_000_000


def _records(data):
    pos = 0
    while pos + 8 <= len(data):
        kind, size = struct.unpack_from("<II", data, pos)
        if size < 8 or size % 4 or pos + size > len(data):
            raise ValueError("malformed EMF record")
        yield kind, data[pos:pos + size]
        pos += size
    if pos != len(data):
        raise ValueError("trailing bytes after the last EMF record")


def _plus_records(comment):
    body = comment[16:12 + struct.unpack_from("<I", comment, 8)[0]]
    at = 0
    while at + 12 <= len(body):
        kind, flags, size, data_size = struct.unpack_from("<HHII", body, at)
        if size < 12 or data_size > size - 12 or at + size > len(body):
            raise ValueError("malformed EMF+ record")
        yield kind, flags, body[at + 12:at + 12 + data_size]
        at += size


def _bitmap(blob):
    _, image_type = struct.unpack_from("<II", blob)
    if image_type != IMAGE_BITMAP:
        return None
    width, height, stride, pixel_format, compressed = struct.unpack_from("<iiiII", blob, 8)
    payload = blob[28:]
    if not 0 < width * height <= MAX_PIXELS:
        return None
    if compressed:
        image = Image.open(io.BytesIO(payload))
        image.load()
        return image
    if pixel_format not in RAW_MODES or abs(stride) < width * 4 or len(payload) < abs(stride) * height:
        return None
    mode, raw = RAW_MODES[pixel_format]
    image = Image.frombuffer(mode, (width, height), payload, "raw", raw, stride, 1)
    return image.convert("RGBA")


def emfplus_bitmap(data):
    """The one bitmap an EMF+ only metafile draws, or None when it is anything else."""
    try:
        records = list(_records(data))
        if not records or records[0][0] != EMR_HEADER or records[0][1][40:44] != b" EMF" or records[-1][0] != EMR_EOF:
            return None
        objects, draws, continued = {}, 0, {}
        for kind, record in records[1:-1]:
            if kind != EMR_COMMENT:
                return None  # a GDI record draws on its own
            if record[12:16] != b"EMF+":
                continue  # Office also stores a GDIC multiformats comment; comments never draw
            for plus_kind, flags, chunk in _plus_records(record):
                if plus_kind in (PLUS_DRAW_IMAGE, PLUS_DRAW_IMAGE_POINTS):
                    draws += 1
                elif plus_kind not in PLUS_NON_DRAWING:
                    return None
                if plus_kind != PLUS_OBJECT or (flags >> 8) & 0x7F != OBJECT_IMAGE:
                    continue
                object_id = flags & 0xFF
                # As LibreOffice reads Office output: every part of a split object, the
                # last one included, starts with the object's total size.
                if flags & CONTINUED or object_id in continued:
                    total = struct.unpack_from("<I", chunk)[0]
                    continued.setdefault(object_id, total)
                    chunk = chunk[4:]
                objects[object_id] = objects.get(object_id, b"") + chunk
        if draws != 1 or len(objects) != 1:
            return None
        object_id, blob = next(iter(objects.items()))
        if object_id in continued:
            # Record data is padded to four bytes; the declared total is the object.
            if len(blob) < continued[object_id]:
                return None
            blob = blob[:continued[object_id]]
        return _bitmap(blob)
    except (struct.error, ValueError, OSError, Image.DecompressionBombError):
        return None


def stretched_bitmap(data):
    """The bitmap of an EMF that only copies one whole DIB onto the page, or None."""
    try:
        records = list(_records(data))
        if not records or records[0][0] != EMR_HEADER or records[0][1][40:44] != b" EMF" or records[-1][0] != EMR_EOF:
            return None
        drawing = [record for kind, record in records[1:-1] if kind != EMR_COMMENT]
        if len(drawing) != 1 or struct.unpack_from("<I", drawing[0])[0] != EMR_STRETCHDIBITS:
            return None
        record = drawing[0]
        x_src, y_src, cx_src, cy_src = struct.unpack_from("<4i", record, 32)
        off_bmi, cb_bmi, off_bits, cb_bits, usage, operation = struct.unpack_from("<6I", record, 48)
        info, bits = record[off_bmi:off_bmi + cb_bmi], record[off_bits:off_bits + cb_bits]
        if operation != SRCCOPY or usage != DIB_RGB_COLORS or len(info) != cb_bmi or len(bits) != cb_bits or cb_bmi < 40:
            return None
        width, height = struct.unpack_from("<ii", info, 4)
        if not 0 < width * abs(height) <= MAX_PIXELS or (x_src, y_src) != (0, 0):
            return None
        # The whole bitmap, give or take the few pixels exporters round the source size up by;
        # a part of it would need the DIB's row direction to place.
        if not all(size <= source <= size + max(4, size // 100) for size, source in ((width, cx_src), (abs(height), cy_src))):
            return None
        bmp = b"BM" + struct.pack("<IHHI", 14 + cb_bmi + cb_bits, 0, 0, 14 + cb_bmi) + info + bits
        image = Image.open(io.BytesIO(bmp))
        image.load()
        return image
    except (struct.error, ValueError, OSError, Image.DecompressionBombError):
        return None


def metafile_bitmap(data):
    """The one bitmap an EMF+ only metafile or a single-DIB EMF draws, or None."""
    return emfplus_bitmap(data) or stretched_bitmap(data)


def open_picture(blob):
    """Pillow's image of a Word picture part, including the bitmap a metafile only copies.

    Raises like Image.open for anything else (a WMF or drawn EMF has no pixels to read here).
    """
    try:
        image = Image.open(io.BytesIO(blob))
        image.load()
        return image
    except Exception:  # Pillow's metafile plugin can fail on a header it cannot size
        decoded = metafile_bitmap(blob)
        if decoded is None:
            raise
        return decoded


def drawable_word_copy(source, folder):
    """A copy of the .docx whose single-bitmap metafiles hold the PNG they draw, for LibreOffice.

    The part keeps its name: LibreOffice reads a picture by its bytes, and Word
    relationships, content types and image hashes stay as they are. None when the
    document has no such picture.
    """
    replaced = {}
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            if info.filename.startswith("word/media/") and info.filename.lower().endswith(".emf"):
                image = metafile_bitmap(archive.read(info))
                if image is not None:
                    buffer = io.BytesIO()
                    image.save(buffer, "PNG")
                    replaced[info.filename] = buffer.getvalue()
        if not replaced:
            return None
        folder = Path(folder)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        target = folder / Path(source).name
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as copy:
            for info in archive.infolist():
                copy.writestr(info, replaced.get(info.filename) or archive.read(info), compress_type=info.compress_type)
    return target
