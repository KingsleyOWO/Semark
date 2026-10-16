"""Optional CPU OCR evidence; bounded pixels, immutable inputs and cancellable execution."""
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
from pathlib import Path

from PIL import Image

from app.adapters.metafile import on_white
from app.config import settings


def ocr_python():
    configured = settings.ocr_python_path
    if configured:
        return configured if Path(configured).is_file() and os.access(configured, os.X_OK) else None
    return sys.executable if importlib.util.find_spec("paddleocr") else None


def ocr_runtime_identity():
    python = ocr_python()
    if not python:
        return None
    stat = Path(python).stat()
    worker = Path(__file__).with_name("ocr_worker.py")
    return hashlib.sha256((str(Path(python).absolute()) + str(stat.st_mtime_ns)
                           + settings.ocr_model_cache).encode() + worker.read_bytes()).hexdigest()


async def read_ocr(source, region, scale, folder, *, timeout=90):
    python = ocr_python()
    if not python:
        raise ValueError("ocr_runtime_unavailable")
    if (len(region) != 4 or not all(type(v) is int for v in region)
            or not 0 <= region[0] < region[2] <= 1000 or not 0 <= region[1] < region[3] <= 1000
            or scale not in {1, 2, 4}):
        raise ValueError("invalid_ocr_region_or_scale")
    source = Path(source)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "ocr-input.png"
    with Image.open(source) as image:
        width, height = image.size
        box = (math.floor(region[0] * width / 1000), math.floor(region[1] * height / 1000),
               math.ceil(region[2] * width / 1000), math.ceil(region[3] * height / 1000))
        size = ((box[2] - box[0]) * scale, (box[3] - box[1]) * scale)
        if image.width * image.height > 20_000_000 or size[0] * size[1] > 20_000_000:
            raise ValueError("ocr_pixel_limit")
        # On white, as printed: the black under transparent pixels hid dark lettering.
        on_white(image).crop(box).resize(size, Image.Resampling.LANCZOS).save(target)
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
           "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"}
    if settings.ocr_model_cache:
        env["PADDLE_PDX_CACHE_HOME"] = settings.ocr_model_cache
    process = await asyncio.create_subprocess_exec(python, "-I", str(Path(__file__).with_name("ocr_worker.py")),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=env, start_new_session=True)
    try:
        output, error = await asyncio.wait_for(process.communicate(
            (json.dumps({"image": str(target.resolve())}) + "\n").encode()), timeout)
    finally:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
    if process.returncode or len(output) > 2_000_000:
        raise ValueError("ocr_execution_failed: " + error.decode(errors="replace")[-500:])
    responses = [line[len(b"OCR_RESULT="):] for line in output.splitlines() if line.startswith(b"OCR_RESULT=")]
    if len(responses) != 1:
        raise ValueError("ocr_response_format_error")
    response = json.loads(responses[0])
    lines = response["lines"]
    if len(lines) > 128 or sum(len(item["text"]) for item in lines) > 16000:
        raise ValueError("ocr_output_capacity_exceeded: select a smaller region")
    for i, item in enumerate(lines):
        score = item["score"]
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid_ocr_score")
        points = item.pop("polygon")
        if len(points) < 3 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
            raise ValueError("invalid_ocr_polygon")
        xs = [max(0, min(1000, round((p[0] / scale + box[0]) * 1000 / width))) for p in points]
        ys = [max(0, min(1000, round((p[1] / scale + box[1]) * 1000 / height))) for p in points]
        item.update(line_id=f"line{i}", bbox=[min(xs), min(ys), max(xs), max(ys)])
        if item["bbox"][0] >= item["bbox"][2] or item["bbox"][1] >= item["bbox"][3]:
            raise ValueError("empty_ocr_source_region")
    if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
        raise ValueError("ocr_source_changed")
    return {"status": "ok" if lines else "empty", "source_sha256": sha, "region": region, "scale": scale,
            "lines": lines, "version": response["version"], "detector": "PP-OCRv5_mobile_det",
            "recognizer": "PP-OCRv5_server_rec", "device": "cpu"}
