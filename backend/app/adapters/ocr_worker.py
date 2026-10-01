"""One bounded PaddleOCR request in an administrator-configured Python runtime."""
import json
import sys


def main():
    request = json.loads(sys.stdin.readline())
    import numpy as np
    import paddleocr
    from paddleocr import PaddleOCR
    from PIL import Image

    engine = PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det",
                       text_recognition_model_name="PP-OCRv5_server_rec",
                       use_doc_orientation_classify=False, use_doc_unwarping=False,
                       use_textline_orientation=False, device="cpu", cpu_threads=4,
                       enable_mkldnn=False)
    with Image.open(request["image"]) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("ocr_pixel_limit")
        pixels = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
    lines = []
    for result in engine.predict(pixels, text_rec_score_thresh=0.0):
        value = result.json["res"]
        for text, score, polygon in zip(value["rec_texts"], value["rec_scores"], value["rec_polys"], strict=True):
            lines.append({"text": text, "score": float(score), "polygon": np.asarray(polygon).tolist()})
    print("OCR_RESULT=" + json.dumps({"lines": lines, "version": paddleocr.__version__}, ensure_ascii=False))


if __name__ == "__main__":
    main()
