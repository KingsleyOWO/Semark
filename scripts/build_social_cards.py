#!/usr/bin/env python3
"""Build deterministic FB comparison cards from the public Semark demos."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "social"
BG = OUT / "semantic-card-background.png"
SCREENSHOT = ROOT / "examples" / "demos" / "zh-screenshot-guide-01" / "source-page-2.png"
FLOWCHART = ROOT / "examples" / "demos" / "zh-flowchart-01" / "source-page.png"

W, H = 1080, 1350
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_MEDIUM = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

INK = "#162B3A"
MUTED = "#526675"
BLUE = "#1976A3"
BLUE_SOFT = "#E8F4F8"
TEAL = "#167D75"
TEAL_SOFT = "#E5F4F1"
RED = "#C54A46"
RED_SOFT = "#FCECEB"
WHITE = "#FFFFFF"
LINE = "#C9DAE2"


def font(size: int, bold: bool = False, medium: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_MEDIUM if medium else FONT_REGULAR
    return ImageFont.truetype(path, size)


def canvas() -> Image.Image:
    bg = Image.open(BG).convert("RGB")
    ratio = W / H
    current = bg.width / bg.height
    if current > ratio:
        crop_w = int(bg.height * ratio)
        left = (bg.width - crop_w) // 2
        bg = bg.crop((left, 0, left + crop_w, bg.height))
    else:
        crop_h = int(bg.width / ratio)
        top = (bg.height - crop_h) // 2
        bg = bg.crop((0, top, bg.width, top + crop_h))
    return bg.resize((W, H), Image.Resampling.LANCZOS)


def shadow_card(base: Image.Image, box: tuple[int, int, int, int], radius: int = 24,
                fill: str = WHITE, outline: str | None = LINE) -> None:
    x0, y0, x1, y1 = box
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((x0 + 5, y0 + 9, x1 + 5, y1 + 9), radius, fill=(18, 54, 72, 35))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    base.paste(shadow, (0, 0), shadow)
    d = ImageDraw.Draw(base)
    d.rounded_rectangle(box, radius, fill=fill, outline=outline, width=2 if outline else 0)


def fit_image(path: Path, box: tuple[int, int, int, int], contain: bool = True) -> Image.Image:
    x0, y0, x1, y1 = box
    target_w, target_h = x1 - x0, y1 - y0
    im = Image.open(path).convert("RGB")
    scale = min(target_w / im.width, target_h / im.height) if contain else max(
        target_w / im.width, target_h / im.height
    )
    resized = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    layer = Image.new("RGB", (target_w, target_h), WHITE)
    left = (target_w - resized.width) // 2
    top = (target_h - resized.height) // 2
    layer.paste(resized, (left, top))
    if not contain:
        layer = layer.crop((0, 0, target_w, target_h))
    return layer


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int,
         color: str = INK, bold: bool = False, medium: bool = False,
         spacing: int = 8, anchor: str | None = None) -> None:
    draw.multiline_text(xy, value, font=font(size, bold=bold, medium=medium), fill=color,
                        spacing=spacing, anchor=anchor)


def pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, fill: str,
         color: str, width: int) -> None:
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + 46), 23, fill=fill)
    draw.text((x + width // 2, y + 22), label, font=font(23, bold=True), fill=color,
              anchor="mm")


def footer(draw: ImageDraw.ImageDraw, value: str) -> None:
    draw.rounded_rectangle((55, 1246, 1025, 1315), 28, fill="#17384A")
    draw.text((540, 1280), value, font=font(29, bold=True), fill=WHITE, anchor="mm")


def build_screenshot_card() -> None:
    im = canvas()
    d = ImageDraw.Draw(im)
    pill(d, (55, 42), "SEMARK · BEFORE / AFTER", BLUE_SOFT, BLUE, 346)
    text(d, (55, 105), "一張操作截圖，RAG 到底看到了什麼？", 48, bold=True)
    text(d, (55, 168), "同一頁文件：純解析結果 vs. 結構化語意文本", 27, color=MUTED)

    shadow_card(im, (55, 222, 1025, 650), radius=22)
    source = fit_image(SCREENSHOT, (76, 243, 1004, 629), contain=True)
    im.paste(source, (76, 243))
    d = ImageDraw.Draw(im)
    pill(d, (76, 242), "原始文件", "#EEF2F4", INK, 136)

    shadow_card(im, (55, 686, 500, 1214), radius=24, fill="#FFF9F8", outline="#E8C6C3")
    shadow_card(im, (525, 686, 1025, 1214), radius=24, fill="#F8FCFB", outline="#BFDCD7")
    d = ImageDraw.Draw(im)
    pill(d, (79, 712), "只做 OCR／Parser", RED_SOFT, RED, 236)
    text(d, (79, 780), "![](images/\nf93289d3a314f46910d41d95\na3175e6b....jpg)", 27, color="#6E4D4A", medium=True, spacing=10)
    d.line((79, 936, 476, 936), fill="#E8C6C3", width=2)
    text(d, (79, 971), "整個資料登打畫面\n只剩一條圖片連結", 33, color=RED, bold=True, spacing=14)
    text(d, (79, 1083), "欄位、紅框與提示值\n都無法被 RAG 檢索", 27, color=MUTED, spacing=10)

    pill(d, (549, 712), "Semark 語意化", TEAL_SOFT, TEAL, 221)
    text(d, (549, 784), "收入科目代號", 24, color=MUTED, medium=True)
    text(d, (549, 821), "12171002103", 36, color=TEAL, bold=True)
    text(d, (549, 883), "收入科目名稱", 24, color=MUTED, medium=True)
    text(d, (549, 920), "其他雜項收入", 32, color=INK, bold=True)
    text(d, (549, 982), "收入／對帳機關代號", 24, color=MUTED, medium=True)
    text(d, (549, 1019), "1710003", 36, color=TEAL, bold=True)
    d.rounded_rectangle((549, 1080, 997, 1176), 16, fill="#EAF6F3")
    text(d, (573, 1097), "✓ 點選右上角「列印」\n✓ 備註輸入聯絡電話及案號", 25,
         color=INK, medium=True, spacing=8)

    footer(d, "從「有圖但搜不到」→「可以檢索、可以回答」")
    im.save(OUT / "fb-01-ocr-vs-semark.png", optimize=True)


def build_flowchart_card() -> None:
    im = canvas()
    d = ImageDraw.Draw(im)
    pill(d, (55, 42), "SEMARK · VISUAL REASONING", BLUE_SOFT, BLUE, 354)
    text(d, (55, 105), "流程圖不是文字清單，而是決策邏輯", 48, bold=True)
    text(d, (55, 168), "把框框、箭頭與條件，轉成可檢索的流程", 27, color=MUTED)

    shadow_card(im, (55, 230, 487, 1216), radius=24)
    shadow_card(im, (512, 230, 1025, 1216), radius=24, fill="#F8FCFB", outline="#BFDCD7")
    d = ImageDraw.Draw(im)
    pill(d, (79, 255), "原始流程圖", "#EEF2F4", INK, 164)
    source = fit_image(FLOWCHART, (79, 321, 463, 1178), contain=True)
    im.paste(source, (79, 321))

    d = ImageDraw.Draw(im)
    pill(d, (536, 255), "Semark 轉換後", TEAL_SOFT, TEAL, 213)
    text(d, (544, 324), "被害人提出申訴", 29, color=INK, bold=True)
    text(d, (556, 368), "↓", 31, color=TEAL, bold=True)
    text(d, (544, 410), "判斷適用法律", 29, color=INK, bold=True)

    d.line((559, 468, 559, 655), fill=TEAL, width=4)
    for y in (492, 568, 644):
        d.line((559, y, 588, y), fill=TEAL, width=4)
    text(d, (596, 473), "性別平等工作法\n→ 機關內部調查", 25, color=INK, medium=True, spacing=6)
    text(d, (596, 549), "性別平等教育法\n→ 學校內部調查", 25, color=INK, medium=True, spacing=6)
    text(d, (596, 625), "性騷擾防治法\n→ 依行為人身分分流", 25, color=INK, medium=True, spacing=6)

    d.rounded_rectangle((536, 724, 1000, 1036), 18, fill="#EAF6F3")
    text(d, (560, 746), "行為人為機關／學校人員", 26, color=TEAL, bold=True)
    text(d, (560, 791), "判斷是否有不予受理情形", 24, color=INK, medium=True)
    text(d, (560, 839), "是 → 移送社會處確認", 24, color=INK)
    text(d, (560, 880), "否 → 依內部規定調查", 24, color=INK)
    d.line((560, 930, 976, 930), fill="#BFDCD7", width=2)
    text(d, (560, 951), "無雇主／身分不明", 26, color=TEAL, bold=True)
    text(d, (560, 993), "→ 向事件發生地警察機關申訴", 23, color=INK, medium=True)

    text(d, (536, 1082), "輸出的不只是文字：", 25, color=MUTED, medium=True)
    text(d, (536, 1125), "條件  ·  分支  ·  負責單位", 30, color=TEAL, bold=True)

    footer(d, "從「框框裡的字」→「完整的流程與判斷關係」")
    im.save(OUT / "fb-02-flowchart-to-semantics.png", optimize=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_screenshot_card()
    build_flowchart_card()
