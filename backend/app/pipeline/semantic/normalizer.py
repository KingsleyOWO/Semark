"""Deterministic semantic normalization for titles, versions, notes, and fields."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schema import SemanticField, SemanticVersion

_VERSION_RE = re.compile(r"(?P<date>\d{2,3}[.．]\d{1,2}[.．]\d{1,2})\s*(?:核定|核准|修正|修訂)?版")
_COMPACT_VERSION_RE = re.compile(r"(?P<date>\d{6,8})\s*(?:核定|核准|修正|修訂)?版")


def clean_title_noise(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip("#:： ")
    text = re.sub(r"(表[一二三四五六七八九十0-9]+)[〇○昇鑑箇]+", r"\1", text)
    text = re.sub(r"(表[一二三四五六七八九十0-9]+)\s*[〇○昇鑑箇]+", r"\1", text)
    text = re.sub(r"[昇鑑](?=台灣|臺灣|國內|國外|大台北|大臺北)", "", text)
    return text.strip("#:： ")


def extract_version(*values: Any) -> SemanticVersion:
    for value in values:
        text = str(value or "")
        match = _VERSION_RE.search(text)
        if match:
            raw = match.group(0).replace("．", ".")
            date = match.group("date").replace("．", ".")
            return SemanticVersion(raw=raw, date=date)
        match = _COMPACT_VERSION_RE.search(text)
        if match:
            raw = match.group(0)
            date = match.group("date")
            return SemanticVersion(raw=raw, date=date)
    return SemanticVersion()


def is_version_text(value: Any) -> bool:
    text = re.sub(r"\s+", "", str(value or ""))
    return bool(
        _VERSION_RE.fullmatch(text)
        or _COMPACT_VERSION_RE.fullmatch(text)
        or re.fullmatch(r"\d{2,3}[.．]\d{1,2}[.．]\d{1,2}(?:核定|核准|修正|修訂)?版?", text)
        or re.fullmatch(r"\d{6,8}(?:核定|核准|修正|修訂)?版?", text)
    )


def source_title_from_path(source_path: str) -> str:
    stem = Path(source_path).stem.strip()
    stem = re.sub(r"^\d+(?:-\d+)*", "", stem)
    stem = _VERSION_RE.sub("", stem)
    stem = _COMPACT_VERSION_RE.sub("", stem).strip(" -_　")
    return clean_title_noise(stem) or Path(source_path).stem[:80] or "文件"


def normalize_notes(notes: list[str]) -> tuple[list[str], SemanticVersion]:
    kept: list[str] = []
    version = SemanticVersion()
    seen: set[str] = set()
    for note in notes or []:
        text = re.sub(r"\s+", " ", str(note or "")).strip()
        if not text:
            continue
        if is_version_text(text):
            if not version.raw:
                version = extract_version(text)
            continue
        key = re.sub(r"\s+", "", text)
        if key in seen:
            continue
        seen.add(key)
        kept.append(text)
    return kept, version


def split_merged_field_label(value: Any) -> list[str]:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip("：:")
    if not text:
        return []
    text = text.replace("☐", "□").replace("☑", "□")
    checkbox_parts = [part.strip(" ：:") for part in re.split(r"(?=□)", text) if part.strip(" ：:")]
    if len(checkbox_parts) > 1:
        result: list[str] = []
        for part in checkbox_parts:
            result.extend(split_merged_field_label(part.lstrip("□")))
        return _dedupe(result)
    compact = text.replace(" ", "")
    known = ["沖預借金額", "應補發金額", "受款人", "應繳回金額", "付款對象", "支票抬頭", "戶名", "銀行", "帳號"]
    matched = [name for name in known if name in compact]
    if len(matched) >= 2:
        return matched
    delimiters = re.split(r"[；;]|(?<=[：:])\s*(?=[^_\s]{2,})", text)
    if len(delimiters) > 1:
        result = []
        for part in delimiters:
            cleaned = part.strip(" ：:_")
            if cleaned:
                result.append(cleaned)
        return _dedupe(result)
    return [text.lstrip("□").strip(" ：:")]


def normalize_fields(fields: Any) -> list[SemanticField]:
    normalized: list[SemanticField] = []
    seen: set[tuple[str, str]] = set()
    for field in fields or []:
        if isinstance(field, dict):
            raw_name = str(field.get("name") or "").strip()
            aliases = [str(item) for item in field.get("aliases", []) if str(item).strip()]
            evidence = str(field.get("evidence_text") or raw_name).strip() or None
            explicit_type = str(field.get("type") or "").strip()
            explicit_req = str(field.get("requirement") or "").strip()
            explicit_required = bool(field.get("required"))
        else:
            raw_name = str(field or "").strip()
            aliases = []
            evidence = raw_name or None
            explicit_type = ""
            explicit_req = ""
            explicit_required = False
        for name in split_merged_field_label(raw_name):
            clean_name = _normalize_field_name(name)
            if not clean_name or is_version_text(clean_name) or is_low_value_field_name(clean_name):
                continue
            field_type = explicit_type or infer_field_type(clean_name)
            requirement = explicit_req or infer_requirement(clean_name)
            required = explicit_required or requirement == "required"
            section = infer_field_section(clean_name)
            key = (re.sub(r"\s+", "", clean_name).lower(), field_type.lower())
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                SemanticField(
                    name=clean_name,
                    normalized_name=clean_name,
                    type=field_type,
                    required=required,
                    requirement=requirement,
                    section=section,
                    aliases=aliases,
                    evidence_text=evidence,
                )
            )
    return normalized


def is_low_value_field_name(name: str) -> bool:
    normalized = _normalize_field_name(name)
    compact = re.sub(r"\s+", "", normalized)
    if not compact:
        return True
    exact_noise = {
        "男", "女", "年", "月", "日", "年月日", "月日", "上午", "下午",
        "職", "等", "與", "無資料", "其他", "不詳", "本院", "主管",
        "方式一", "方式二", "表單", "填寫規則", "一式三聯", "第一聯", "第二聯", "第三聯",
    }
    if compact.strip("()（）") in exact_noise:
        return True
    if re.fullmatch(r"[上午下午時分年月日:：]+", compact):
        return True
    if re.fullmatch(r"[()（）]?(?:\d{1,3}(?:-\d{1,3})?分|無資料)[()（）]?", compact):
        return True
    short_meaningful = r"章|簽|金|額|名|日|期|費|機|車|幣|率|註|備|款|帳|號|單|點|地|由|職|稱"
    if len(compact) <= 2 and not re.search(short_meaningful, compact):
        return True
    if compact.count("□") >= 3 and len(compact) <= 18:
        return True
    if len(compact) > 34 and not re.search(r"日期|期間|地址|說明|備註|事由|內容", compact):
        return True
    if len(compact) > 48:
        return True
    if re.search(r"存查|核定版|正本存查|出勤紀錄|經申請人及單位主管雙方同意|會計單位[︵(]?白|採購單位[︵(]?藍|請購單位[︵(]?紅", compact):
        return True
    if re.fullmatch(r"[□☐☑、,，/A-Za-z0-9]+", compact):
        return True
    return False


def infer_field_type(name: str) -> str:
    normalized = _normalize_field_name(name)
    if re.search(r"簽名|簽章|核章", normalized):
        return "signature"
    if normalized.startswith("□") or re.search(r"保險|報支單位|示範院|APEC|PECC|其他$", normalized):
        return "checkbox"
    if re.search(r"日期|期間|年月日", normalized):
        return "date"
    if re.search(r"金額|費用|交通費|宿費|膳雜費|生活費|辦公費|匯率|折合台幣|合計|小計|雜費", normalized):
        return "number"
    if "身份證" in normalized or "身分證" in normalized:
        return "id"
    if "姓名" in normalized or "申請人" in normalized or "受款人" in normalized or "領款人" in normalized:
        return "name"
    if re.search(r"飛機|汽車|火車|高鐵|計程車", normalized):
        return "choice"
    return "text"


def infer_requirement(name: str) -> str:
    normalized = _normalize_field_name(name)
    if normalized.startswith("□"):
        return "conditional"
    if re.search(r"預估|預借|需用日期|報支單位|付款|支票|戶名|銀行|帳號|預算|計畫|保險|申根|其他|變更|備註|代理人|主任秘書|副院長|院長|董事長", normalized):
        return "conditional"
    if re.search(r"申請單位|申請人|申請日期|姓名|職級|職稱|出差地點|出差事由|出差期間|單位主管|出差人簽", normalized):
        return "required"
    return "situational"


def infer_field_section(name: str) -> str:
    mapping = [
        (r"主任|主管|秘書|副院長|院長|董事長|人事|處長|簽|章|核|對保", "簽核/用印"),
        (r"採購|請購|預付款|預付|承辦單位|詢價|議價|招標|報價|廠商|電腦中心|資服中心", "採購/請購資訊"),
        (r"飛機|汽車|火車|高鐵|計程車", "交通工具"),
        (r"出差地點|出差事由|出差期間|代理人|變更|起訖地點|起始地點|到達地點|工作紀要", "出差/行程資訊"),
        (r"□|保險|報支單位|預估費用|預借金額|金額|費用|交通費|宿費|膳雜費|生活費|辦公費|匯率|幣別|折合台幣|合計|小計|預算|付款|支票|匯款|受款人|領款人|應繳回|應補發|沖預借|單據編號", "費用/報支資訊"),
        (r"保證|保證人|商號|營業|資本|負責人|被保人|關係", "保證人/商號資料"),
        (r"學校|系所|科系|學位|進修|選修|課程|學科|減免|受訓|訓練", "進修/訓練資訊"),
        (r"姓名|出生|身分證|身份證|電話|手機|E-?mail|地址|緊急|申請|日期|單位|職級|職稱|員工", "申請/基本資料"),
        (r"附件|合約|切結|證明", "附件/佐證資料"),
    ]
    for pattern, section in mapping:
        if re.search(pattern, name, re.IGNORECASE):
            return section
    return "表單欄位"


def fields_to_dicts(fields: list[SemanticField]) -> list[dict[str, Any]]:
    return [field.to_dict() for field in fields]


def _normalize_field_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip("：:")
    text = text.lstrip("□☐☑").strip(" ：:") if text.startswith(("□", "☐", "☑")) else text
    text = re.sub(r"_+", "", text).strip()
    return text


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = re.sub(r"\s+", "", item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
