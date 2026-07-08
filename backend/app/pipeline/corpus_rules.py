"""Data-driven corpus-specific heuristics.

Quality heuristics that are tuned to a specific document corpus (regex fixes,
bilingual label maps, keyword lists, named markers) live in a JSON ruleset
instead of being hardcoded, so a new corpus can swap them out without
inheriting fixture-specific behavior.

The bundled default ruleset (``rulesets/default.json``) reproduces the
historical hardcoded values verbatim, so default behavior is unchanged. Set
``SEMARK_CORPUS_RULES_PATH`` to point at a custom ruleset file; an invalid or
missing custom file logs a warning and falls back to the bundled default.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RULES_PATH_ENV_VAR = "SEMARK_CORPUS_RULES_PATH"
DEFAULT_RULESET_PATH = Path(__file__).resolve().parent / "rulesets" / "default.json"

_FLAG_NAMES: dict[str, re.RegexFlag] = {
    "ASCII": re.ASCII,
    "DOTALL": re.DOTALL,
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "UNICODE": re.UNICODE,
    "VERBOSE": re.VERBOSE,
}


def _parse_flags(value: Any) -> int:
    if not value:
        return 0
    if isinstance(value, str):
        names = [part.strip() for part in value.split("|") if part.strip()]
    elif isinstance(value, (list, tuple)):
        names = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise ValueError(f"unsupported regex flags value: {value!r}")
    flags = 0
    for name in names:
        try:
            flags |= _FLAG_NAMES[name.upper()]
        except KeyError:
            raise ValueError(f"unknown regex flag: {name!r}") from None
    return flags


@dataclass(frozen=True)
class RegexRule:
    """One regex fix: a compiled ``pattern`` plus a ``replacement``.

    ``sub`` rewrites matching spans in place. Callers that need whole-value
    overrides (or ``re.Match.expand`` templates) use ``pattern.search`` and
    apply ``replacement`` themselves.
    """

    pattern: re.Pattern[str]
    replacement: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegexRule:
        return cls(
            pattern=re.compile(str(data["pattern"]), _parse_flags(data.get("flags"))),
            replacement=str(data.get("replacement", "")),
        )

    def sub(self, text: str) -> str:
        return self.pattern.sub(self.replacement, text)


def _rule_list(data: Any) -> tuple[RegexRule, ...]:
    return tuple(RegexRule.from_dict(item) for item in (data or []))


def _string_tuple(data: Any) -> tuple[str, ...]:
    if data is None:
        return ()
    if isinstance(data, str):
        return (data,)
    return tuple(str(item) for item in data)


@dataclass(frozen=True)
class CorpusRules:
    """Typed view over one corpus ruleset (see ``rulesets/default.json``).

    Sections:
    - ``title_fixes``: search-and-expand rules for source-title inference.
    - ``text_fixes``: ``re.sub`` rules for display-line OCR/spacing repair.
    - ``field_label_fixes``: ``re.sub`` rules for inferred form-field labels.
    - ``field_label_overrides``: search rules that replace the whole label
      with ``replacement`` when the pattern matches.
    - ``visual_label_map``: ordered pattern -> replacement map for bilingual
      visual (flowchart) node labels.
    - ``flow_role_terms``: role names surfaced from flowchart nodes.
    - ``document_markers``: named literal lists (exact-match markers).
    - ``plan_keywords``: named keyword lists for document-plan detection.
    """

    title_fixes: tuple[RegexRule, ...] = ()
    text_fixes: tuple[RegexRule, ...] = ()
    field_label_fixes: tuple[RegexRule, ...] = ()
    field_label_overrides: tuple[RegexRule, ...] = ()
    visual_label_map: dict[str, str] = field(default_factory=dict)
    flow_role_terms: tuple[str, ...] = ()
    document_markers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    plan_keywords: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> CorpusRules:
        if not isinstance(data, dict):
            raise ValueError("corpus ruleset must be a JSON object")
        return cls(
            title_fixes=_rule_list(data.get("title_fixes")),
            text_fixes=_rule_list(data.get("text_fixes")),
            field_label_fixes=_rule_list(data.get("field_label_fixes")),
            field_label_overrides=_rule_list(data.get("field_label_overrides")),
            visual_label_map={
                str(key): str(value)
                for key, value in (data.get("visual_label_map") or {}).items()
            },
            flow_role_terms=_string_tuple(data.get("flow_role_terms")),
            document_markers={
                str(key): _string_tuple(value)
                for key, value in (data.get("document_markers") or {}).items()
            },
            plan_keywords={
                str(key): _string_tuple(value)
                for key, value in (data.get("plan_keywords") or {}).items()
            },
        )

    def marker_list(self, name: str) -> tuple[str, ...]:
        return self.document_markers.get(name, ())

    def keyword_list(self, name: str) -> tuple[str, ...]:
        return self.plan_keywords.get(name, ())


_cached_rules: CorpusRules | None = None


def _load_ruleset_file(path: Path) -> CorpusRules:
    return CorpusRules.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_rules() -> CorpusRules:
    override = os.environ.get(RULES_PATH_ENV_VAR, "").strip()
    if override:
        try:
            return _load_ruleset_file(Path(override))
        except Exception as exc:
            logger.warning(
                "Failed to load corpus ruleset from %s=%s (%s); falling back to bundled default",
                RULES_PATH_ENV_VAR,
                override,
                exc,
            )
    return _load_ruleset_file(DEFAULT_RULESET_PATH)


def get_rules() -> CorpusRules:
    """Return the active corpus ruleset, loading and caching it on first use."""

    global _cached_rules
    if _cached_rules is None:
        _cached_rules = _load_rules()
    return _cached_rules


def reset_rules_cache() -> None:
    """Clear the cached ruleset so the next ``get_rules()`` reloads it."""

    global _cached_rules
    _cached_rules = None
