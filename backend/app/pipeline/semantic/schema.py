"""Stable semantic document schema used before rendering RAG outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DocumentType = Literal["form_document", "reference_table", "diagram_document", "article_document"]


@dataclass
class SemanticSource:
    file_name: str
    pages: list[int] = field(default_factory=list)
    parent_title: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "pages": self.pages,
            "parent_title": self.parent_title,
        }


@dataclass
class SemanticVersion:
    raw: str | None = None
    date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"raw": self.raw, "date": self.date}


@dataclass
class SemanticField:
    name: str
    normalized_name: str
    type: str = "text"
    required: bool = False
    requirement: str = "situational"
    section: str = "表單欄位"
    aliases: list[str] = field(default_factory=list)
    evidence_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "normalized_name": self.normalized_name,
            "type": self.type,
            "required": self.required,
            "requirement": self.requirement,
            "section": self.section,
            "aliases": self.aliases,
            "evidence_text": self.evidence_text,
        }


@dataclass
class SemanticDocument:
    document_type: DocumentType
    title: str
    source: SemanticSource
    version: SemanticVersion = field(default_factory=SemanticVersion)
    purpose: str = ""
    usage_scenarios: list[str] = field(default_factory=list)
    fields: list[SemanticField] = field(default_factory=list)
    approval_flow: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    rag_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type,
            "title": self.title,
            "source": self.source.to_dict(),
            "version": self.version.to_dict(),
            "purpose": self.purpose,
            "usage_scenarios": self.usage_scenarios,
            "fields": [field.to_dict() for field in self.fields],
            "approval_flow": self.approval_flow,
            "notes": self.notes,
            "records": self.records,
            "sections": self.sections,
            "rag_hints": self.rag_hints,
        }


@dataclass
class SemanticIssue:
    code: str
    severity: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class SemanticQualityReport:
    correctness_score: float
    rag_readiness_score: float
    issues: list[SemanticIssue] = field(default_factory=list)
    recommended_repairs: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return round((self.correctness_score * 0.55) + (self.rag_readiness_score * 0.45), 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "correctness_score": self.correctness_score,
            "rag_readiness_score": self.rag_readiness_score,
            "issues": [issue.to_dict() for issue in self.issues],
            "recommended_repairs": self.recommended_repairs,
        }
