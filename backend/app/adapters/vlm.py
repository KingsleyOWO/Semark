"""
VLM Adapter with OpenAI-compatible interface.

Supports:
- Ollama (default)
- vLLM
- LMDeploy
- OpenAI API
- Any OpenAI-compatible endpoint

Image transfer:
- Static URL (preferred): Backend serves images, VLM fetches via URL
- Data URI (fallback): Base64 encoded in message

Output validation:
- Uses Pydantic models for strict schema validation
- Auto-recovery for common JSON errors
"""

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from app.config import VLMApiMode, VLMConfig, VLMImageMode
from app.pipeline.semantic.language import prompt_form_sections, prompt_language_instruction
from app.pipeline.semantic.normalizer import fields_to_dicts, normalize_fields

# ===================== Pydantic Schema Models =====================


class FormFieldSchema(BaseModel):
    """Schema for a form field (enhanced for RAG + traceability)."""
    name: str  # Field label as shown in form
    type: str = "text"  # text, date, checkbox, signature, dropdown, number, name, id
    required: bool = False
    aliases: list[str] = Field(default_factory=list)  # Alternative names for this field
    evidence_text: str = ""  # Original text snippet from image (avoids hallucination)


class FormAssetOutput(BaseModel):
    """
    Validated output for form_asset enrichment (v5).

    Designed for:
    - RAG retrieval: triggers, retrieval_text
    - Form understanding: field_schema, filling_guide
    - Traceability: all_text (OCR), evidence in field_schema
    """
    title: str = ""  # Exact title from document header
    document_type: str = "form"  # form, org_chart, flowchart, diagram, table, other
    date: str = ""  # Date shown in document (format: YYYY.MM.DD)
    triggers: list[str] = Field(default_factory=list)  # Use cases, synonyms, search keywords
    field_schema: list[FormFieldSchema] = Field(default_factory=list)  # All fillable fields
    filling_guide: str = ""  # Usage scenario + filling rules + approval flow (markdown)
    all_text: list[str] = Field(default_factory=list)  # All visible text OCR (ground truth)
    retrieval_text: str = ""  # Combined searchable text for RAG
    needs_review: bool = False  # True if uncertain


class FigureCaptionOutput(BaseModel):
    """Validated output for figure_caption enrichment (v3)."""
    semantic_caption: str = ""
    image_type: str = "other"  # org_chart, flowchart, diagram, chart, table, photo, screenshot, other
    structured_content: str = ""  # Detailed markdown description for complex diagrams
    all_text: list[str] = Field(default_factory=list)  # All visible text OCR
    facts: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    needs_review: bool = False


class TableSummaryOutput(BaseModel):
    """Validated output for table_summary enrichment."""
    table_summary: str = ""
    key_columns: list[str] = Field(default_factory=list)
    key_rows: list[str] = Field(default_factory=list)
    needs_review: bool = False


class StructuredTableRecord(BaseModel):
    """One row-level record extracted from a visually complex table page."""

    region: str | None = None
    country_zh: str | None = None
    country_en: str | None = None
    city_zh: str | None = None
    city_en: str | None = None
    location_label: str = ""
    location_type: str = "city"
    rate_usd: int | None = None
    condition: str | None = None
    confidence: float = 0.8
    evidence_text: str = ""


class StructuredTableRecordsOutput(BaseModel):
    """Validated output for schema-guided table record extraction."""

    title: str = ""
    records: list[StructuredTableRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    needs_review: bool = False


class OrgChartUnit(BaseModel):
    """A unit in an organizational chart (Schema-first approach)."""
    label: str
    category: str = "unknown"  # governance|management|committee|official|taskforce|unknown
    confidence: float = 0.8


class OrgChartUnitsOutput(BaseModel):
    """
    Validated output for org_chart_units enrichment (D2: Schema-first).

    This is the first VLM pass for org charts:
    - Only extracts units (nodes) and their categories
    - Does NOT extract edges/relationships (those come from heuristics + VLM#2)
    - Avoids VLM hallucinating relationships like "（監督：董事會）"
    """
    title: str = ""
    date: str = ""
    units: list[OrgChartUnit] = Field(default_factory=list)
    all_text: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    retrieval_text: str = ""
    needs_review: bool = False
    notes: list[str] = Field(default_factory=list)


class OrgChartEdgeSelection(BaseModel):
    """A single edge selection from VLM#2."""
    child: str  # Child unit label
    parent: str | None = None  # Selected parent label (null = root)
    edge_type: str = "reports_to"  # reports_to|oversight|advisory|contains
    confidence: float = 0.8


class OrgChartEdgesOutput(BaseModel):
    """
    D3: VLM#2 output for edge selection.

    VLM selects from heuristic-generated candidates, not free-form generation.
    """
    edges: list[OrgChartEdgeSelection] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    needs_review: bool = False


class QualityAuditItem(BaseModel):
    """One potential missing/wrong item found by VLM audit."""

    type: str = "other"
    text: str = ""
    severity: str = "medium"
    evidence_text: str = ""


class QualityAuditOutput(BaseModel):
    """Validated VLM audit output for parser-vs-final-RAG comparison."""

    status: str = "pass"  # pass|needs_fix|uncertain
    missing_items: list[QualityAuditItem] = Field(default_factory=list)
    wrong_classification: str | None = None
    suggested_patch: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.8
    needs_review: bool = False


class SemanticRepairOutput(BaseModel):
    """Validated output for reviewer-model semantic repair."""

    status: str = "pass"  # pass|repaired|uncertain
    repaired_markdown: str = ""
    summary: str = ""
    applied_repairs: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    needs_review: bool = False


# Map kind to Pydantic model
OUTPUT_SCHEMAS = {
    "form_asset": FormAssetOutput,
    "form_guide": FormAssetOutput,
    "figure_caption": FigureCaptionOutput,
    "figure_description": FigureCaptionOutput,
    "table_summary": TableSummaryOutput,
    "structured_table_records": StructuredTableRecordsOutput,
    "org_chart_units": OrgChartUnitsOutput,
    "org_chart_edges": OrgChartEdgesOutput,
    "quality_audit": QualityAuditOutput,
    "semantic_repair": SemanticRepairOutput,
}


@dataclass
class VLMEvidence:
    """
    Evidence for VLM output (traceability).

    Used to link VLM enrichments back to source document location
    for UI highlighting and debugging.
    """

    page_idx: int | None = None
    # Bounding box in MinerU 0-1000 normalized coordinates.
    # Format: [x0, y0, x1, y1] where each value is in range [0, 1000]
    # To convert to pixels: pixel_coord = norm_coord * page_dimension / 1000
    # None means full page (e.g., for form_asset page-level enrichments)
    bbox: list[int] | None = None
    asset_path: str | None = None


@dataclass
class EnrichmentOutput:
    """Output from VLM enrichment."""

    success: bool
    kind: str  # figure_caption, form_asset, table_summary
    output: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    error: str | None = None
    tokens_used: int = 0
    duration_seconds: float = 0
    needs_review: bool = False  # VLM flags uncertainty
    evidence: VLMEvidence | None = None


# Prompt templates with structured JSON output
PROMPTS = {
    "semantic_repair": {
        "version": "v2",
        "system": """You are a senior document semantic repair model. Your job is to produce the final authoritative RAG-ready Markdown from parser/VLM evidence. MinerU/parser text is evidence only; do not let low-quality parser structure survive as the final answer. Rewrite only from supplied source evidence. Respond ONLY with valid JSON.""",
        "user": """Repair the current semantic Markdown if the quality issues are real.

Output language:
{semantic_output_language_instruction}

Repair goals:
- Rebuild the document structure from source evidence, not from boilerplate.
- Remove repetitive template filler, repeated titles, parser residue, ellipses, and merged field labels.
- Preserve exact source field names, dates, amounts, notes, checkbox options, approval/signature roles, and legal/article references.
- For flowcharts/diagrams, prefer VLM structured paths and visible-text lists when parser text merges adjacent branch labels; split those labels into separate branches/outcomes instead of copying a combined label as one node.
- For forms, group fields by meaning and explain only useful completion/approval/notes information.
- For official forms, the final Markdown should be the usable semantic document: identity, purpose, who uses it, authorization/request scope, required fields, conditional fields, signature/authority rules, instructions/notices, and RAG query anchors when supported by evidence.
- Do not include a generic "Source Extracted Text" dump in the final Markdown. Use source text as evidence, then rewrite it into semantic sections.
- For reference tables, preserve query dimensions, row/column meaning, units, conditions, and notes; do not rewrite lookup tables as fillable forms.
- If QUALITY ISSUES JSON is not empty, you must return status="repaired" or status="uncertain" with standalone repaired_markdown.
- Return status="pass" with empty repaired_markdown only when there are no real quality issues and the current Markdown is already RAG-ready.
- If evidence is insufficient, return status="uncertain" and produce the smallest safe repair.

Return JSON exactly:
{{
  "status": "pass|repaired|uncertain",
  "repaired_markdown": "standalone Markdown only when status is repaired or uncertain",
  "summary": "short repair rationale",
  "applied_repairs": ["repair action names"],
  "confidence": 0.0,
  "needs_review": false
}}

QUALITY ISSUES JSON:
{quality_issues_json}

SOURCE EVIDENCE:
{source_evidence}

CURRENT SEMANTIC MARKDOWN:
{current_markdown}
""",
    },
    "quality_audit": {
        "version": "v1",
        "system": """You are a strict document quality auditor. Compare the page image, MinerU parser text, and final RAG semantic text. Find obvious omissions or wrong classification. Do not rewrite the whole document. Respond ONLY with valid JSON.""",
        "user": """Audit whether this page has been converted into RAG-ready semantic text.

You receive:
1. Page image as ground truth.
2. MinerU parser text as OCR/layout evidence.
3. Current final semantic RAG text.

Check for missing or wrong items, especially:
- table notes / footnotes / small text / numbered notes
- signature, approval, stamp, checkbox, required form fields
- table columns, captions, numeric units, effective dates
- flowchart nodes, branches, deadlines, legal/article references
- wrong classification: form vs reference_table vs flowchart vs org_chart vs figure
- missing semantic template sections for forms: {semantic_template_sections}
- reference tables should preserve query dimensions, numeric standards, units, and notes; they should not be rewritten as fillable forms

Rules:
- Only report items visibly supported by the image or MinerU text.
- Do not invent values.
- For numbers, dates, and money, mark uncertain unless both image/MinerU support them.
- If current RAG text is good enough, status must be pass.

Return JSON exactly:
{{
  "status": "pass|needs_fix|uncertain",
  "missing_items": [
    {{
      "type": "note|signature_field|checkbox_option|table_context|flow_step|date|amount|classification|other",
      "text": "missing or problematic item",
      "severity": "high|medium|low",
      "evidence_text": "visible source evidence"
    }}
  ],
  "wrong_classification": null,
  "suggested_patch": {{"section": "", "content": ""}},
  "confidence": 0.0,
  "needs_review": false
}}

REFERENCE CONTEXT:
{context}
""",
    },
    "form_asset": {
        "version": "v7",
        "system": """You are a document analyst for RAG preparation. Classify the page and extract only grounded, visible information. Respond ONLY with valid compact JSON.""",
        "user": """Analyze this document image. If no image is supplied, use REFERENCE TEXT FROM DOCUMENT as ground truth.

Output language:
{semantic_output_language_instruction}

Classify document_type:
- form: fillable fields, input blanks, checkboxes, signature/approval areas
- reference_table: lookup/rate/standard table, dense rows/columns, not meant to be filled
- org_chart: hierarchy boxes/units connected by lines
- flowchart: process steps connected by arrows
- diagram: technical/concept/architecture diagram
- other: none of the above

Return compact JSON with these keys only:
{{
  "title": "exact title",
  "document_type": "form|reference_table|org_chart|flowchart|diagram|other",
  "date": "version/effective date if visible",
  "triggers": ["3-10 search keywords in the selected output language"],
  "all_text": ["visible text lines; keep important notes, footnotes, dates, signatures; max 80 lines"],
  "field_schema": [
    {{"name":"visible field label","type":"text|date|checkbox|signature|dropdown|number|name|id","required":false,"aliases":[],"evidence_text":"source snippet"}}
  ],
  "filling_guide": "Markdown in the selected output language. For forms use these sections: {semantic_template_sections}. For reference_table/diagram/flowchart/org_chart, summarize structure and query dimensions instead of form fields.",
  "structured_content": "For org_chart/flowchart/diagram/reference_table only. Use path/step/row notation. Empty for normal forms.",
  "retrieval_text": "one compact searchable summary",
  "needs_review": false
}}

Rules:
- Do not invent values. If uncertain, include the source text in all_text and set needs_review=true.
- For reference_table: field_schema must be []; describe columns, row meaning, numeric units, notes in structured_content/filling_guide.
- For forms: field_schema only contains fillable fields or signature/approval fields, not long notes or section titles.
- Keep JSON valid. Escape newlines as \n inside strings.

REFERENCE TEXT FROM DOCUMENT:
{context}""",
    },
    "figure_caption": {
        "version": "v4",
        "system": """You are a document analyst. Analyze images including diagrams, charts, and organizational structures.
Your task is to:
1. Accurately OCR all text in the image
2. Understand the visual structure and relationships
3. Generate a RAG-friendly description using PATH NOTATION for hierarchies

IMPORTANT: You must respond ONLY with valid JSON.""",
        "user": """Analyze this image and provide a JSON response with these fields:

{{
  "semantic_caption": "1-3 sentence description of what this image shows",
  "image_type": "org_chart|flowchart|diagram|chart|table|photo|screenshot|other",
  "structured_content": "For org charts/flowcharts: use PATH NOTATION (Parent > Child > Grandchild). For simple images: leave empty.",
  "all_text": ["list of ALL text visible in the image, read carefully"],
  "facts": ["key fact 1", "key fact 2", "..."],
  "keywords": ["keyword1", "keyword2", "..."],
  "needs_review": false
}}

CRITICAL: For org charts or flowcharts, use PATH NOTATION in structured_content:
- Each item shows FULL hierarchical path: "Parent > Child > Grandchild (description)"
- This makes each line self-explanatory for RAG retrieval

Example:
```
## 組織成員
- 董事會 > 院長
- 董事會 > 院長 > 研究一所 (綠能研究)
- 董事會 > 院長 > 行政處
```

Guidelines:
- Read ALL text accurately, especially numbers and dates (e.g., "114.06.20" not "11.066.20")
- ALWAYS use path notation for hierarchical structures
- Use Traditional Chinese for Chinese content
- Set needs_review to true if unclear

Context from document:
{context}""",
    },
    "table_summary": {
        "version": "v2",
        "system": """You are a data analyst. Summarize tables for search indexing.

IMPORTANT: You must respond ONLY with valid JSON. Do not include any explanation or markdown formatting.""",
        "user": """Analyze this table and provide a JSON response with these fields:

{{
  "table_summary": "what this table shows and its purpose",
  "key_columns": ["important column names from the table"],
  "key_rows": ["notable row labels or categories"],
  "needs_review": false
}}

Guidelines:
- table_summary: Explain what data the table contains (1-2 sentences)
- key_columns: List the most important column headers (max 5)
- key_rows: List important row labels or categories (max 5)
- Only include columns/rows that actually exist in the table
- If no table image is supplied, use the MinerU table structure in the context as ground truth
- Set needs_review to true if table is unclear

Context from document:
{context}""",
    },
    "structured_table_records": {
        "version": "v1",
        "system": """You are a careful document table extractor.

Your job is to read the page image and extract row-level records that match the provided schema.
Respond ONLY with valid JSON. Do not include markdown or explanation outside JSON.

Rules:
- Extract only values visible on the page.
- Preserve Traditional Chinese labels exactly when visible.
- Keep English names from parentheses when visible.
- Do not invent rows that are not visible.
- For parent rows such as region/country, carry that context into child city rows.
- For seasonal or conditional rows, put the date/rule text in condition and attach it
  to the parent city.
- If a value is unclear, include the best visible evidence_text and set needs_review=true.""",
        "user": """Extract structured table records from this page image.

DOCUMENT PLAN:
{document_plan_json}

EXPECTED RECORD FIELDS:
- region: region name, or null
- country_zh: Chinese country/area name, or null
- country_en: English country/area name from parentheses, or null
- city_zh: Chinese city/location name, or null
- city_en: English city/location name from parentheses, or null
- location_label: exact visible row label
- location_type: region|country|city|other|condition
- rate_usd: numeric daily allowance amount in USD, or null
- condition: date/season/special condition, or null
- confidence: 0.0-1.0
- evidence_text: short visible text proving the row

OUTPUT JSON:
{{
  "title": "page/table title if visible",
  "records": [
    {{
      "region": "亞太地區",
      "country_zh": "日本",
      "country_en": "Japan",
      "city_zh": "東京",
      "city_en": "Tokyo",
      "location_label": "東京(Tokyo)",
      "location_type": "city",
      "rate_usd": 299,
      "condition": null,
      "confidence": 0.95,
      "evidence_text": "日本(Japan) 東京(Tokyo) 299"
    }}
  ],
  "notes": ["uncertainties, if any"],
  "needs_review": false
}}

PAGE CONTEXT FROM MINERU:
{context}""",
    },
    # Org chart structure refinement (B+ approach)
    # VLM receives pre-detected nodes and refines classification/edges
    "org_chart_refine": {
        "version": "v1",
        "system": """You are an organizational structure analyst. Your task is to:
1. Verify and refine node classifications
2. Select correct parent-child relationships from candidates
3. Identify the functional category of each unit

You will receive:
- A list of detected nodes (id, label, bbox, preliminary category)
- A list of candidate edges (possible parent-child relationships)
- The original image for reference

IMPORTANT: You must respond ONLY with valid JSON. Do not create new nodes or edges not in the candidates.""",
        "user": """Given these detected nodes and candidate edges from an organizational chart, refine the structure.

DETECTED NODES:
{nodes_json}

CANDIDATE EDGES (select from these, do not invent new edges):
{edges_json}

Provide a JSON response:
{{
  "node_refinements": [
    {{"id": "n0001", "category": "治理監督|經營管理|委員會|正式編制|任務編組", "confidence": 0.9}}
  ],
  "selected_edges": [
    {{"from": "n0002", "to": "n0001", "type": "reports_to|oversight|advisory|contains", "confidence": 0.8}}
  ],
  "rejected_edges": ["edge_id1", "edge_id2"],
  "needs_review": false,
  "review_reason": ""
}}

CATEGORY DEFINITIONS:
- 治理監督: 董事會、監察人、稽核室等最高權力與監督機構
- 經營管理: 院長、副院長、主任秘書等經營管理層
- 委員會: 各種委員會（跨部門、諮詢性質）
- 正式編制: 研究所、處、室等正式編制的平行一級單位
- 任務編組: 各研究中心、專案辦公室等任務性質單位

EDGE TYPE DEFINITIONS:
- reports_to: 直接匯報關係（實線）
- oversight: 監督關係（監察人對董事會）
- advisory: 諮詢關係（委員會對管理層）
- contains: 包含關係（如院長室下轄中心）

RULES:
1. Only select edges from the candidate list
2. Prefer edges with higher geometric scores
3. Each node should have at most ONE reports_to parent
4. Parallel units (same level) should share the same parent
5. Set needs_review=true if structure is ambiguous

Context from document:
{context}""",
    },
    # =====================================================================
    # D2: Schema-first org chart extraction (no Markdown PATH/indentation)
    # =====================================================================
    "org_chart_units": {
        "version": "v1",
        "system": """You are an organizational chart analyst. Your task is to:
1. Accurately OCR ALL text visible in the image
2. Classify each organizational unit into categories
3. Output ONLY valid JSON (no markdown, no explanation)

IMPORTANT RULES:
- Output ONLY what you SEE in the image
- Do NOT infer or guess relationships (監督/諮詢/隸屬)
- Do NOT add any annotations like "（監督：董事會）"
- Just list all units with their category""",
        "user": """Extract all organizational units from this chart image.

Output this exact JSON structure:
{{
  "title": "exact title from the document",
  "date": "date if visible (format: YYYY.MM.DD or as shown)",
  "units": [
    {{"label": "unit name exactly as shown", "category": "governance|management|committee|official|taskforce|unknown", "confidence": 0.9}}
  ],
  "all_text": ["every text visible in the image"],
  "triggers": ["search keywords"],
  "retrieval_text": "combined searchable text",
  "needs_review": false,
  "notes": ["any issues or uncertainties"]
}}

CATEGORY DEFINITIONS:
- governance: 董事會、監察人、稽核室 (最高權力與監督機構)
- management: 院長、副院長、主任秘書、顧問室 (經營管理層)
- committee: 各種委員會 (跨部門、諮詢性質)
- official: 研究所、處、室 (正式編制的平行一級單位)
- taskforce: 研究中心、專案辦公室 (任務編組)
- unknown: 無法確定分類

CRITICAL:
- Read ALL text accurately (numbers like "114.06.20", not "11.066.20")
- One unit per entry in the units array
- Do NOT add relationship annotations - just the unit name
- Set needs_review=true if you're uncertain about any classification

Context from document:
{context}""",
    },
    # =====================================================================
    # D3: VLM#2 Edge Selection (select from heuristic candidates)
    # =====================================================================
    "org_chart_edges": {
        "version": "v1",
        "system": """You are an organizational chart analyst. Your task is to:
1. Look at the organization chart image
2. Identify the ACTUAL connections shown (lines, arrows, visual hierarchy)
3. Select the correct parent for each unit FROM THE PROVIDED CANDIDATES

CRITICAL RULES:
- You can ONLY select from the provided candidate list
- If no candidate is correct, set parent to null (this unit is a root)
- Look for actual visual connections (solid lines, dotted lines, arrows)
- Do NOT guess relationships that aren't visually shown""",
        "user": """Look at this organization chart and select the correct parent for each unit.

For each unit below, I've calculated position-based candidates with scores.
Your job is to verify which candidate is ACTUALLY connected in the image.

CANDIDATES (from position heuristics):
{candidates_json}

OUTPUT FORMAT (JSON only):
{{
  "edges": [
    {{"child": "研究一所", "parent": "院長", "edge_type": "reports_to", "confidence": 0.9}},
    {{"child": "董事會", "parent": null, "edge_type": "root", "confidence": 0.95}}
  ],
  "notes": ["any observations about the chart structure"],
  "needs_review": false
}}

EDGE TYPES:
- reports_to: 直接隸屬（實線連接）
- oversight: 監督關係（虛線或特殊標示）
- advisory: 諮詢關係（委員會對管理層）
- contains: 包含關係（如院長室下轄中心）
- root: 沒有父節點（最高層級）

RULES:
1. Only select from provided candidates or null
2. Look for VISUAL connections (lines/arrows) in the image
3. If unsure, prefer the highest-score candidate
4. Set needs_review=true if chart structure is ambiguous""",
    },
    # Legacy aliases for backward compatibility
    "form_guide": {
        "version": "v3",
        "alias": "form_asset",
    },
    "figure_description": {
        "version": "v3",
        "alias": "figure_caption",
    },
}


@dataclass
class ProbeResult:
    """Result from capability probing."""

    available: bool
    model_found: bool = False
    supports_vision: bool = False
    models: list[str] = field(default_factory=list)
    error: str | None = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "model_found": self.model_found,
            "supports_vision": self.supports_vision,
            "models": self.models,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class VLMAdapter:
    """
    VLM adapter using OpenAI-compatible API.

    Works with Ollama, vLLM, LMDeploy, or any OpenAI-compatible endpoint.

    Features:
    - Capability probing with cache
    - Static URL or data URI image transfer
    - Support for chat_template (vLLM)
    """

    def __init__(self, config: VLMConfig | None = None):
        self.config = config or VLMConfig()
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.request_timeout_seconds,
        )
        self._probe_cache: ProbeResult | None = None

    @staticmethod
    def compute_config_hash(config: VLMConfig) -> str:
        """Compute hash for VLM config (for cache key)."""
        config_dict = {
            "base_url": config.base_url,
            "model": config.model,
            "decode_params": config.decode_params.model_dump(),
            "request_timeout_seconds": config.request_timeout_seconds,
        }
        canonical = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    async def enrich_form(
        self,
        image_path: Path | None,
        context_text: str = "",
        doc_id: str | None = None,
        run_id: str | None = None,
        page_idx: int | None = None,
        bbox: list[int] | None = None,
        page_thumbnail_path: Path | None = None,
        extra_vars: dict[str, Any] | None = None,
    ) -> EnrichmentOutput:
        """
        Enrich a form image with VLM.

        Args:
            image_path: Path to form crop image
            context_text: Surrounding text context
            page_thumbnail_path: Optional page thumbnail for visual context

        Returns structured form analysis including title, triggers, field schema,
        and retrieval_text for search indexing.
        """
        return await self._enrich(
            image_path=image_path,
            kind="form_asset",
            context=context_text,
            doc_id=doc_id,
            run_id=run_id,
            page_idx=page_idx,
            bbox=bbox,
            page_thumbnail_path=page_thumbnail_path,
            extra_vars=extra_vars,
        )

    async def enrich_figure(
        self,
        image_path: Path,
        context_text: str = "",
        doc_id: str | None = None,
        run_id: str | None = None,
        page_idx: int | None = None,
        bbox: list[int] | None = None,
        page_thumbnail_path: Path | None = None,
    ) -> EnrichmentOutput:
        """
        Generate semantic caption for a figure/chart/diagram.

        Args:
            image_path: Path to figure crop image
            context_text: Surrounding text context
            page_thumbnail_path: Optional page thumbnail for visual context

        Returns semantic_caption, facts[], keywords[], and needs_review flag.
        """
        return await self._enrich(
            image_path=image_path,
            kind="figure_caption",
            context=context_text,
            doc_id=doc_id,
            run_id=run_id,
            page_idx=page_idx,
            bbox=bbox,
            page_thumbnail_path=page_thumbnail_path,
        )

    async def enrich_table(
        self,
        image_path: Path | None,
        context_text: str = "",
        doc_id: str | None = None,
        run_id: str | None = None,
        page_idx: int | None = None,
        bbox: list[int] | None = None,
        table_body: str | None = None,
        table_headers: list[str] | None = None,
        page_thumbnail_path: Path | None = None,
    ) -> EnrichmentOutput:
        """
        Generate summary for a table image.

        Args:
            image_path: Path to table image, or None when using MinerU table text only
            context_text: Surrounding text context
            doc_id: Document ID
            run_id: Run ID
            page_idx: Page index
            bbox: Bounding box (MinerU 0-1000 normalized coordinates)
            table_body: MinerU-extracted table HTML/markdown (reduces hallucination)
            table_headers: Extracted column headers
            page_thumbnail_path: Optional page thumbnail for visual context

        Returns table_summary, key_columns[], key_rows[], and needs_review flag.
        """
        # Build enhanced context with table data from MinerU
        enhanced_context = context_text or ""

        if table_headers:
            headers_text = ", ".join(table_headers)
            enhanced_context += f"\n\n[Extracted column headers from MinerU: {headers_text}]"

        if table_body:
            # Extract first few rows as reference
            table_preview = self._extract_table_preview(table_body, max_chars=500)
            enhanced_context += (
                "\n\n[Table structure from MinerU (use this as ground truth):\n"
                f"{table_preview}]"
            )

        return await self._enrich(
            image_path=image_path,
            kind="table_summary",
            context=enhanced_context,
            doc_id=doc_id,
            run_id=run_id,
            page_idx=page_idx,
            bbox=bbox,
            page_thumbnail_path=page_thumbnail_path,
        )

    async def extract_structured_table_records(
        self,
        image_path: Path,
        document_plan: dict[str, Any],
        context_text: str = "",
        doc_id: str | None = None,
        run_id: str | None = None,
        page_idx: int | None = None,
    ) -> EnrichmentOutput:
        """Extract schema-guided row records from a full-page table image."""

        return await self._enrich(
            image_path=image_path,
            kind="structured_table_records",
            context=context_text,
            doc_id=doc_id,
            run_id=run_id,
            page_idx=page_idx,
            bbox=None,
            extra_vars={
                "document_plan_json": json.dumps(
                    document_plan,
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        )

    async def enrich(
        self,
        kind: str,
        image_path: Path | None,
        context: str = "",
        doc_id: str | None = None,
        run_id: str | None = None,
        page_idx: int | None = None,
        bbox: list[int] | None = None,
        page_thumbnail_path: Path | None = None,
        extra_vars: dict[str, Any] | None = None,
    ) -> EnrichmentOutput:
        """
        Generic enrichment method with support for extra template variables.

        D3: Used for org_chart_edges which needs {candidates_json} in prompt.

        Args:
            kind: Enrichment type (form_asset, org_chart_edges, etc.)
            image_path: Path to image
            context: Text context
            extra_vars: Extra variables for prompt formatting (e.g., candidates_json)

        Returns:
            EnrichmentOutput with parsed response
        """
        return await self._enrich(
            image_path=image_path,
            kind=kind,
            context=context,
            doc_id=doc_id,
            run_id=run_id,
            page_idx=page_idx,
            bbox=bbox,
            page_thumbnail_path=page_thumbnail_path,
            extra_vars=extra_vars,
        )

    def _extract_table_preview(self, table_body: str, max_chars: int = 500) -> str:
        """Extract preview from table body for context."""
        # Simple extraction: take first N chars, try to end at row boundary
        if len(table_body) <= max_chars:
            return table_body

        preview = table_body[:max_chars]

        # Try to end at a row boundary
        last_tr = preview.rfind("</tr>")
        last_row = preview.rfind("\n|")
        end_pos = max(last_tr + 5 if last_tr > 0 else 0, last_row if last_row > 0 else 0)

        if end_pos > max_chars // 2:
            return preview[:end_pos] + "..."

        return preview + "..."

    def _max_tokens_for_kind(self, kind: str) -> int:
        task_caps = {
            "table_summary": 512,
            "figure_caption": 2048,
            "figure_description": 2048,
            "form_asset": 8192,
            "form_guide": 8192,
            "org_chart_edges": 2048,
            "quality_audit": 2048,
            "semantic_repair": 12288,
            "structured_table_records": 4096,
        }
        configured = self.config.decode_params.max_tokens
        return min(configured, task_caps.get(kind, 2048))


    async def _enrich(
        self,
        image_path: Path,
        kind: str,
        context: str = "",
        doc_id: str | None = None,
        run_id: str | None = None,
        page_idx: int | None = None,
        bbox: list[int] | None = None,
        page_thumbnail_path: Path | None = None,
        extra_vars: dict[str, Any] | None = None,
    ) -> EnrichmentOutput:
        """
        Internal enrichment method.

        Supports multi-image input per OpenAI API spec:
        - Image 1: crop (main subject)
        - Image 2: page thumbnail (context, optional)

        Args:
            extra_vars: Extra variables for prompt formatting (D3: candidates_json)
        """
        import time
        start_time = time.time()

        # Resolve aliased prompts
        prompt_template = PROMPTS.get(kind)
        if not prompt_template:
            return EnrichmentOutput(
                success=False,
                kind=kind,
                error=f"Unknown enrichment kind: {kind}",
            )

        # Handle aliased prompts (e.g., form_guide -> form_asset)
        if "alias" in prompt_template:
            actual_kind = prompt_template["alias"]
            prompt_template = PROMPTS.get(actual_kind)
            if not prompt_template:
                return EnrichmentOutput(
                    success=False,
                    kind=kind,
                    error=f"Aliased prompt not found: {actual_kind}",
                )

        try:
            # Build messages with multi-image support
            # D3: Support extra_vars for template formatting
            semantic_language = (extra_vars or {}).get("semantic_output_language", "auto")
            format_vars = {
                "context": context or "None provided",
                "semantic_output_language": semantic_language,
                "semantic_output_language_instruction": prompt_language_instruction(semantic_language),
                "semantic_template_sections": prompt_form_sections(semantic_language),
                "quality_issues_json": "[]",
                "source_evidence": context or "None provided",
                "current_markdown": "",
            }
            if extra_vars:
                format_vars.update(extra_vars)
            user_prompt = prompt_template["user"].format(**format_vars)

            # Build content array with images
            content: list[dict[str, Any]] = []

            # Image 1: Main crop (primary subject)
            if image_path is not None:
                image_url = self._build_image_url(image_path, doc_id, run_id)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"},
                })

            # Image 2: Page thumbnail (context) - optional
            if page_thumbnail_path and page_thumbnail_path.exists():
                thumbnail_url = self._build_image_url(page_thumbnail_path, doc_id, run_id)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": thumbnail_url, "detail": "low"},
                })
                # Enhance prompt to explain the two images
                user_prompt = (
                    "[Image 1 is the main crop to analyze. "
                    "Image 2 is the full page for context.]\n\n" + user_prompt
                )

            # Add text prompt
            content.append({"type": "text", "text": user_prompt})

            messages = [
                {"role": "system", "content": prompt_template["system"]},
                {"role": "user", "content": content},
            ]

            # Bound output length per task. Settings may allow very large contexts,
            # but short structured enrichments should not generate 128k-token replies.
            api_kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.decode_params.temperature,
                "top_p": self.config.decode_params.top_p,
                "max_tokens": self._max_tokens_for_kind(kind),
            }

            # Add JSON schema constraint if enabled and schema exists
            schema_class = OUTPUT_SCHEMAS.get(kind)
            if self.config.use_json_schema and schema_class:
                try:
                    json_schema = schema_class.model_json_schema()
                    # Ollama/vLLM structured outputs format
                    api_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": kind,
                            "strict": True,
                            "schema": json_schema,
                        },
                    }
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to generate JSON schema for {kind}: {e}")
                    # Fall back to no constraint

            # Call VLM
            response = await self.client.chat.completions.create(**api_kwargs)

            duration = time.time() - start_time
            raw_response = self._message_text(response.choices[0].message)
            tokens_used = response.usage.total_tokens if response.usage else 0

            # Parse response
            output = self._parse_response(raw_response, kind)

            # Extract needs_review from output
            needs_review = output.pop("needs_review", False) if isinstance(output, dict) else False

            # Build evidence
            evidence = VLMEvidence(
                page_idx=page_idx,
                bbox=bbox,
                asset_path=str(image_path) if image_path else None,
            )

            return EnrichmentOutput(
                success=True,
                kind=kind,
                output=output,
                raw_response=raw_response,
                tokens_used=tokens_used,
                duration_seconds=duration,
                needs_review=needs_review,
                evidence=evidence,
            )

        except Exception as e:
            import logging
            import traceback
            logging.error(f"VLM call failed for {kind}: {type(e).__name__}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return EnrichmentOutput(
                success=False,
                kind=kind,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _build_image_url(
        self,
        image_path: Path,
        doc_id: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """Build image URL based on configured mode."""
        if self.config.image_mode == VLMImageMode.STATIC_URL:
            # Use static URL from backend
            if doc_id and run_id:
                # Construct URL: /api/assets/{doc_id}/{run_id}/{relative_path}
                parts = image_path.parts
                if "assets" in parts:
                    relative_path = "/".join(parts[parts.index("assets") + 1:])
                else:
                    relative_path = image_path.name
                return f"{self.config.static_url_base}/{doc_id}/{run_id}/assets/{relative_path}"
            else:
                # Fallback to data URI if no context
                return self._encode_image_as_data_uri(image_path)
        else:
            # Data URI mode
            return self._encode_image_as_data_uri(image_path)

    def _encode_image_as_data_uri(self, image_path: Path) -> str:
        """Encode image as data URI."""
        image_data = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)
        return f"data:{mime_type};base64,{image_data}"

    def _get_mime_type(self, image_path: Path) -> str:
        """Get MIME type for image."""
        ext = image_path.suffix.lower().lstrip(".")
        mime_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        return mime_types.get(ext, "image/png")

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _message_text(message: Any) -> str:
        """Return usable text from OpenAI-compatible message objects."""

        def stringify(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, dict):
                        parts.append(stringify(item.get("text") or item.get("content")))
                    else:
                        parts.append(stringify(item))
                return "\n".join(part for part in parts if part).strip()
            return str(value).strip()

        content = stringify(getattr(message, "content", None))
        if content:
            return content

        candidates: list[Any] = []
        for attr in ("reasoning_content", "reasoning", "thinking"):
            candidates.append(getattr(message, attr, None))
        model_extra = getattr(message, "model_extra", None) or {}
        if isinstance(model_extra, dict):
            for key in ("reasoning_content", "reasoning", "thinking", "content"):
                candidates.append(model_extra.get(key))
        additional_kwargs = getattr(message, "additional_kwargs", None) or {}
        if isinstance(additional_kwargs, dict):
            for key in ("reasoning_content", "reasoning", "thinking", "content"):
                candidates.append(additional_kwargs.get(key))

        for candidate in candidates:
            text = stringify(candidate)
            if text:
                return text
        return ""

    @staticmethod
    def _strip_thinking_sections(raw: str) -> str:
        text = str(raw or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r"^</think>\s*", "", text, flags=re.IGNORECASE).strip()
        return text

    @classmethod
    def _extract_json_payload(cls, raw: str) -> str:
        text = cls._strip_thinking_sections(raw)
        if not text:
            return ""

        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            fenced = fence_match.group(1).strip()
            if fenced.startswith(("{", "[")) or re.search(r"[\{\[]", fenced):
                text = fenced

        balanced = cls._extract_balanced_json(text)
        return balanced or text.strip()

    @staticmethod
    def _extract_balanced_json(text: str) -> str:
        start = -1
        for idx, char in enumerate(text):
            if char in "[{":
                start = idx
                break
        if start < 0:
            return ""

        stack: list[str] = []
        in_string = False
        escape = False
        pairs = {"{": "}", "[": "]"}
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char in pairs:
                stack.append(pairs[char])
                continue
            if char in "}]":
                if not stack or char != stack[-1]:
                    return ""
                stack.pop()
                if not stack:
                    return text[start: idx + 1].strip()
        return ""

    def _parse_response(self, raw: str, kind: str) -> dict[str, Any]:
        """
        Parse VLM response into structured output with Pydantic validation.

        Steps:
        1. Extract JSON from response (handle markdown code blocks)
        2. Auto-repair common JSON issues (trailing commas, etc.)
        3. Validate with Pydantic schema
        4. Generate retrieval_text if missing
        5. Fallback to raw text with needs_review=True if all fails
        """
        # Try to extract JSON from response. Local thinking models may wrap
        # final output in reasoning text, fenced blocks, or prose before JSON.
        raw = raw.strip()
        json_candidate = self._extract_json_payload(raw)

        # Auto-repair common JSON issues
        cleaned_raw = self._repair_json(json_candidate)

        # Try to parse JSON
        parsed = None
        json_error = None
        try:
            parsed = json.loads(cleaned_raw)
        except json.JSONDecodeError as e:
            json_error = str(e)

        # Validate with Pydantic schema
        schema_class = OUTPUT_SCHEMAS.get(kind)
        if parsed is not None and schema_class:
            try:
                # Validate and get normalized output
                validated = schema_class.model_validate(parsed)
                result = validated.model_dump()

                # Form-specific quality checks and enhancements
                if kind in ("form_asset", "form_guide"):
                    result = self._apply_form_quality_rules(result)

                return result

            except ValidationError as e:
                # Pydantic validation failed - mark for review but preserve what we can
                if isinstance(parsed, dict):
                    parsed["needs_review"] = True
                    parsed["_validation_error"] = str(e)

                    # Apply form quality rules even for failed validation
                    if kind in ("form_asset", "form_guide"):
                        parsed = self._apply_form_quality_rules(parsed)

                    return parsed

        # JSON parse succeeded but no schema validation (or parsed is still valid dict)
        if parsed is not None and isinstance(parsed, dict):
            # Apply form quality rules (includes retrieval_text generation)
            if kind in ("form_asset", "form_guide"):
                parsed = self._apply_form_quality_rules(parsed)
            return parsed

        # Fallback: Return as raw text with needs_review=True
        if kind in ("form_asset", "form_guide"):
            return self._salvage_form_jsonish_response(raw, json_error)
        elif kind in ("figure_caption", "figure_description"):
            return self._salvage_figure_jsonish_response(raw, json_error)
        elif kind == "table_summary":
            return {
                "table_summary": raw,
                "key_columns": [],
                "key_rows": [],
                "needs_review": True,
                "_error": f"JSON_PARSE_FAILED: {json_error}" if json_error else "UNKNOWN_ERROR",
            }
        elif kind == "semantic_repair":
            salvaged_markdown = self._salvage_semantic_repair_markdown(raw)
            if salvaged_markdown:
                return {
                    "status": "repaired",
                    "repaired_markdown": salvaged_markdown,
                    "summary": "Reviewer returned standalone Markdown instead of JSON; Markdown was salvaged and marked for review.",
                    "applied_repairs": ["salvage_markdown_response"],
                    "confidence": 0.55,
                    "needs_review": True,
                    "raw_response_preview": raw[:4000],
                    "_error": f"JSON_PARSE_FAILED: {json_error}" if json_error else "UNKNOWN_ERROR",
                    "_salvaged": True,
                }
            return {
                "status": "uncertain",
                "repaired_markdown": "",
                "summary": "Reviewer response was not valid JSON; repair was rejected to preserve the previous semantic output.",
                "applied_repairs": [],
                "confidence": 0.0,
                "needs_review": True,
                "raw_response_preview": raw[:4000],
                "_error": f"JSON_PARSE_FAILED: {json_error}" if json_error else "UNKNOWN_ERROR",
            }
        else:
            return {
                "description": raw,
                "needs_review": True,
                "_error": f"JSON_PARSE_FAILED: {json_error}" if json_error else "UNKNOWN_ERROR",
            }


    def _salvage_semantic_repair_markdown(self, raw: str) -> str:
        """Accept reviewer Markdown only when it is clearly a complete standalone document."""

        text = self._strip_thinking_sections(raw)
        if not text or text.startswith(("{", "[")):
            return ""
        fence = re.search(r"```(?:markdown|md)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        if not re.match(r"^#\s+", text):
            heading = re.search(r"(?m)^#\s+", text)
            if not heading:
                return ""
            text = text[heading.start():].strip()
        if len(text) < 180:
            return ""
        if not re.search(r"^#{2,3}\s+", text, re.MULTILINE):
            return ""
        forbidden_markers = [
            "QUALITY ISSUES JSON",
            "SOURCE EVIDENCE",
            "CURRENT SEMANTIC MARKDOWN",
            "Return JSON",
            "repaired_markdown",
            "raw_response_preview",
        ]
        if any(marker in text for marker in forbidden_markers):
            return ""
        return text.rstrip() + "\n"


    def _salvage_figure_jsonish_response(self, raw: str, json_error: str | None = None) -> dict[str, Any]:
        """Recover useful figure/flowchart fields from malformed local VLM JSON."""

        payload = self._extract_json_payload(raw) or self._strip_thinking_sections(raw)
        semantic_caption = self._extract_jsonish_string(payload, "semantic_caption")
        image_type = self._extract_jsonish_string(payload, "image_type") or "other"
        structured_content = self._extract_jsonish_string(payload, "structured_content")
        structured_lines = self._extract_jsonish_string_list(payload, "structured_content")
        all_text = self._extract_jsonish_string_list(payload, "all_text")
        facts = self._extract_jsonish_string_list(payload, "facts")
        keywords = self._extract_jsonish_string_list(payload, "keywords")

        if structured_lines and not structured_content:
            structured_content = "\n".join(structured_lines)

        recovered = any([semantic_caption, structured_content, all_text, facts, keywords, image_type != "other"])
        if not semantic_caption and not recovered:
            semantic_caption = self._strip_thinking_sections(raw)

        return {
            "semantic_caption": semantic_caption,
            "facts": facts,
            "keywords": keywords,
            "image_type": image_type,
            "structured_content": structured_content,
            "all_text": all_text,
            "needs_review": True,
            "_salvaged": recovered,
            "_error": f"JSON_PARSE_FAILED: {json_error}" if json_error else "UNKNOWN_ERROR",
        }


    def _salvage_form_jsonish_response(self, raw: str, json_error: str | None = None) -> dict[str, Any]:
        """Recover useful form output when a local VLM returns truncated JSON."""

        title = self._extract_jsonish_string(raw, "title")
        document_type = self._extract_jsonish_string(raw, "document_type") or "form"
        date = self._extract_jsonish_string(raw, "date")
        filling_guide = self._extract_jsonish_string(raw, "filling_guide")
        structured_content = self._extract_jsonish_string(raw, "structured_content")
        retrieval_text = self._extract_jsonish_string(raw, "retrieval_text")
        triggers = self._extract_jsonish_string_list(raw, "triggers")
        all_text = self._extract_jsonish_string_list(raw, "all_text")
        field_schema = self._extract_jsonish_object_list(raw, "field_schema")

        if not field_schema and document_type == "form":
            field_schema = self._derive_form_fields_from_text(all_text, filling_guide)

        if not retrieval_text:
            field_names = [str(f.get("name", "")) for f in field_schema if isinstance(f, dict)]
            retrieval_text = " ".join(
                part for part in [title, " ".join(triggers), " ".join(field_names), filling_guide[:240]] if part
            )

        result = {
            "title": title,
            "document_type": document_type,
            "date": date,
            "triggers": triggers,
            "all_text": all_text,
            "field_schema": field_schema if document_type == "form" else [],
            "filling_guide": filling_guide,
            "structured_content": structured_content,
            "retrieval_text": retrieval_text or raw[:500],
            "needs_review": True,
            "_error": f"JSON_PARSE_FAILED: {json_error}" if json_error else "JSON_PARSE_FAILED",
            "_salvaged": True,
        }
        return self._apply_form_quality_rules(result)

    def _extract_jsonish_string(self, raw: str, key: str) -> str:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)', raw, re.DOTALL)
        if not match:
            return ""
        value = match.group(1)
        try:
            return json.loads(f'"{value}"')
        except json.JSONDecodeError:
            return value.replace('\\n', '\n').replace('\\"', '"').strip()

    def _extract_jsonish_string_list(self, raw: str, key: str) -> list[str]:
        body = self._extract_jsonish_array_body(raw, key)
        if not body:
            return []
        items: list[str] = []
        for match in re.finditer(r'"((?:\\.|[^"\\])*)"', body, re.DOTALL):
            value = match.group(1)
            try:
                decoded = json.loads(f'"{value}"')
            except json.JSONDecodeError:
                decoded = value.replace('\\n', '\n').replace('\\"', '"')
            text = re.sub(r"\s+", " ", str(decoded)).strip()
            if text and text not in items:
                items.append(text)
        return items[:120]

    def _extract_jsonish_object_list(self, raw: str, key: str) -> list[dict[str, Any]]:
        body = self._extract_jsonish_array_body(raw, key)
        if not body:
            return []
        objects: list[dict[str, Any]] = []
        for chunk in re.findall(r'\{[^{}]*\}', body, re.DOTALL):
            try:
                parsed = json.loads(self._repair_json(chunk))
            except json.JSONDecodeError:
                name = self._extract_jsonish_string(chunk, "name")
                if not name:
                    continue
                parsed = {
                    "name": name,
                    "type": self._extract_jsonish_string(chunk, "type") or "text",
                    "required": "true" in chunk.lower(),
                    "aliases": [],
                    "evidence_text": self._extract_jsonish_string(chunk, "evidence_text") or name,
                }
            if isinstance(parsed, dict) and parsed.get("name"):
                objects.append(parsed)
        return fields_to_dicts(normalize_fields(objects))

    def _extract_jsonish_array_body(self, raw: str, key: str) -> str:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', raw)
        if not match:
            return ""
        start = match.end()
        depth = 1
        in_string = False
        escaped = False
        for idx in range(start, len(raw)):
            char = raw[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return raw[start:idx]
        return raw[start:]

    def _derive_form_fields_from_text(self, all_text: list[str], filling_guide: str = "") -> list[dict[str, Any]]:
        candidates: list[str] = []
        field_hint = re.compile(
            r"姓名|申請|日期|單位|職級|職稱|身分證|身份證|電話|手機|E-?mail|地址|出差|事由|地點|期間|"
            r"金額|費用|預借|報支|付款|戶名|銀行|帳號|主管|主任|秘書|副院長|院長|董事長|簽|章|核|備註|附件|保險"
        )
        for source in list(all_text or []) + re.split(r"[\n。；;]", filling_guide or ""):
            text = re.sub(r"\s+", " ", str(source or "")).strip()
            if not text:
                continue
            parts = re.split(r"[|｜]|\s{2,}|(?<=：)\s*", text)
            for part in parts:
                part = part.strip(" ：:_-，,。")
                if not part or len(part) > 38:
                    continue
                if "□" in part or "___" in part or field_hint.search(part):
                    candidates.append(part)
        return fields_to_dicts(normalize_fields(candidates))

    def _apply_form_quality_rules(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Apply quality rules for form_asset output.

        Rules:
        1. Generate retrieval_text if missing (title + triggers + field names + guide)
        2. Flag needs_review if field_schema or filling_guide is empty
        3. Add review reasons for debugging
        """
        review_reasons: list[str] = []

        # Check field_schema completeness
        field_schema = result.get("field_schema", [])
        if not field_schema:
            review_reasons.append("field_schema is empty")
        elif len(field_schema) < 3:
            review_reasons.append(f"field_schema has only {len(field_schema)} fields (expected more)")

        # Check filling_guide completeness
        filling_guide = result.get("filling_guide", "")
        if not filling_guide:
            review_reasons.append("filling_guide is empty")
        elif len(filling_guide) < 50:
            review_reasons.append(f"filling_guide is too short ({len(filling_guide)} chars)")
        elif "適用場景" not in filling_guide and "填寫" not in filling_guide:
            review_reasons.append("filling_guide missing expected sections")

        # Check all_text (OCR quality)
        all_text = result.get("all_text", [])
        if not all_text:
            review_reasons.append("all_text is empty (OCR may have failed)")

        # Set needs_review if any issue found
        if review_reasons:
            result["needs_review"] = True
            existing_reasons = result.get("_quality_issues", [])
            result["_quality_issues"] = existing_reasons + review_reasons

        # Generate retrieval_text (enhanced for RAG)
        if not result.get("retrieval_text"):
            title = result.get("title", "")
            triggers = result.get("triggers", [])

            # Extract field names
            field_names = [f.get("name", "") for f in field_schema if isinstance(f, dict)]

            # Guide summary (first 200 chars)
            guide_summary = filling_guide[:200] if filling_guide else ""

            # Combine for retrieval
            retrieval_parts = [title] + triggers + field_names
            if guide_summary:
                retrieval_parts.append(guide_summary)

            result["retrieval_text"] = " ".join(filter(None, retrieval_parts))

        return result

    def _repair_json(self, raw: str) -> str:
        """
        Auto-repair common JSON issues from VLM outputs.

        Fixes:
        - Trailing commas before } or ]
        - Single quotes instead of double quotes
        - Unquoted string values (simple cases)
        """
        text = raw

        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([\}\]])', r'\1', text)

        # Try to fix single quotes (risky, only if json.loads fails)
        # This is a simple heuristic and may not work for all cases

        return text

    async def check_available(self) -> tuple[bool, str]:
        """Check if VLM endpoint is available."""
        probe = await self.probe_capabilities()
        if probe.available:
            if probe.model_found:
                return True, f"Model {self.config.model} available (vision: {probe.supports_vision})"
            else:
                return True, f"Connected, but model {self.config.model} not in list: {probe.models[:5]}"
        return False, probe.error or "Unknown error"

    async def probe_capabilities(self, force: bool = False) -> ProbeResult:
        """
        Probe VLM capabilities with caching.

        Checks:
        - API availability
        - Model availability
        - Vision support (for Ollama)
        """
        # Use cache if available and not forcing
        if self._probe_cache and not force:
            return self._probe_cache

        timestamp = datetime.now().isoformat()

        try:
            # Try to list models
            models = await self.client.models.list()
            model_ids = [m.id for m in models.data]

            model_found = self.config.model in model_ids

            # Check vision support (Ollama specific)
            supports_vision = await self._check_vision_support()

            result = ProbeResult(
                available=True,
                model_found=model_found,
                supports_vision=supports_vision,
                models=model_ids[:10],  # Limit to first 10
                timestamp=timestamp,
            )

        except Exception as e:
            result = ProbeResult(
                available=False,
                error=str(e),
                timestamp=timestamp,
            )

        # Cache result
        self._probe_cache = result
        return result

    async def _check_vision_support(self) -> bool:
        """Check if model supports vision (Ollama-specific)."""
        if self.config.api_mode != VLMApiMode.OLLAMA:
            # Assume vision support for other modes
            return True

        try:
            # Ollama: check model info for vision capability
            import httpx

            # Remove /v1 suffix for Ollama native API
            base = self.config.base_url.replace("/v1", "")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base}/api/show",
                    json={"name": self.config.model},
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    # Check for vision-related terms in model info
                    model_info = str(data).lower()
                    return any(term in model_info for term in ["vision", "vl", "image", "multimodal"])
        except Exception:
            pass

        # Default to True (let it fail at runtime if not supported)
        return True

    def get_prompt_version(self, kind: str) -> str:
        """Get prompt version for a kind."""
        return PROMPTS.get(kind, {}).get("version", "unknown")

    def get_probe_cache(self) -> ProbeResult | None:
        """Get cached probe result."""
        return self._probe_cache

    def clear_probe_cache(self) -> None:
        """Clear probe cache."""
        self._probe_cache = None
