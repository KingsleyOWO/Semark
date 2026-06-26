// API Types matching backend models

export type RunStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'canceled'
export type StageStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'canceled'
export type StageName = 'ingest' | 'parse' | 'normalize' | 'enrich' | 'package' | 'chunk'
export type ProfileName = 'fast' | 'accurate'
export type BlockType = 'text' | 'table' | 'image' | 'equation' | 'code' | 'list'

export interface Doc {
  doc_id: string
  source_path: string
  sha256?: string  // Optional in DocResponse
  ext: string
  size_bytes: number
  created_at: string
  run_count: number
  meta?: Record<string, unknown>
}

export interface RunStageProgress {
  phase?: string
  total?: number
  completed?: number
  current?: {
    kind?: string
    page_idx?: number | null
    block_id?: string | null
  } | null
  percent?: number
  message?: string
  updated_at?: string
}

export interface RunResponse {
  run_id: string
  doc_id: string
  profile: string
  status: RunStatus
  use_cache: boolean
  created_at: string
  updated_at: string
  current_stage?: StageName | null
  stage_progress?: RunStageProgress | null
}

export interface StageResponse {
  stage: StageName
  status: StageStatus
  started_at: string | null
  finished_at: string | null
  duration_seconds: number | null
  error: Record<string, unknown> | null
  stats: Record<string, unknown> | null
}

export interface RunDetailResponse extends RunResponse {
  config: Record<string, unknown>
  stages: StageResponse[]
}

export interface RunListResponse {
  runs: RunResponse[]
  total: number
}

export interface DocListResponse {
  docs: Doc[]
  total: number
}

// DocumentIR types
export interface SourceInfo {
  path: string
  ext: string
  sha256: string
  size_bytes: number
}

export interface EngineInfo {
  name: string
  backend: string
  version?: string
  method: string
  lang?: string
  table: boolean
  formula: boolean
}

export interface PageInfo {
  page_idx: number
  width_px?: number
  height_px?: number
  page_image_path?: string
}

export interface Block {
  block_id: string
  type: BlockType
  page_idx: number
  bbox_norm: number[] // [x0, y0, x1, y1] 0-1000
  reading_order: number
  payload: Record<string, unknown>
  enrichment_ref?: string
}

export interface DocumentIR {
  doc_id: string
  run_id: string
  source: SourceInfo
  engine: EngineInfo
  pages: PageInfo[]
  blocks: Block[]
}

// Source map types
export interface MdAnchor {
  anchor_id: string
  md_range: [number, number]
  block_ids: string[]
}

export interface SourceMap {
  md_anchors: MdAnchor[]
}

// Split document exports
export interface SplitDocumentMeta {
  document_id: string
  kind: string
  title?: string
  page_indices?: number[]
  page_label?: string
  page_image_path?: string | null
  asset_path?: string | null
  logical_doc_id?: string
  parent_doc_id?: string
  filename?: string
}

export interface SplitDocumentsResponse {
  run_id: string
  doc_id: string
  documents: SplitDocumentMeta[]
  total: number
}

export interface SplitDocumentResponse {
  run_id: string
  doc_id: string
  document: SplitDocumentMeta
  content: string
}

// Asset types
export type AssetType = 'form_asset' | 'figure_asset' | 'table_asset'

export interface FieldSchemaEntry {
  field_name: string
  field_type?: string
  required?: boolean
  description?: string
}

export interface AssetEntry {
  type: AssetType
  asset_id: string
  doc_id: string
  run_id: string
  title: string
  triggers: string[]
  page_idx: number
  asset_path: string
  block_id: string
  guide_ref?: string
  retrieval_text: string
  // Form-specific fields
  filling_guide?: string
  field_schema?: FieldSchemaEntry[]
  // Figure-specific fields
  semantic_caption?: string
  facts?: string[]
  keywords?: string[]
  // Quality flag
  needs_review: boolean
}

// Enrichment types (from enrich stage)
export interface EnrichmentEvidence {
  page_idx: number
  bbox: number[] | null  // [x0, y0, x1, y1] 0-1000 normalized, null for full page
  asset_path: string | null
}

export interface EnrichmentQuality {
  needs_review: boolean
  tokens_used?: number
  duration_seconds?: number
}

export interface EnrichmentEntry {
  block_id: string
  kind: 'form_asset' | 'figure_caption' | 'table_summary' | 'form_guide' | 'figure_description'
  prompt_version: string
  model: string
  decode: Record<string, unknown>
  input: Record<string, unknown>
  output: Record<string, unknown>
  quality: EnrichmentQuality
  evidence: EnrichmentEvidence
}

// Quality report (matches backend QualityResponse)
export interface PageQualityInfo {
  page_idx: number
  block_count: number
  text_length: number
  has_table: boolean
  has_image: boolean
}

export interface QualityReport {
  doc_id: string
  run_id: string
  blocks: Record<string, number>  // type -> count
  pages: PageQualityInfo[]
  enrich_coverage: Record<string, unknown>
  warnings: string[]
}


export interface QualityGateIssue {
  code: string
  severity: 'high' | 'medium' | 'warning' | 'low' | string
  message: string
  page_idx?: number | null
  block_id?: string | null
  document_id?: string | null
  evidence?: Record<string, unknown>
}

export interface QualityGateReport {
  status: 'pass' | 'warning' | 'needs_review' | 'unknown' | string
  score: number
  issues: QualityGateIssue[]
  vlm_audit_candidates: Array<Record<string, unknown>>
  vlm_audits: Array<Record<string, unknown>>
  stats: Record<string, unknown>
}

// Request types
export interface IngestRequest {
  path?: string
  url?: string
}

// Response types
export interface IngestResponse {
  doc_id: string
  source_path: string
  ext: string
  size_bytes: number
  already_exists: boolean
}

export interface EnrichmentsResponse {
  enrichments: EnrichmentEntry[]
  total: number
}

export interface RunCreateRequest {
  doc_id: string
  profile: ProfileName
  config_overrides?: Record<string, unknown>
  use_cache?: boolean
  force_stages?: StageName[]
}
