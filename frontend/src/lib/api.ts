import type {
  Doc,
  DocListResponse,
  RunResponse,
  RunDetailResponse,
  RunListResponse,
  RunCreateRequest,
  IngestRequest,
  IngestResponse,
  DocumentIR,
  SourceMap,
  AssetEntry,
  QualityReport,
  QualityGateReport,
  EnrichmentEntry,
  EnrichmentsResponse,
  SplitDocumentResponse,
  SplitDocumentsResponse,
  SplitDocumentMeta,
  RunStatus,
  StageName,
} from '@/types/api'

export const SUPPORTED_UPLOAD_EXTENSIONS = ['pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls', 'odt', 'odp', 'ods', 'html', 'htm', 'png', 'jpg', 'jpeg'] as const
export const SUPPORTED_UPLOAD_ACCEPT = SUPPORTED_UPLOAD_EXTENSIONS.map((ext) => `.${ext}`).join(',')

const API_BASE = '/api'

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(error.detail || `HTTP ${res.status}`)
  }

  return res.json()
}

// Documents API
export async function listDocs(limit = 100, offset = 0): Promise<DocListResponse> {
  return fetchJson(`${API_BASE}/docs?limit=${limit}&offset=${offset}`)
}

export async function getDoc(docId: string): Promise<Doc> {
  return fetchJson(`${API_BASE}/docs/${docId}`)
}

export async function ingestPath(path: string): Promise<IngestResponse> {
  return fetchJson(`${API_BASE}/ingest`, {
    method: 'POST',
    body: JSON.stringify({ path } as IngestRequest),
  })
}

export async function uploadFile(file: File): Promise<IngestResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(`${API_BASE}/ingest/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(error.detail || `HTTP ${res.status}`)
  }

  return res.json()
}

export interface BatchUploadProgress {
  total: number
  completed: number
  failed: number
  inProgress: number
  results: Array<{ file: string; success: boolean; docId?: string; error?: string }>
}

export interface BatchUploadOptions {
  concurrency?: number // 同時上傳數量，預設 3
  onProgress?: (progress: BatchUploadProgress) => void
}

/**
 * 批量上傳多個文件，支持並行節流和進度回調
 */
export async function uploadMultipleFiles(
  files: File[],
  options: BatchUploadOptions = {}
): Promise<BatchUploadProgress> {
  const { concurrency = 3, onProgress } = options
  const total = files.length
  const results: BatchUploadProgress['results'] = []
  let completed = 0
  let failed = 0
  let inProgress = 0

  const emitProgress = () => {
    const progress: BatchUploadProgress = {
      total,
      completed,
      failed,
      inProgress,
      results: [...results],
    }
    onProgress?.(progress)
    return progress
  }

  // 建立工作隊列
  const queue = [...files]
  const workers: Promise<void>[] = []

  const processNext = async (): Promise<void> => {
    while (queue.length > 0) {
      const file = queue.shift()!
      inProgress++
      emitProgress()

      try {
        const response = await uploadFile(file)
        results.push({
          file: file.name,
          success: true,
          docId: response.doc_id,
        })
        completed++
      } catch (err) {
        results.push({
          file: file.name,
          success: false,
          error: err instanceof Error ? err.message : String(err),
        })
        completed++
        failed++
      }
      inProgress--
      emitProgress()
    }
  }

  // 啟動並行 workers
  for (let i = 0; i < Math.min(concurrency, files.length); i++) {
    workers.push(processNext())
  }

  await Promise.all(workers)
  return emitProgress()
}

export async function ingestFolder(path: string): Promise<{ doc_ids: string[] }> {
  return fetchJson(`${API_BASE}/ingest/folder`, {
    method: 'POST',
    body: JSON.stringify({ path }),
  })
}

// Runs API
export async function listRuns(
  status?: RunStatus,
  limit = 100,
  offset = 0,
  includeHidden = false
): Promise<RunListResponse> {
  const params = new URLSearchParams()
  if (status) params.set('status', status)
  params.set('limit', String(limit))
  params.set('offset', String(offset))
  if (includeHidden) params.set('include_hidden', 'true')
  return fetchJson(`${API_BASE}/runs?${params}`)
}

export async function getRun(runId: string): Promise<RunDetailResponse> {
  return fetchJson(`${API_BASE}/runs/${runId}`)
}

export interface RunsStats {
  total: number
  pending: number
  running: number
  succeeded: number
  failed: number
  canceled: number
}

export async function getRunsStats(): Promise<RunsStats> {
  return fetchJson(`${API_BASE}/runs/stats`)
}

export async function createRun(request: RunCreateRequest): Promise<RunResponse> {
  return fetchJson(`${API_BASE}/runs`, {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

export async function executeRun(
  runId: string,
  background = true
): Promise<{ message: string; run_id: string }> {
  return fetchJson(`${API_BASE}/runs/${runId}/execute?background=${background}`, {
    method: 'POST',
  })
}

export async function cancelRun(runId: string): Promise<{ message: string }> {
  return fetchJson(`${API_BASE}/runs/${runId}/cancel`, {
    method: 'POST',
  })
}

export async function deleteRun(runId: string): Promise<{ message: string }> {
  return fetchJson(`${API_BASE}/runs/${runId}`, {
    method: 'DELETE',
  })
}

// Batch APIs
export async function batchDeleteDocs(
  docIds: string[]
): Promise<{ message: string; deleted: string[]; errors: Array<{ doc_id: string; error: string }> }> {
  return fetchJson(`${API_BASE}/docs/batch-delete`, {
    method: 'POST',
    body: JSON.stringify(docIds),
  })
}

export async function batchDeleteRuns(
  runIds: string[]
): Promise<{ message: string; deleted: string[]; errors: Array<{ run_id: string; error: string }> }> {
  return fetchJson(`${API_BASE}/runs/batch-delete`, {
    method: 'POST',
    body: JSON.stringify(runIds),
  })
}

export async function batchCancelRuns(
  runIds: string[]
): Promise<{
  message: string
  canceled: string[]
  skipped: Array<{ run_id: string; status: string }>
  errors: Array<{ run_id: string; error: string }>
}> {
  return fetchJson(`${API_BASE}/runs/batch-cancel`, {
    method: 'POST',
    body: JSON.stringify(runIds),
  })
}

export async function batchCreateRuns(
  docIds: string[],
  profile: 'fast' | 'accurate' = 'accurate',
  options?: { use_cache?: boolean }
): Promise<{
  message: string
  profile: string
  use_cache: boolean
  created: Array<{ doc_id: string; run_id: string }>
  errors: Array<{ doc_id: string; error: string }>
}> {
  const params = new URLSearchParams({ profile, use_cache: String(options?.use_cache ?? false) })
  return fetchJson(`${API_BASE}/runs/batch-create?${params}`, {
    method: 'POST',
    body: JSON.stringify(docIds),
  })
}

export async function deleteDoc(docId: string): Promise<{ message: string }> {
  return fetchJson(`${API_BASE}/docs/${docId}`, {
    method: 'DELETE',
  })
}

export async function getTaskStatus(runId: string): Promise<{
  run_id: string
  in_queue: boolean
  status?: string
  current_stage?: string
}> {
  return fetchJson(`${API_BASE}/runs/${runId}/task_status`)
}


export interface OutputRunSummary {
  run_id: string
  doc_id: string
  profile: string
  status: RunStatus
  created_at: string
  updated_at: string
  source_path: string
  source_name: string
  documents_total: number
  main_document_count: number
  extracted_document_count: number
  documents: SplitDocumentMeta[]
  quality_gate_status: string
  quality_score: number | null
  quality_issue_count: number
}

export interface OutputsSummaryResponse {
  runs: OutputRunSummary[]
  total: number
}

export async function getOutputsSummary(
  limit = 100,
  offset = 0,
  options?: { include_hidden?: boolean; has_documents_only?: boolean }
): Promise<OutputsSummaryResponse> {
  const params = new URLSearchParams()
  params.set('status', 'succeeded')
  params.set('limit', String(limit))
  params.set('offset', String(offset))
  params.set('include_hidden', String(options?.include_hidden ?? true))
  params.set('has_documents_only', String(options?.has_documents_only ?? true))
  return fetchJson(`${API_BASE}/runs/outputs-summary?${params}`)
}

// Run artifacts
export async function getDocumentIR(runId: string): Promise<DocumentIR> {
  return fetchJson(`${API_BASE}/runs/${runId}/document_ir`)
}

export async function getSourceMap(runId: string): Promise<SourceMap> {
  return fetchJson(`${API_BASE}/runs/${runId}/source_map`)
}

export async function getQuality(runId: string): Promise<QualityReport> {
  return fetchJson(`${API_BASE}/runs/${runId}/quality`)
}

export async function getQualityGate(runId: string): Promise<QualityGateReport> {
  return fetchJson(`${API_BASE}/runs/${runId}/quality_gate`)
}

export async function getAssetsIndex(runId: string): Promise<AssetEntry[]> {
  return fetchJson(`${API_BASE}/runs/${runId}/assets_index`)
}

export async function getEnrichments(
  runId: string,
  options?: {
    block_id?: string
    kind?: EnrichmentEntry['kind']
    needs_review?: boolean
  }
): Promise<EnrichmentsResponse> {
  const params = new URLSearchParams()
  if (options?.block_id) params.set('block_id', options.block_id)
  if (options?.kind) params.set('kind', options.kind)
  if (options?.needs_review !== undefined) params.set('needs_review', String(options.needs_review))
  const query = params.toString()
  return fetchJson(`${API_BASE}/runs/${runId}/enrichments${query ? `?${query}` : ''}`)
}

export type OutputView = 'source'

export async function getOutput(runId: string, view: OutputView): Promise<string> {
  const res = await fetch(`${API_BASE}/runs/${runId}/output?view=${view}`)
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`)
  }
  return res.text()
}

export async function getSplitDocuments(runId: string): Promise<SplitDocumentsResponse> {
  return fetchJson(`${API_BASE}/runs/${runId}/documents`)
}

export async function getSplitDocument(
  runId: string,
  documentId: string
): Promise<SplitDocumentResponse> {
  return fetchJson(`${API_BASE}/runs/${runId}/documents/${encodeURIComponent(documentId)}`)
}

// Assets API
export function getAssetUrl(docId: string, runId: string, assetPath: string): string {
  // Strip 'assets/' prefix if present (backend API already has /assets/ in path)
  const cleanPath = assetPath.replace(/^assets\//, '')
  return `${API_BASE}/assets/${docId}/${runId}/${cleanPath}`
}

// Profiles
export async function getProfiles(): Promise<string[]> {
  return fetchJson(`${API_BASE}/profiles`)
}

// Re-run with cache control
export async function rerunWithForce(
  docId: string,
  profile: string,
  options?: { use_cache?: boolean; force_stages?: StageName[] }
): Promise<RunResponse> {
  const run = await createRun({
    doc_id: docId,
    profile: profile as 'fast' | 'accurate',
    use_cache: options?.use_cache ?? false,
    force_stages: options?.force_stages,
  })
  await executeRun(run.run_id, true)
  return run
}

// Get manifest for a run
export async function getManifest(runId: string): Promise<{
  doc_id: string
  run_id: string
  config_hash: string
  created_at: string
  engines: Record<string, unknown>
  pipeline_config: Record<string, unknown>
}> {
  return fetchJson(`${API_BASE}/runs/${runId}/manifest`)
}

// Invalidate cache for specific stages
export async function invalidateCache(
  runId: string,
  stages: string[] = ['parse', 'enrich']
): Promise<{
  message: string
  doc_id: string
  run_id: string
  invalidated: Record<string, number>
}> {
  const params = new URLSearchParams()
  stages.forEach((stage) => params.append('stages', stage))
  return fetchJson(`${API_BASE}/runs/${runId}/invalidate?${params}`, {
    method: 'POST',
  })
}

// Get cache stats for a run
export async function getCacheStats(runId: string): Promise<{
  doc_id: string
  run_id: string
  cache: {
    parse: { entries: number; keys: string[] }
    enrich: { entries: number; blocks: string[] }
  }
}> {
  return fetchJson(`${API_BASE}/runs/${runId}/cache_stats`)
}

// ========== D5: Org Chart API ==========

export interface OrgChartNode {
  id: string
  label: string
  bbox?: [number, number, number, number]
  page_idx?: number
  category?: string
  category_hint?: string
  confidence?: number
  level?: number
  chosen_parent?: string | null
  parent_decision?: string
  source?: string
}

export interface OrgChartEdge {
  parent_id: string
  child_id: string
  relation?: string
  confidence?: number
}

export interface OrgChartGroup {
  name: string
  members: string[]
  confidence?: number
  description?: string
}

export interface OrgChartGraph {
  title?: string
  date?: string
  page_idx?: number
  nodes: OrgChartNode[]
  edges: OrgChartEdge[]
  groups?: OrgChartGroup[]
  derived_paths?: string[]
  needs_review?: boolean
  review_reasons?: string[]
}

export interface OrgChartResponse {
  found: boolean
  doc_id: string
  run_id: string
  page_idx: number
  graph: OrgChartGraph | null
  render_md: string
  warnings: string[]
  decision_trace: Record<string, unknown>
  message?: string
}

export interface OrgChartDebugFile {
  name: string
  size: number
  mtime: string
}

export interface OrgChartDebugIndex {
  doc_id: string
  run_id: string
  files: OrgChartDebugFile[]
}

/** D5: 取得組織圖處理結果 */
export async function getOrgChart(runId: string): Promise<OrgChartResponse> {
  return fetchJson(`${API_BASE}/runs/${runId}/org-chart`)
}

/** D5: 取得組織圖 debug 檔案索引 */
export async function getOrgChartDebugIndex(runId: string): Promise<OrgChartDebugIndex> {
  return fetchJson(`${API_BASE}/runs/${runId}/org-chart/debug/index`)
}

/** D5: 取得組織圖 debug 檔案內容 */
export async function getOrgChartDebugFile(runId: string, fileName: string): Promise<unknown> {
  return fetchJson(`${API_BASE}/runs/${runId}/org-chart/debug/file?name=${encodeURIComponent(fileName)}`)
}

/** D5: 取得組織圖 debug 檔案內容（文字格式） */
export async function getOrgChartDebugFileText(runId: string, fileName: string): Promise<string> {
  const res = await fetch(`${API_BASE}/runs/${runId}/org-chart/debug/file?name=${encodeURIComponent(fileName)}`)
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`)
  }
  return res.text()
}

// ========== Download API ==========

export type DownloadFileType = 'source' | 'documents' | 'quality' | 'assets_index' | 'enrichments'
export type DownloadOutputFormat = 'md' | 'docx' | 'txt' | 'json'

export interface DownloadRequest {
  run_ids: string[]
  file_types: DownloadFileType[]
  format: DownloadOutputFormat
  document_ids?: string[]
}

/**
 * Download multiple runs as a ZIP file.
 * Returns a Response object that can be used to trigger browser download.
 */
export async function downloadRuns(request: DownloadRequest): Promise<Response> {
  const res = await fetch(`${API_BASE}/runs/download`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!res.ok) {
    const errorText = await res.text()
    console.error('Download error response:', errorText)
    let errorDetail: string
    try {
      const errorJson = JSON.parse(errorText)
      errorDetail = errorJson.detail || JSON.stringify(errorJson)
    } catch {
      errorDetail = errorText || res.statusText
    }
    throw new Error(`HTTP ${res.status}: ${errorDetail}`)
  }

  return res
}


export async function deleteSplitDocuments(
  runId: string,
  documentIds: string[]
): Promise<{
  message: string
  run_id: string
  deleted: string[]
  errors: Array<{ document_id: string; error: string }>
  remaining: number
}> {
  return fetchJson(`${API_BASE}/runs/${runId}/documents`, {
    method: 'DELETE',
    body: JSON.stringify(documentIds),
  })
}

export function getSplitDocumentDownloadUrl(
  runId: string,
  documentId: string,
  format: Exclude<DownloadOutputFormat, 'json'> = 'md'
): string {
  return `${API_BASE}/runs/${runId}/documents/${encodeURIComponent(documentId)}/download?format=${format}`
}

// ========== Profile Settings API ==========

export interface ProfileDescription {
  name: string
  description: string
  features: string[]
}

export interface ProfileConfig {
  mineru: {
    method: string
    backend: string
    lang: string
    table: boolean
    formula: boolean
  }
  enrich: {
    enable_vlm: boolean
    vlm_enrich_forms: boolean
    vlm_enrich_figures: boolean
    vlm_enrich_tables: boolean
    table_vlm_budget: number
    table_min_cells: number
    table_max_cells: number
  }
  package: {
    generate_dataset_md: boolean
    generate_rag_md: boolean
    chunk_max_tokens: number
    chunk_overlap_tokens: number
    semantic_output_language: string
  }
}

export interface ProfileOverrides {
  method?: string
  formula?: boolean
  enable_vlm?: boolean
  vlm_enrich_forms?: boolean
  vlm_enrich_figures?: boolean
  vlm_enrich_tables?: boolean
  table_vlm_budget?: number
  table_min_cells?: number
  table_max_cells?: number
  chunk_max_tokens?: number
  chunk_overlap_tokens?: number
  semantic_output_language?: string
}

export interface ProfileWithOverrides {
  name: string
  description: ProfileDescription
  is_default: boolean
  has_overrides: boolean
  config: ProfileConfig
  overrides: ProfileOverrides
}

/**
 * Get a specific profile with user overrides merged.
 */
export async function getProfile(profileName: string): Promise<ProfileWithOverrides> {
  return fetchJson(`${API_BASE}/settings/profiles/${profileName}`)
}

/**
 * Update profile-specific overrides.
 */
export async function updateProfileOverrides(
  profileName: string,
  overrides: Partial<ProfileOverrides>
): Promise<{ message: string; profile: string; overrides: ProfileOverrides }> {
  return fetchJson(`${API_BASE}/settings/profiles/${profileName}`, {
    method: 'PUT',
    body: JSON.stringify(overrides),
  })
}

/**
 * Reset profile overrides to defaults.
 */
export async function resetProfileOverrides(
  profileName: string
): Promise<{ message: string }> {
  return fetchJson(`${API_BASE}/settings/profiles/${profileName}`, {
    method: 'DELETE',
  })
}
