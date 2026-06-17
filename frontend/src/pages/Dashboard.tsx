import { useMemo, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  listRuns,
  listDocs,
  createRun,
  executeRun,
  cancelRun,
  deleteRun,
  ingestPath,
  rerunWithForce,
  invalidateCache,
  uploadFile,
  SUPPORTED_UPLOAD_ACCEPT,
  uploadMultipleFiles,
  getAssetsIndex,
  batchDeleteDocs,
  batchDeleteRuns,
  batchCancelRuns,
  batchCreateRuns,
  getRunsStats,
} from '@/lib/api'
import type { BatchUploadProgress } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { formatDate, formatBytes } from '@/lib/utils'
import { useI18n } from '@/lib/i18n'
import type { RunStatus, ProfileName, RunStageProgress } from '@/types/api'
import {
  Play,
  Pause,
  Square,
  Trash2,
  FolderOpen,
  RefreshCw,
  FileDown,
  Eye,
  Plus,
  RotateCcw,
  Eraser,
  Upload,
  AlertTriangle,
  CheckSquare,
  Square as SquareIcon,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
} from 'lucide-react'

const statusVariants: Record<RunStatus, 'default' | 'secondary' | 'destructive' | 'success' | 'warning'> = {
  pending: 'secondary',
  running: 'warning',
  succeeded: 'success',
  failed: 'destructive',
  canceled: 'secondary',
}

const PAGE_SIZE = 100

// 分頁組件
function Pagination({
  currentPage,
  totalPages,
  totalItems,
  onPageChange,
}: {
  currentPage: number
  totalPages: number
  totalItems: number
  onPageChange: (page: number) => void
}) {
  const { t } = useI18n()
  const startItem = (currentPage - 1) * PAGE_SIZE + 1
  const endItem = Math.min(currentPage * PAGE_SIZE, totalItems)

  return (
    <div className="flex items-center justify-between mt-4 text-sm">
      <div className="text-muted-foreground">
        {t('pagination.range', { start: startItem, end: endItem, total: totalItems })}
      </div>
      <div className="flex items-center gap-1">
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8"
          onClick={() => onPageChange(1)}
          disabled={currentPage === 1}
          title={t('pagination.first')}
        >
          <ChevronsLeft className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8"
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          title={t('pagination.previous')}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <div className="flex items-center gap-1 mx-2">
          <span>{t('pagination.pagePrefix')}</span>
          <Input
            type="number"
            min={1}
            max={totalPages}
            value={currentPage}
            onChange={(e) => {
              const page = parseInt(e.target.value, 10)
              if (page >= 1 && page <= totalPages) {
                onPageChange(page)
              }
            }}
            className="w-16 h-8 text-center"
          />
          <span>{t('pagination.pageOf', { totalPages })}</span>
        </div>
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8"
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          title={t('pagination.next')}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8"
          onClick={() => onPageChange(totalPages)}
          disabled={currentPage === totalPages}
          title={t('pagination.last')}
        >
          <ChevronsRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}


function WorkbenchMetric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border bg-background px-2 py-2">
      <div className="text-base font-semibold leading-none">{value}</div>
      <div className="mt-1 text-muted-foreground">{label}</div>
    </div>
  )
}

// Component to show needs_review count for a run
function RunReviewBadge({ runId, status }: { runId: string; status: RunStatus }) {
  const { data: assets } = useQuery({
    queryKey: ['assets', runId],
    queryFn: () => getAssetsIndex(runId),
    enabled: status === 'succeeded',
    staleTime: 60000, // Cache for 1 minute
  })

  if (status !== 'succeeded' || !assets) return null

  const reviewCount = assets.filter((a) => a.needs_review).length
  if (reviewCount === 0) return null

  return (
    <Badge variant="warning" className="text-xs">
      <AlertTriangle className="h-3 w-3 mr-1" />
      {reviewCount}
    </Badge>
  )
}

function RunProgress({
  status,
  currentStage,
  progress,
}: {
  status: RunStatus
  currentStage?: string | null
  progress?: RunStageProgress | null
}) {
  const { t } = useI18n()

  if (status === 'pending') {
    return <span className="text-xs text-muted-foreground">{t('dashboard.pendingSchedule')}</span>
  }

  if (status !== 'running') {
    return <span className="text-xs text-muted-foreground">-</span>
  }

  const stageLabel = currentStage ? t(`stage.${currentStage}`) : t('status.running')
  const total = progress?.total ?? 0
  const completed = progress?.completed ?? 0
  const percent = typeof progress?.percent === 'number' ? progress.percent : null
  const message = progress?.message ?? t('dashboard.processingStage', { stage: stageLabel })
  const current = progress?.current
  const pageLabel = typeof current?.page_idx === 'number' ? t('dashboard.currentPage', { page: current.page_idx + 1 }) : null
  const kindLabel = current?.kind ? String(current.kind) : null

  return (
    <div className="min-w-[220px] space-y-1">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="truncate text-muted-foreground" title={message}>
          {message}
        </span>
        {total > 0 && (
          <span className="shrink-0 font-mono text-muted-foreground">
            {completed}/{total}
          </span>
        )}
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${percent ?? 12}%` }}
        />
      </div>
      <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
        <span>{stageLabel}</span>
        {pageLabel && <span>{pageLabel}</span>}
        {kindLabel && <span>{kindLabel}</span>}
      </div>
    </div>
  )
}

export function Dashboard() {
  const queryClient = useQueryClient()
  const { t } = useI18n()
  const [ingestPathValue, setIngestPathValue] = useState('')
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
  const [selectedProfile, setSelectedProfile] = useState<ProfileName>('accurate')

  // Selection state for batch operations
  const [selectedDocIds, setSelectedDocIds] = useState<Set<string>>(new Set())
  const [selectedRunIds, setSelectedRunIds] = useState<Set<string>>(new Set())

  // 分頁狀態
  const [docsPage, setDocsPage] = useState(1)
  const [runsPage, setRunsPage] = useState(1)

  // Queries - 使用分頁
  const { data: runsData, isLoading: runsLoading } = useQuery({
    queryKey: ['runs', runsPage],
    queryFn: () => listRuns(undefined, PAGE_SIZE, (runsPage - 1) * PAGE_SIZE),
    refetchInterval: 5000, // Poll every 5s for status updates
  })

  const { data: docsData } = useQuery({
    queryKey: ['docs', docsPage],
    queryFn: () => listDocs(PAGE_SIZE, (docsPage - 1) * PAGE_SIZE),
  })

  // Runs statistics query
  const { data: runsStatsData } = useQuery({
    queryKey: ['runs-stats'],
    queryFn: getRunsStats,
    refetchInterval: 5000, // Poll every 5s to stay in sync with runs
  })

  // 計算總頁數
  const docsTotalPages = Math.max(1, Math.ceil((docsData?.total ?? 0) / PAGE_SIZE))
  const runsTotalPages = Math.max(1, Math.ceil((runsData?.total ?? 0) / PAGE_SIZE))

  // Mutations
  const ingestMutation = useMutation({
    mutationFn: ingestPath,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['docs'] })
      setIngestPathValue('')
    },
  })

  const createRunMutation = useMutation({
    mutationFn: async (docId: string) => {
      const run = await createRun({
        doc_id: docId,
        profile: selectedProfile,
      })
      await executeRun(run.run_id, true)
      return run
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      setSelectedDocId(null)
    },
  })

  const cancelMutation = useMutation({
    mutationFn: cancelRun,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })

  const removeRunMutation = useMutation({
    mutationFn: deleteRun,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      queryClient.invalidateQueries({ queryKey: ['runs-stats'] })
    },
  })

  const rerunMutation = useMutation({
    mutationFn: async ({ docId, profile }: { docId: string; profile: string }) => {
      return rerunWithForce(docId, profile, { use_cache: false })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })

  const invalidateMutation = useMutation({
    mutationFn: async ({ runId, stages }: { runId: string; stages: string[] }) => {
      return invalidateCache(runId, stages)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })

  const uploadMutation = useMutation({
    mutationFn: uploadFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['docs'] })
    },
  })

  // 批量上傳狀態
  const [uploadProgress, setUploadProgress] = useState<BatchUploadProgress | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  const handleBatchUpload = async (files: File[]) => {
    if (files.length === 0) return

    // 限制最多 200 個文件
    const MAX_FILES = 200
    if (files.length > MAX_FILES) {
      alert(t('dashboard.confirmMaxFiles', { max: MAX_FILES, count: files.length }))
      return
    }

    setIsUploading(true)
    setUploadProgress({ total: files.length, completed: 0, failed: 0, inProgress: 0, results: [] })

    try {
      await uploadMultipleFiles(files, {
        concurrency: 3,
        onProgress: setUploadProgress,
      })
      queryClient.invalidateQueries({ queryKey: ['docs'] })
    } finally {
      setIsUploading(false)
      // 3 秒後清除進度顯示
      setTimeout(() => setUploadProgress(null), 3000)
    }
  }

  // Batch delete mutations
  const batchDeleteDocsMutation = useMutation({
    mutationFn: batchDeleteDocs,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['docs'] })
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      setSelectedDocIds(new Set())
    },
  })

  const batchDeleteRunsMutation = useMutation({
    mutationFn: batchDeleteRuns,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      setSelectedRunIds(new Set())
    },
  })

  const batchCancelRunsMutation = useMutation({
    mutationFn: batchCancelRuns,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      setSelectedRunIds(new Set())
    },
  })

  // Batch execute runs mutation (for resuming canceled/failed runs)
  const batchExecuteRunsMutation = useMutation({
    mutationFn: async (runIds: string[]) => {
      const results = await Promise.allSettled(
        runIds.map((runId) => executeRun(runId, true))
      )
      const succeeded = results.filter((r) => r.status === 'fulfilled').length
      const failed = results.filter((r) => r.status === 'rejected').length
      return { succeeded, failed, total: runIds.length }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      setSelectedRunIds(new Set())
    },
  })

  // Batch create runs mutation
  const batchCreateRunsMutation = useMutation({
    mutationFn: ({ docIds, profile }: { docIds: string[]; profile: ProfileName }) =>
      batchCreateRuns(docIds, profile, { use_cache: false }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      setSelectedDocIds(new Set())
    },
  })

  const [isDragging, setIsDragging] = useState(false)

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleBatchUpload(files)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleBatchUpload(Array.from(files))
    }
    // 清除 input 值，允許重複選擇相同文件
    e.target.value = ''
  }

  const runs = useMemo(() => runsData?.runs ?? [], [runsData])
  const docs = useMemo(() => docsData?.docs ?? [], [docsData])
  const latestSucceededRun = useMemo(
    () => runs.find((run) => run.status === 'succeeded') ?? null,
    [runs]
  )
  const activeResultRun = useMemo(() => {
    if (selectedRunIds.size === 1) {
      const selectedRunId = Array.from(selectedRunIds)[0]
      return runs.find((run) => run.run_id === selectedRunId && run.status === 'succeeded') ?? latestSucceededRun
    }
    return latestSucceededRun
  }, [latestSucceededRun, runs, selectedRunIds])

  // Use API-provided statistics (accurate across all pages)
  const statusStats = runsStatsData ?? {
    total: 0,
    pending: 0,
    running: 0,
    succeeded: 0,
    failed: 0,
    canceled: 0,
  }

  // Selection helpers
  const toggleDocSelection = (docId: string) => {
    setSelectedDocIds((prev) => {
      const next = new Set(prev)
      if (next.has(docId)) {
        next.delete(docId)
      } else {
        next.add(docId)
      }
      return next
    })
  }

  const toggleRunSelection = (runId: string) => {
    setSelectedRunIds((prev) => {
      const next = new Set(prev)
      if (next.has(runId)) {
        next.delete(runId)
      } else {
        next.add(runId)
      }
      return next
    })
  }

  // 選擇當前頁所有文檔
  const selectAllDocsOnPage = () => {
    if (selectedDocIds.size === docs.length && docs.every(d => selectedDocIds.has(d.doc_id))) {
      setSelectedDocIds(new Set())
    } else {
      setSelectedDocIds(new Set(docs.map((d) => d.doc_id)))
    }
  }

  // 選擇所有頁的文檔（需要獲取所有 ID）
  const [isSelectingAll, setIsSelectingAll] = useState(false)
  const selectAllDocsAllPages = async () => {
    if (!docsData?.total) return
    setIsSelectingAll(true)
    try {
      // 獲取所有文檔（只需要 ID）
      const allDocs = await listDocs(docsData.total, 0)
      setSelectedDocIds(new Set(allDocs.docs.map((d) => d.doc_id)))
    } finally {
      setIsSelectingAll(false)
    }
  }

  // 選擇當前頁所有 Runs
  const selectAllRunsOnPage = () => {
    if (runs.every(r => selectedRunIds.has(r.run_id)) && runs.length > 0) {
      setSelectedRunIds(new Set())
    } else {
      setSelectedRunIds(new Set(runs.map((r) => r.run_id)))
    }
  }

  // 選擇所有頁的 Runs
  const [isSelectingAllRuns, setIsSelectingAllRuns] = useState(false)
  const selectAllRunsAllPages = async () => {
    if (!runsData?.total) return
    setIsSelectingAllRuns(true)
    try {
      const allRuns = await listRuns(undefined, runsData.total, 0)
      setSelectedRunIds(new Set(allRuns.runs.map((r) => r.run_id)))
    } finally {
      setIsSelectingAllRuns(false)
    }
  }

  const handleBatchDeleteDocs = () => {
    if (selectedDocIds.size === 0) return
    if (confirm(t('dashboard.confirmDeleteDocs', { count: selectedDocIds.size }))) {
      batchDeleteDocsMutation.mutate(Array.from(selectedDocIds))
    }
  }

  const handleBatchRunDocs = () => {
    if (selectedDocIds.size === 0) return
    batchCreateRunsMutation.mutate({
      docIds: Array.from(selectedDocIds),
      profile: selectedProfile,
    })
  }

  const handleBatchDeleteRuns = () => {
    if (selectedRunIds.size === 0) return
    if (confirm(t('dashboard.confirmRemoveRuns', { count: selectedRunIds.size }))) {
      batchDeleteRunsMutation.mutate(Array.from(selectedRunIds))
    }
  }

  const handleBatchCancelRuns = () => {
    if (selectedRunIds.size === 0) return
    if (confirm(t('dashboard.confirmCancelRuns', { count: selectedRunIds.size }))) {
      batchCancelRunsMutation.mutate(Array.from(selectedRunIds))
    }
  }

  const handleBatchExecuteRuns = () => {
    if (selectedRunIds.size === 0) return
    // Filter to only canceled/failed runs
    const eligibleRuns = runs.filter(
      (r) => selectedRunIds.has(r.run_id) && (r.status === 'canceled' || r.status === 'failed')
    )
    if (eligibleRuns.length === 0) {
      alert(t('dashboard.noExecutableRuns'))
      return
    }
    if (confirm(t('dashboard.confirmExecuteRuns', { count: eligibleRuns.length }))) {
      batchExecuteRunsMutation.mutate(eligibleRuns.map((r) => r.run_id))
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t('nav.dashboard')}</h1>
        <Button
          variant="outline"
          size="sm"
          onClick={() => queryClient.invalidateQueries({ queryKey: ['runs'] })}
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          {t('common.refresh')}
        </Button>
      </div>

      {/* Workbench */}
      <div className="grid gap-4 xl:grid-cols-[1fr_1fr_1.1fr]">
        <Card className="min-h-[260px]">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Upload className="h-5 w-5" />
              {t('dashboard.uploadDocuments')}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div
              className={`relative rounded-md border-2 border-dashed p-5 text-center transition-colors ${
                isDragging
                  ? 'border-primary bg-primary/5'
                  : 'border-muted-foreground/25 hover:border-muted-foreground/50'
              }`}
              onDragOver={(e) => {
                e.preventDefault()
                setIsDragging(true)
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <Upload className="mx-auto mb-3 h-8 w-8 text-muted-foreground" />
              <p className="mb-2 text-sm text-muted-foreground">{t('dashboard.dropFiles')}</p>
              <p className="mb-3 text-xs text-muted-foreground">{t('dashboard.supportedFiles')}</p>
              <label>
                <input
                  type="file"
                  className="hidden"
                  accept={SUPPORTED_UPLOAD_ACCEPT}
                  onChange={handleFileSelect}
                  disabled={isUploading}
                  multiple
                />
                <Button variant="outline" size="sm" disabled={isUploading} asChild>
                  <span className="cursor-pointer">{t('dashboard.chooseFiles')}</span>
                </Button>
              </label>
              <p className="mt-2 text-xs text-muted-foreground">{t('dashboard.fileLimit')}</p>
            </div>

            {uploadProgress && (
              <div className="rounded-md bg-muted/50 p-3 text-sm">
                <div className="mb-2 flex items-center justify-between">
                  <span>
                    {t('dashboard.uploading', { completed: uploadProgress.completed, total: uploadProgress.total })}
                    {uploadProgress.failed > 0 && (
                      <span className="ml-2 text-destructive">{t('dashboard.uploadFailed', { failed: uploadProgress.failed })}</span>
                    )}
                  </span>
                  {uploadProgress.completed === uploadProgress.total && (
                    <span className="text-green-600">{t('dashboard.done')}</span>
                  )}
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${(uploadProgress.completed / uploadProgress.total) * 100}%` }}
                  />
                </div>
              </div>
            )}

            <div className="flex gap-2">
              <Input
                placeholder={t('dashboard.serverPath')}
                value={ingestPathValue}
                onChange={(e) => setIngestPathValue(e.target.value)}
                className="flex-1"
              />
              <Button
                variant="outline"
                onClick={() => ingestMutation.mutate(ingestPathValue)}
                disabled={!ingestPathValue || ingestMutation.isPending}
              >
                <Plus className="mr-2 h-4 w-4" />
                {t('stage.ingest')}
              </Button>
            </div>

            {(ingestMutation.error || uploadMutation.error) && (
              <p className="text-sm text-destructive">
                {((ingestMutation.error || uploadMutation.error) as Error).message}
              </p>
            )}
          </CardContent>
        </Card>

        <Card className="min-h-[260px]">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Play className="h-5 w-5" />
              {t('dashboard.processing')}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <WorkbenchMetric label={t('common.documents')} value={docsData?.total ?? 0} />
              <WorkbenchMetric label={t('common.selected')} value={selectedDocIds.size} />
              <WorkbenchMetric label={t('common.running')} value={statusStats.running} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">{t('dashboard.pipelineProfile')}</label>
              <div className="grid grid-cols-2 gap-2">
                {(['accurate', 'fast'] as ProfileName[]).map((profile) => (
                  <Button
                    key={profile}
                    type="button"
                    variant={selectedProfile === profile ? 'default' : 'outline'}
                    onClick={() => setSelectedProfile(profile)}
                    className="capitalize"
                  >
                    {t(`profile.${profile}`)}
                  </Button>
                ))}
              </div>
            </div>

            <Button
              className="w-full gap-2"
              onClick={handleBatchRunDocs}
              disabled={selectedDocIds.size === 0 || batchCreateRunsMutation.isPending}
            >
              <Play className="h-4 w-4" />
              {selectedDocIds.size > 0 ? t('dashboard.startSelected', { count: selectedDocIds.size }) : t('dashboard.selectDocsFirst')}
            </Button>

            <div className="rounded-md border bg-muted/30 p-3 text-xs text-muted-foreground">
              {t('dashboard.flowHint')}
            </div>
          </CardContent>
        </Card>

        <Card className="min-h-[260px]">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <FileDown className="h-5 w-5" />
              {t('dashboard.results')}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <WorkbenchMetric label={t('dashboard.done')} value={statusStats.succeeded} />
              <WorkbenchMetric label={t('common.failed')} value={statusStats.failed} />
              <WorkbenchMetric label={t('common.canceled')} value={statusStats.canceled} />
            </div>

            {selectedRunIds.size === 0 && latestSucceededRun && (
              <div className="rounded-md border bg-muted/30 p-3 text-xs text-muted-foreground">
                {t('dashboard.latestRunHint')}
                <span className="ml-1 font-mono">{latestSucceededRun.run_id.slice(0, 12)}...</span>
              </div>
            )}

            {activeResultRun ? (
              <div className="flex flex-wrap gap-2">
                <Button asChild variant="outline" size="sm">
                  <Link to={`/viewer/${activeResultRun.run_id}`}>
                    <Eye className="mr-2 h-4 w-4" />
                    {t('viewer.title')}
                  </Link>
                </Button>
                <Button asChild variant="outline" size="sm">
                  <Link to={`/assets?run=${activeResultRun.run_id}`}>
                    <FolderOpen className="mr-2 h-4 w-4" />
                    {t('nav.assets')}
                  </Link>
                </Button>
              </div>
            ) : (
              <div className="rounded-md border bg-muted/30 p-3 text-sm text-muted-foreground">
                {t('dashboard.noCompletedRuns')}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Documents Section */}
      {(docsData?.total ?? 0) > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>{t('common.documents')} ({docsData?.total ?? 0})</CardTitle>
              <div className="flex items-center gap-2">
                {/* {t('common.selectPage')} */}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={selectAllDocsOnPage}
                  className="gap-1"
                >
                  {docs.every(d => selectedDocIds.has(d.doc_id)) && docs.length > 0 ? (
                    <CheckSquare className="h-4 w-4" />
                  ) : (
                    <SquareIcon className="h-4 w-4" />
                  )}
                  {t('common.selectPage')}
                </Button>
                {/* 全選所有（多頁時顯示） */}
                {docsTotalPages > 1 && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={selectAllDocsAllPages}
                    disabled={isSelectingAll || selectedDocIds.size === (docsData?.total ?? 0)}
                    className="gap-1"
                  >
                    {selectedDocIds.size === (docsData?.total ?? 0) ? (
                      <CheckSquare className="h-4 w-4" />
                    ) : (
                      <SquareIcon className="h-4 w-4" />
                    )}
                    {isSelectingAll ? t('common.loading') : t('dashboard.selectAllCount', { count: docsData?.total ?? 0 })}
                  </Button>
                )}
                {/* {t('common.clearSelection')} */}
                {selectedDocIds.size > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedDocIds(new Set())}
                    className="gap-1 text-muted-foreground"
                  >
                    {t('common.clearSelection')}
                  </Button>
                )}
                {selectedDocIds.size > 0 && (
                  <>
                    <select
                      className="h-8 rounded-md border px-2 text-sm bg-background"
                      value={selectedProfile}
                      onChange={(e) => setSelectedProfile(e.target.value as ProfileName)}
                    >
                      <option value="fast">{t('profile.fast')}</option>
                      <option value="accurate">{t('profile.accurate')}</option>
                    </select>
                    <Button
                      size="sm"
                      onClick={handleBatchRunDocs}
                      disabled={batchCreateRunsMutation.isPending}
                      className="gap-1"
                    >
                      <Play className="h-4 w-4" />
                      {t('dashboard.runCount', { count: selectedDocIds.size })}
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleBatchDeleteDocs}
                      disabled={batchDeleteDocsMutation.isPending}
                      className="gap-1"
                    >
                      <Trash2 className="h-4 w-4" />
                      {t('dashboard.deleteCount', { count: selectedDocIds.size })}
                    </Button>
                  </>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-[400px] overflow-y-auto">
              {docs.map((doc) => (
                <div
                  key={doc.doc_id}
                  className={`flex items-center justify-between rounded-lg border p-3 ${
                    selectedDocIds.has(doc.doc_id) ? 'border-primary bg-primary/5' : ''
                  }`}
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <button
                      type="button"
                      onClick={() => toggleDocSelection(doc.doc_id)}
                      className="flex-shrink-0 p-1 rounded hover:bg-muted"
                    >
                      {selectedDocIds.has(doc.doc_id) ? (
                        <CheckSquare className="h-5 w-5 text-primary" />
                      ) : (
                        <SquareIcon className="h-5 w-5 text-muted-foreground" />
                      )}
                    </button>
                    <div className="flex-1 min-w-0">
                      <p className="truncate font-mono text-sm">{doc.source_path}</p>
                      <p className="text-xs text-muted-foreground">
                        {doc.ext.toUpperCase()} · {formatBytes(doc.size_bytes)} ·{' '}
                        {doc.doc_id.slice(0, 12)}...
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 ml-4">
                    <select
                      className="h-8 rounded-md border px-2 text-sm bg-background"
                      value={selectedDocId === doc.doc_id ? selectedProfile : 'accurate'}
                      onChange={(e) => {
                        setSelectedDocId(doc.doc_id)
                        setSelectedProfile(e.target.value as ProfileName)
                      }}
                    >
                      <option value="fast">{t('profile.fast')}</option>
                      <option value="accurate">{t('profile.accurate')}</option>
                    </select>
                    <Button
                      size="sm"
                      onClick={() => createRunMutation.mutate(doc.doc_id)}
                      disabled={createRunMutation.isPending}
                    >
                      <Play className="mr-1 h-3 w-3" />
                      {t('dashboard.run')}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
            {/* 分頁控制 */}
            {docsTotalPages > 1 && (
              <Pagination
                currentPage={docsPage}
                totalPages={docsTotalPages}
                totalItems={docsData?.total ?? 0}
                onPageChange={setDocsPage}
              />
            )}
          </CardContent>
        </Card>
      )}

      {/* Runs Table */}
      <Card>
        <CardHeader className="space-y-4">
          <div className="flex items-center justify-between">
            <CardTitle>{t('dashboard.pipelineRuns')} ({runsData?.total ?? 0})</CardTitle>
            <div className="flex items-center gap-2">
              {runs.length > 0 && (
                <>
                  {/* {t('common.selectPage')} */}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={selectAllRunsOnPage}
                    className="gap-1"
                  >
                    {runs.every(r => selectedRunIds.has(r.run_id)) && runs.length > 0 ? (
                      <CheckSquare className="h-4 w-4" />
                    ) : (
                      <SquareIcon className="h-4 w-4" />
                    )}
                    {t('common.selectPage')}
                  </Button>
                  {/* 全選所有（多頁時顯示） */}
                  {runsTotalPages > 1 && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={selectAllRunsAllPages}
                      disabled={isSelectingAllRuns || selectedRunIds.size === (runsData?.total ?? 0)}
                      className="gap-1"
                    >
                      {selectedRunIds.size === (runsData?.total ?? 0) ? (
                        <CheckSquare className="h-4 w-4" />
                      ) : (
                        <SquareIcon className="h-4 w-4" />
                      )}
                      {isSelectingAllRuns ? t('common.loading') : t('dashboard.selectAllCount', { count: runsData?.total ?? 0 })}
                    </Button>
                  )}
                  {/* {t('common.clearSelection')} */}
                  {selectedRunIds.size > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSelectedRunIds(new Set())}
                      className="gap-1 text-muted-foreground"
                    >
                      {t('common.clearSelection')}
                    </Button>
                  )}
                  {selectedRunIds.size > 0 && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleBatchCancelRuns}
                      disabled={batchCancelRunsMutation.isPending}
                      className="gap-1"
                    >
                      <Pause className="h-4 w-4" />
                      {t('dashboard.pauseCount', { count: selectedRunIds.size })}
                    </Button>
                  )}
                  {selectedRunIds.size > 0 && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleBatchExecuteRuns}
                      disabled={batchExecuteRunsMutation.isPending}
                      className="gap-1"
                    >
                      <Play className="h-4 w-4" />
                      {t('dashboard.resumeCount', { count: selectedRunIds.size })}
                    </Button>
                  )}
                  {selectedRunIds.size > 0 && (
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleBatchDeleteRuns}
                      disabled={batchDeleteRunsMutation.isPending}
                      className="gap-1"
                    >
                      <Trash2 className="h-4 w-4" />
                      {t('dashboard.removeCount', { count: selectedRunIds.size })}
                    </Button>
                  )}
                </>
              )}
            </div>
          </div>
          {/* Status Statistics Cards */}
          {(runsData?.total ?? 0) > 0 && (
            <div className="flex flex-wrap gap-2">
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-muted/50 text-sm">
                <span className="text-muted-foreground">{t('common.total')}:</span>
                <span className="font-medium">{statusStats.total}</span>
              </div>
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-green-500/10 text-sm">
                <span className="w-2 h-2 rounded-full bg-green-500" />
                <span className="text-green-700 dark:text-green-400">{t('common.success')}:</span>
                <span className="font-medium text-green-700 dark:text-green-400">{statusStats.succeeded}</span>
              </div>
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-yellow-500/10 text-sm">
                <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
                <span className="text-yellow-700 dark:text-yellow-400">{t('common.running')}:</span>
                <span className="font-medium text-yellow-700 dark:text-yellow-400">{statusStats.running}</span>
              </div>
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-slate-500/10 text-sm">
                <span className="w-2 h-2 rounded-full bg-slate-400" />
                <span className="text-slate-600 dark:text-slate-400">{t('common.pending')}:</span>
                <span className="font-medium text-slate-600 dark:text-slate-400">{statusStats.pending}</span>
              </div>
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-red-500/10 text-sm">
                <span className="w-2 h-2 rounded-full bg-red-500" />
                <span className="text-red-700 dark:text-red-400">{t('common.failed')}:</span>
                <span className="font-medium text-red-700 dark:text-red-400">{statusStats.failed}</span>
              </div>
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-slate-500/10 text-sm">
                <span className="w-2 h-2 rounded-full bg-slate-400" />
                <span className="text-slate-600 dark:text-slate-400">{t('common.canceled')}:</span>
                <span className="font-medium text-slate-600 dark:text-slate-400">{statusStats.canceled}</span>
              </div>
            </div>
          )}
        </CardHeader>
        <CardContent>
          {runsLoading ? (
            <p className="text-muted-foreground">{t('common.loading')}</p>
          ) : runs.length === 0 ? (
            <p className="text-muted-foreground">{t('dashboard.noRuns')}</p>
          ) : (
            <div className="rounded-md border max-h-[500px] overflow-y-auto">
              <table className="w-full">
                <thead className="sticky top-0 bg-background z-10">
                  <tr className="border-b bg-muted/50">
                    <th className="p-3 text-left text-sm font-medium w-10">
                      <button
                        type="button"
                        onClick={selectAllRunsOnPage}
                        className="p-1 rounded hover:bg-muted"
                        title={t('common.selectPage')}
                      >
                        {runs.every(r => selectedRunIds.has(r.run_id)) && runs.length > 0 ? (
                          <CheckSquare className="h-4 w-4 text-primary" />
                        ) : (
                          <SquareIcon className="h-4 w-4 text-muted-foreground" />
                        )}
                      </button>
                    </th>
                    <th className="p-3 text-left text-sm font-medium">{t('dashboard.runId')}</th>
                    <th className="p-3 text-left text-sm font-medium">{t('dashboard.docId')}</th>
                    <th className="p-3 text-left text-sm font-medium">{t('common.profile')}</th>
                    <th className="p-3 text-left text-sm font-medium">{t('common.status')}</th>
                    <th className="p-3 text-left text-sm font-medium">{t('common.progress')}</th>
                    <th className="p-3 text-left text-sm font-medium">{t('common.review')}</th>
                    <th className="p-3 text-left text-sm font-medium">{t('common.created')}</th>
                    <th className="p-3 text-left text-sm font-medium">{t('common.actions')}</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <tr
                      key={run.run_id}
                      className={`border-b ${
                        selectedRunIds.has(run.run_id) ? 'bg-primary/5' : ''
                      }`}
                    >
                      <td className="p-3">
                        <button
                          type="button"
                          onClick={() => toggleRunSelection(run.run_id)}
                          className="p-1 rounded hover:bg-muted"
                        >
                          {selectedRunIds.has(run.run_id) ? (
                            <CheckSquare className="h-4 w-4 text-primary" />
                          ) : (
                            <SquareIcon className="h-4 w-4 text-muted-foreground" />
                          )}
                        </button>
                      </td>
                      <td className="p-3">
                        <code className="text-xs">{run.run_id.slice(0, 12)}...</code>
                      </td>
                      <td className="p-3">
                        <code className="text-xs">{run.doc_id.slice(0, 12)}...</code>
                      </td>
                      <td className="p-3">
                        <Badge variant="outline">{t(`profile.${run.profile}`)}</Badge>
                      </td>
                      <td className="p-3">
                        <Badge variant={statusVariants[run.status]}>
                          {t(`status.${run.status}`)}
                        </Badge>
                      </td>
                      <td className="p-3">
                        <RunProgress
                          status={run.status}
                          currentStage={run.current_stage}
                          progress={run.stage_progress}
                        />
                      </td>
                      <td className="p-3">
                        <RunReviewBadge runId={run.run_id} status={run.status} />
                      </td>
                      <td className="p-3 text-sm text-muted-foreground">
                        {formatDate(run.created_at)}
                      </td>
                      <td className="p-3">
                        <div className="flex items-center gap-1">
                          {run.status === 'succeeded' && (
                            <>
                              <Button size="icon" variant="ghost" asChild>
                                <Link to={`/viewer/${run.run_id}`}>
                                  <Eye className="h-4 w-4" />
                                </Link>
                              </Button>
                              <Button
                                size="icon"
                                variant="ghost"
                                title={t('dashboard.forceRerun')}
                                onClick={() => rerunMutation.mutate({
                                  docId: run.doc_id,
                                  profile: run.profile,
                                })}
                                disabled={rerunMutation.isPending}
                              >
                                <RotateCcw className="h-4 w-4" />
                              </Button>
                              <Button
                                size="icon"
                                variant="ghost"
                                title={t('dashboard.invalidateCache')}
                                onClick={() => invalidateMutation.mutate({
                                  runId: run.run_id,
                                  stages: ['parse', 'enrich'],
                                })}
                                disabled={invalidateMutation.isPending}
                              >
                                <Eraser className="h-4 w-4" />
                              </Button>
                            </>
                          )}
                          {run.status === 'failed' && (
                            <Button
                              size="icon"
                              variant="ghost"
                              title={t('dashboard.retryRun')}
                              onClick={() => executeRun(run.run_id, true).then(() => {
                                queryClient.invalidateQueries({ queryKey: ['runs'] })
                              })}
                            >
                              <Play className="h-4 w-4" />
                            </Button>
                          )}
                          {(run.status === 'pending' || run.status === 'running') && (
                            <Button
                              size="icon"
                              variant="ghost"
                              onClick={() => cancelMutation.mutate(run.run_id)}
                            >
                              <Square className="h-4 w-4" />
                            </Button>
                          )}
                          <Button
                            size="icon"
                            variant="ghost"
                            title={t('dashboard.removeRun')}
                            onClick={() => removeRunMutation.mutate(run.run_id)}
                            disabled={removeRunMutation.isPending}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {/* 分頁控制 */}
          {runsTotalPages > 1 && (
            <Pagination
              currentPage={runsPage}
              totalPages={runsTotalPages}
              totalItems={runsData?.total ?? 0}
              onPageChange={setRunsPage}
            />
          )}
        </CardContent>
      </Card>

    </div>
  )
}
