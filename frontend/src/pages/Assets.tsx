import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useSearchParams } from 'react-router-dom'
import {
  deleteSplitDocuments,
  downloadRuns,
  getAssetUrl,
  getQualityGate,
  getSplitDocument,
  getSplitDocumentDownloadUrl,
  getSplitDocuments,
  getOutputsSummary,
  type DownloadOutputFormat,
  type OutputRunSummary,
} from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useI18n } from '@/lib/i18n'
import type { QualityGateReport, SplitDocumentMeta } from '@/types/api'
import {
  Archive,
  CheckSquare,
  Download,
  Eye,
  FileText,
  FolderOpen,
  Search,
  SplitSquareHorizontal,
  Square,
  Trash2,
} from 'lucide-react'

const formatOptions: Array<Exclude<DownloadOutputFormat, 'json'>> = ['md', 'docx', 'txt']

interface DocumentSearchItem {
  meta: SplitDocumentMeta
  content: string
}

export function Assets() {
  const queryClient = useQueryClient()
  const { t } = useI18n()
  const [searchParams] = useSearchParams()
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedRunId, setSelectedRunId] = useState<string | null>(() => searchParams.get('run'))
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null)
  const [selectedDownloadIds, setSelectedDownloadIds] = useState<Set<string>>(new Set())
  const [format, setFormat] = useState<Exclude<DownloadOutputFormat, 'json'>>('md')
  const [isDownloadingAll, setIsDownloadingAll] = useState(false)
  const [isDownloadingRuns, setIsDownloadingRuns] = useState(false)
  const [selectedRunDownloadIds, setSelectedRunDownloadIds] = useState<Set<string>>(new Set())

  const { data: outputsSummary } = useQuery({
    queryKey: ['outputs-summary'],
    queryFn: () => getOutputsSummary(100, 0, { include_hidden: true, has_documents_only: true }),
  })

  const runs = useMemo(() => outputsSummary?.runs ?? [], [outputsSummary])

  useEffect(() => {
    if (runs.length === 0) {
      setSelectedRunId(null)
      return
    }
    if (!selectedRunId || !runs.some((run) => run.run_id === selectedRunId)) {
      setSelectedRunId(runs[0].run_id)
    }
  }, [runs, selectedRunId])

  useEffect(() => {
    setSelectedRunDownloadIds((current) => {
      const visibleRunIds = new Set(runs.map((run) => run.run_id))
      return new Set(Array.from(current).filter((runId) => visibleRunIds.has(runId)))
    })
  }, [runs])

  const selectedRun = useMemo(
    () => runs.find((run) => run.run_id === selectedRunId) ?? null,
    [runs, selectedRunId]
  )

  const splitDocumentsQuery = useQuery({
    queryKey: ['processed-documents', selectedRunId],
    queryFn: () => getSplitDocuments(selectedRunId!),
    enabled: !!selectedRunId,
  })

  const splitDocuments = useMemo(
    () => splitDocumentsQuery.data?.documents ?? [],
    [splitDocumentsQuery.data]
  )

  useEffect(() => {
    setSelectedDownloadIds(new Set())
  }, [selectedRunId])

  useEffect(() => {
    if (splitDocuments.length === 0) {
      setSelectedDocumentId(null)
      return
    }
    if (!selectedDocumentId || !splitDocuments.some((doc) => doc.document_id === selectedDocumentId)) {
      setSelectedDocumentId(splitDocuments[0].document_id)
    }
  }, [splitDocuments, selectedDocumentId])

  const documentContentsQuery = useQuery({
    queryKey: ['processed-document-search', selectedRunId, splitDocuments.map((doc) => doc.document_id).join('|')],
    queryFn: async (): Promise<DocumentSearchItem[]> => {
      const items = await Promise.all(
        splitDocuments.map(async (document) => {
          const detail = await getSplitDocument(selectedRunId!, document.document_id)
          return { meta: document, content: detail.content }
        })
      )
      return items
    },
    enabled: !!selectedRunId && splitDocuments.length > 0,
  })

  const selectedDocumentQuery = useQuery({
    queryKey: ['processed-document-preview', selectedRunId, selectedDocumentId],
    queryFn: () => getSplitDocument(selectedRunId!, selectedDocumentId!),
    enabled: !!selectedRunId && !!selectedDocumentId,
  })

  const qualityGateQuery = useQuery({
    queryKey: ['quality-gate', selectedRunId],
    queryFn: () => getQualityGate(selectedRunId!),
    enabled: !!selectedRunId,
    retry: false,
  })

  const qualityGate = qualityGateQuery.data
  const searchableDocuments = useMemo(
    () => documentContentsQuery.data ?? [],
    [documentContentsQuery.data]
  )
  const filteredDocuments = useMemo(() => {
    const documents = searchableDocuments.length > 0
      ? searchableDocuments
      : splitDocuments.map((meta) => ({ meta, content: '' }))

    const query = searchQuery.trim().toLowerCase()
    if (!query) return documents

    return documents.filter(({ meta, content }) => {
      const haystack = [
        meta.title,
        meta.filename,
        meta.kind,
        meta.page_label,
        meta.page_indices?.map((page) => String(page + 1)).join(' '),
        content,
      ]
        .filter(Boolean)
        .join('\n')
        .toLowerCase()
      return haystack.includes(query)
    })
  }, [searchQuery, searchableDocuments, splitDocuments])

  const selectedDocument = selectedDocumentQuery.data?.document
  const selectedContent = selectedDocumentQuery.data?.content ?? ''
  const mainDocumentCount = splitDocuments.filter((document) => document.kind === 'main').length
  const extractedDocumentCount = Math.max(splitDocuments.length - mainDocumentCount, 0)
  const selectedDownloadCount = selectedDownloadIds.size

  const selectedRunDownloadCount = selectedRunDownloadIds.size

  function toggleRunDownloadSelection(runId: string) {
    setSelectedRunDownloadIds((current) => {
      const next = new Set(current)
      if (next.has(runId)) {
        next.delete(runId)
      } else {
        next.add(runId)
      }
      return next
    })
  }

  function selectAllRunDownloads() {
    setSelectedRunDownloadIds(new Set(runs.map((run) => run.run_id)))
  }

  function clearRunDownloads() {
    setSelectedRunDownloadIds(new Set())
  }

  async function downloadSelectedRuns() {
    const runIds = Array.from(selectedRunDownloadIds)
    if (runIds.length === 0) return

    setIsDownloadingRuns(true)
    try {
      const response = await downloadRuns({
        run_ids: runIds,
        file_types: ['documents'],
        format,
      })
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `processed_documents_${runIds.length}_runs_${format}.zip`
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } finally {
      setIsDownloadingRuns(false)
    }
  }

  function toggleDownloadSelection(documentId: string) {
    setSelectedDownloadIds((current) => {
      const next = new Set(current)
      if (next.has(documentId)) {
        next.delete(documentId)
      } else {
        next.add(documentId)
      }
      return next
    })
  }


  const deleteDocumentsMutation = useMutation({
    mutationFn: async (documentIds: string[]) => {
      if (!selectedRunId) throw new Error('No run selected')
      return deleteSplitDocuments(selectedRunId, documentIds)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['processed-documents', selectedRunId] })
      queryClient.invalidateQueries({ queryKey: ['processed-document-search', selectedRunId] })
      queryClient.invalidateQueries({ queryKey: ['processed-document-preview', selectedRunId] })
      queryClient.invalidateQueries({ queryKey: ['outputs-summary'] })
      setSelectedDownloadIds(new Set())
      setSelectedDocumentId(null)
    },
  })

  const deleteRunOutputsMutation = useMutation({
    mutationFn: async (runIds: string[]) => {
      const results = []
      for (const runId of runIds) {
        const documents = await getSplitDocuments(runId)
        const documentIds = documents.documents.map((document) => document.document_id)
        if (documentIds.length === 0) {
          results.push({ runId, deleted: [] })
          continue
        }
        const result = await deleteSplitDocuments(runId, documentIds)
        results.push({ runId, deleted: result.deleted })
      }
      return results
    },
    onSuccess: (_results, runIds) => {
      for (const runId of runIds) {
        queryClient.invalidateQueries({ queryKey: ['processed-documents', runId] })
        queryClient.invalidateQueries({ queryKey: ['processed-document-search', runId] })
        queryClient.invalidateQueries({ queryKey: ['processed-document-preview', runId] })
      }
      queryClient.invalidateQueries({ queryKey: ['outputs-summary'] })
      setSelectedRunDownloadIds(new Set())
      setSelectedDownloadIds(new Set())
      setSelectedDocumentId(null)
    },
  })

  async function downloadAllDocuments() {
    if (!selectedRunId) return
    setIsDownloadingAll(true)
    try {
      const response = await downloadRuns({
        run_ids: [selectedRunId],
        file_types: ['documents'],
        format,
        document_ids: Array.from(selectedDownloadIds),
      })
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${selectedRunId}_documents_${format}.zip`
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } finally {
      setIsDownloadingAll(false)
    }
  }


  function deleteSelectedDocuments() {
    if (!selectedRunId || selectedDownloadIds.size === 0) return
    const count = selectedDownloadIds.size
    if (!confirm(t('assets.confirmDeleteDocs', { count }))) {
      return
    }
    deleteDocumentsMutation.mutate(Array.from(selectedDownloadIds))
  }

  function deleteSelectedRunOutputs() {
    const runIds = Array.from(selectedRunDownloadIds)
    if (runIds.length === 0) return
    if (!confirm(t('assets.confirmDeleteRunDocs', { count: runIds.length }))) {
      return
    }
    deleteRunOutputsMutation.mutate(runIds)
  }

  function deleteSingleRunOutputs(runId: string) {
    if (!confirm(t('assets.confirmDeleteRunOutput'))) {
      return
    }
    deleteRunOutputsMutation.mutate([runId])
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-2xl font-bold">{t('assets.title')}</h1>
          <p className="text-sm text-muted-foreground">
{t('assets.description')}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {formatOptions.map((option) => (
            <Button
              key={option}
              type="button"
              size="sm"
              variant={format === option ? 'default' : 'outline'}
              onClick={() => setFormat(option)}
              className="uppercase"
            >
              {option}
            </Button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(280px,0.75fr)_minmax(360px,0.95fr)_minmax(520px,1.45fr)] 2xl:grid-cols-[minmax(320px,0.7fr)_minmax(420px,0.9fr)_minmax(680px,1.6fr)]">
        <Card className="flex min-h-[520px] flex-col xl:h-[calc(100vh-12rem)] xl:min-h-[620px]">
          <CardHeader className="space-y-3 pb-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <CardTitle className="flex items-center gap-2 text-sm">
                  <FolderOpen className="h-4 w-4" />
                  {t('assets.runs')}
                </CardTitle>
                <p className="mt-1 text-xs text-muted-foreground">{t('assets.runsHint')}</p>
              </div>
              <Badge variant="outline">{selectedRunDownloadCount} / {runs.length}</Badge>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={selectAllRunDownloads}
                disabled={runs.length === 0}
              >
                <CheckSquare className="mr-2 h-4 w-4" />
                {t('common.selectAll')}
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={clearRunDownloads}
                disabled={selectedRunDownloadCount === 0}
              >
                <Square className="mr-2 h-4 w-4" />
                {t('common.cancel')}
              </Button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant="default"
                size="sm"
                onClick={downloadSelectedRuns}
                disabled={selectedRunDownloadCount === 0 || isDownloadingRuns}
              >
                <Archive className="mr-2 h-4 w-4" />
                {t('common.download')} {selectedRunDownloadCount}
              </Button>
              <Button
                type="button"
                variant="destructive"
                size="sm"
                onClick={deleteSelectedRunOutputs}
                disabled={selectedRunDownloadCount === 0 || deleteRunOutputsMutation.isPending}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                {t('common.delete')} {selectedRunDownloadCount}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="flex min-h-0 flex-1 flex-col p-0">
            <ScrollArea className="min-h-0 flex-1">
              <div className="space-y-1 p-2">
                {runs.length === 0 ? (
                  <p className="p-2 text-sm text-muted-foreground">{t('assets.noCompletedRuns')}</p>
                ) : (
                  runs.map((run) => (
                    <RunButton
                      key={run.run_id}
                      run={run}
                      selected={selectedRunId === run.run_id}
                      downloadSelected={selectedRunDownloadIds.has(run.run_id)}
                      onToggleDownload={() => toggleRunDownloadSelection(run.run_id)}
                      onClick={() => {
                        setSelectedRunId(run.run_id)
                        setSelectedDocumentId(null)
                      }}
                      onDeleteOutputs={() => deleteSingleRunOutputs(run.run_id)}
                    />
                  ))
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>

        <Card className="flex min-h-[520px] flex-col xl:h-[calc(100vh-12rem)] xl:min-h-[620px]">
          <CardHeader className="space-y-3 pb-3">
            <div className="flex items-center justify-between gap-3">
              <CardTitle className="flex items-center gap-2 text-sm">
                <SplitSquareHorizontal className="h-4 w-4" />
                {t('assets.title')}
              </CardTitle>
              <Badge variant="outline">{selectedDownloadCount} / {splitDocuments.length}</Badge>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <Metric label={t('common.mainDocument')} value={mainDocumentCount} />
              <Metric label={t('common.childDocuments')} value={extractedDocumentCount} />
              <Metric label={t('common.selected')} value={selectedDownloadCount} />
            </div>
            <div className="relative">
              <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder={t('assets.searchPlaceholder')}
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                className="pl-8"
              />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setSelectedDownloadIds(new Set(splitDocuments.map((document) => document.document_id)))}
                disabled={splitDocuments.length === 0}
              >
                <CheckSquare className="mr-2 h-4 w-4" />
                {t('common.selectAll')}
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setSelectedDownloadIds(new Set())}
                disabled={selectedDownloadCount === 0}
              >
                <Square className="mr-2 h-4 w-4" />
                {t('common.clearSelection')}
              </Button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant="default"
                size="sm"
                onClick={downloadAllDocuments}
                disabled={!selectedRunId || selectedDownloadCount === 0 || isDownloadingAll}
              >
                <Archive className="mr-2 h-4 w-4" />
                {t('assets.downloadSelected', { count: selectedDownloadCount })}
              </Button>
              <Button
                type="button"
                variant="destructive"
                size="sm"
                onClick={deleteSelectedDocuments}
                disabled={!selectedRunId || selectedDownloadCount === 0 || deleteDocumentsMutation.isPending}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                {t('assets.deleteSelected', { count: selectedDownloadCount })}
              </Button>
            </div>
            <div className="rounded-md border bg-muted/30 p-3 text-xs text-muted-foreground">
              {t('assets.outputOnlyHint')}
            </div>
          </CardHeader>
          <CardContent className="flex min-h-0 flex-1 flex-col p-0">
            <ScrollArea className="min-h-0 flex-1">
              {!selectedRunId ? (
                <p className="p-4 text-sm text-muted-foreground">{t('assets.selectRunFirst')}</p>
              ) : splitDocumentsQuery.isLoading ? (
                <p className="p-4 text-sm text-muted-foreground">{t('assets.loadingDocuments')}</p>
              ) : filteredDocuments.length === 0 ? (
                <p className="p-4 text-sm text-muted-foreground">{t('assets.noSearchResults')}</p>
              ) : (
                <div className="space-y-2 p-2">
                  {filteredDocuments.map(({ meta, content }) => {
                    const checked = selectedDownloadIds.has(meta.document_id)
                    return (
                      <div
                        key={meta.document_id}
                        className={`rounded-md border p-3 transition-colors ${
                          selectedDocumentId === meta.document_id
                            ? 'border-primary bg-primary/5'
                            : 'border-border hover:bg-muted/60'
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          <button
                            type="button"
                            className="mt-0.5 rounded text-muted-foreground hover:text-foreground"
                            onClick={() => toggleDownloadSelection(meta.document_id)}
                            aria-label={checked ? t('common.clearSelection') : t('common.selectAll')}
                          >
                            {checked ? <CheckSquare className="h-4 w-4" /> : <Square className="h-4 w-4" />}
                          </button>
                          <button
                            type="button"
                            className="min-w-0 flex-1 text-left"
                            onClick={() => setSelectedDocumentId(meta.document_id)}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <div className="min-w-0">
                                <div className="truncate text-sm font-medium">
                                  {meta.title || meta.filename || meta.document_id}
                                </div>
                                <div className="mt-1 flex flex-wrap items-center gap-1">
                                  <Badge variant={meta.kind === 'main' ? 'default' : 'secondary'} className="text-[11px]">
                                    {formatKind(meta.kind)}
                                  </Badge>
                                  {formatPageLabel(meta) && (
                                    <Badge variant="outline" className="text-[11px]">
                                      {formatPageLabel(meta)}
                                    </Badge>
                                  )}
                                </div>
                              </div>
                              <FileText className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                            </div>
                            <p className="mt-2 line-clamp-2 text-xs text-muted-foreground">
                              <HighlightedText
                                text={buildSearchSnippet(content, searchQuery) || meta.filename || meta.document_id}
                                query={searchQuery}
                              />
                            </p>
                          </button>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>

        <Card className="flex min-h-[520px] flex-col xl:h-[calc(100vh-12rem)] xl:min-h-[620px]">
          <CardHeader className="space-y-3 pb-3">
            <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
              <div className="min-w-0">
                <CardTitle className="truncate text-sm">
                  {selectedDocument?.title || selectedDocument?.filename || t('assets.documentContent')}
                </CardTitle>
                <div className="mt-1 flex flex-wrap items-center gap-1">
                  {selectedDocument && (
                    <>
                      <Badge variant={selectedDocument.kind === 'main' ? 'default' : 'secondary'}>
                        {formatKind(selectedDocument.kind)}
                      </Badge>
                      {formatPageLabel(selectedDocument) && (
                        <Badge variant="outline">{formatPageLabel(selectedDocument)}</Badge>
                      )}
                    </>
                  )}
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {selectedRunId && selectedDocumentId && (
                  <Button size="sm" variant="outline" asChild>
                    <a href={getSplitDocumentDownloadUrl(selectedRunId, selectedDocumentId, format)}>
                      <Download className="mr-2 h-4 w-4" />
                      {t('assets.downloadDocument')}
                    </a>
                  </Button>
                )}
                {selectedRunId && selectedDocument?.page_indices?.[0] !== undefined && (
                  <Button size="sm" variant="outline" asChild>
                    <Link to={`/viewer/${selectedRunId}?page=${selectedDocument.page_indices[0]}`}>
                      <Eye className="mr-2 h-4 w-4" />
                      Viewer
                    </Link>
                  </Button>
                )}
              </div>
            </div>
            {selectedRun && (
              <div className="rounded-md border bg-muted/35 p-3 text-xs text-muted-foreground">
                <div className="truncate">{t('assets.sourcePrefix', { source: selectedRun.source_name || getFileName(selectedRun.source_path) || selectedRun.doc_id })}</div>
                <div className="mt-1 font-mono">Run：{selectedRun.run_id}</div>
              </div>
            )}
            {qualityGate && <QualityGateSummary report={qualityGate} />}
            {selectedRunId && selectedDocument && (
              <SourcePreview
                docId={selectedRun?.doc_id}
                runId={selectedRunId}
                document={selectedDocument}
              />
            )}
          </CardHeader>
          <CardContent className="flex min-h-0 flex-1 flex-col p-0">
            <ScrollArea className="min-h-0 flex-1 border-t">
              {!selectedDocumentId ? (
                <p className="p-4 text-sm text-muted-foreground">{t('assets.selectDocumentFirst')}</p>
              ) : selectedDocumentQuery.isLoading ? (
                <p className="p-4 text-sm text-muted-foreground">{t('assets.loadingContent')}</p>
              ) : (
                <DocumentPreview content={selectedContent || t('assets.emptyContent')} query={searchQuery} />
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}



function QualityGateSummary({ report }: { report: QualityGateReport }) {
  const { t } = useI18n()
  const issues = report.issues ?? []
  const statusLabel = formatQualityGateStatus(report.status, t)
  const badgeVariant = report.status === 'pass' ? 'default' : report.status === 'warning' ? 'secondary' : 'destructive'

  return (
    <div className="rounded-md border bg-muted/20 p-3 text-xs">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Badge variant={badgeVariant}>{statusLabel}</Badge>
          <span className="text-muted-foreground">{t('assets.qualityScore', { score: Math.round((report.score ?? 0) * 100) })}</span>
        </div>
        <span className="text-muted-foreground">{t('assets.auditSummary', { issues: issues.length, audits: report.vlm_audits?.length ?? 0 })}</span>
      </div>
      {issues.length > 0 && (
        <div className="mt-2 space-y-1 text-muted-foreground">
          {issues.slice(0, 3).map((issue, index) => (
            <div key={`${issue.code}-${index}`} className="line-clamp-2">
              {formatIssueSeverity(issue.severity)} · {issue.message}
              {issue.page_idx !== null && issue.page_idx !== undefined ? ` (${t('common.page')} ${issue.page_idx + 1})` : ''}
            </div>
          ))}
          {issues.length > 3 && <div>{t('assets.moreIssues', { count: issues.length - 3 })}</div>}
        </div>
      )}
    </div>
  )
}

function formatQualityGateStatus(status: string, t: ReturnType<typeof useI18n>['t']) {
  if (status === 'pass') return t('assets.qualityPass')
  if (status === 'warning') return t('assets.qualityWarning')
  if (status === 'needs_review') return t('assets.qualityReview')
  return t('assets.qualityUnknown')
}

function formatIssueSeverity(severity: string) {
  if (severity === 'high') return 'High'
  if (severity === 'medium') return 'Medium'
  if (severity === 'warning') return 'Warning'
  return severity
}

function SourcePreview({
  docId,
  runId,
  document,
}: {
  docId?: string
  runId: string
  document: SplitDocumentMeta
}) {
  const { t } = useI18n()
  const pageImagePath = document.page_image_path
  const assetPath = document.asset_path
  const imagePath = pageImagePath || assetPath
  const pageLabel = formatPageLabel(document)

  if (!docId || !imagePath) {
    return (
      <div className="rounded-md border bg-muted/20 p-3 text-xs text-muted-foreground">
        {t('assets.noPageImage', { pageLabel: pageLabel ? `${pageLabel} · ` : '' })}
      </div>
    )
  }

  return (
    <div className="overflow-hidden rounded-md border bg-muted/20">
      <div className="border-b px-3 py-2 text-xs text-muted-foreground">{t('assets.sourcePreview', { pageLabel: pageLabel ? ` · ${pageLabel}` : '' })}</div>
      <div className="flex max-h-56 items-center justify-center bg-background p-2">
        <img
          src={getAssetUrl(docId, runId, imagePath)}
          alt={document.title || document.document_id}
          className="max-h-52 max-w-full object-contain"
        />
      </div>
    </div>
  )
}

function RunButton({
  run,
  selected,
  downloadSelected,
  onToggleDownload,
  onClick,
  onDeleteOutputs,
}: {
  run: OutputRunSummary
  selected: boolean
  downloadSelected: boolean
  onToggleDownload: () => void
  onClick: () => void
  onDeleteOutputs: () => void
}) {
  const { t } = useI18n()
  return (
    <div
      className={`flex w-full items-start gap-2 rounded-md p-2 text-sm transition-colors ${
        selected ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'
      }`}
    >
      <button
        type="button"
        className={`mt-0.5 rounded ${selected ? 'text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`}
        onClick={(event) => {
          event.stopPropagation()
          onToggleDownload()
        }}
        aria-label={downloadSelected ? t('common.clearSelection') : t('common.selectAll')}
      >
        {downloadSelected ? <CheckSquare className="h-4 w-4" /> : <Square className="h-4 w-4" />}
      </button>
      <button type="button" className="min-w-0 flex-1 text-left" onClick={onClick}>
        <div className="truncate font-medium">{run.source_name || getFileName(run.source_path) || run.doc_id}</div>
        <div className="mt-1 flex items-center justify-between gap-2 text-xs opacity-75">
          <span className="font-mono">{run.run_id.slice(0, 12)}...</span>
          <span>{run.profile}</span>
        </div>
      </button>
      <button
        type="button"
        className={`mt-0.5 rounded ${selected ? 'text-primary-foreground/85 hover:text-primary-foreground' : 'text-muted-foreground hover:text-destructive'}`}
        onClick={(event) => {
          event.stopPropagation()
          onDeleteOutputs()
        }}
        aria-label={t('assets.confirmDeleteRunOutput')}
        title={t('assets.confirmDeleteRunOutput')}
      >
        <Trash2 className="h-4 w-4" />
      </button>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border bg-background px-2 py-2">
      <div className="text-base font-semibold leading-none">{value}</div>
      <div className="mt-1 text-muted-foreground">{label}</div>
    </div>
  )
}

function formatKind(kind: string) {
  const labels: Record<string, string> = {
    main: 'Main',
    form: 'Form',
    figure: 'Figure',
    table: 'Table',
    attachment: 'Attachment',
    contract: 'Contract',
  }
  return labels[kind] ?? kind
}

function formatPageLabel(document: SplitDocumentMeta) {
  if (document.page_label) return document.page_label
  if (!document.page_indices || document.page_indices.length === 0) return ''
  const pages = document.page_indices.map((page) => page + 1)
  if (pages.length === 1) return `p.${pages[0]}`
  return `p.${pages[0]}-${pages[pages.length - 1]}`
}

function buildSearchSnippet(content: string, query: string) {
  const normalized = stripMarkdown(content)
  const term = query.trim().toLowerCase()
  if (!term) return normalized.slice(0, 180)

  const index = normalized.toLowerCase().indexOf(term)
  if (index < 0) return normalized.slice(0, 180)

  const start = Math.max(index - 70, 0)
  const end = Math.min(index + term.length + 110, normalized.length)
  const prefix = start > 0 ? '...' : ''
  const suffix = end < normalized.length ? '...' : ''
  return `${prefix}${normalized.slice(start, end)}${suffix}`
}

function stripMarkdown(content: string) {
  return content
    .replace(/^#.+$/gm, '')
    .replace(/[#*_`>\-|]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function DocumentPreview({ content, query }: { content: string; query: string }) {
  const blocks = content.split(/\n{2,}/).filter((block) => block.trim().length > 0)

  return (
    <div className="space-y-3 p-4 text-sm leading-6">
      {blocks.map((block, index) => {
        const trimmed = block.trim()
        const isHeading = trimmed.startsWith('#')
        return (
          <div
            key={`${index}-${trimmed.slice(0, 24)}`}
            className={isHeading ? 'font-semibold text-foreground' : 'whitespace-pre-wrap break-words text-foreground/90'}
          >
            <HighlightedText text={trimmed.replace(/^#+\s*/, '')} query={query} />
          </div>
        )
      })}
    </div>
  )
}

function HighlightedText({ text, query }: { text: string; query: string }) {
  const term = query.trim()
  if (!term) return <>{text}</>

  const lowerText = text.toLowerCase()
  const lowerTerm = term.toLowerCase()
  const parts: Array<{ text: string; match: boolean }> = []
  let cursor = 0
  let index = lowerText.indexOf(lowerTerm)

  while (index >= 0) {
    if (index > cursor) {
      parts.push({ text: text.slice(cursor, index), match: false })
    }
    parts.push({ text: text.slice(index, index + term.length), match: true })
    cursor = index + term.length
    index = lowerText.indexOf(lowerTerm, cursor)
  }

  if (cursor < text.length) {
    parts.push({ text: text.slice(cursor), match: false })
  }

  return (
    <>
      {parts.map((part, index) =>
        part.match ? (
          <mark key={index} className="rounded bg-yellow-200 px-0.5 text-yellow-950">
            {part.text}
          </mark>
        ) : (
          <span key={index}>{part.text}</span>
        )
      )}
    </>
  )
}

function getFileName(path: string) {
  return path.split(/[\\/]/).pop() || path
}
