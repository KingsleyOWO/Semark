import { useState, useMemo, useEffect } from 'react'
import { useParams, Link, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import ReactMarkdown from 'react-markdown'
import {
  getRun,
  getDocumentIR,
  getSourceMap,
  getOutput,
  getSplitDocument,
  getSplitDocuments,
  getQuality,
  getAssetUrl,
  getAssetsIndex,
  getEnrichments,
} from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { AssetEntry, Block, EnrichmentEntry, PageInfo, SplitDocumentMeta } from '@/types/api'
import { useI18n } from '@/lib/i18n'
import {
  ArrowLeft,
  FileText,
  Image as ImageIcon,
  Info,
  Sparkles,
  AlertTriangle,
  Copy,
  Check,
} from 'lucide-react'

export function Viewer() {
  const { t } = useI18n()
  const { runId } = useParams<{ runId: string }>()
  const [searchParams] = useSearchParams()
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null)
  const [currentPage, setCurrentPage] = useState(0)
  const [mdView] = useState<'source'>('source')
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null)
  const [initialized, setInitialized] = useState(false)
  const [copiedField, setCopiedField] = useState<string | null>(null)

  // Queries
  const { data: run } = useQuery({
    queryKey: ['run', runId],
    queryFn: () => getRun(runId!),
    enabled: !!runId,
  })

  const { data: documentIR } = useQuery({
    queryKey: ['documentIR', runId],
    queryFn: () => getDocumentIR(runId!),
    enabled: !!runId,
  })

  const { data: sourceMap } = useQuery({
    queryKey: ['sourceMap', runId],
    queryFn: () => getSourceMap(runId!),
    enabled: !!runId,
  })

  const { data: mdContent } = useQuery({
    queryKey: ['output', runId, mdView],
    queryFn: () => getOutput(runId!, mdView),
    enabled: !!runId,
    retry: false,
  })

  const { data: splitDocumentsData } = useQuery({
    queryKey: ['splitDocuments', runId],
    queryFn: () => getSplitDocuments(runId!),
    enabled: !!runId,
    retry: false,
  })

  const { data: quality } = useQuery({
    queryKey: ['quality', runId],
    queryFn: () => getQuality(runId!),
    enabled: !!runId,
    retry: false,
  })

  const splitDocuments = useMemo(
    () => splitDocumentsData?.documents ?? [],
    [splitDocumentsData]
  )

  useEffect(() => {
    if (!selectedDocumentId && splitDocuments.length > 0) {
      setSelectedDocumentId(splitDocuments[0].document_id)
    }
  }, [selectedDocumentId, splitDocuments])

  const { data: selectedSplitDocument } = useQuery({
    queryKey: ['splitDocument', runId, selectedDocumentId],
    queryFn: () => getSplitDocument(runId!, selectedDocumentId!),
    enabled: !!runId && !!selectedDocumentId,
    retry: false,
  })

  const { data: assets } = useQuery({
    queryKey: ['assets', runId],
    queryFn: () => getAssetsIndex(runId!),
    enabled: !!runId,
  })

  const { data: enrichmentsData } = useQuery({
    queryKey: ['enrichments', runId],
    queryFn: () => getEnrichments(runId!),
    enabled: !!runId,
  })

  // Initialize from URL params when documentIR loads
  useEffect(() => {
    if (initialized || !documentIR) return

    const pageParam = searchParams.get('page')
    const blockParam = searchParams.get('block')

    if (pageParam !== null) {
      const pageIdx = parseInt(pageParam, 10)
      if (!isNaN(pageIdx) && pageIdx >= 0 && pageIdx < documentIR.pages.length) {
        setCurrentPage(pageIdx)
      }
    }

    if (blockParam) {
      const block = documentIR.blocks.find((b) => b.block_id === blockParam)
      if (block) {
        setSelectedBlockId(blockParam)
        if (pageParam === null) {
          setCurrentPage(block.page_idx)
        }
      }
    }

    setInitialized(true)
  }, [documentIR, searchParams, initialized])

  // Computed values
  const pages = useMemo(() => documentIR?.pages ?? [], [documentIR])
  const blocks = useMemo(() => documentIR?.blocks ?? [], [documentIR])
  const enrichments = useMemo(() => enrichmentsData?.enrichments ?? [], [enrichmentsData])

  const currentPageBlocks = useMemo(
    () => blocks.filter((b) => b.page_idx === currentPage),
    [blocks, currentPage]
  )

  const currentPageInfo = useMemo(
    () => pages.find((p) => p.page_idx === currentPage),
    [pages, currentPage]
  )

  const selectedDocument = selectedSplitDocument?.document
  const selectedDocumentContent = selectedSplitDocument?.content ?? mdContent ?? ''
  const selectedDocumentAnchors =
    selectedDocument?.document_id === 'main' ? sourceMap?.md_anchors ?? [] : []
  const selectedDocumentPages = useMemo(
    () => selectedDocument?.page_indices ?? [],
    [selectedDocument]
  )
  const selectedDocumentBlocks = useMemo(
    () =>
      selectedDocumentPages.length > 0
        ? blocks.filter((block) => selectedDocumentPages.includes(block.page_idx))
        : blocks,
    [blocks, selectedDocumentPages]
  )
  const currentPageAssets = useMemo(
    () => (assets ?? []).filter((asset) => asset.page_idx === currentPage),
    [assets, currentPage]
  )
  const currentPageEnrichments = useMemo(
    () =>
      enrichments.filter((entry) => {
        const inputPage = entry.input?.page_idx
        const evidencePage = entry.evidence?.page_idx
        return inputPage === currentPage || evidencePage === currentPage
      }),
    [enrichments, currentPage]
  )

  // Selected block data
  const selectedBlock = useMemo(
    () => blocks.find((b) => b.block_id === selectedBlockId),
    [blocks, selectedBlockId]
  )

  const selectedEnrichment = useMemo(
    () => enrichments.find((e) => e.block_id === selectedBlockId),
    [enrichments, selectedBlockId]
  )

  const selectedAsset = useMemo(
    () => assets?.find((a) => a.block_id === selectedBlockId),
    [assets, selectedBlockId]
  )

  const activeAsset = selectedAsset ?? currentPageAssets[0]
  const activeEnrichment = selectedEnrichment ?? currentPageEnrichments[0]
  const reviewIssues = useMemo(
    () =>
      buildReviewIssues({
        t,
        splitDocuments,
        blocks,
        assets: assets ?? [],
        enrichments,
        qualityWarnings: quality?.warnings ?? [],
      }),
    [splitDocuments, blocks, assets, enrichments, quality, t]
  )

  // Handlers
  const handleBlockClick = (blockId: string) => {
    setSelectedBlockId(blockId)
    const block = blocks.find((b) => b.block_id === blockId)
    if (block && block.page_idx !== currentPage) {
      setCurrentPage(block.page_idx)
    }
  }

  const handleMdClick = (anchorId: string) => {
    const anchor = sourceMap?.md_anchors.find((a) => a.anchor_id === anchorId)
    if (anchor && anchor.block_ids.length > 0) {
      setSelectedBlockId(anchor.block_ids[0])
      const firstBlock = blocks.find((b) => anchor.block_ids.includes(b.block_id))
      if (firstBlock) {
        setCurrentPage(firstBlock.page_idx)
      }
    }
  }

  const handleDocumentSelect = (documentId: string) => {
    setSelectedDocumentId(documentId)
    const document = splitDocuments.find((item) => item.document_id === documentId)
    const firstPage = document?.page_indices?.[0]
    if (typeof firstPage === 'number') {
      setCurrentPage(firstPage)
    }
  }

  const copyToClipboard = async (text: string, field: string) => {
    await navigator.clipboard.writeText(text)
    setCopiedField(field)
    setTimeout(() => setCopiedField(null), 2000)
  }

  if (!runId) {
    return <div>{t('viewer.invalidRun')}</div>
  }

  return (
    <div className="flex min-h-[calc(100vh-5.5rem)] flex-col xl:h-[calc(100vh-5.5rem)] xl:overflow-hidden">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3 border-b p-3 shrink-0">
        <Button variant="ghost" size="sm" asChild>
          <Link to="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            {t('common.back')}
          </Link>
        </Button>
        <div className="min-w-0 flex-1">
          <h1 className="text-lg font-bold">{t('viewer.title')}</h1>
          <p className="text-xs text-muted-foreground">
            {t('viewer.runMeta', { run: runId?.slice(0, 16) ?? '', profile: run?.profile ? t(`profile.${run.profile}`) : '' })}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
          <span>{t('viewer.pageCount', { count: pages.length })}</span>
          <span>·</span>
          <span>{t('viewer.blockCount', { count: blocks.length })}</span>
          <span>·</span>
          <span>{t('viewer.assetCount', { count: assets?.length ?? 0 })}</span>
        </div>
      </div>

      {/* Three-column layout */}
      <div className="grid min-h-0 flex-1 gap-3 p-2 xl:grid-cols-[minmax(360px,0.95fr)_minmax(520px,1.35fr)_minmax(360px,0.9fr)] 2xl:grid-cols-[minmax(440px,1fr)_minmax(720px,1.35fr)_minmax(440px,0.95fr)]">
        {/* Left: Split document markdown view (4 cols) */}
        <div className="flex min-h-[420px] flex-col rounded-lg border xl:min-h-0">
          <div className="flex items-center justify-between p-2 border-b shrink-0">
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              <span className="text-sm font-medium">{t('viewer.splitDocuments')}</span>
            </div>
            <Badge variant="secondary" className="text-xs">
              {splitDocuments.length > 0 ? `${splitDocuments.length} ${t('common.files')}` : t('common.mainDocument')}
            </Badge>
          </div>
          {splitDocuments.length > 0 && (
            <div className="border-b p-2 shrink-0">
              <div className="flex max-h-48 flex-col gap-1 overflow-y-auto pr-1">
                {splitDocuments.map((document) => {
                  const pageLabel =
                    document.page_label ??
                    (document.page_indices?.length
                      ? document.page_indices.map((page) => `P${page + 1}`).join(', ')
                      : t('viewer.wholeDocument'))
                  return (
                    <button
                      key={document.document_id}
                      type="button"
                      onClick={() => handleDocumentSelect(document.document_id)}
                      className={[
                        'w-full rounded-md border px-2 py-2 text-left transition-colors',
                        selectedDocumentId === document.document_id
                          ? 'border-primary bg-primary/5'
                          : 'border-transparent hover:bg-muted',
                      ].join(' ')}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-sm font-medium">
                          {document.title || document.document_id}
                        </span>
                        <Badge variant={document.kind === 'main' ? 'secondary' : 'outline'} className="text-xs">
                          {document.kind === 'main' ? t('common.mainDocument') : document.kind}
                        </Badge>
                      </div>
                      <div className="mt-1 flex items-center justify-between gap-2 text-xs text-muted-foreground">
                        <span className="truncate">{document.filename}</span>
                        <span className="shrink-0">{pageLabel}</span>
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>
          )}
          {selectedDocument && (
            <div className="border-b p-2 text-xs text-muted-foreground shrink-0">
              <div className="flex items-center justify-between gap-2">
                <span className="truncate">{selectedDocument.filename}</span>
                <span className="shrink-0">
                  {selectedDocumentPages.length > 0
                    ? t('viewer.sourcePages', { pages: selectedDocumentPages.map((page) => page + 1).join(', ') })
                    : t('viewer.sourceAll')}
                </span>
              </div>
            </div>
          )}
          <ScrollArea className="flex-1">
            <div className="p-3 prose prose-sm max-w-none dark:prose-invert">
              <MarkdownWithAnchors
                content={selectedDocumentContent}
                sourceMap={selectedDocumentAnchors}
                selectedBlockId={selectedBlockId}
                onAnchorClick={handleMdClick}
              />
            </div>
          </ScrollArea>
        </div>

        {/* Center: Page view (5 cols) */}
        <div className="flex min-h-[520px] flex-col rounded-lg border xl:min-h-0">
          <div className="flex items-center justify-between p-2 border-b shrink-0">
            <div className="flex items-center gap-2">
              <ImageIcon className="h-4 w-4" />
              <span className="text-sm font-medium">
                Page {currentPage + 1} / {pages.length}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2"
                onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
                disabled={currentPage === 0}
              >
                Prev
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2"
                onClick={() => setCurrentPage((p) => Math.min(pages.length - 1, p + 1))}
                disabled={currentPage >= pages.length - 1}
              >
                Next
              </Button>
            </div>
          </div>
          <ScrollArea className="flex-1">
            <PageWithBboxOverlay
              pageInfo={currentPageInfo}
              blocks={currentPageBlocks}
              selectedBlockId={selectedBlockId}
              docId={documentIR?.doc_id ?? ''}
              runId={runId}
              onBlockClick={handleBlockClick}
            />
          </ScrollArea>
          {/* Block chips */}
          <div className="p-2 border-t shrink-0">
            <div className="flex flex-wrap gap-1">
              {currentPageBlocks.slice(0, 10).map((block) => (
                <Badge
                  key={block.block_id}
                  variant={selectedBlockId === block.block_id ? 'default' : 'outline'}
                  className="cursor-pointer text-xs"
                  onClick={() => handleBlockClick(block.block_id)}
                >
                  {block.type}
                </Badge>
              ))}
              {currentPageBlocks.length > 10 && (
                <Badge variant="secondary" className="text-xs">
                  +{currentPageBlocks.length - 10}
                </Badge>
              )}
            </div>
          </div>
        </div>

        {/* Right: Review panel (3 cols) */}
        <div className="flex min-h-[420px] flex-col rounded-lg border xl:min-h-0">
          <div className="flex items-center justify-between p-2 border-b shrink-0">
            <span className="text-sm font-medium">{t('viewer.tabs.review')}</span>
            <Badge variant={reviewIssues.length > 0 ? 'warning' : 'success'} className="text-xs">
              {reviewIssues.length > 0 ? `${reviewIssues.length} checks` : 'OK'}
            </Badge>
          </div>
          <Tabs defaultValue="documents" className="flex-1 flex flex-col min-h-0">
            <TabsList className="mx-2 mt-2 grid grid-cols-4 shrink-0">
              <TabsTrigger value="documents" className="text-xs">
                <FileText className="h-3 w-3 mr-1" />
                {t('viewer.tabs.documents')}
              </TabsTrigger>
              <TabsTrigger value="page" className="text-xs">
                <Info className="h-3 w-3 mr-1" />
                {t('viewer.tabs.pages')}
              </TabsTrigger>
              <TabsTrigger value="semantic" className="text-xs">
                <Sparkles className="h-3 w-3 mr-1" />
                {t('viewer.tabs.semantic')}
              </TabsTrigger>
              <TabsTrigger value="checks" className="text-xs">
                <AlertTriangle className="h-3 w-3 mr-1" />
                {t('viewer.tabs.review')}
              </TabsTrigger>
            </TabsList>

            <ScrollArea className="flex-1">
              <TabsContent value="documents" className="m-0 p-3">
                <div className="space-y-3 text-sm">
                  <div className="grid grid-cols-3 gap-2">
                    <Metric label={t('common.documents')} value={splitDocuments.length || 1} />
                    <Metric label={t('settings.forms')} value={splitDocuments.filter((document) => document.kind === 'form').length} />
                    <Metric label={t('viewer.assets')} value={assets?.length ?? 0} />
                  </div>

                  <div className="space-y-2">
                    {splitDocuments.length > 0 ? (
                      splitDocuments.map((document) => {
                        const selected = selectedDocumentId === document.document_id
                        return (
                          <button
                            key={document.document_id}
                            type="button"
                            onClick={() => handleDocumentSelect(document.document_id)}
                            className={[
                              'w-full rounded-md border p-2 text-left transition-colors',
                              selected ? 'border-primary bg-primary/5' : 'hover:bg-muted',
                            ].join(' ')}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="truncate text-xs font-medium">
                                {document.title || document.document_id}
                              </span>
                              <Badge variant={document.kind === 'main' ? 'secondary' : 'outline'} className="text-xs">
                                {document.kind === 'main' ? t('common.mainDocument') : document.kind}
                              </Badge>
                            </div>
                            <div className="mt-1 flex items-center justify-between gap-2 text-xs text-muted-foreground">
                              <span className="truncate">{document.filename ?? `${document.document_id}.md`}</span>
                              <span className="shrink-0">{formatPageLabel(document)}</span>
                            </div>
                          </button>
                        )
                      })
                    ) : (
                      <p className="text-sm text-muted-foreground">{t('viewer.noSplitIndex')}</p>
                    )}
                  </div>

                  <div className="border-t pt-3">
                    <div className="flex items-center justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-xs font-medium">
                          {selectedDocument?.title || t('common.mainDocument')}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {selectedDocumentPages.length > 0
                            ? t('viewer.sourcePages', { pages: selectedDocumentPages.map((page) => page + 1).join(', ') })
                            : t('viewer.sourceAll')}
                        </p>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2"
                        onClick={() => copyToClipboard(selectedDocumentContent, 'document')}
                      >
                        {copiedField === 'document' ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                      </Button>
                    </div>
                    <div className="mt-3 grid grid-cols-3 gap-2">
                      <Metric label={t('viewer.blocks')} value={selectedDocumentBlocks.length} />
                      <Metric
                        label={t('viewer.tables')}
                        value={selectedDocumentBlocks.filter((block) => block.type === 'table').length}
                      />
                      <Metric
                        label={t('viewer.images')}
                        value={selectedDocumentBlocks.filter((block) => block.type === 'image').length}
                      />
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="page" className="m-0 p-3">
                <div className="space-y-3 text-sm">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs font-medium">Page {currentPage + 1}</p>
                      <p className="text-xs text-muted-foreground">
                        {currentPageBlocks.length} blocks · {currentPageAssets.length} assets · {currentPageEnrichments.length} semantic results
                      </p>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {currentPageInfo?.page_image_path ? 'image ok' : 'no image'}
                    </Badge>
                  </div>

                  <div className="grid grid-cols-3 gap-2">
                    <Metric label={t('viewer.text')} value={currentPageBlocks.filter((block) => block.type === 'text').length} />
                    <Metric label={t('viewer.tables')} value={currentPageBlocks.filter((block) => block.type === 'table').length} />
                    <Metric label={t('viewer.images')} value={currentPageBlocks.filter((block) => block.type === 'image').length} />
                  </div>

                  <div className="space-y-1">
                    {currentPageBlocks.length > 0 ? (
                      currentPageBlocks.map((block) => (
                        <button
                          key={block.block_id}
                          type="button"
                          onClick={() => handleBlockClick(block.block_id)}
                          className={[
                            'w-full rounded border px-2 py-1.5 text-left text-xs transition-colors',
                            selectedBlockId === block.block_id ? 'border-primary bg-primary/5' : 'hover:bg-muted',
                          ].join(' ')}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="font-mono">{block.block_id}</span>
                            <Badge variant="outline" className="text-xs">{block.type}</Badge>
                          </div>
                          <p className="mt-1 line-clamp-2 text-muted-foreground">
                            {getBlockPreview(block)}
                          </p>
                        </button>
                      ))
                    ) : (
                      <p className="text-sm text-muted-foreground">{t('viewer.noBlocks')}</p>
                    )}
                  </div>

                  {selectedBlock && (
                    <div className="border-t pt-3">
                      <p className="text-xs font-medium">{t('viewer.selectedBlock')}</p>
                      <div className="mt-2 space-y-2 text-xs">
                        <div className="flex justify-between gap-2">
                          <span className="text-muted-foreground">{t('viewer.type')}</span>
                          <Badge variant="outline" className="text-xs">{selectedBlock.type}</Badge>
                        </div>
                        <div className="flex justify-between gap-2">
                          <span className="text-muted-foreground">{t('viewer.readingOrder')}</span>
                          <span>{selectedBlock.reading_order}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">BBox</span>
                          <p className="mt-1 font-mono">
                            [{selectedBlock.bbox_norm.map((value) => value.toFixed(0)).join(', ')}]
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="semantic" className="m-0 p-3">
                <div className="space-y-3 text-sm">
                  {activeAsset || activeEnrichment ? (
                    <>
                      {activeAsset && (
                        <div className="space-y-3">
                          <div className="flex items-center justify-between gap-2">
                            <div className="min-w-0">
                              <p className="truncate text-xs font-medium">{activeAsset.title}</p>
                              <p className="text-xs text-muted-foreground">Page {activeAsset.page_idx + 1}</p>
                            </div>
                            <Badge variant={activeAsset.needs_review ? 'warning' : 'secondary'} className="text-xs">
                              {activeAsset.type.replace('_asset', '')}
                            </Badge>
                          </div>
                          {activeAsset.asset_path && (
                            <div className="overflow-hidden rounded border">
                              <img
                                src={getAssetUrl(activeAsset.doc_id, activeAsset.run_id, activeAsset.asset_path)}
                                alt={activeAsset.title}
                                className="max-h-36 w-full object-contain"
                              />
                            </div>
                          )}
                          <SemanticText
                            label={t('viewer.retrievalText')}
                            value={activeAsset.retrieval_text}
                            onCopy={() => copyToClipboard(activeAsset.retrieval_text, 'retrieval')}
                            copied={copiedField === 'retrieval'}
                          />
                          {activeAsset.filling_guide && (
                            <SemanticText
                              label={t('viewer.fillingGuide')}
                              value={activeAsset.filling_guide}
                              onCopy={() => copyToClipboard(activeAsset.filling_guide ?? '', 'filling_guide')}
                              copied={copiedField === 'filling_guide'}
                            />
                          )}
                          {activeAsset.field_schema && activeAsset.field_schema.length > 0 && (
                            <div>
                              <p className="text-xs text-muted-foreground">Fields ({activeAsset.field_schema.length})</p>
                              <div className="mt-1 space-y-1">
                                {activeAsset.field_schema.slice(0, 8).map((field) => (
                                  <div key={field.field_name} className="flex justify-between gap-2 text-xs">
                                    <span className="truncate font-mono">{field.field_name}</span>
                                    <span className="shrink-0 text-muted-foreground">{field.field_type || 'text'}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {activeEnrichment && (
                        <div className="border-t pt-3">
                          <div className="flex items-center justify-between gap-2">
                            <p className="text-xs font-medium">VLM / Semantic Output</p>
                            <Badge variant={activeEnrichment.quality?.needs_review ? 'warning' : 'secondary'} className="text-xs">
                              {activeEnrichment.kind}
                            </Badge>
                          </div>
                          <p className="mt-1 text-xs text-muted-foreground">{activeEnrichment.model}</p>
                          <SemanticOutput output={activeEnrichment.output} />
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="rounded-md border border-dashed p-3 text-sm text-muted-foreground">
                      {t('viewer.noSemantic')}
                    </div>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="checks" className="m-0 p-3">
                <div className="space-y-3 text-sm">
                  {reviewIssues.length > 0 ? (
                    reviewIssues.map((issue) => (
                      <div key={`${issue.level}-${issue.title}`} className="rounded-md border p-2">
                        <div className="flex items-center justify-between gap-2">
                          <p className="text-xs font-medium">{issue.title}</p>
                          <Badge variant={issue.level === 'warning' ? 'warning' : 'secondary'} className="text-xs">
                            {issue.level === 'warning' ? t('common.warning') : t('common.info')}
                          </Badge>
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground">{issue.detail}</p>
                      </div>
                    ))
                  ) : (
                    <div className="rounded-md border border-dashed p-3 text-sm text-muted-foreground">
                      {t('viewer.noReviewIssues')}
                    </div>
                  )}

                  <div className="border-t pt-3">
                    <p className="text-xs font-medium">{t('viewer.debug')}</p>
                    <details className="mt-2 rounded-md border p-2 text-xs">
                      <summary className="cursor-pointer text-muted-foreground">{t('viewer.showBlockJson')}</summary>
                      <pre className="mt-2 max-h-48 overflow-auto rounded bg-muted p-2">
                        {selectedBlock ? JSON.stringify(selectedBlock, null, 2) : 'No block selected'}
                      </pre>
                    </details>
                    <details className="mt-2 rounded-md border p-2 text-xs">
                      <summary className="cursor-pointer text-muted-foreground">{t('viewer.showSemanticJson')}</summary>
                      <pre className="mt-2 max-h-48 overflow-auto rounded bg-muted p-2">
                        {activeEnrichment ? JSON.stringify(activeEnrichment, null, 2) : 'No semantic result'}
                      </pre>
                    </details>
                  </div>
                </div>
              </TabsContent>
            </ScrollArea>
          </Tabs>
        </div>

      </div>
    </div>
  )
}


interface ReviewIssue {
  level: 'info' | 'warning'
  title: string
  detail: string
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-md border bg-muted/30 px-2 py-1.5">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="text-sm font-semibold">{value}</div>
    </div>
  )
}

function formatPageLabel(document: SplitDocumentMeta) {
  if (document.page_label) return document.page_label
  if (document.page_indices?.length) {
    return document.page_indices.map((page) => `P${page + 1}`).join(', ')
  }
  return 'Whole document'
}

function getBlockPreview(block: Block) {
  const text = block.payload?.text
  if (typeof text === 'string' && text.trim()) return text.trim()
  const tableBody = block.payload?.table_body
  if (typeof tableBody === 'string' && tableBody.trim()) {
    return tableBody.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()
  }
  const imagePath = block.payload?.image_path
  if (typeof imagePath === 'string') return imagePath
  return `${block.type} block`
}

function buildReviewIssues({
  t,
  splitDocuments,
  blocks,
  assets,
  enrichments,
  qualityWarnings,
}: {
  t: ReturnType<typeof useI18n>['t']
  splitDocuments: SplitDocumentMeta[]
  blocks: Block[]
  assets: AssetEntry[]
  enrichments: EnrichmentEntry[]
  qualityWarnings: string[]
}): ReviewIssue[] {
  const issues: ReviewIssue[] = []
  const formDocuments = splitDocuments.filter((document) => document.kind === 'form')
  const formAssets = assets.filter((asset) => asset.type === 'form_asset')
  const tableBlocks = blocks.filter((block) => block.type === 'table')
  const reviewAssets = assets.filter((asset) => asset.needs_review)
  const reviewEnrichments = enrichments.filter((entry) => entry.quality?.needs_review)

  if (splitDocuments.length <= 1 && (formAssets.length > 0 || tableBlocks.length > 0)) {
    issues.push({
      level: 'warning',
      title: t('viewer.splitMayBeInsufficient'),
      detail: t('viewer.splitMayBeInsufficientDetail', {
        documents: splitDocuments.length || 1,
        forms: formAssets.length,
        tables: tableBlocks.length,
      }),
    })
  }

  const documentsWithoutPages = splitDocuments.filter(
    (document) => document.kind !== 'main' && !document.page_indices?.length
  )
  if (documentsWithoutPages.length > 0) {
    issues.push({
      level: 'warning',
      title: t('viewer.childMissingPages'),
      detail: t('viewer.childMissingPagesDetail', { count: documentsWithoutPages.length }),
    })
  }

  if (formDocuments.length > 0) {
    issues.push({
      level: 'info',
      title: t('viewer.formsSplit'),
      detail: t('viewer.formsSplitDetail', { count: formDocuments.length }),
    })
  }

  if (reviewAssets.length > 0) {
    issues.push({
      level: 'warning',
      title: t('viewer.assetNeedsReview'),
      detail: t('viewer.assetNeedsReviewDetail', { count: reviewAssets.length }),
    })
  }

  if (reviewEnrichments.length > 0) {
    issues.push({
      level: 'warning',
      title: t('viewer.vlmNeedsReview'),
      detail: t('viewer.vlmNeedsReviewDetail', { count: reviewEnrichments.length }),
    })
  }

  qualityWarnings.forEach((warning) => {
    issues.push({ level: 'warning', title: 'Quality warning', detail: warning })
  })

  return issues
}

function SemanticText({
  label,
  value,
  onCopy,
  copied,
}: {
  label: string
  value: string
  onCopy: () => void
  copied: boolean
}) {
  return (
    <div>
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-muted-foreground">{label}</p>
        <Button variant="ghost" size="sm" className="h-6 px-1" onClick={onCopy}>
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
        </Button>
      </div>
      <p className="mt-1 whitespace-pre-line text-xs text-muted-foreground">{value}</p>
    </div>
  )
}

function SemanticOutput({ output }: { output: Record<string, unknown> }) {
  const { t } = useI18n()
  const visibleEntries = Object.entries(output).filter(([, value]) => {
    if (value == null) return false
    if (Array.isArray(value)) return value.length > 0
    return String(value).trim().length > 0
  })

  if (visibleEntries.length === 0) {
    return <p className="mt-2 text-xs text-muted-foreground">{t('viewer.noSemanticFields')}</p>
  }

  return (
    <div className="mt-2 space-y-2">
      {visibleEntries.slice(0, 6).map(([key, value]) => (
        <div key={key}>
          <p className="text-xs text-muted-foreground">{key}</p>
          {Array.isArray(value) ? (
            <div className="mt-1 flex flex-wrap gap-1">
              {value.slice(0, 8).map((item, index) => (
                <Badge key={`${key}-${index}`} variant="outline" className="text-xs">
                  {String(item)}
                </Badge>
              ))}
            </div>
          ) : typeof value === 'object' ? (
            <pre className="mt-1 max-h-32 overflow-auto rounded bg-muted p-2 text-xs">
              {JSON.stringify(value, null, 2)}
            </pre>
          ) : (
            <p className="mt-1 whitespace-pre-line text-xs">{String(value)}</p>
          )}
        </div>
      ))}
    </div>
  )
}

// Markdown with clickable anchors
interface MarkdownWithAnchorsProps {
  content: string
  sourceMap: Array<{ anchor_id: string; md_range: number[]; block_ids: string[] }>
  selectedBlockId: string | null
  onAnchorClick: (anchorId: string) => void
}

function MarkdownWithAnchors({
  content,
  sourceMap,
  selectedBlockId,
  onAnchorClick,
}: MarkdownWithAnchorsProps) {
  // Find which anchor contains the selected block
  const selectedAnchor = sourceMap.find((a) =>
    selectedBlockId ? a.block_ids.includes(selectedBlockId) : false
  )

  return (
    <div className="markdown-content">
      <ReactMarkdown
        components={{
          p: ({ children, ...props }) => {
            // Try to find matching anchor by content position
            const textContent = String(children)
            const matchingAnchor = sourceMap.find((a) => {
              const anchorText = content.slice(a.md_range[0], a.md_range[1])
              return anchorText.includes(textContent.slice(0, 50))
            })

            const isSelected = matchingAnchor && matchingAnchor === selectedAnchor

            return (
              <p
                {...props}
                className={`cursor-pointer rounded px-1 -mx-1 transition-colors ${
                  isSelected
                    ? 'bg-primary/20 ring-1 ring-primary'
                    : 'hover:bg-accent/50'
                }`}
                onClick={() => {
                  if (matchingAnchor) {
                    onAnchorClick(matchingAnchor.anchor_id)
                  }
                }}
              >
                {children}
              </p>
            )
          },
          img: ({ src, alt, ...props }) => (
            <img
              {...props}
              src={src}
              alt={alt}
              className="rounded border cursor-pointer hover:ring-2 hover:ring-primary"
            />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

// Page with bbox overlay
interface PageWithBboxOverlayProps {
  pageInfo?: PageInfo
  blocks: Block[]
  selectedBlockId: string | null
  docId: string
  runId: string
  onBlockClick: (blockId: string) => void
}

function PageWithBboxOverlay({
  pageInfo,
  blocks,
  selectedBlockId,
  docId,
  runId,
  onBlockClick,
}: PageWithBboxOverlayProps) {
  const imageUrl = pageInfo?.page_image_path
    ? getAssetUrl(docId, runId, pageInfo.page_image_path)
    : null

  const width = pageInfo?.width_px ?? 800
  const height = pageInfo?.height_px ?? 1100

  return (
    <div className="relative p-2">
      <div
        className="relative bg-muted rounded overflow-hidden"
        style={{ aspectRatio: `${width} / ${height}` }}
      >
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={`Page ${pageInfo?.page_idx}`}
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-muted-foreground text-sm">
            No page image
          </div>
        )}

        {/* Bbox overlays */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox={`0 0 1000 ${(height / width) * 1000}`}
          preserveAspectRatio="none"
        >
          {blocks.map((block) => {
            const [x0, y0, x1, y1] = block.bbox_norm
            const isSelected = selectedBlockId === block.block_id

            return (
              <rect
                key={block.block_id}
                x={x0}
                y={y0}
                width={x1 - x0}
                height={y1 - y0}
                fill={isSelected ? 'rgba(59, 130, 246, 0.3)' : 'transparent'}
                stroke={isSelected ? 'rgb(59, 130, 246)' : 'rgba(100, 100, 100, 0.3)'}
                strokeWidth={isSelected ? 3 : 1}
                className="pointer-events-auto cursor-pointer hover:fill-primary/20"
                onClick={() => onBlockClick(block.block_id)}
              />
            )
          })}
        </svg>
      </div>
    </div>
  )
}
