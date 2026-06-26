/**
 * D5: Org Chart Panel - 組織圖Debug面板
 *
 * 四個頁籤：
 * - Result: 最終輸出 (render_md + warnings)
 * - Graph: Node/Edge 表格
 * - Candidates: Candidates排錯 (VLM Raw output vs Validated)
 * - Debug: 檔案瀏覽器
 */

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import ReactMarkdown from 'react-markdown'
import {
  getOrgChart,
  getOrgChartDebugIndex,
  getOrgChartDebugFile,
  getOrgChartDebugFileText,
  type OrgChartResponse,
  type OrgChartDebugIndex,
} from '@/lib/api'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  FileText,
  Network,
  Search,
  FolderOpen,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ChevronRight,
  RefreshCw,
} from 'lucide-react'

interface OrgChartPanelProps {
  runId: string
}

export function OrgChartPanel({ runId }: OrgChartPanelProps) {
  const [activeTab, setActiveTab] = useState('result')
  const [selectedFile, setSelectedFile] = useState<string | null>(null)

  // 取得組織圖Result
  const { data: orgChart, isLoading, error, refetch } = useQuery({
    queryKey: ['orgChart', runId],
    queryFn: () => getOrgChart(runId),
    enabled: !!runId,
    retry: false,
  })

  // 取得 debug 檔案索引
  const { data: debugIndex } = useQuery({
    queryKey: ['orgChartDebugIndex', runId],
    queryFn: () => getOrgChartDebugIndex(runId),
    enabled: !!runId && activeTab === 'debug',
    retry: false,
  })

  // 取得選中的 debug 檔案內容
  const { data: debugFileContent, isLoading: isLoadingFile } = useQuery({
    queryKey: ['orgChartDebugFile', runId, selectedFile],
    queryFn: async () => {
      if (!selectedFile) return null
      if (selectedFile.endsWith('.md')) {
        return getOrgChartDebugFileText(runId, selectedFile)
      }
      return getOrgChartDebugFile(runId, selectedFile)
    },
    enabled: !!runId && !!selectedFile,
    retry: false,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
        Loading...
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-4">
        <XCircle className="h-8 w-8 mb-2 text-destructive" />
        <p className="text-sm">Unable to load org chart data</p>
        <p className="text-xs text-muted-foreground mt-1">{String(error)}</p>
        <Button variant="outline" size="sm" className="mt-4" onClick={() => refetch()}>
          Retry
        </Button>
      </div>
    )
  }

  if (!orgChart) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        No org chart data
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <TabsList className="mx-2 mt-2 shrink-0 grid grid-cols-4">
          <TabsTrigger value="result" className="text-xs">
            <FileText className="h-3 w-3 mr-1" />
            Result
          </TabsTrigger>
          <TabsTrigger value="graph" className="text-xs">
            <Network className="h-3 w-3 mr-1" />
            Graph
          </TabsTrigger>
          <TabsTrigger value="candidates" className="text-xs">
            <Search className="h-3 w-3 mr-1" />
            Candidates
          </TabsTrigger>
          <TabsTrigger value="debug" className="text-xs">
            <FolderOpen className="h-3 w-3 mr-1" />
            Debug
          </TabsTrigger>
        </TabsList>

        <ScrollArea className="flex-1">
          {/* Tab A: Result */}
          <TabsContent value="result" className="m-0 p-3">
            <ResultTab orgChart={orgChart} />
          </TabsContent>

          {/* Tab B: Graph */}
          <TabsContent value="graph" className="m-0 p-3">
            <GraphTab orgChart={orgChart} />
          </TabsContent>

          {/* Tab C: Candidates */}
          <TabsContent value="candidates" className="m-0 p-3">
            <CandidatesTab orgChart={orgChart} runId={runId} />
          </TabsContent>

          {/* Tab D: Debug */}
          <TabsContent value="debug" className="m-0 p-3">
            <DebugTab
              debugIndex={debugIndex}
              selectedFile={selectedFile}
              onSelectFile={setSelectedFile}
              fileContent={debugFileContent}
              isLoadingFile={isLoadingFile}
            />
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  )
}

// ========== Tab A: Result ==========

function ResultTab({ orgChart }: { orgChart: OrgChartResponse }) {
  const hasGraph = orgChart.graph && orgChart.graph.nodes && orgChart.graph.nodes.length > 0
  const hasWarnings = orgChart.warnings && orgChart.warnings.length > 0
  const decisionTrace = orgChart.decision_trace || {}

  // 如果沒有找到資料
  if (!orgChart.found) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
        <XCircle className="h-8 w-8 mb-2" />
        <p className="text-sm">{orgChart.message || 'This run has no org chart result'}</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Decision Trace 摘要 */}
      <div className="bg-muted/50 rounded-lg p-3">
        <h4 className="text-xs font-medium text-muted-foreground mb-2">Processing decision</h4>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">BBox available:</span>
            {decisionTrace.bbox_available ? (
              <Badge variant="default" className="text-[10px] px-1 py-0">
                <CheckCircle className="h-2 w-2 mr-0.5" />Yes
              </Badge>
            ) : (
              <Badge variant="secondary" className="text-[10px] px-1 py-0">
                <XCircle className="h-2 w-2 mr-0.5" />No
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">VLM fallback:</span>
            {decisionTrace.used_vlm_fallback ? (
              <Badge variant="outline" className="text-[10px] px-1 py-0">Yes</Badge>
            ) : (
              <Badge variant="secondary" className="text-[10px] px-1 py-0">No</Badge>
            )}
          </div>
          {decisionTrace.skipped_vlm2 != null && (
            <div className="col-span-2 flex items-center gap-1">
              <span className="text-muted-foreground">Skipped VLM#2:</span>
              <Badge variant="secondary" className="text-[10px] px-1 py-0">
                {String(decisionTrace.skipped_vlm2)}
              </Badge>
            </div>
          )}
        </div>
      </div>

      {/* Warnings */}
      {hasWarnings && (
        <div className="space-y-1">
          <h4 className="text-xs font-medium text-muted-foreground flex items-center gap-1">
            <AlertTriangle className="h-3 w-3 text-yellow-500" />
            Warnings ({orgChart.warnings.length})
          </h4>
          <div className="space-y-1">
            {orgChart.warnings.map((w, i) => (
              <div key={i} className="text-xs bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200 rounded px-2 py-1">
                {w}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Render MD */}
      <div className="space-y-2">
        <h4 className="text-xs font-medium text-muted-foreground">
          {hasGraph ? 'Org chart Markdown' : 'Unable to generate org chart'}
        </h4>
        {orgChart.render_md ? (
          <div className="prose prose-sm max-w-none dark:prose-invert bg-muted/30 rounded-lg p-3">
            <ReactMarkdown>{orgChart.render_md}</ReactMarkdown>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground italic">
            {hasGraph ? 'No Markdown output' : 'Org chart processing failed. Check the Debug tab.'}
          </div>
        )}
      </div>
    </div>
  )
}

// ========== Tab B: Graph ==========

function GraphTab({ orgChart }: { orgChart: OrgChartResponse }) {
  const graph = orgChart.graph

  if (!graph || graph.nodes.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-8">
        No node data
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Graph Title */}
      {graph.title && (
        <div className="bg-muted/50 rounded-lg p-2">
          <div className="text-sm font-medium">{graph.title}</div>
          {graph.date && <div className="text-xs text-muted-foreground">Date: {graph.date}</div>}
        </div>
      )}

      {/* Nodes */}
      <div>
        <h4 className="text-xs font-medium text-muted-foreground mb-2">
          Nodes ({graph.nodes.length})
        </h4>
        <div className="border rounded-lg overflow-hidden max-h-48 overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="bg-muted/50 sticky top-0">
              <tr>
                <th className="px-2 py-1 text-left font-medium">ID</th>
                <th className="px-2 py-1 text-left font-medium">Label</th>
                <th className="px-2 py-1 text-left font-medium">Type</th>
                <th className="px-2 py-1 text-left font-medium">Parent</th>
              </tr>
            </thead>
            <tbody>
              {graph.nodes.map((node) => (
                <tr key={node.id} className="border-t hover:bg-muted/30">
                  <td className="px-2 py-1 font-mono text-muted-foreground">{node.id}</td>
                  <td className="px-2 py-1">{node.label}</td>
                  <td className="px-2 py-1 text-muted-foreground">{node.category || '-'}</td>
                  <td className="px-2 py-1 font-mono text-muted-foreground">{node.chosen_parent || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Edges */}
      <div>
        <h4 className="text-xs font-medium text-muted-foreground mb-2">
          Edges ({graph.edges.length})
        </h4>
        {graph.edges.length > 0 ? (
          <div className="border rounded-lg overflow-hidden max-h-32 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="bg-muted/50 sticky top-0">
                <tr>
                  <th className="px-2 py-1 text-left font-medium">Parent</th>
                  <th className="px-2 py-1 text-center font-medium"></th>
                  <th className="px-2 py-1 text-left font-medium">Child</th>
                  <th className="px-2 py-1 text-left font-medium">Relation</th>
                </tr>
              </thead>
              <tbody>
                {graph.edges.map((edge, i) => (
                  <tr key={i} className="border-t hover:bg-muted/30">
                    <td className="px-2 py-1 font-mono">{edge.parent_id}</td>
                    <td className="px-2 py-1 text-center text-muted-foreground">
                      <ChevronRight className="h-3 w-3 inline" />
                    </td>
                    <td className="px-2 py-1 font-mono">{edge.child_id}</td>
                    <td className="px-2 py-1 text-muted-foreground">{edge.relation || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground italic">No edge data. Using derived_paths.</div>
        )}
      </div>

      {/* Groups */}
      {graph.groups && graph.groups.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2">
            Groups ({graph.groups.length})
          </h4>
          <div className="space-y-1">
            {graph.groups.map((group, i) => (
              <div key={i} className="border rounded p-2 text-xs">
                <div className="font-medium">{group.name}</div>
                <div className="text-muted-foreground">
                  Members: {group.members.join(', ')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ========== Tab C: Candidates ==========

function CandidatesTab({ orgChart, runId }: { orgChart: OrgChartResponse; runId: string }) {
  const [view, setView] = useState<'nodes' | 'edges'>('nodes')

  // 載入Candidates資料
  const { data: nodesCandidates } = useQuery({
    queryKey: ['orgChartDebugFile', runId, 'org_nodes_candidates.json'],
    queryFn: () => getOrgChartDebugFile(runId, 'org_nodes_candidates.json'),
    enabled: !!runId,
    retry: false,
  })

  const { data: edgeCandidates } = useQuery({
    queryKey: ['orgChartDebugFile', runId, 'org_edge_candidates.json'],
    queryFn: () => getOrgChartDebugFile(runId, 'org_edge_candidates.json'),
    enabled: !!runId,
    retry: false,
  })

  const { data: vlm1Raw } = useQuery({
    queryKey: ['orgChartDebugFile', runId, 'org_vlm1_units.raw.json'],
    queryFn: () => getOrgChartDebugFile(runId, 'org_vlm1_units.raw.json'),
    enabled: !!runId,
    retry: false,
  })

  const { data: vlm1Validated } = useQuery({
    queryKey: ['orgChartDebugFile', runId, 'org_vlm1_units.validated.json'],
    queryFn: () => getOrgChartDebugFile(runId, 'org_vlm1_units.validated.json'),
    enabled: !!runId,
    retry: false,
  })

  const nodes = (nodesCandidates as unknown[]) || []
  const edges = (edgeCandidates as unknown[]) || []
  const rawUnits = vlm1Raw as Record<string, unknown> | null
  const validatedUnits = vlm1Validated as Record<string, unknown> | null

  return (
    <div className="space-y-4">
      {/* VLM#1 對比 */}
      <div>
        <h4 className="text-xs font-medium text-muted-foreground mb-2">VLM#1 output comparison</h4>
        <div className="grid grid-cols-2 gap-2">
          <div className="border rounded-lg p-2">
            <div className="text-[10px] text-muted-foreground mb-1">Raw output</div>
            <pre className="text-[10px] bg-muted/50 p-2 rounded overflow-auto max-h-32">
              {rawUnits ? JSON.stringify(rawUnits, null, 2) : 'No data'}
            </pre>
          </div>
          <div className="border rounded-lg p-2">
            <div className="text-[10px] text-muted-foreground mb-1">Validated</div>
            <pre className="text-[10px] bg-muted/50 p-2 rounded overflow-auto max-h-32">
              {validatedUnits ? JSON.stringify(validatedUnits, null, 2) : 'No data'}
            </pre>
          </div>
        </div>
      </div>

      {/* 切換 nodes/edges */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <h4 className="text-xs font-medium text-muted-foreground">Candidate list</h4>
          <div className="flex gap-1">
            <Badge
              variant={view === 'nodes' ? 'default' : 'outline'}
              className="text-[10px] cursor-pointer"
              onClick={() => setView('nodes')}
            >
              Nodes ({nodes.length})
            </Badge>
            <Badge
              variant={view === 'edges' ? 'default' : 'outline'}
              className="text-[10px] cursor-pointer"
              onClick={() => setView('edges')}
            >
              Edges ({edges.length})
            </Badge>
          </div>
        </div>

        <div className="border rounded-lg overflow-hidden">
          <pre className="text-[10px] p-2 overflow-auto max-h-48">
            {view === 'nodes'
              ? nodes.length > 0
                ? JSON.stringify(nodes, null, 2)
                : 'No node candidates'
              : edges.length > 0
                ? JSON.stringify(edges, null, 2)
                : 'No edge candidates'}
          </pre>
        </div>
      </div>

      {/* Decision Trace 詳情 */}
      <div>
        <h4 className="text-xs font-medium text-muted-foreground mb-2">Processing trace</h4>
        <pre className="text-[10px] bg-muted/50 p-2 rounded overflow-auto max-h-32">
          {JSON.stringify(orgChart.decision_trace, null, 2)}
        </pre>
      </div>
    </div>
  )
}

// ========== Tab D: Debug ==========

interface DebugTabProps {
  debugIndex: OrgChartDebugIndex | undefined
  selectedFile: string | null
  onSelectFile: (fileName: string | null) => void
  fileContent: unknown
  isLoadingFile: boolean
}

function DebugTab({ debugIndex, selectedFile, onSelectFile, fileContent, isLoadingFile }: DebugTabProps) {
  const files = debugIndex?.files || []

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="space-y-4">
      {/* 檔案列表 */}
      <div>
        <h4 className="text-xs font-medium text-muted-foreground mb-2">
          Debug files ({files.length})
        </h4>
        <div className="space-y-1">
          {files.map((file) => (
            <Button
              key={file.name}
              variant={selectedFile === file.name ? 'secondary' : 'ghost'}
              size="sm"
              className="w-full justify-start text-xs h-8"
              onClick={() => onSelectFile(file.name)}
            >
              <FileText className="h-3 w-3 mr-2 shrink-0" />
              <span className="flex-1 text-left truncate">{file.name}</span>
              <span className="text-muted-foreground shrink-0 ml-2">
                {formatFileSize(file.size)}
              </span>
            </Button>
          ))}
          {files.length === 0 && (
            <div className="text-xs text-muted-foreground text-center py-4">
              No debug files
            </div>
          )}
        </div>
      </div>

      {/* 檔案內容 */}
      {selectedFile && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2">
            {selectedFile}
          </h4>
          {isLoadingFile ? (
            <div className="flex items-center justify-center py-8 text-muted-foreground">
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              Loading...
            </div>
          ) : (
            <pre className="text-[10px] bg-muted/50 p-2 rounded overflow-auto max-h-64">
              {typeof fileContent === 'string'
                ? fileContent
                : JSON.stringify(fileContent, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}
