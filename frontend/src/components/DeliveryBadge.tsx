import { useQuery } from '@tanstack/react-query'
import { getQualityGate } from '@/lib/api'
import { useI18n } from '@/lib/i18n'
import type { QualityGateReport } from '@/types/api'
import { Badge } from '@/components/ui/badge'

export function DeliveryBadge({ report }: { report?: QualityGateReport }) {
  const { t } = useI18n()
  const checked = report?.stats?.final_delivery_checked === true
  const review = report?.stats?.agentic_review as {
    complete?: boolean
    total_pages?: number
    verified_pages?: number
    needs_review_pages?: number
    not_reviewed_pages?: number
  } | undefined
  const recovery = report?.stats?.agentic_recovery as {
    applied?: number
    unresolved?: number
    budget_exhausted?: number
  } | undefined
  const automatic = report?.stats?.automatic_delivery as {
    status?: 'complete' | 'partial' | 'unverified'
    rag_chunks?: number
    gap_count?: number
  } | undefined
  const ready = automatic ? automatic.status === 'complete'
    : checked && review?.complete === true && report?.stats?.auto_rag_ready === true
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <Badge variant={ready ? 'success' : checked ? 'warning' : 'outline'}>
        {t(automatic ? `delivery.${automatic.status}` : ready ? 'delivery.ready' : checked && !review ? 'delivery.sourceNotChecked' : checked ? 'delivery.review' : 'delivery.notChecked')}
      </Badge>
      {automatic && (
        <span className="text-xs text-muted-foreground">
          {t('delivery.automaticCounts', { chunks: automatic.rag_chunks ?? 0, gaps: automatic.gap_count ?? 0 })}
        </span>
      )}
      {review && (
        <span className="text-xs text-muted-foreground">
          {t('delivery.sourceReview', {
            verified: review.verified_pages ?? 0,
            total: review.total_pages ?? 0,
            pending: review.needs_review_pages ?? 0,
            unchecked: review.not_reviewed_pages ?? 0,
          })}
        </span>
      )}
      {recovery && (
        <span className="text-xs text-muted-foreground">
          {t('delivery.recovery', {
            applied: recovery.applied ?? 0,
            pending: recovery.unresolved ?? 0,
            exhausted: recovery.budget_exhausted ?? 0,
          })}
        </span>
      )}
    </span>
  )
}

export function RunDeliveryBadge({ runId, enabled = true }: { runId: string; enabled?: boolean }) {
  const { data } = useQuery({
    queryKey: ['qualityGate', runId],
    queryFn: () => getQualityGate(runId),
    enabled,
    staleTime: 60000,
    retry: false,
  })
  if (!enabled) return null
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      <DeliveryBadge report={data} />
      {data?.stats?.automatic_delivery != null && <AutomaticDownloads runId={runId} />}
    </span>
  )
}



function AutomaticDownloads({ runId }: { runId: string }) {
  const { t } = useI18n()
  return (
    <span className="inline-flex gap-2 text-xs">
      <a className="underline" href={`/api/runs/${runId}/output?view=delivery`} download="delivery.md">{t('delivery.download')}</a>
      <a className="underline" href={`/api/runs/${runId}/rag-chunks`}>{t('delivery.downloadRag')}</a>
      <a className="underline" href={`/api/runs/${runId}/processing-report`}>{t('delivery.report')}</a>
    </span>
  )
}
