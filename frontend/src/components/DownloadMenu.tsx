import { useEffect, useState, type ReactNode } from 'react'
import * as DropdownMenu from '@radix-ui/react-dropdown-menu'
import { Check, ChevronDown, Download, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/lib/i18n'
import { cn } from '@/lib/utils'

export type DownloadContent = 'main' | 'all'
export type DownloadFormat = 'md' | 'docx' | 'txt'

export interface DownloadSelection {
  content: DownloadContent
  format: DownloadFormat
}

const FORMAT_OPTIONS: DownloadFormat[] = ['md', 'docx', 'txt']
const FORMAT_STORAGE_KEY = 'semark.download.format'
const CONTENT_STORAGE_KEY = 'semark.download.content'

function storedFormat(): DownloadFormat {
  const value = typeof window === 'undefined' ? null : window.localStorage.getItem(FORMAT_STORAGE_KEY)
  return value === 'md' || value === 'docx' || value === 'txt' ? value : 'md'
}

function storedContent(): DownloadContent {
  const value = typeof window === 'undefined' ? null : window.localStorage.getItem(CONTENT_STORAGE_KEY)
  return value === 'all' ? 'all' : 'main'
}

interface DownloadMenuProps {
  /** Text on the trigger button (icon is added automatically). */
  triggerLabel: ReactNode
  /** Runs when the user confirms; reject/throw keeps the menu open with the error. */
  onDownload: (selection: DownloadSelection) => Promise<void> | void
  disabled?: boolean
  /** Show the 主文/全部文件 choice (batch across runs). */
  showContentChoice?: boolean
  /** Optional scope line shown at the top of the menu. */
  summary?: string
  triggerVariant?: 'default' | 'outline'
  align?: 'start' | 'center' | 'end'
  className?: string
}

export function DownloadMenu({
  triggerLabel,
  onDownload,
  disabled = false,
  showContentChoice = false,
  summary,
  triggerVariant = 'default',
  align = 'start',
  className,
}: DownloadMenuProps) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const [content, setContent] = useState<DownloadContent>(storedContent)
  const [format, setFormat] = useState<DownloadFormat>(storedFormat)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) setError(null)
  }, [open])

  function chooseContent(next: DownloadContent) {
    setContent(next)
    window.localStorage.setItem(CONTENT_STORAGE_KEY, next)
  }

  function chooseFormat(next: DownloadFormat) {
    setFormat(next)
    window.localStorage.setItem(FORMAT_STORAGE_KEY, next)
  }

  async function confirm() {
    setBusy(true)
    setError(null)
    try {
      await onDownload({ content: showContentChoice ? content : 'all', format })
      setOpen(false)
    } catch (err) {
      setError(err instanceof Error && err.message ? err.message : t('assets.downloadFailed'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <DropdownMenu.Root open={open} onOpenChange={(next) => !busy && setOpen(next)}>
      <DropdownMenu.Trigger asChild>
        <Button type="button" size="sm" variant={triggerVariant} disabled={disabled} className={className}>
          <Download className="mr-2 h-4 w-4" />
          {triggerLabel}
          <ChevronDown className="ml-1 h-3 w-3 opacity-70" />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align={align}
          sideOffset={6}
          className="z-50 w-72 rounded-md border bg-background p-3 shadow-lg"
        >
          {summary && (
            <p className="mb-2 rounded-md bg-muted/60 px-2 py-1.5 text-xs text-muted-foreground">{summary}</p>
          )}
          {showContentChoice && (
            <div className="space-y-1.5">
              <div className="text-xs font-medium text-muted-foreground">{t('assets.downloadContent')}</div>
              <ContentOption
                selected={content === 'main'}
                label={t('assets.contentMain')}
                hint={t('assets.contentMainHint')}
                onSelect={() => chooseContent('main')}
              />
              <ContentOption
                selected={content === 'all'}
                label={t('assets.contentAll')}
                hint={t('assets.contentAllHint')}
                onSelect={() => chooseContent('all')}
              />
            </div>
          )}
          <div className={cn('space-y-1.5', showContentChoice && 'mt-3')}>
            <div className="text-xs font-medium text-muted-foreground">{t('assets.format')}</div>
            <div className="grid grid-cols-3 gap-1">
              {FORMAT_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => chooseFormat(option)}
                  className={cn(
                    'rounded-md border px-2 py-1.5 text-xs font-medium uppercase transition-colors',
                    format === option
                      ? 'border-primary bg-primary text-primary-foreground'
                      : 'border-input bg-background text-foreground hover:bg-muted'
                  )}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
          {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
          <Button type="button" size="sm" className="mt-3 w-full" onClick={confirm} disabled={busy}>
            {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Download className="mr-2 h-4 w-4" />}
            {t('assets.startDownload')}
          </Button>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  )
}

function ContentOption({
  selected,
  label,
  hint,
  onSelect,
}: {
  selected: boolean
  label: string
  hint: string
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'flex w-full items-start gap-2 rounded-md border p-2 text-left transition-colors',
        selected ? 'border-primary bg-primary/5' : 'border-input hover:bg-muted'
      )}
    >
      <span
        className={cn(
          'mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border',
          selected ? 'border-primary bg-primary text-primary-foreground' : 'border-muted-foreground/50'
        )}
      >
        {selected && <Check className="h-3 w-3" />}
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-medium">{label}</span>
        <span className="block text-xs text-muted-foreground">{hint}</span>
      </span>
    </button>
  )
}
