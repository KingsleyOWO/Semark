import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { FileText, FolderOpen, LayoutDashboard, Settings } from 'lucide-react'
import { useI18n, type Language } from '@/lib/i18n'

const navItems = [
  { href: '/', labelKey: 'nav.dashboard', icon: LayoutDashboard },
  { href: '/assets', labelKey: 'nav.assets', icon: FolderOpen },
  { href: '/settings', labelKey: 'nav.settings', icon: Settings },
]

interface LayoutProps {
  children: React.ReactNode
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const { language, setLanguage, t } = useI18n()

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur">
        <div className="flex h-14 w-full items-center px-4 sm:px-5 lg:px-6">
          <Link to="/" className="flex items-center gap-2 font-semibold">
            <FileText className="h-5 w-5" />
            <span>{t('app.name')}</span>
          </Link>

          <nav className="ml-8 flex items-center gap-6">
            {navItems.map((item) => (
              <Link
                key={item.href}
                to={item.href}
                className={cn(
                  'flex items-center gap-2 text-sm font-medium transition-colors hover:text-foreground/80',
                  location.pathname === item.href
                    ? 'text-foreground'
                    : 'text-foreground/60'
                )}
              >
                <item.icon className="h-4 w-4" />
                {t(item.labelKey)}
              </Link>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            <label htmlFor="language-select" className="sr-only">{t('language.label')}</label>
            <select
              id="language-select"
              value={language}
              onChange={(event) => setLanguage(event.target.value as Language)}
              className="h-8 rounded-md border border-input bg-background px-2 text-sm"
              aria-label={t('language.label')}
            >
              <option value="en">{t('language.en')}</option>
              <option value="zh-TW">{t('language.zhTW')}</option>
            </select>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="min-w-0 w-full flex-1 px-4 py-4 sm:px-5 lg:px-6">{children}</main>
    </div>
  )
}
