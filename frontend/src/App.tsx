import { Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/Layout'
import { Dashboard } from '@/pages/Dashboard'
import { Viewer } from '@/pages/Viewer'
import { Assets } from '@/pages/Assets'
import { Settings } from '@/pages/Settings'
import { I18nProvider } from '@/lib/i18n'

function App() {
  return (
    <I18nProvider>
      <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/viewer/:runId" element={<Viewer />} />
        <Route path="/assets" element={<Assets />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
      </Layout>
    </I18nProvider>
  )
}

export default App
