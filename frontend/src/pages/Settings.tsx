import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Settings as SettingsIcon,
  Server,
  Cpu,
  RefreshCw,
  CheckCircle,
  XCircle,
  Loader2,
  Save,
  RotateCcw,
  Info,
  Sliders,
} from 'lucide-react'
import { useI18n } from '@/lib/i18n'
import {
  getProfile,
  updateProfileOverrides,
  resetProfileOverrides,
  type ProfileWithOverrides,
  type ProfileOverrides,
} from '@/lib/api'

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

interface VLMSettings {
  base_url: string
  api_key: string
  model: string
  api_mode: string
  image_mode: string
  decode_params: {
    temperature: number
    top_p: number
    top_k: number | null
    max_tokens: number
    repetition_penalty: number
  }
  available_modes: string[]
  available_image_modes: string[]
}

interface ProbeResult {
  available: boolean
  model_found: boolean
  supports_vision: boolean
  models: string[]
  error: string | null
  timestamp: string
}

interface MinerUSettings {
  api_url: string | null
  method: string
  backend: string
  model_source: string
}

interface MinerUProbeResult {
  available: boolean
  version: string | null
  error: string | null
  cli_path: string
  api_url: string | null
  api_probe: {
    configured: boolean
    available: boolean
    url: string | null
    host?: string
    port?: number
    error: string | null
  }
  fallback_enabled: boolean
}

interface VLMFormData {
  base_url: string
  api_key: string
  model: string
  api_mode: string
  image_mode: string
  temperature: number
  top_p: number
  top_k: string
  max_tokens: number
  repetition_penalty: number
}

// Profile form data
interface ProfileFormData {
  method: string
  formula: boolean
  enable_vlm: boolean
  vlm_enrich_forms: boolean
  vlm_enrich_figures: boolean
  vlm_enrich_tables: boolean
  table_vlm_budget: number
  table_min_cells: number
  table_max_cells: number
  chunk_max_tokens: number
  chunk_overlap_tokens: number
  semantic_output_language: string
}

const PARAM_DESCRIPTION_KEYS = {
  method: 'settings.desc.method',
  formula: 'settings.desc.formula',
  enable_vlm: 'settings.desc.enable_vlm',
  vlm_enrich_forms: 'settings.desc.vlm_enrich_forms',
  vlm_enrich_figures: 'settings.desc.vlm_enrich_figures',
  vlm_enrich_tables: 'settings.desc.vlm_enrich_tables',
  table_vlm_budget: 'settings.desc.table_vlm_budget',
  table_min_cells: 'settings.desc.table_min_cells',
  table_max_cells: 'settings.desc.table_max_cells',
  chunk_max_tokens: 'settings.desc.chunk_max_tokens',
  chunk_overlap_tokens: 'settings.desc.chunk_overlap_tokens',
  semantic_output_language: 'settings.desc.semantic_output_language',
  temperature: 'settings.desc.temperature',
  top_p: 'settings.desc.top_p',
  top_k: 'settings.desc.top_k',
  max_tokens: 'settings.desc.max_tokens',
  repetition_penalty: 'settings.desc.repetition_penalty',
} as const

export function Settings() {
  const queryClient = useQueryClient()
  const { t } = useI18n()
  const [activeTab, setActiveTab] = useState('vlm')
  const [selectedProfile, setSelectedProfile] = useState('accurate')
  const [vlmRole, setVlmRole] = useState<'enrich' | 'review'>('enrich')
  const vlmSettingsPath = vlmRole === 'enrich' ? 'vlm' : 'review-vlm'

  // VLM form state
  const [vlmFormData, setVlmFormData] = useState<VLMFormData>({
    base_url: '',
    api_key: '',
    model: '',
    api_mode: 'ollama',
    image_mode: 'static_url',
    temperature: 0.2,
    top_p: 0.8,
    top_k: '',
    max_tokens: 1024,
    repetition_penalty: 1.0,
  })
  const [vlmHasChanges, setVlmHasChanges] = useState(false)
  const [mineruApiUrl, setMineruApiUrl] = useState('')
  const [mineruHasChanges, setMineruHasChanges] = useState(false)

  // Profile form state
  const [profileFormData, setProfileFormData] = useState<ProfileFormData>({
    method: 'auto',
    formula: true,
    enable_vlm: true,
    vlm_enrich_forms: true,
    vlm_enrich_figures: true,
    vlm_enrich_tables: false,
    table_vlm_budget: 10,
    table_min_cells: 4,
    table_max_cells: 200,
    chunk_max_tokens: 512,
    chunk_overlap_tokens: 50,
    semantic_output_language: 'auto',
  })
  const [profileHasChanges, setProfileHasChanges] = useState(false)

  // Queries
  const { data: vlmSettings, isLoading: vlmLoading } = useQuery<VLMSettings>({
    queryKey: ['settings', vlmSettingsPath],
    queryFn: () => fetchJson(`${API_BASE}/settings/${vlmSettingsPath}`),
  })

  const { data: mineruSettings, isLoading: mineruLoading } = useQuery<MinerUSettings>({
    queryKey: ['settings', 'mineru'],
    queryFn: () => fetchJson(`${API_BASE}/settings/mineru`),
  })

  const { data: profileData, isLoading: profileLoading } = useQuery<ProfileWithOverrides>({
    queryKey: ['settings', 'profile', selectedProfile],
    queryFn: () => getProfile(selectedProfile),
  })

  // Initialize VLM form when settings load
  useEffect(() => {
    if (vlmSettings) {
      setVlmFormData({
        base_url: vlmSettings.base_url,
        api_key: '',
        model: vlmSettings.model,
        api_mode: vlmSettings.api_mode,
        image_mode: vlmSettings.image_mode,
        temperature: vlmSettings.decode_params.temperature,
        top_p: vlmSettings.decode_params.top_p,
        top_k: vlmSettings.decode_params.top_k?.toString() ?? '',
        max_tokens: vlmSettings.decode_params.max_tokens,
        repetition_penalty: vlmSettings.decode_params.repetition_penalty,
      })
      setVlmHasChanges(false)
    }
  }, [vlmSettings])

  useEffect(() => {
    if (mineruSettings) {
      setMineruApiUrl(mineruSettings.api_url ?? '')
      setMineruHasChanges(false)
    }
  }, [mineruSettings])

  // Initialize profile form when profile data loads
  useEffect(() => {
    if (profileData) {
      setProfileFormData({
        method: profileData.config.mineru.method,
        formula: profileData.config.mineru.formula,
        enable_vlm: profileData.config.enrich.enable_vlm,
        vlm_enrich_forms: profileData.config.enrich.vlm_enrich_forms,
        vlm_enrich_figures: profileData.config.enrich.vlm_enrich_figures,
        vlm_enrich_tables: profileData.config.enrich.vlm_enrich_tables,
        table_vlm_budget: profileData.config.enrich.table_vlm_budget,
        table_min_cells: profileData.config.enrich.table_min_cells,
        table_max_cells: profileData.config.enrich.table_max_cells,
        chunk_max_tokens: profileData.config.package.chunk_max_tokens,
        chunk_overlap_tokens: profileData.config.package.chunk_overlap_tokens,
        semantic_output_language: profileData.config.package.semantic_output_language,
      })
      setProfileHasChanges(false)
    }
  }, [profileData])

  // VLM Mutations
  const vlmUpdateMutation = useMutation({
    mutationFn: (data: Record<string, unknown>) =>
      fetchJson(`${API_BASE}/settings/${vlmSettingsPath}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings', vlmSettingsPath] })
      setVlmHasChanges(false)
    },
  })

  const vlmResetMutation = useMutation({
    mutationFn: () =>
      fetchJson(`${API_BASE}/settings/${vlmSettingsPath}`, { method: 'DELETE' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings', vlmSettingsPath] })
    },
  })

  const vlmProbeMutation = useMutation<ProbeResult>({
    mutationFn: () => fetchJson(`${API_BASE}/settings/${vlmSettingsPath}/probe`),
  })

  const mineruUpdateMutation = useMutation({
    mutationFn: (data: Record<string, unknown>) =>
      fetchJson(`${API_BASE}/settings/mineru`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings', 'mineru'] })
      setMineruHasChanges(false)
    },
  })

  const mineruResetMutation = useMutation({
    mutationFn: () => fetchJson(`${API_BASE}/settings/mineru`, { method: 'DELETE' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings', 'mineru'] })
      setMineruHasChanges(false)
    },
  })

  const mineruProbeMutation = useMutation<MinerUProbeResult>({
    mutationFn: () => fetchJson(`${API_BASE}/settings/mineru/probe`),
  })

  // Profile Mutations
  const profileUpdateMutation = useMutation({
    mutationFn: (overrides: Partial<ProfileOverrides>) =>
      updateProfileOverrides(selectedProfile, overrides),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings', 'profile', selectedProfile] })
      setProfileHasChanges(false)
    },
  })

  const profileResetMutation = useMutation({
    mutationFn: () => resetProfileOverrides(selectedProfile),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings', 'profile', selectedProfile] })
    },
  })

  const handleMineruApiUrlChange = (value: string) => {
    setMineruApiUrl(value)
    setMineruHasChanges(true)
  }

  const handleMineruSave = () => {
    mineruUpdateMutation.mutate({ api_url: mineruApiUrl.trim() })
  }

  // VLM handlers
  const handleVlmInputChange = (field: keyof VLMFormData, value: string | number) => {
    setVlmFormData((prev) => ({ ...prev, [field]: value }))
    setVlmHasChanges(true)
  }

  const handleVlmSave = () => {
    const updateData: Record<string, unknown> = {}

    if (vlmFormData.base_url !== vlmSettings?.base_url) {
      updateData.base_url = vlmFormData.base_url
    }
    if (vlmFormData.api_key) {
      updateData.api_key = vlmFormData.api_key
    }
    if (vlmFormData.model !== vlmSettings?.model) {
      updateData.model = vlmFormData.model
    }
    if (vlmFormData.api_mode !== vlmSettings?.api_mode) {
      updateData.api_mode = vlmFormData.api_mode
    }
    if (vlmFormData.image_mode !== vlmSettings?.image_mode) {
      updateData.image_mode = vlmFormData.image_mode
    }
    if (vlmFormData.temperature !== vlmSettings?.decode_params.temperature) {
      updateData.temperature = vlmFormData.temperature
    }
    if (vlmFormData.top_p !== vlmSettings?.decode_params.top_p) {
      updateData.top_p = vlmFormData.top_p
    }
    if (vlmFormData.top_k !== (vlmSettings?.decode_params.top_k?.toString() ?? '')) {
      updateData.top_k = vlmFormData.top_k ? parseInt(vlmFormData.top_k) : null
    }
    if (vlmFormData.max_tokens !== vlmSettings?.decode_params.max_tokens) {
      updateData.max_tokens = vlmFormData.max_tokens
    }
    if (vlmFormData.repetition_penalty !== vlmSettings?.decode_params.repetition_penalty) {
      updateData.repetition_penalty = vlmFormData.repetition_penalty
    }

    if (Object.keys(updateData).length > 0) {
      vlmUpdateMutation.mutate(updateData)
    }
  }

  // Profile handlers
  const handleProfileInputChange = <K extends keyof ProfileFormData>(
    field: K,
    value: ProfileFormData[K]
  ) => {
    setProfileFormData((prev) => ({ ...prev, [field]: value }))
    setProfileHasChanges(true)
  }

  const handleProfileSave = () => {
    if (!profileData) return

    const overrides: Partial<ProfileOverrides> = {}

    // Only include changed fields
    if (profileFormData.method !== profileData.config.mineru.method) {
      overrides.method = profileFormData.method
    }
    if (profileFormData.formula !== profileData.config.mineru.formula) {
      overrides.formula = profileFormData.formula
    }
    if (profileFormData.enable_vlm !== profileData.config.enrich.enable_vlm) {
      overrides.enable_vlm = profileFormData.enable_vlm
    }
    if (profileFormData.vlm_enrich_forms !== profileData.config.enrich.vlm_enrich_forms) {
      overrides.vlm_enrich_forms = profileFormData.vlm_enrich_forms
    }
    if (profileFormData.vlm_enrich_figures !== profileData.config.enrich.vlm_enrich_figures) {
      overrides.vlm_enrich_figures = profileFormData.vlm_enrich_figures
    }
    if (profileFormData.vlm_enrich_tables !== profileData.config.enrich.vlm_enrich_tables) {
      overrides.vlm_enrich_tables = profileFormData.vlm_enrich_tables
    }
    if (profileFormData.table_vlm_budget !== profileData.config.enrich.table_vlm_budget) {
      overrides.table_vlm_budget = profileFormData.table_vlm_budget
    }
    if (profileFormData.table_min_cells !== profileData.config.enrich.table_min_cells) {
      overrides.table_min_cells = profileFormData.table_min_cells
    }
    if (profileFormData.table_max_cells !== profileData.config.enrich.table_max_cells) {
      overrides.table_max_cells = profileFormData.table_max_cells
    }
    if (profileFormData.chunk_max_tokens !== profileData.config.package.chunk_max_tokens) {
      overrides.chunk_max_tokens = profileFormData.chunk_max_tokens
    }
    if (profileFormData.chunk_overlap_tokens !== profileData.config.package.chunk_overlap_tokens) {
      overrides.chunk_overlap_tokens = profileFormData.chunk_overlap_tokens
    }
    if (profileFormData.semantic_output_language !== profileData.config.package.semantic_output_language) {
      overrides.semantic_output_language = profileFormData.semantic_output_language
    }

    profileUpdateMutation.mutate(overrides)
  }

  // Helper component for parameter with description
  const ParamLabel = ({ label, description }: { label: string; description: string }) => (
    <div className="space-y-1">
      <label className="text-sm font-medium">{label}</label>
      <p className="text-xs text-muted-foreground">{description}</p>
    </div>
  )

  // Toggle switch component
  const Toggle = ({
    checked,
    onChange,
    disabled,
  }: {
    checked: boolean
    onChange: (v: boolean) => void
    disabled?: boolean
  }) => (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      disabled={disabled}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        checked ? 'bg-primary' : 'bg-muted'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          checked ? 'translate-x-6' : 'translate-x-1'
        }`}
      />
    </button>
  )

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <SettingsIcon className="h-6 w-6" />
          {t('settings.title')}
        </h1>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="vlm">{t('settings.vlmTab')}</TabsTrigger>
          <TabsTrigger value="profiles">{t('settings.profileTab')}</TabsTrigger>
        </TabsList>
      </Tabs>

      {activeTab === 'vlm' && (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                {t('settings.mineruTitle')}
              </CardTitle>
              <CardDescription>
                {t('settings.mineruDescription')}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {mineruLoading ? (
                <p className="text-muted-foreground">{t('common.loading')}</p>
              ) : (
                <>
                  <div className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-end">
                    <div>
                      <label className="text-sm font-medium">MinerU API URL</label>
                      <Input
                        value={mineruApiUrl}
                        onChange={(e) => handleMineruApiUrlChange(e.target.value)}
                        placeholder={t('settings.mineruPlaceholder')}
                        className="mt-1"
                      />
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={handleMineruSave}
                        disabled={!mineruHasChanges || mineruUpdateMutation.isPending}
                      >
                        {mineruUpdateMutation.isPending ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                          <Save className="mr-2 h-4 w-4" />
                        )}
                        {t('common.save')}
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => mineruResetMutation.mutate()}
                        disabled={mineruResetMutation.isPending}
                      >
                        {mineruResetMutation.isPending ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                          <RotateCcw className="mr-2 h-4 w-4" />
                        )}
                        {t('common.reset')}
                      </Button>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 border-t pt-4">
                    <Button
                      variant="outline"
                      onClick={() => mineruProbeMutation.mutate()}
                      disabled={mineruProbeMutation.isPending}
                    >
                      {mineruProbeMutation.isPending ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <RefreshCw className="mr-2 h-4 w-4" />
                      )}
                      {t('settings.testMineru')}
                    </Button>
                    {mineruSettings && (
                      <>
                        <Badge variant="outline">method: {mineruSettings.method}</Badge>
                        <Badge variant="outline">backend: {mineruSettings.backend}</Badge>
                        <Badge variant="outline">model: {mineruSettings.model_source}</Badge>
                      </>
                    )}
                  </div>

                  {mineruProbeMutation.data && (
                    <div className="rounded-lg border p-4 space-y-2">
                      <div className="flex items-center gap-2">
                        {mineruProbeMutation.data.available ? (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        ) : (
                          <XCircle className="h-5 w-5 text-red-500" />
                        )}
                        <span className="font-medium">
                          {t('settings.cliAvailable', { state: mineruProbeMutation.data.available ? t('settings.available') : t('settings.unavailable') })}
                        </span>
                        {mineruProbeMutation.data.version && (
                          <Badge variant="secondary">{mineruProbeMutation.data.version}</Badge>
                        )}
                      </div>
                      <div className="text-sm text-muted-foreground">CLI Path：{mineruProbeMutation.data.cli_path}</div>
                      <div className="text-sm">
                        <span className="text-muted-foreground">API URL：</span>
                        {mineruProbeMutation.data.api_url || t('settings.notConfiguredAuto')}
                      </div>
                      {mineruProbeMutation.data.api_probe.configured ? (
                        <div className="flex items-center gap-2 text-sm">
                          {mineruProbeMutation.data.api_probe.available ? (
                            <CheckCircle className="h-4 w-4 text-green-500" />
                          ) : (
                            <XCircle className="h-4 w-4 text-amber-500" />
                          )}
                          <span>
                            {mineruProbeMutation.data.api_probe.available ? t('settings.apiConnected') : t('settings.apiDisconnected')}
                          </span>
                          {!mineruProbeMutation.data.api_probe.available && mineruProbeMutation.data.fallback_enabled && (
                            <Badge variant="secondary">{t('settings.autoFallback')}</Badge>
                          )}
                        </div>
                      ) : (
                        <div className="text-sm text-muted-foreground">{t('settings.mineruAutoHint')}</div>
                      )}
                      {mineruProbeMutation.data.error && (
                        <p className="text-sm text-destructive">{mineruProbeMutation.data.error}</p>
                      )}
                      {mineruProbeMutation.data.api_probe.error && (
                        <p className="text-sm text-muted-foreground">API：{mineruProbeMutation.data.api_probe.error}</p>
                      )}
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>

          {/* VLM Connection */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                {t('settings.vlmTitle')}
              </CardTitle>
              <CardDescription>
                {t('settings.vlmDescription')}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <Button
                  type="button"
                  variant={vlmRole === 'enrich' ? 'default' : 'outline'}
                  onClick={() => setVlmRole('enrich')}
                >
                  {t('settings.vlmEnrichRole')}
                </Button>
                <Button
                  type="button"
                  variant={vlmRole === 'review' ? 'default' : 'outline'}
                  onClick={() => setVlmRole('review')}
                >
                  {t('settings.vlmReviewRole')}
                </Button>
              </div>
              <p className="text-sm text-muted-foreground">
                {t(vlmRole === 'enrich' ? 'settings.vlmEnrichRoleDesc' : 'settings.vlmReviewRoleDesc')}
              </p>
              {vlmLoading ? (
                <p className="text-muted-foreground">{t('common.loading')}</p>
              ) : vlmSettings ? (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium">Base URL</label>
                      <Input
                        value={vlmFormData.base_url}
                        onChange={(e) => handleVlmInputChange('base_url', e.target.value)}
                        placeholder="http://localhost:11434/v1"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium">Model</label>
                      <Input
                        value={vlmFormData.model}
                        onChange={(e) => handleVlmInputChange('model', e.target.value)}
                        placeholder="qwen2.5-vl:7b"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium">{t('settings.apiKeyBlank')}</label>
                      <Input
                        type="password"
                        value={vlmFormData.api_key}
                        onChange={(e) => handleVlmInputChange('api_key', e.target.value)}
                        placeholder="••••••••"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium">{t('settings.apiMode')}</label>
                      <select
                        value={vlmFormData.api_mode}
                        onChange={(e) => handleVlmInputChange('api_mode', e.target.value)}
                        className="mt-1 w-full h-10 rounded-md border px-3 text-sm bg-background text-foreground"
                      >
                        {vlmSettings.available_modes.map((mode) => (
                          <option key={mode} value={mode}>
                            {mode}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-sm font-medium">{t('settings.imageMode')}</label>
                      <select
                        value={vlmFormData.image_mode}
                        onChange={(e) => handleVlmInputChange('image_mode', e.target.value)}
                        className="mt-1 w-full h-10 rounded-md border px-3 text-sm bg-background text-foreground"
                      >
                        {vlmSettings.available_image_modes.map((mode) => (
                          <option key={mode} value={mode}>
                            {mode}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="border-t pt-4">
                    <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
                      <Sliders className="h-4 w-4" />
                      {t('settings.decodeParameters')}
                    </h4>
                    <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                      <div>
                        <ParamLabel label="Temperature" description={t(PARAM_DESCRIPTION_KEYS.temperature)} />
                        <Input
                          type="number"
                          step="0.1"
                          min="0"
                          max="2"
                          value={vlmFormData.temperature}
                          onChange={(e) => handleVlmInputChange('temperature', parseFloat(e.target.value))}
                          className="mt-1"
                        />
                      </div>
                      <div>
                        <ParamLabel label="Top P" description={t(PARAM_DESCRIPTION_KEYS.top_p)} />
                        <Input
                          type="number"
                          step="0.1"
                          min="0"
                          max="1"
                          value={vlmFormData.top_p}
                          onChange={(e) => handleVlmInputChange('top_p', parseFloat(e.target.value))}
                          className="mt-1"
                        />
                      </div>
                      <div>
                        <ParamLabel label={t('settings.optionalTopK')} description={t(PARAM_DESCRIPTION_KEYS.top_k)} />
                        <Input
                          type="number"
                          min="1"
                          value={vlmFormData.top_k}
                          onChange={(e) => handleVlmInputChange('top_k', e.target.value)}
                          placeholder={t('settings.blankDefault')}
                          className="mt-1"
                        />
                      </div>
                      <div>
                        <ParamLabel label="Max Tokens" description={t(PARAM_DESCRIPTION_KEYS.max_tokens)} />
                        <Input
                          type="number"
                          step="128"
                          min="128"
                          max="8192"
                          value={vlmFormData.max_tokens}
                          onChange={(e) => handleVlmInputChange('max_tokens', parseInt(e.target.value))}
                          className="mt-1"
                        />
                      </div>
                      <div>
                        <ParamLabel label="Repetition Penalty" description={t(PARAM_DESCRIPTION_KEYS.repetition_penalty)} />
                        <Input
                          type="number"
                          step="0.1"
                          min="1"
                          max="2"
                          value={vlmFormData.repetition_penalty}
                          onChange={(e) => handleVlmInputChange('repetition_penalty', parseFloat(e.target.value))}
                          className="mt-1"
                        />
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-2 pt-4 border-t">
                    <Button
                      onClick={handleVlmSave}
                      disabled={!vlmHasChanges || vlmUpdateMutation.isPending}
                    >
                      {vlmUpdateMutation.isPending ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Save className="mr-2 h-4 w-4" />
                      )}
                      {t('settings.saveSettings')}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => vlmResetMutation.mutate()}
                      disabled={vlmResetMutation.isPending}
                    >
                      {vlmResetMutation.isPending ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <RotateCcw className="mr-2 h-4 w-4" />
                      )}
                      {t('settings.resetDefaults')}
                    </Button>
                  </div>

                  {vlmUpdateMutation.isSuccess && (
                    <p className="text-sm text-green-600">{t('settings.settingsSaved')}</p>
                  )}
                  {vlmUpdateMutation.error && (
                    <p className="text-sm text-destructive">
                      {(vlmUpdateMutation.error as Error).message}
                    </p>
                  )}
                </>
              ) : null}
            </CardContent>
          </Card>

          {/* VLM Probe */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="h-5 w-5" />
                {t('settings.vlmTestTitle')}
              </CardTitle>
              <CardDescription>
                {t('settings.vlmTestDescription')}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={() => vlmProbeMutation.mutate()}
                disabled={vlmProbeMutation.isPending}
              >
                {vlmProbeMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="mr-2 h-4 w-4" />
                )}
                {t('settings.testConnection')}
              </Button>

              {vlmProbeMutation.data && (
                <div className="rounded-lg border p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    {vlmProbeMutation.data.available ? (
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    ) : (
                      <XCircle className="h-5 w-5 text-red-500" />
                    )}
                    <span className="font-medium">
                      {vlmProbeMutation.data.available ? t('settings.connectionSuccess') : t('settings.connectionFailed')}
                    </span>
                  </div>

                  {vlmProbeMutation.data.available && (
                    <>
                      <div className="text-sm">
                        <span className="text-muted-foreground">{t('settings.modelFound')}:</span>
                        <Badge variant={vlmProbeMutation.data.model_found ? 'default' : 'destructive'} className="ml-2">
                          {vlmProbeMutation.data.model_found ? t('common.yes') : t('common.no')}
                        </Badge>
                      </div>
                      <div className="text-sm">
                        <span className="text-muted-foreground">{t('settings.visionSupport')}:</span>
                        <Badge variant={vlmProbeMutation.data.supports_vision ? 'default' : 'secondary'} className="ml-2">
                          {vlmProbeMutation.data.supports_vision ? t('common.yes') : t('common.no')}
                        </Badge>
                      </div>
                      {vlmProbeMutation.data.models.length > 0 && (
                        <div className="text-sm">
                          <span className="text-muted-foreground">{t('settings.availableModels')}:</span>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {vlmProbeMutation.data.models.slice(0, 5).map((m) => (
                              <Badge key={m} variant="outline" className="text-xs">
                                {m}
                              </Badge>
                            ))}
                            {vlmProbeMutation.data.models.length > 5 && (
                              <Badge variant="outline" className="text-xs">
                                {t('settings.moreModels', { count: vlmProbeMutation.data.models.length - 5 })}
                              </Badge>
                            )}
                          </div>
                        </div>
                      )}
                    </>
                  )}

                  {vlmProbeMutation.data.error && (
                    <p className="text-sm text-destructive">{vlmProbeMutation.data.error}</p>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === 'profiles' && (
        <div className="space-y-4">
          {/* Profile Selector */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sliders className="h-5 w-5" />
                {t('settings.profileTitle')}
              </CardTitle>
              <CardDescription>
                {t('settings.profileDescription')}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Profile tabs */}
              <div className="flex gap-2">
                {['fast', 'accurate'].map((profile) => (
                  <Button
                    key={profile}
                    variant={selectedProfile === profile ? 'default' : 'outline'}
                    onClick={() => setSelectedProfile(profile)}
                    className="gap-2"
                  >
                    {profile.toUpperCase()}
                    {profile === 'accurate' && (
                      <Badge variant="secondary" className="text-xs">{t('common.default')}</Badge>
                    )}
                  </Button>
                ))}
              </div>

              {profileLoading ? (
                <p className="text-muted-foreground">{t('common.loading')}</p>
              ) : profileData ? (
                <>
                  {/* Profile description */}
                  <div className="rounded-lg border p-4 bg-muted/50">
                    <div className="flex items-start gap-2">
                      <Info className="h-5 w-5 text-muted-foreground mt-0.5" />
                      <div>
                        <h4 className="font-medium">{profileData.description.name}</h4>
                        <p className="text-sm text-muted-foreground mt-1">
                          {profileData.description.description}
                        </p>
                        <div className="flex flex-wrap gap-2 mt-2">
                          {profileData.description.features.map((f, i) => (
                            <Badge key={i} variant="outline" className="text-xs">
                              {f}
                            </Badge>
                          ))}
                        </div>
                        {profileData.has_overrides && (
                          <Badge variant="warning" className="mt-2">
                            {t('settings.customized')}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* MinerU Settings */}
                  <div>
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2 text-muted-foreground">
                      {t('settings.mineruParser')}
                    </h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <ParamLabel label={t('settings.method')} description={t(PARAM_DESCRIPTION_KEYS.method)} />
                        <select
                          value={profileFormData.method}
                          onChange={(e) => handleProfileInputChange('method', e.target.value)}
                          className="mt-1 w-full h-10 rounded-md border px-3 text-sm bg-background text-foreground"
                        >
                          <option value="auto">{t('settings.methodAuto')}</option>
                          <option value="txt">{t('settings.methodTxt')}</option>
                          <option value="ocr">{t('settings.methodOcr')}</option>
                        </select>
                      </div>
                      <div>
                        <ParamLabel label={t('settings.formula')} description={t(PARAM_DESCRIPTION_KEYS.formula)} />
                        <div className="mt-2">
                          <Toggle
                            checked={profileFormData.formula}
                            onChange={(v) => handleProfileInputChange('formula', v)}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Enrich Settings */}
                  <div className="border-t pt-4">
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2 text-muted-foreground">
                      {t('settings.vlmEnrichment')}
                    </h4>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                        <div>
                          <ParamLabel label={t('settings.enableVlm')} description={t(PARAM_DESCRIPTION_KEYS.enable_vlm)} />
                          <div className="mt-2">
                            <Toggle
                              checked={profileFormData.enable_vlm}
                              onChange={(v) => handleProfileInputChange('enable_vlm', v)}
                            />
                          </div>
                        </div>
                        <div>
                          <ParamLabel label={t('settings.forms')} description={t(PARAM_DESCRIPTION_KEYS.vlm_enrich_forms)} />
                          <div className="mt-2">
                            <Toggle
                              checked={profileFormData.vlm_enrich_forms}
                              onChange={(v) => handleProfileInputChange('vlm_enrich_forms', v)}
                              disabled={!profileFormData.enable_vlm}
                            />
                          </div>
                        </div>
                        <div>
                          <ParamLabel label={t('settings.figures')} description={t(PARAM_DESCRIPTION_KEYS.vlm_enrich_figures)} />
                          <div className="mt-2">
                            <Toggle
                              checked={profileFormData.vlm_enrich_figures}
                              onChange={(v) => handleProfileInputChange('vlm_enrich_figures', v)}
                              disabled={!profileFormData.enable_vlm}
                            />
                          </div>
                        </div>
                        <div>
                          <ParamLabel label={t('settings.tables')} description={t(PARAM_DESCRIPTION_KEYS.vlm_enrich_tables)} />
                          <div className="mt-2">
                            <Toggle
                              checked={profileFormData.vlm_enrich_tables}
                              onChange={(v) => handleProfileInputChange('vlm_enrich_tables', v)}
                              disabled={!profileFormData.enable_vlm}
                            />
                          </div>
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <ParamLabel label={t('settings.tableBudget')} description={t(PARAM_DESCRIPTION_KEYS.table_vlm_budget)} />
                          <Input
                            type="number"
                            min="0"
                            value={profileFormData.table_vlm_budget}
                            onChange={(e) => handleProfileInputChange('table_vlm_budget', parseInt(e.target.value) || 0)}
                            className="mt-1"
                            disabled={!profileFormData.enable_vlm}
                          />
                        </div>
                        <div>
                          <ParamLabel label={t('settings.minCells')} description={t(PARAM_DESCRIPTION_KEYS.table_min_cells)} />
                          <Input
                            type="number"
                            min="1"
                            value={profileFormData.table_min_cells}
                            onChange={(e) => handleProfileInputChange('table_min_cells', parseInt(e.target.value) || 1)}
                            className="mt-1"
                            disabled={!profileFormData.enable_vlm}
                          />
                        </div>
                        <div>
                          <ParamLabel label={t('settings.maxCells')} description={t(PARAM_DESCRIPTION_KEYS.table_max_cells)} />
                          <Input
                            type="number"
                            min="1"
                            value={profileFormData.table_max_cells}
                            onChange={(e) => handleProfileInputChange('table_max_cells', parseInt(e.target.value) || 1)}
                            className="mt-1"
                            disabled={!profileFormData.enable_vlm}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Package Settings */}
                  <div className="border-t pt-4">
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2 text-muted-foreground">
                      {t('settings.outputPackage')}
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <ParamLabel label={t('settings.semanticOutputLanguage')} description={t(PARAM_DESCRIPTION_KEYS.semantic_output_language)} />
                        <select
                          value={profileFormData.semantic_output_language}
                          onChange={(e) => handleProfileInputChange('semantic_output_language', e.target.value)}
                          className="mt-1 w-full h-10 rounded-md border px-3 text-sm bg-background text-foreground"
                        >
                          <option value="auto">{t('settings.semanticAuto')}</option>
                          <option value="zh-TW">{t('settings.semanticZhTW')}</option>
                          <option value="en">{t('settings.semanticEn')}</option>
                        </select>
                      </div>
                      <div>
                        <ParamLabel label="Chunk Size (tokens)" description={t(PARAM_DESCRIPTION_KEYS.chunk_max_tokens)} />
                        <Input
                          type="number"
                          min="64"
                          max="8192"
                          step="64"
                          value={profileFormData.chunk_max_tokens}
                          onChange={(e) => handleProfileInputChange('chunk_max_tokens', parseInt(e.target.value) || 512)}
                          className="mt-1"
                        />
                      </div>
                      <div>
                        <ParamLabel label="Overlap (tokens)" description={t(PARAM_DESCRIPTION_KEYS.chunk_overlap_tokens)} />
                        <Input
                          type="number"
                          min="0"
                          max="1024"
                          step="10"
                          value={profileFormData.chunk_overlap_tokens}
                          onChange={(e) => handleProfileInputChange('chunk_overlap_tokens', parseInt(e.target.value) || 0)}
                          className="mt-1"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Save/Reset buttons */}
                  <div className="flex gap-2 pt-4 border-t">
                    <Button
                      onClick={handleProfileSave}
                      disabled={!profileHasChanges || profileUpdateMutation.isPending}
                    >
                      {profileUpdateMutation.isPending ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Save className="mr-2 h-4 w-4" />
                      )}
                      {t('settings.saveSettings')}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => profileResetMutation.mutate()}
                      disabled={profileResetMutation.isPending || !profileData.has_overrides}
                    >
                      {profileResetMutation.isPending ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <RotateCcw className="mr-2 h-4 w-4" />
                      )}
                      {t('settings.resetDefaults')}
                    </Button>
                  </div>

                  {profileUpdateMutation.isSuccess && (
                    <p className="text-sm text-green-600">{t('settings.settingsSaved')}</p>
                  )}
                  {profileUpdateMutation.error && (
                    <p className="text-sm text-destructive">
                      {(profileUpdateMutation.error as Error).message}
                    </p>
                  )}
                  {profileResetMutation.isSuccess && (
                    <p className="text-sm text-green-600">{t('settings.resetDone')}</p>
                  )}
                </>
              ) : null}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
