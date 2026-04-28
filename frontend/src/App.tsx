import { startTransition, useEffect, useRef, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import './App.css'
import BoundingBoxOverlay from './BoundingBoxOverlay'
import ChatPanel from './ChatPanel'
import NerdStatsTab from './NerdStatsTab'
import ProcessingOverlay from './ProcessingOverlay'
import { exportReportPdf } from './reportPdf'
import type {
  AppConfig,
  HistoryEntry,
  Modality,
  Page,
  PatientInfo,
  PredictionResponse,
  ThemeMode,
  TumorLocation,
} from './types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const HISTORY_KEY = 'neurovision-history-v2'

// ─── Icons ────────────────────────────────────────────────────────────────────

function IconBrain() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.16Z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.16Z" />
    </svg>
  )
}

function IconHistory() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M12 7v5l4 2" />
    </svg>
  )
}

function IconChat() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
      <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z" />
    </svg>
  )
}

function IconSun() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
    </svg>
  )
}

function IconMoon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  )
}

function IconUpload() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}

function IconDownload() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  )
}

function IconLocation() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="10" r="3" />
      <path d="M12 21.7C17.3 17 20 13 20 10a8 8 0 1 0-16 0c0 3 2.7 6.9 8 11.7z" />
    </svg>
  )
}

function IconDelete() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14H6L5 6" />
      <path d="M10 11v6M14 11v6" />
      <path d="M9 6V4h6v2" />
    </svg>
  )
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function formatLabel(label: string) {
  return label.replaceAll('_', ' ')
}

function getInitialTheme(): ThemeMode {
  if (typeof window === 'undefined') return 'light'
  const stored = window.localStorage.getItem('neurovision-theme')
  if (stored === 'light' || stored === 'dark') return stored
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function generateAnalysisId(): string {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
  return 'NV-' + Array.from({ length: 6 }, () => chars[Math.floor(Math.random() * chars.length)]).join('')
}

function formatTimestamp(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.75) return 'var(--success)'
  if (confidence >= 0.5) return 'var(--warning)'
  return 'var(--error)'
}

function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    if (!raw) return []
    return JSON.parse(raw) as HistoryEntry[]
  } catch {
    return []
  }
}

function saveToHistory(entry: HistoryEntry) {
  const current = loadHistory()
  const updated = [entry, ...current].slice(0, 100)
  localStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
}

// ─── Image with tumor location overlay ────────────────────────────────────────

function ImageWithOverlay({
  src,
  alt,
  location,
  isPredictionTumor,
  label,
  confidence,
}: {
  src: string
  alt: string
  location?: TumorLocation
  isPredictionTumor: boolean
  label?: string
  confidence?: number
}) {
  return (
    <div className="image-overlay-wrapper">
      <img src={src} alt={alt} className="scan-img" />
      {location && isPredictionTumor && label && confidence !== undefined && (
        <BoundingBoxOverlay location={location} label={label} confidence={confidence} />
      )}
    </div>
  )
}

// ─── Probability bar ────────────────────────────────────────────────────────

function ProbabilityRow({ label, probability, isTop }: { label: string; probability: number; isTop: boolean }) {
  return (
    <div className="prob-row">
      <div className="prob-meta">
        <span className={`prob-label ${isTop ? 'prob-label--top' : ''}`}>{formatLabel(label)}</span>
        <span className={`prob-pct ${isTop ? 'prob-pct--top' : ''}`}>{(probability * 100).toFixed(1)}%</span>
      </div>
      <div className="prob-track">
        <div
          className={`prob-fill ${isTop ? 'prob-fill--top' : ''}`}
          style={{ width: `${Math.max(probability * 100, 2)}%` }}
        />
      </div>
    </div>
  )
}

// ─── History Card ─────────────────────────────────────────────────────────────

function HistoryCard({ entry, onDelete }: { entry: HistoryEntry; onDelete: (id: string) => void }) {
  const isTumor = entry.predicted_label !== 'no_tumor'
  const colorClass = isTumor ? 'badge--tumor' : 'badge--clear'

  return (
    <article className="history-card">
      <div className="history-card-header">
        <div className="history-patient">
          <span className="history-patient-name">{entry.patient.name || 'Anonymous'}</span>
          <span className="history-patient-id">{entry.patient.id}</span>
        </div>
        <button
          className="icon-btn icon-btn--ghost"
          onClick={() => onDelete(entry.id)}
          aria-label="Delete this history entry"
          title="Delete"
        >
          <IconDelete />
        </button>
      </div>

      <div className="history-card-body">
        <div className="history-modality-chip">{entry.modality.toUpperCase()}</div>
        <div className={`history-badge ${colorClass}`}>{formatLabel(entry.predicted_label)}</div>
        <div className="history-confidence" style={{ color: getConfidenceColor(entry.confidence) }}>
          {(entry.confidence * 100).toFixed(1)}%
        </div>
      </div>

      {entry.tumor_location && isTumor && (
        <div className="history-location">
          <IconLocation />
          <span>{entry.tumor_location.description}</span>
        </div>
      )}

      <div className="history-card-footer">
        <span className="history-date">{formatTimestamp(entry.timestamp)}</span>
        <span className="history-provider">{entry.report_provider === 'ollama' ? 'Ollama report' : 'Template report'}</span>
      </div>
    </article>
  )
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [page, setPage] = useState<Page>('analysis')
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme)
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [backendError, setBackendError] = useState<string | null>(null)
  const [navOpen, setNavOpen] = useState(false)

  // Patient
  const [patientName, setPatientName] = useState('')
  const [analysisId] = useState(generateAnalysisId)

  // Scan form
  const [modality, setModality] = useState<Modality>('mri')
  const [mriFile, setMriFile] = useState<File | null>(null)
  const [ctFile, setCtFile] = useState<File | null>(null)
  const [mriPreviewUrl, setMriPreviewUrl] = useState<string | null>(null)
  const [ctPreviewUrl, setCtPreviewUrl] = useState<string | null>(null)
  const [useOllamaReport, setUseOllamaReport] = useState(false)
  const [imageTab, setImageTab] = useState<'original' | 'gradcam'>('gradcam')

  // Results
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [resultsTab, setResultsTab] = useState<'results' | 'stats'>('results')

  // History
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory)

  // Chat
  const [chatOpen, setChatOpen] = useState(false)

  const resultsRef = useRef<HTMLDivElement>(null)

  // Theme sync
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('neurovision-theme', theme)
  }, [theme])

  // Fetch config
  useEffect(() => {
    const controller = new AbortController()
    fetch(`${API_BASE_URL}/config`, { signal: controller.signal })
      .then(async (r) => {
        if (!r.ok) throw new Error('Unable to reach the NeuroVision backend.')
        return r.json() as Promise<AppConfig>
      })
      .then((data) => {
        setConfig(data)
        setBackendError(null)
      })
      .catch((err: Error) => {
        if (!controller.signal.aborted) setBackendError(err.message)
      })
    return () => controller.abort()
  }, [])

  // Cleanup object URLs
  useEffect(() => {
    return () => {
      if (mriPreviewUrl) URL.revokeObjectURL(mriPreviewUrl)
      if (ctPreviewUrl) URL.revokeObjectURL(ctPreviewUrl)
    }
  }, [mriPreviewUrl, ctPreviewUrl])

  function onFileChange(e: ChangeEvent<HTMLInputElement>, kind: 'mri' | 'ct') {
    const file = e.target.files?.[0] ?? null
    setResult(null)
    setError(null)
    if (kind === 'mri') {
      setMriFile(file)
      if (mriPreviewUrl) URL.revokeObjectURL(mriPreviewUrl)
      setMriPreviewUrl(file ? URL.createObjectURL(file) : null)
    } else {
      setCtFile(file)
      if (ctPreviewUrl) URL.revokeObjectURL(ctPreviewUrl)
      setCtPreviewUrl(file ? URL.createObjectURL(file) : null)
    }
  }

  function onModalityChange(next: Modality) {
    setModality(next)
    setResult(null)
    setError(null)
  }

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setError(null)
    setResult(null)

    const availableNow = new Set(config?.available_now ?? [])
    if (!availableNow.has(modality)) {
      setError('This modality is not active on the backend. Check that the server is running and model weights are loaded.')
      return
    }

    const formData = new FormData()
    let endpoint = ''

    if (modality === 'mri') {
      if (!mriFile) { setError('Please upload an MRI image.'); return }
      formData.append('file', mriFile)
      formData.append('use_ollama', String(useOllamaReport))
      endpoint = '/predict/mri'
    } else if (modality === 'ct') {
      if (!ctFile) { setError('Please upload a CT image.'); return }
      formData.append('file', ctFile)
      formData.append('use_ollama', String(useOllamaReport))
      endpoint = '/predict/ct'
    } else {
      if (!mriFile || !ctFile) { setError('Fusion mode requires both an MRI and a CT image.'); return }
      formData.append('mri_file', mriFile)
      formData.append('ct_file', ctFile)
      formData.append('use_ollama', String(useOllamaReport))
      endpoint = '/predict/fusion'
    }

    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, { method: 'POST', body: formData })
      const payload = (await response.json()) as PredictionResponse | { detail?: string }
      if (!response.ok) {
        throw new Error('detail' in payload ? payload.detail ?? 'Prediction failed.' : 'Prediction failed.')
      }
      const pred = payload as PredictionResponse
      startTransition(() => {
        setResult(pred)
        setImageTab('gradcam')
        setResultsTab('results')
      })

      // Save to history (no base64 images to stay within localStorage limits)
      const patient: PatientInfo = { name: patientName.trim() || 'Anonymous', id: analysisId }
      const entry: HistoryEntry = {
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        patient,
        modality: pred.modality,
        predicted_label: pred.predicted_label,
        confidence: pred.confidence,
        probabilities: pred.probabilities,
        tumor_location: pred.tumor_location,
        report_provider: pred.report_provider,
      }
      saveToHistory(entry)
      setHistory(loadHistory())

      // Scroll to results
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong during analysis.')
    } finally {
      setLoading(false)
    }
  }

  function onDownloadReport() {
    if (!result) return
    exportReportPdf(result, {
      generatedAt: new Date(),
      patientName: patientName.trim() || 'Anonymous',
      patientId: analysisId,
      authorName: 'Pranesh Dharani',
    })
  }

  function onDeleteHistory(id: string) {
    const updated = history.filter((e) => e.id !== id)
    setHistory(updated)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
  }

  function onClearHistory() {
    if (!confirm('Clear all history? This cannot be undone.')) return
    setHistory([])
    localStorage.removeItem(HISTORY_KEY)
  }

  const availableNow = new Set(config?.available_now ?? [])
  const isTumorPrediction = result ? result.predicted_label !== 'no_tumor' : false

  return (
    <div className="app-shell">
      {/* ── Top App Bar ───────────────────────────────────────── */}
      <header className="top-bar">
        <div className="top-bar-start">
          <button
            className="icon-btn nav-hamburger"
            onClick={() => setNavOpen((o) => !o)}
            aria-label="Toggle navigation"
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
              <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z" />
            </svg>
          </button>
          <div className="app-logo">
            <div className="app-logo-icon">
              <IconBrain />
            </div>
            <div className="app-logo-text">
              <span className="app-logo-name">NeuroVision</span>
              <span className="app-logo-sub">AI · Brain Tumor Classifier</span>
            </div>
          </div>
        </div>

        <div className="top-bar-end">
          <div className={`status-pill ${config ? (availableNow.size > 0 ? 'status-pill--online' : 'status-pill--warn') : 'status-pill--offline'}`}>
            <span className="status-dot" />
            <span className="status-text">
              {config
                ? availableNow.size > 0
                  ? `${[...availableNow].map((m) => m.toUpperCase()).join(' · ')} ready`
                  : 'No models loaded'
                : backendError
                  ? 'Backend offline'
                  : 'Connecting…'}
            </span>
          </div>
          <button
            className="icon-btn"
            onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
            aria-label="Toggle theme"
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? <IconSun /> : <IconMoon />}
          </button>
        </div>
      </header>

      {/* ── Navigation Drawer ─────────────────────────────────── */}
      <div className={`nav-backdrop ${navOpen ? 'nav-backdrop--open' : ''}`} onClick={() => setNavOpen(false)} aria-hidden="true" />

      <nav className={`nav-drawer ${navOpen ? 'nav-drawer--open' : ''}`}>
        <div className="nav-header">
          <span className="nav-section-label">Navigation</span>
        </div>

        <button
          className={`nav-item ${page === 'analysis' ? 'nav-item--active' : ''}`}
          onClick={() => { setPage('analysis'); setNavOpen(false) }}
        >
          <span className="nav-item-icon"><IconBrain /></span>
          <span className="nav-item-label">Analysis</span>
        </button>

        <button
          className={`nav-item ${page === 'history' ? 'nav-item--active' : ''}`}
          onClick={() => { setPage('history'); setNavOpen(false) }}
        >
          <span className="nav-item-icon"><IconHistory /></span>
          <span className="nav-item-label">Patient History</span>
          {history.length > 0 && (
            <span className="nav-badge">{history.length}</span>
          )}
        </button>

        <div className="nav-divider" />
        <div className="nav-footer">
          <p className="nav-footer-text">NeuroVision AI v0.1</p>
          <p className="nav-footer-text">Built by Pranesh Dharani</p>
        </div>
      </nav>

      {/* ── Main Content ──────────────────────────────────────── */}
      <main className="main-content">

        {/* ── PAGE: Analysis ─────────────────────────────────── */}
        {page === 'analysis' && (
          <div className="analysis-page">
            {/* Page header */}
            <div className="page-header">
              <div>
                <h1 className="page-title">Brain Scan Analysis</h1>
                <p className="page-subtitle">Upload a scan, run AI classification, and review the findings.</p>
              </div>
            </div>

            {/* Backend error banner */}
            {backendError && (
              <div className="alert alert--error">
                <strong>Backend unreachable:</strong> {backendError}. Start the server with{' '}
                <code>uvicorn app.main:app --reload --port 8000</code>.
              </div>
            )}

            {/* Patient info card */}
            <div className="patient-card">
              <div className="patient-card-row">
                <div className="field-group">
                  <label className="field-label" htmlFor="patient-name">Patient Name</label>
                  <input
                    id="patient-name"
                    className="text-input"
                    type="text"
                    value={patientName}
                    onChange={(e) => setPatientName(e.target.value)}
                    placeholder="Enter patient name (optional)"
                  />
                </div>
                <div className="field-group field-group--id">
                  <span className="field-label">Analysis ID</span>
                  <span className="analysis-id-chip">{analysisId}</span>
                </div>
                <div className="field-group field-group--id">
                  <span className="field-label">Date</span>
                  <span className="analysis-id-chip">{new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}</span>
                </div>
              </div>
            </div>

            <div className="analysis-grid">
              {/* ── Left: Input form ─────────────────────────── */}
              <form className="input-card" onSubmit={onSubmit}>
                <div className="card-header">
                  <h2 className="card-title">Scan Input</h2>
                </div>

                {/* Modality selector */}
                <div className="field-section">
                  <span className="field-label">Imaging Modality</span>
                  <div className="modality-chips">
                    {(['mri', 'ct', 'fusion'] as Modality[]).map((opt) => {
                      const live = availableNow.has(opt)
                      return (
                        <button
                          key={opt}
                          type="button"
                          className={`modality-chip ${modality === opt ? 'modality-chip--selected' : ''} ${!live ? 'modality-chip--pending' : ''}`}
                          onClick={() => onModalityChange(opt)}
                        >
                          <span className="modality-chip-name">{opt.toUpperCase()}</span>
                          <span className={`modality-chip-dot ${live ? 'modality-chip-dot--live' : ''}`} />
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Upload zones */}
                {(modality === 'mri' || modality === 'fusion') && (
                  <div className="field-section">
                    <span className="field-label">MRI Image</span>
                    <label className={`upload-zone ${mriFile ? 'upload-zone--filled' : ''}`}>
                      <input type="file" accept="image/*" onChange={(e) => onFileChange(e, 'mri')} />
                      {mriPreviewUrl ? (
                        <img src={mriPreviewUrl} alt="MRI preview" className="upload-preview" />
                      ) : (
                        <div className="upload-placeholder">
                          <IconUpload />
                          <span className="upload-label">Drop MRI image here or click to browse</span>
                          <span className="upload-hint">JPEG, PNG, DICOM-export</span>
                        </div>
                      )}
                      {mriFile && (
                        <div className="upload-filename">{mriFile.name}</div>
                      )}
                    </label>
                  </div>
                )}

                {(modality === 'ct' || modality === 'fusion') && (
                  <div className="field-section">
                    <span className="field-label">CT Image</span>
                    <label className={`upload-zone ${ctFile ? 'upload-zone--filled' : ''}`}>
                      <input type="file" accept="image/*" onChange={(e) => onFileChange(e, 'ct')} />
                      {ctPreviewUrl ? (
                        <img src={ctPreviewUrl} alt="CT preview" className="upload-preview" />
                      ) : (
                        <div className="upload-placeholder">
                          <IconUpload />
                          <span className="upload-label">Drop CT image here or click to browse</span>
                          <span className="upload-hint">JPEG, PNG, DICOM-export</span>
                        </div>
                      )}
                      {ctFile && (
                        <div className="upload-filename">{ctFile.name}</div>
                      )}
                    </label>
                  </div>
                )}

                {/* Ollama toggle */}
                <div className="field-section">
                  <div className={`toggle-row ${!config?.ollama_available ? 'toggle-row--disabled' : ''}`}>
                    <div className="toggle-info">
                      <span className="field-label">AI Report Enhancement</span>
                      <span className="toggle-hint">
                        {config?.ollama_available
                          ? `Ollama · ${config.ollama_model ?? 'local model'}`
                          : 'Ollama not available — template report will be used'}
                      </span>
                    </div>
                    <label className="toggle-switch">
                      <input
                        type="checkbox"
                        checked={useOllamaReport}
                        onChange={(e) => setUseOllamaReport(e.target.checked)}
                        disabled={!config?.ollama_available}
                      />
                      <span className="toggle-track" />
                    </label>
                  </div>
                </div>

                {/* Error */}
                {error && <div className="alert alert--error">{error}</div>}

                {/* Submit */}
                <button
                  className={`btn-primary ${loading ? 'btn-primary--loading' : ''}`}
                  type="submit"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner" />
                      Analyzing scan…
                    </>
                  ) : (
                    'Run Analysis'
                  )}
                </button>

                {config && !backendError && (
                  <p className="subtle-note">
                    {availableNow.size > 0
                      ? `Active: ${[...availableNow].map((m) => m.toUpperCase()).join(', ')}`
                      : 'No modalities are currently active on the backend.'}
                  </p>
                )}
              </form>

              {/* ── Right: Results ──────────────────────────── */}
              <div className="results-column" ref={resultsRef}>
                {!result && !loading && (
                  <div className="results-empty">
                    <div className="results-empty-icon"><IconBrain /></div>
                    <h3>No analysis yet</h3>
                    <p>Upload a scan and click "Run Analysis" to see results here.</p>
                  </div>
                )}

                <ProcessingOverlay visible={loading} />

                {result && (
                  <>
                    {/* Results / Stats for Nerds toggle */}
                    <div className="results-view-toggle">
                      <button
                        className={`tab-btn ${resultsTab === 'results' ? 'tab-btn--active' : ''}`}
                        type="button"
                        onClick={() => setResultsTab('results')}
                      >
                        Analysis
                      </button>
                      <button
                        className={`tab-btn ${resultsTab === 'stats' ? 'tab-btn--active' : ''}`}
                        type="button"
                        onClick={() => setResultsTab('stats')}
                      >
                        Stats for Nerds
                      </button>
                    </div>

                    {/* Stats for Nerds view */}
                    {resultsTab === 'stats' && (
                      <NerdStatsTab result={result} config={config} />
                    )}

                    {/* Analysis results */}
                    {resultsTab === 'results' && <>

                    {/* Image viewer */}
                    <div className="result-card">
                      <div className="card-header card-header--with-tabs">
                        <h2 className="card-title">Scan Viewer</h2>
                        <div className="tab-row">
                          <button
                            className={`tab-btn ${imageTab === 'original' ? 'tab-btn--active' : ''}`}
                            onClick={() => setImageTab('original')}
                            type="button"
                          >
                            Original
                          </button>
                          <button
                            className={`tab-btn ${imageTab === 'gradcam' ? 'tab-btn--active' : ''}`}
                            onClick={() => setImageTab('gradcam')}
                            type="button"
                          >
                            Attention Map
                          </button>
                        </div>
                      </div>

                      {imageTab === 'original' ? (
                        <ImageWithOverlay
                          src={result.original_preview}
                          alt="Original scan"
                          isPredictionTumor={false}
                        />
                      ) : (
                        <ImageWithOverlay
                          src={result.gradcam_overlay}
                          alt="Grad-CAM attention map"
                          location={result.tumor_location}
                          isPredictionTumor={isTumorPrediction}
                          label={result.predicted_label}
                          confidence={result.confidence}
                        />
                      )}

                      {result.tumor_location && isTumorPrediction && imageTab === 'gradcam' && (
                        <div className="location-badge">
                          <IconLocation />
                          <span>{result.tumor_location.description}</span>
                        </div>
                      )}

                      <div className="report-provider-row">
                        <span className="field-label">Report engine</span>
                        <span className={`provider-chip ${result.report_provider === 'ollama' ? 'provider-chip--ollama' : ''}`}>
                          {result.report_provider === 'ollama'
                            ? `Ollama${config?.ollama_model ? ` · ${config.ollama_model}` : ''}`
                            : 'Template'}
                        </span>
                      </div>
                    </div>

                    {/* Prediction summary */}
                    <div className="result-card">
                      <div className="card-header">
                        <h2 className="card-title">Prediction</h2>
                        <button className="btn-secondary" type="button" onClick={onDownloadReport}>
                          <IconDownload />
                          Export PDF
                        </button>
                      </div>

                      <div className="prediction-main">
                        <div className={`prediction-label-chip ${isTumorPrediction ? 'prediction-label-chip--tumor' : 'prediction-label-chip--clear'}`}>
                          {formatLabel(result.predicted_label)}
                        </div>
                        <div
                          className="prediction-confidence"
                          style={{ color: getConfidenceColor(result.confidence) }}
                        >
                          {(result.confidence * 100).toFixed(1)}% confidence
                        </div>
                        <div className="prediction-modality-tag">{result.modality.toUpperCase()}</div>
                      </div>

                      {/* Probability bars */}
                      <div className="prob-list">
                        {[...result.probabilities]
                          .sort((a, b) => b.probability - a.probability)
                          .map((entry, idx) => (
                            <ProbabilityRow
                              key={entry.label}
                              label={entry.label}
                              probability={entry.probability}
                              isTop={idx === 0}
                            />
                          ))}
                      </div>
                    </div>

                    {/* Tumor location card */}
                    {result.tumor_location && isTumorPrediction && (
                      <div className="result-card">
                        <div className="card-header">
                          <h2 className="card-title">
                            <IconLocation />
                            Predicted Location
                          </h2>
                        </div>
                        <div className="location-details">
                          <div className="location-stat">
                            <span className="field-label">Quadrant</span>
                            <strong>{result.tumor_location.quadrant}</strong>
                          </div>
                          <div className="location-stat">
                            <span className="field-label">Horizontal position</span>
                            <strong>{(result.tumor_location.cx * 100).toFixed(0)}% from left</strong>
                          </div>
                          <div className="location-stat">
                            <span className="field-label">Vertical position</span>
                            <strong>{(result.tumor_location.cy * 100).toFixed(0)}% from top</strong>
                          </div>
                          <div className="location-stat">
                            <span className="field-label">Activation radius</span>
                            <strong>{(result.tumor_location.radius * 100).toFixed(0)}% of image width</strong>
                          </div>
                        </div>
                        <p className="location-disclaimer">
                          Location is derived from Grad-CAM attention, not a pathological contour. Treat as qualitative saliency only.
                        </p>
                      </div>
                    )}

                    {/* Structured report */}
                    <div className="result-card">
                      <div className="card-header">
                        <h2 className="card-title">Structured Report</h2>
                      </div>
                      <div className="report-sections">
                        {result.report.map((section) => (
                          <details key={section.title} className="report-section" open>
                            <summary className="report-section-title">{section.title}</summary>
                            <p className="report-section-body">{section.body}</p>
                          </details>
                        ))}
                      </div>
                    </div>

                    {/* Operational notes */}
                    {result.notes.length > 0 && (
                      <div className="result-card result-card--notes">
                        <div className="card-header">
                          <h2 className="card-title">Operational Notes</h2>
                        </div>
                        <ul className="notes-list">
                          {result.notes.map((note) => (
                            <li key={note} className="notes-item">{note}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    </>}
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── PAGE: History ──────────────────────────────────── */}
        {page === 'history' && (
          <div className="history-page">
            <div className="page-header">
              <div>
                <h1 className="page-title">Patient History</h1>
                <p className="page-subtitle">Past analyses stored locally on this device.</p>
              </div>
              {history.length > 0 && (
                <button className="btn-secondary btn-secondary--danger" type="button" onClick={onClearHistory}>
                  Clear all
                </button>
              )}
            </div>

            {history.length === 0 ? (
              <div className="history-empty">
                <div className="results-empty-icon"><IconHistory /></div>
                <h3>No history yet</h3>
                <p>Analyses will appear here after you run your first scan.</p>
                <button className="btn-primary" type="button" onClick={() => setPage('analysis')}>
                  Go to Analysis
                </button>
              </div>
            ) : (
              <div className="history-grid">
                {history.map((entry) => (
                  <HistoryCard key={entry.id} entry={entry} onDelete={onDeleteHistory} />
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      {/* ── Chat FAB ──────────────────────────────────────────── */}
      <button
        className={`chat-fab ${result ? 'chat-fab--active' : ''}`}
        onClick={() => setChatOpen(true)}
        aria-label="Open AI chat assistant"
        title="Chat with AI about the scan"
      >
        <IconChat />
        {result && <span className="chat-fab-label">Ask AI</span>}
      </button>

      {/* ── Chat Panel ────────────────────────────────────────── */}
      <ChatPanel
        isOpen={chatOpen}
        onClose={() => setChatOpen(false)}
        result={result}
        patientName={patientName}
        config={config}
      />
    </div>
  )
}
