import { startTransition, useEffect, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import './App.css'
import { exportReportPdf } from './reportPdf'

type Modality = 'mri' | 'ct' | 'fusion'
type ThemeMode = 'light' | 'dark'

type AppConfig = {
  supported_modalities: Modality[]
  available_now: Modality[]
  pending_datasets: Modality[]
}

type Probability = {
  label: string
  probability: number
}

type ReportSection = {
  title: string
  body: string
}

type PredictionResponse = {
  modality: Modality
  predicted_label: string
  confidence: number
  probabilities: Probability[]
  gradcam_overlay: string
  original_preview: string
  report: ReportSection[]
  notes: string[]
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

function formatLabel(label: string) {
  return label.replaceAll('_', ' ')
}

function getInitialTheme(): ThemeMode {
  if (typeof window === 'undefined') {
    return 'dark'
  }

  const storedTheme = window.localStorage.getItem('neurovision-theme')
  if (storedTheme === 'light' || storedTheme === 'dark') {
    return storedTheme
  }

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function buildExplainer(result: PredictionResponse | null) {
  if (!result) {
    return 'Once analysis completes, this panel will translate the model output into a clinical-style narrative with attention-map context and confidence framing.'
  }

  const ranked = [...result.probabilities].sort((left, right) => right.probability - left.probability)
  const runnerUp = ranked[1]
  const modalityLead =
    result.modality === 'fusion'
      ? 'The multimodal fusion baseline assigns the highest posterior support'
      : `The ${result.modality.toUpperCase()} classifier assigns the highest posterior support`

  return `${modalityLead} to ${formatLabel(result.predicted_label)} at ${(result.confidence * 100).toFixed(1)}% confidence. The attention map highlights visually influential regions rather than a true lesion contour, so it should be interpreted as qualitative saliency only. The nearest competing class is ${runnerUp ? `${formatLabel(runnerUp.label)} at ${(runnerUp.probability * 100).toFixed(1)}%` : 'not available'}.`
}

function buildClinicalNote(result: PredictionResponse | null) {
  if (!result) {
    return 'Clinical note output will appear here after a scan is analyzed, including scope-of-use guidance and recommended clinical correlation.'
  }

  const modeNote =
    result.modality === 'fusion'
      ? 'Fusion mode currently combines MRI and CT probabilities and displays MRI-branch saliency for visual context.'
      : `${result.modality.toUpperCase()} mode reflects a single-modality classifier output from the submitted image.`

  return `${modeNote} Correlate this preliminary AI impression with the complete study, lesion location, mass effect, edema pattern, contrast behavior when available, prior imaging, and the treating clinician's formal interpretation before any diagnostic or management decision is made.`
}

function App() {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme)
  const [modality, setModality] = useState<Modality>('mri')
  const [mriFile, setMriFile] = useState<File | null>(null)
  const [ctFile, setCtFile] = useState<File | null>(null)
  const [studyLabel, setStudyLabel] = useState('')
  const [mriPreviewUrl, setMriPreviewUrl] = useState<string | null>(null)
  const [ctPreviewUrl, setCtPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    window.localStorage.setItem('neurovision-theme', theme)
  }, [theme])

  useEffect(() => {
    const controller = new AbortController()

    fetch(`${API_BASE_URL}/config`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error('Unable to load API configuration.')
        }

        return response.json() as Promise<AppConfig>
      })
      .then((data) => {
        setConfig(data)
        setError(null)
      })
      .catch((fetchError: Error) => {
        if (!controller.signal.aborted) {
          setError(fetchError.message)
        }
      })

    return () => controller.abort()
  }, [])

  useEffect(() => {
    return () => {
      if (mriPreviewUrl) URL.revokeObjectURL(mriPreviewUrl)
      if (ctPreviewUrl) URL.revokeObjectURL(ctPreviewUrl)
    }
  }, [ctPreviewUrl, mriPreviewUrl])

  function onFileChange(event: ChangeEvent<HTMLInputElement>, kind: 'mri' | 'ct') {
    const nextFile = event.target.files?.[0] ?? null
    setResult(null)
    setError(null)

    if (kind === 'mri') {
      setMriFile(nextFile)
      if (mriPreviewUrl) URL.revokeObjectURL(mriPreviewUrl)
      setMriPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null)
      return
    }

    setCtFile(nextFile)
    if (ctPreviewUrl) URL.revokeObjectURL(ctPreviewUrl)
    setCtPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null)
  }

  function onDownloadReport() {
    if (!result) {
      return
    }

    exportReportPdf(result, {
      generatedAt: new Date(),
      patientName: studyLabel,
      authorName: 'Pranesh Dharani',
    })
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setResult(null)

    const formData = new FormData()
    let endpoint = ''

    if (modality === 'mri') {
      if (!mriFile) {
        setError('Choose an MRI image before running analysis.')
        return
      }
      formData.append('file', mriFile)
      endpoint = '/predict/mri'
    } else if (modality === 'ct') {
      if (!ctFile) {
        setError('Choose a CT image before running analysis.')
        return
      }
      formData.append('file', ctFile)
      endpoint = '/predict/ct'
    } else {
      if (!mriFile || !ctFile) {
        setError('Fusion mode needs both an MRI image and a CT image.')
        return
      }
      formData.append('mri_file', mriFile)
      formData.append('ct_file', ctFile)
      endpoint = '/predict/fusion'
    }

    if (!availableNow.has(modality)) {
      setError('This modality is not active on the backend yet. Confirm the server is running and the required model weights are loaded.')
      return
    }

    setLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
      })

      const payload = (await response.json()) as PredictionResponse | { detail?: string }

      if (!response.ok) {
        throw new Error('detail' in payload ? payload.detail ?? 'Prediction failed.' : 'Prediction failed.')
      }

      startTransition(() => {
        setResult(payload as PredictionResponse)
      })
    } catch (submitError) {
      setError(
        submitError instanceof Error ? submitError.message : 'Something went wrong during analysis.'
      )
    } finally {
      setLoading(false)
    }
  }

  const availableNow = new Set(config?.available_now ?? [])
  const selectedPreviewUrl = modality === 'ct' ? ctPreviewUrl : mriPreviewUrl
  const statusLabel = availableNow.has(modality) ? 'Ready' : 'Standby'
  const confidenceDescriptor =
    result?.modality === 'fusion'
      ? 'current multimodal fusion baseline'
      : result
        ? `current ${result.modality.toUpperCase()} model`
        : `selected ${modality.toUpperCase()} workflow`

  return (
    <main className="app-shell">
      <header className="console-bar">
        <div className="console-branding">
          <p className="eyebrow">NeuroVision AI</p>
          <strong>Diagnostic Workstation</strong>
        </div>
        <div className="console-actions">
          <div className="mode-indicator">
            <span className="indicator-dot" />
            {config ? `Online: ${config.available_now.join(', ').toUpperCase()}` : 'Awaiting backend handshake'}
          </div>
          <button
            className="theme-toggle"
            type="button"
            onClick={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? 'Light mode' : 'Dark mode'}
          </button>
        </div>
      </header>

      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Neuroradiology Console</p>
          <h1>NeuroVision</h1>
          <p className="lede">
            Brain tumor screening with MRI, CT, and fusion review in one focused diagnostic workspace.
          </p>
        </div>
        <div className="hero-stats">
          <article>
            <span className="stat-label">Modalities</span>
            <strong>MR | CT | Fusion</strong>
            <p>Single-modality review plus a baseline multimodal fusion path for paired inputs.</p>
          </article>
          <article>
            <span className="stat-label">Model</span>
            <strong>ResNet50 + Grad-CAM</strong>
            <p>ResNet50 performs the image classification, and Grad-CAM highlights the regions that most influenced the prediction.</p>
          </article>
        </div>
      </section>

      <section className="workspace">
        <form className="control-panel" onSubmit={onSubmit}>
          <div className="panel-header">
            <h2>Scan intake</h2>
            <p>Load the study, choose the active workflow, and dispatch the analysis request.</p>
          </div>

          <div className="control-meta">
            <div className="meta-chip">
              <span className="stat-label">Workflow</span>
              <strong>{modality.toUpperCase()}</strong>
            </div>
            <div className="meta-chip">
              <span className="stat-label">Backend status</span>
              <strong>{statusLabel}</strong>
            </div>
          </div>

          <label className="study-label-field">
            <span className="stat-label">Study label</span>
            <input
              type="text"
              value={studyLabel}
              onChange={(event) => setStudyLabel(event.target.value)}
              placeholder="Optional patient or study identifier"
            />
          </label>

          <div className="modality-grid">
            {(['mri', 'ct', 'fusion'] as Modality[]).map((option) => {
              const live = availableNow.has(option)
              return (
                <button
                  key={option}
                  className={`modality-card ${modality === option ? 'selected' : ''}`}
                  type="button"
                  onClick={() => setModality(option)}
                >
                  <span className="modality-name">{option.toUpperCase()}</span>
                  <span className={`modality-badge ${live ? 'live' : 'pending'}`}>
                    {live ? 'Ready for analysis' : 'Standby'}
                  </span>
                </button>
              )
            })}
          </div>

          {(modality === 'mri' || modality === 'fusion') && (
            <label className="upload-zone">
              <input type="file" accept="image/*" onChange={(event) => onFileChange(event, 'mri')} />
              <span className="upload-title">
                {mriFile ? mriFile.name : 'Load MRI study image'}
              </span>
              <span className="upload-subtitle">
                MRI powers single-modality review and serves as the visual branch in fusion mode.
              </span>
            </label>
          )}

          {(modality === 'ct' || modality === 'fusion') && (
            <label className="upload-zone ct-zone">
              <input type="file" accept="image/*" onChange={(event) => onFileChange(event, 'ct')} />
              <span className="upload-title">
                {ctFile ? ctFile.name : 'Load CT study image'}
              </span>
              <span className="upload-subtitle">
                CT supports single-modality review and contributes to the fusion probability baseline.
              </span>
            </label>
          )}

          <button className="primary-action" type="submit" disabled={loading}>
            {loading ? 'Processing study...' : 'Run analysis'}
          </button>

          {error ? <p className="message error">{error}</p> : null}
          {!error && config ? (
            <p className="message subtle">
              Active backend modes: {config.available_now.join(', ').toUpperCase()}.
            </p>
          ) : null}
        </form>

        <section className="results-panel">
          <div className="panel-header">
            <h2>Reading console</h2>
            <p>Review the uploaded study, the model saliency map, and the structured clinical output.</p>
          </div>

          <div className="results-toolbar">
            <button className="secondary-action" type="button" onClick={onDownloadReport} disabled={!result}>
              Download PDF report
            </button>
          </div>

          <div className="preview-grid">
            <article className="preview-card">
              <div className="preview-header">
                <span>{modality === 'ct' ? 'Uploaded CT' : 'Uploaded primary study'}</span>
              </div>
              {selectedPreviewUrl ? (
                <img
                  src={selectedPreviewUrl}
                  alt="Uploaded scan preview"
                  className="scan-image"
                />
              ) : (
                <div className="empty-state">
                  {modality === 'ct'
                    ? 'Upload a CT image to preview it here.'
                    : 'Upload an MRI image to preview it here.'}
                </div>
              )}
            </article>

            <article className="preview-card">
              <div className="preview-header">
                <span>{modality === 'fusion' ? 'Fusion attention (MRI branch)' : 'Model attention map'}</span>
              </div>
              {result ? (
                <img src={result.gradcam_overlay} alt="Grad-CAM attention map" className="scan-image" />
              ) : (
                <div className="empty-state">Attention-map output will appear after analysis.</div>
              )}
            </article>
          </div>

          <div className="insight-grid">
            <article className="insight-card diagnosis-card">
              <span className="section-label">Impression</span>
              <h3>{result ? formatLabel(result.predicted_label) : 'Awaiting analysis'}</h3>
              <p>
                {result
                  ? `${(result.confidence * 100).toFixed(1)}% confidence on the ${confidenceDescriptor}.`
                  : 'The leading diagnostic class will appear here after analysis.'}
              </p>
            </article>

            <article className="insight-card">
              <span className="section-label">Class confidence</span>
              <div className="probability-list">
                {(result?.probabilities ?? []).map((entry) => (
                  <div key={entry.label} className="probability-row">
                    <div className="probability-meta">
                      <span>{formatLabel(entry.label)}</span>
                      <span>{(entry.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="probability-track">
                      <div
                        className="probability-fill"
                        style={{ width: `${Math.max(entry.probability * 100, 4)}%` }}
                      />
                    </div>
                  </div>
                ))}
                {!result ? <div className="empty-inline">Run analysis to populate class scores.</div> : null}
              </div>
            </article>
          </div>

          <div className="narrative-grid">
            <article className="narrative-card">
              <span className="section-label">AI explainer</span>
              <p>{buildExplainer(result)}</p>
            </article>

            <article className="narrative-card">
              <span className="section-label">Clinical note</span>
              <p>{buildClinicalNote(result)}</p>
            </article>
          </div>

          <article className="report-card">
            <span className="section-label">Structured report</span>
            <div className="report-list">
              {(result?.report ?? []).map((section) => (
                <div key={section.title} className="report-item">
                  <h4>{section.title}</h4>
                  <p>{section.body}</p>
                </div>
              ))}
              {!result ? (
                <div className="report-item">
                  <h4>No report yet</h4>
                  <p>The backend will generate a clinical-style report once a study is analyzed.</p>
                </div>
              ) : null}
            </div>
          </article>

          {result?.notes?.length ? (
            <article className="notes-card">
              <span className="section-label">Operational notes</span>
              <div className="notes-list">
                {result.notes.map((note) => (
                  <p key={note}>{note}</p>
                ))}
              </div>
            </article>
          ) : null}
        </section>
      </section>

      <footer className="app-footer">
        <p>Designed and built by Pranesh Dharani</p>
      </footer>
    </main>
  )
}

export default App
