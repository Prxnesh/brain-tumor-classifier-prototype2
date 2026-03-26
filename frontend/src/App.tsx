import { startTransition, useEffect, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import './App.css'

type Modality = 'mri' | 'ct' | 'fusion'

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

function App() {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [modality, setModality] = useState<Modality>('mri')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()

    fetch(`${API_BASE_URL}/config`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error('Unable to load API configuration.')
        }

        return response.json() as Promise<AppConfig>
      })
      .then((data) => setConfig(data))
      .catch((fetchError: Error) => {
        if (!controller.signal.aborted) {
          setError(fetchError.message)
        }
      })

    return () => controller.abort()
  }, [])

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null
    setFile(nextFile)
    setResult(null)
    setError(null)

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null)
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setResult(null)

    if (!file) {
      setError('Choose an image before running analysis.')
      return
    }

    if (modality !== 'mri') {
      setError(
        modality === 'ct'
          ? 'CT analysis is wired in the app, but still needs a CT dataset and trained weights.'
          : 'Fusion mode is wired in the app, but still needs paired CT and MRI data.'
      )
      return
    }

    const formData = new FormData()
    formData.append('file', file)
    setLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/predict/mri`, {
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

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">NeuroVision AI</p>
          <h1>Detect brain tumor patterns from MRI now, with CT and fusion ready next.</h1>
          <p className="lede">
            A full-stack diagnostic prototype for scan upload, AI classification, confidence
            scoring, and visual attention maps clinicians can inspect.
          </p>
        </div>
        <div className="hero-stats">
          <article>
            <span className="stat-label">Modalities</span>
            <strong>3 modes</strong>
            <p>MRI live, CT and multimodal fusion scaffolded for the next dataset drop.</p>
          </article>
          <article>
            <span className="stat-label">Visualization</span>
            <strong>Grad-CAM</strong>
            <p>Heatmaps reveal which image regions pushed the classifier toward its prediction.</p>
          </article>
        </div>
      </section>

      <section className="workspace">
        <form className="control-panel" onSubmit={onSubmit}>
          <div className="panel-header">
            <h2>Scan intake</h2>
            <p>Upload a brain scan image and choose how the system should analyze it.</p>
          </div>

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
                    {live ? 'Available now' : 'Dataset needed'}
                  </span>
                </button>
              )
            })}
          </div>

          <label className="upload-zone">
            <input type="file" accept="image/*" onChange={onFileChange} />
            <span className="upload-title">
              {file ? file.name : 'Drop an MRI image here or click to browse'}
            </span>
            <span className="upload-subtitle">
              JPG and PNG work well for the current pipeline.
            </span>
          </label>

          <button className="primary-action" type="submit" disabled={loading}>
            {loading ? 'Analyzing scan...' : 'Run analysis'}
          </button>

          {error ? <p className="message error">{error}</p> : null}
          {!error && config ? (
            <p className="message subtle">
              Live now: {config.available_now.join(', ').toUpperCase()}. Pending datasets:{' '}
              {config.pending_datasets.join(', ').toUpperCase()}.
            </p>
          ) : null}
        </form>

        <section className="results-panel">
          <div className="panel-header">
            <h2>Clinical preview</h2>
            <p>The right side updates with the scan preview, attention map, and report summary.</p>
          </div>

          <div className="preview-grid">
            <article className="preview-card">
              <div className="preview-header">
                <span>Uploaded scan</span>
              </div>
              {previewUrl ? (
                <img src={previewUrl} alt="Uploaded scan preview" className="scan-image" />
              ) : (
                <div className="empty-state">Upload an MRI image to preview it here.</div>
              )}
            </article>

            <article className="preview-card">
              <div className="preview-header">
                <span>Model attention</span>
              </div>
              {result ? (
                <img src={result.gradcam_overlay} alt="Grad-CAM attention map" className="scan-image" />
              ) : (
                <div className="empty-state">Grad-CAM visualization will appear after analysis.</div>
              )}
            </article>
          </div>

          <div className="insight-grid">
            <article className="insight-card diagnosis-card">
              <span className="section-label">Prediction</span>
              <h3>{result ? result.predicted_label.replaceAll('_', ' ') : 'Awaiting analysis'}</h3>
              <p>
                {result
                  ? `${(result.confidence * 100).toFixed(1)}% confidence on the current MRI model.`
                  : 'The classifier result and confidence score will show up here.'}
              </p>
            </article>

            <article className="insight-card">
              <span className="section-label">Class confidence</span>
              <div className="probability-list">
                {(result?.probabilities ?? []).map((entry) => (
                  <div key={entry.label} className="probability-row">
                    <div className="probability-meta">
                      <span>{entry.label.replaceAll('_', ' ')}</span>
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

          <article className="report-card">
            <span className="section-label">Decision-support report</span>
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
                  <p>The backend will generate a structured summary once a scan is analyzed.</p>
                </div>
              ) : null}
            </div>
          </article>

          {result?.notes?.length ? (
            <article className="notes-card">
              {result.notes.map((note) => (
                <p key={note}>{note}</p>
              ))}
            </article>
          ) : null}
        </section>
      </section>
    </main>
  )
}

export default App
