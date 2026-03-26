import { startTransition, useEffect, useEffectEvent, useMemo, useState } from 'react'
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

type SliceStatus = 'idle' | 'analyzing' | 'ready' | 'error'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

function App() {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [modality, setModality] = useState<Modality>('mri')
  const [mriFile, setMriFile] = useState<File | null>(null)
  const [ctFile, setCtFile] = useState<File | null>(null)
  const [mriPreviewUrl, setMriPreviewUrl] = useState<string | null>(null)
  const [ctPreviewUrl, setCtPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mriSlices, setMriSlices] = useState<File[]>([])
  const [mriSliceUrls, setMriSliceUrls] = useState<string[]>([])
  const [activeSliceIndex, setActiveSliceIndex] = useState(0)
  const [isSlicePlaying, setIsSlicePlaying] = useState(false)
  const [playbackFps, setPlaybackFps] = useState(8)
  const [showCineHeatmap, setShowCineHeatmap] = useState(true)
  const [cineHeatmapOpacity, setCineHeatmapOpacity] = useState(72)
  const [mriSliceResults, setMriSliceResults] = useState<Record<number, PredictionResponse>>({})
  const [mriSliceStatus, setMriSliceStatus] = useState<Record<number, SliceStatus>>({})

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
      if (mriPreviewUrl) URL.revokeObjectURL(mriPreviewUrl)
      if (ctPreviewUrl) URL.revokeObjectURL(ctPreviewUrl)
    }
  }, [ctPreviewUrl, mriPreviewUrl])

  useEffect(() => {
    return () => {
      mriSliceUrls.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [mriSliceUrls])

  useEffect(() => {
    if (!isSlicePlaying || mriSliceUrls.length <= 1) {
      return
    }

    const interval = window.setInterval(() => {
      setActiveSliceIndex((current) => (current + 1) % mriSliceUrls.length)
    }, Math.max(1000 / playbackFps, 60))

    return () => window.clearInterval(interval)
  }, [isSlicePlaying, mriSliceUrls.length, playbackFps])

  function onFileChange(event: ChangeEvent<HTMLInputElement>, kind: 'mri' | 'ct') {
    const nextFile = event.target.files?.[0] ?? null
    setResult(null)
    setError(null)
    setMriSliceResults({})
    setMriSliceStatus({})

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

  function onMriSequenceChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? [])
      .filter((file) => file.type.startsWith('image/'))
      .sort((left, right) =>
        left.name.localeCompare(right.name, undefined, { numeric: true, sensitivity: 'base' })
      )

    mriSliceUrls.forEach((url) => URL.revokeObjectURL(url))
    setResult(null)
    setMriSliceResults({})
    setMriSliceStatus({})
    setMriSlices(files)
    setMriSliceUrls(files.map((file) => URL.createObjectURL(file)))
    setActiveSliceIndex(0)
    setIsSlicePlaying(files.length > 1)
  }

  async function performAnalysis(options?: { silent?: boolean; preferActiveSlice?: boolean }) {
    const silent = options?.silent ?? false
    const preferActiveSlice = options?.preferActiveSlice ?? false

    if (!silent) {
      setError(null)
      setResult(null)
    }

    const formData = new FormData()
    let endpoint = ''

    const activeMriSliceFile = mriSlices[activeSliceIndex] ?? mriSlices[0] ?? null
    const targetSliceIndex = activeMriSliceFile ? activeSliceIndex : null

    if (modality === 'mri') {
      const analysisFile = preferActiveSlice ? activeMriSliceFile ?? mriFile : mriFile ?? activeMriSliceFile
      if (!analysisFile) {
        if (!silent) {
          setError('Choose an MRI image or load an MRI slice stack before running analysis.')
        }
        return
      }
      formData.append('file', analysisFile)
      endpoint = '/predict/mri'
    } else if (modality === 'ct') {
      if (!ctFile) {
        if (!silent) {
          setError('Choose a CT image before running analysis.')
        }
        return
      }
      formData.append('file', ctFile)
      endpoint = '/predict/ct'
    } else {
      const analysisFile = preferActiveSlice ? activeMriSliceFile ?? mriFile : mriFile ?? activeMriSliceFile
      if (!analysisFile || !ctFile) {
        if (!silent) {
          setError('Fusion mode needs a CT image and either an MRI image or MRI slice stack.')
        }
        return
      }
      formData.append('mri_file', analysisFile)
      formData.append('ct_file', ctFile)
      endpoint = '/predict/fusion'
    }

    if (targetSliceIndex !== null && (modality === 'mri' || modality === 'fusion')) {
      setMriSliceStatus((current) => ({
        ...current,
        [targetSliceIndex]: 'analyzing',
      }))
    }

    if (!availableNow.has(modality)) {
      if (!silent) {
        setError(
          modality === 'ct'
            ? 'CT dataset is present, but CT weights still need to be trained before this endpoint can run.'
            : 'Fusion needs both trained MRI and CT models before it can run.'
        )
      }
      return
    }

    if (!silent) {
      setLoading(true)
    }

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
        const prediction = payload as PredictionResponse
        setResult(prediction)
        if (targetSliceIndex !== null && (modality === 'mri' || modality === 'fusion')) {
          setMriSliceResults((current) => ({
            ...current,
            [targetSliceIndex]: prediction,
          }))
          setMriSliceStatus((current) => ({
            ...current,
            [targetSliceIndex]: 'ready',
          }))
        }
      })
    } catch (submitError) {
      if (targetSliceIndex !== null && (modality === 'mri' || modality === 'fusion')) {
        setMriSliceStatus((current) => ({
          ...current,
          [targetSliceIndex]: 'error',
        }))
      }
      if (!silent) {
        setError(
          submitError instanceof Error ? submitError.message : 'Something went wrong during analysis.'
        )
      }
    } finally {
      if (!silent) {
        setLoading(false)
      }
    }
  }

  const runSliceAutoAnalysis = useEffectEvent(() => {
    void performAnalysis({ silent: true, preferActiveSlice: true })
  })

  useEffect(() => {
    if (isSlicePlaying || !mriSlices.length || modality === 'ct') {
      return
    }

    if (modality === 'fusion' && !ctFile) {
      return
    }

    const timeout = window.setTimeout(() => {
      runSliceAutoAnalysis()
    }, 320)

    return () => window.clearTimeout(timeout)
  }, [activeSliceIndex, config, ctFile, isSlicePlaying, modality, mriSlices.length])

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    await performAnalysis({ silent: false, preferActiveSlice: true })
  }

  const availableNow = new Set(config?.available_now ?? [])
  const activeMriSliceUrl = mriSliceUrls[activeSliceIndex] ?? null
  const activeSliceResult = mriSliceResults[activeSliceIndex] ?? null
  const activeSliceStatus = mriSliceStatus[activeSliceIndex] ?? 'idle'
  const viewerHeatmap = activeSliceResult?.gradcam_overlay ?? result?.gradcam_overlay ?? null
  const sliceProgressLabel = useMemo(() => {
    if (!mriSlices.length) {
      return 'No MRI stack loaded'
    }

    return `Slice ${activeSliceIndex + 1} of ${mriSlices.length}`
  }, [activeSliceIndex, mriSlices.length])

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

          {(modality === 'mri' || modality === 'fusion') && (
            <label className="upload-zone">
              <input type="file" accept="image/*" onChange={(event) => onFileChange(event, 'mri')} />
              <span className="upload-title">
                {mriFile ? mriFile.name : 'Drop an MRI image here or click to browse'}
              </span>
              <span className="upload-subtitle">
                This file powers MRI-only mode and the MRI branch of fusion mode.
              </span>
            </label>
          )}

          {(modality === 'mri' || modality === 'fusion') && (
            <label className="upload-zone sequence-zone">
              <input type="file" accept="image/*" multiple onChange={onMriSequenceChange} />
              <span className="upload-title">
                {mriSlices.length
                  ? `${mriSlices.length} MRI layers loaded for the cine viewer`
                  : 'Load an MRI slice stack for the presentation viewer'}
              </span>
              <span className="upload-subtitle">
                Select multiple ordered MRI images and the viewer will animate them with a slider.
                The active slice can also be used for heatmap analysis.
              </span>
            </label>
          )}

          {(modality === 'ct' || modality === 'fusion') && (
            <label className="upload-zone ct-zone">
              <input type="file" accept="image/*" onChange={(event) => onFileChange(event, 'ct')} />
              <span className="upload-title">
                {ctFile ? ctFile.name : 'Drop a CT image here or click to browse'}
              </span>
              <span className="upload-subtitle">
                CT support is dataset-ready and will go live as soon as CT weights are trained.
              </span>
            </label>
          )}

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
                <span>{modality === 'ct' ? 'Uploaded CT' : 'Uploaded MRI'}</span>
              </div>
              {(modality === 'ct' ? ctPreviewUrl : mriPreviewUrl) ? (
                <img
                  src={modality === 'ct' ? (ctPreviewUrl as string) : (mriPreviewUrl as string)}
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
                <span>{modality === 'fusion' ? 'Fusion attention (MRI branch)' : 'Model attention'}</span>
              </div>
              {result ? (
                <img src={result.gradcam_overlay} alt="Grad-CAM attention map" className="scan-image" />
              ) : (
                <div className="empty-state">Grad-CAM visualization will appear after analysis.</div>
              )}
            </article>
          </div>

          {(modality === 'mri' || modality === 'fusion') && (
            <article className="cine-card">
              <div className="cine-header">
                <div>
                  <span className="section-label">MRI Cine Viewer</span>
                  <h3>Presentation mode for layered MRI slices</h3>
                </div>
                <div className="cine-chip-row">
                  <span className="cine-chip">{sliceProgressLabel}</span>
                  <span className="cine-chip">{playbackFps} fps</span>
                  <span className={`cine-chip status-chip ${activeSliceStatus}`}>
                    {activeSliceStatus === 'analyzing'
                      ? 'Analyzing slice...'
                      : activeSliceStatus === 'ready'
                        ? 'Heatmap ready'
                        : activeSliceStatus === 'error'
                          ? 'Analysis failed'
                          : 'Not analyzed'}
                  </span>
                </div>
              </div>

              <div className="cine-stage">
                {activeMriSliceUrl ? (
                  <>
                    <img
                      src={activeMriSliceUrl}
                      alt={`MRI slice ${activeSliceIndex + 1}`}
                      className="cine-image"
                    />
                    {viewerHeatmap && showCineHeatmap ? (
                      <img
                        src={viewerHeatmap}
                        alt="MRI heatmap overlay"
                        className="cine-heatmap"
                        style={{ opacity: cineHeatmapOpacity / 100 }}
                      />
                    ) : null}
                    <div className="cine-overlay">
                      <span>{sliceProgressLabel}</span>
                      <span>{mriSlices[activeSliceIndex]?.name}</span>
                    </div>
                  </>
                ) : (
                  <div className="empty-state">
                    Load multiple MRI layers and this viewer will animate them like a scan stack.
                  </div>
                )}
              </div>

              <div className="cine-controls">
                <button
                  type="button"
                  className="secondary-action"
                  onClick={() => setIsSlicePlaying((current) => !current)}
                  disabled={mriSliceUrls.length <= 1}
                >
                  {isSlicePlaying ? 'Pause' : 'Play'}
                </button>
                <button
                  type="button"
                  className="secondary-action"
                  onClick={() =>
                    setActiveSliceIndex((current) => Math.max(current - 1, 0))
                  }
                  disabled={!mriSliceUrls.length}
                >
                  Prev
                </button>
                <button
                  type="button"
                  className="secondary-action"
                  onClick={() =>
                    setActiveSliceIndex((current) => Math.min(current + 1, mriSliceUrls.length - 1))
                  }
                  disabled={!mriSliceUrls.length}
                >
                  Next
                </button>
                <div className="speed-group" role="group" aria-label="Playback speed">
                  {[4, 8, 12].map((fps) => (
                    <button
                      key={fps}
                      type="button"
                      className={`speed-pill ${playbackFps === fps ? 'active' : ''}`}
                      onClick={() => setPlaybackFps(fps)}
                    >
                      {fps} fps
                    </button>
                  ))}
                </div>
              </div>

              <div className="heatmap-controls">
                <button
                  type="button"
                  className={`secondary-action ${showCineHeatmap ? 'active-toggle' : ''}`}
                  onClick={() => setShowCineHeatmap((current) => !current)}
                  disabled={!viewerHeatmap}
                >
                  {showCineHeatmap ? 'Hide heatmap' : 'Show heatmap'}
                </button>
                <div className="opacity-group">
                  <label htmlFor="heatmap-opacity">Heatmap opacity</label>
                  <input
                    id="heatmap-opacity"
                    type="range"
                    min={15}
                    max={100}
                    value={cineHeatmapOpacity}
                    onChange={(event) => setCineHeatmapOpacity(Number(event.target.value))}
                    disabled={!viewerHeatmap || !showCineHeatmap}
                  />
                  <span>{cineHeatmapOpacity}%</span>
                </div>
                {!viewerHeatmap ? (
                  <p className="heatmap-hint">
                    Run MRI analysis first. If you only loaded a slice stack, the active slice will be analyzed.
                  </p>
                ) : null}
              </div>

              <div className="slider-wrap">
                <input
                  type="range"
                  min={0}
                  max={Math.max(mriSliceUrls.length - 1, 0)}
                  value={Math.min(activeSliceIndex, Math.max(mriSliceUrls.length - 1, 0))}
                  onChange={(event) => {
                    setIsSlicePlaying(false)
                    setActiveSliceIndex(Number(event.target.value))
                  }}
                  disabled={!mriSliceUrls.length}
                  className="slice-slider"
                />
                <div className="slider-labels">
                  <span>Base layer</span>
                  <span>Focus layer</span>
                  <span>Final layer</span>
                </div>
              </div>
            </article>
          )}

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
