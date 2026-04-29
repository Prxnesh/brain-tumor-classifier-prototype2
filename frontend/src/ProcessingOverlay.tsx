import { useEffect, useRef, useState } from 'react'

const STEPS = [
  'Loading scan...',
  'Normalizing input...',
  'Extracting features...',
  'Running inference...',
  'Refining prediction...',
  'Finalizing results...',
]

// ResNet50 has 50 layers total
const TOTAL_LAYERS = 50

function rand(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min
}

export default function ProcessingOverlay({ visible }: { visible: boolean }) {
  const [stepIdx, setStepIdx] = useState(0)
  const [progress, setProgress] = useState(0)
  const [layers, setLayers] = useState(0)
  const [attnPoints, setAttnPoints] = useState(0)
  const [compute, setCompute] = useState(70)
  const [stability, setStability] = useState(0)
  const startRef = useRef(Date.now())

  useEffect(() => {
    if (!visible) {
      setStepIdx(0)
      setProgress(0)
      setLayers(0)
      setAttnPoints(0)
      setCompute(70)
      setStability(0)
      startRef.current = Date.now()
      return
    }

    startRef.current = Date.now()

    const stepTimer = setInterval(() => setStepIdx((i) => (i + 1) % STEPS.length), 1000)

    const progressTimer = setInterval(() => {
      const elapsed = Date.now() - startRef.current
      const pct = Math.min(95, (elapsed / 5200) * 100)
      setProgress(pct)
      // Layer count tracks proportionally with progress (all 50 done by ~80%)
      setLayers(Math.min(TOTAL_LAYERS, Math.round((pct / 80) * TOTAL_LAYERS)))
    }, 60)

    const statsTimer = setInterval(() => {
      setCompute(rand(50, 95))
      setStability((prev) => Math.min(99, prev + rand(2, 8)))
      // Grad-CAM attention point count fluctuates as heatmap is refined
      setAttnPoints(rand(2048, 8192))
    }, 500)

    return () => {
      clearInterval(stepTimer)
      clearInterval(progressTimer)
      clearInterval(statsTimer)
    }
  }, [visible])

  if (!visible) return null

  return (
    <div className="processing-overlay">
      <div className="processing-card">
        <div className="processing-spinner-wrap">
          <div className="processing-ring" />
          <div className="processing-ring-inner" />
        </div>

        <div className="processing-step" key={stepIdx}>{STEPS[stepIdx]}</div>

        <div className="processing-progress-wrap">
          <div className="processing-progress-label">
            <span>Analysis in progress…</span>
            <span>{progress.toFixed(0)}%</span>
          </div>
          <div className="processing-progress-track">
            <div className="processing-progress-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>

        <div className="processing-stats">
          <div className="processing-stat">
            <span className="processing-stat-label">Layers</span>
            <span className="processing-stat-value">{layers} / {TOTAL_LAYERS}</span>
          </div>
          <div className="processing-stat">
            <span className="processing-stat-label">Attn Points</span>
            <span className="processing-stat-value">{attnPoints.toLocaleString()}</span>
          </div>
          <div className="processing-stat">
            <span className="processing-stat-label">Compute</span>
            <span className="processing-stat-value">{compute}%</span>
          </div>
          <div className="processing-stat">
            <span className="processing-stat-label">Conf. Stability</span>
            <span className="processing-stat-value processing-stat-value--green">{stability}%</span>
          </div>
        </div>
      </div>
    </div>
  )
}
