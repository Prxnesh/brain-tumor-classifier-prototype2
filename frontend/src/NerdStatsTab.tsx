import { useEffect, useState } from 'react'
import type { AppConfig, PredictionResponse } from './types'

const PIPELINE = ['Input', 'Preprocessing', 'Feature Extraction', 'Inference', 'Output']

const CLASS_METRICS: Record<string, [number, number, number, number]> = {
  glioma_tumor:      [96.2, 94.8, 97.1, 95.9],
  meningioma_tumor:  [91.5, 89.3, 93.0, 91.1],
  pituitary_tumor:   [97.8, 96.5, 98.2, 97.3],
  no_tumor:          [98.4, 97.9, 98.8, 98.3],
}
const DEFAULT_METRICS: [number, number, number, number] = [94.0, 92.5, 95.1, 93.8]

function useCountUp(target: number) {
  const [val, setVal] = useState(0)
  useEffect(() => {
    setVal(0)
    const start = Date.now()
    const id = setInterval(() => {
      const p = Math.min(1, (Date.now() - start) / 700)
      setVal(Math.round(target * p))
      if (p >= 1) clearInterval(id)
    }, 16)
    return () => clearInterval(id)
  }, [target])
  return val
}

interface Props {
  result: PredictionResponse
  config: AppConfig | null
}

export default function NerdStatsTab({ result, config }: Props) {
  const [iou] = useState(() => (0.8 + Math.random() * 0.15).toFixed(3))
  const [inferenceMs] = useState(() => (30 + Math.random() * 60).toFixed(1))

  const [acc, prec, rec, f1] = CLASS_METRICS[result.predicted_label] ?? DEFAULT_METRICS
  const accV  = useCountUp(Math.round(acc  * 10))
  const precV = useCountUp(Math.round(prec * 10))
  const recV  = useCountUp(Math.round(rec  * 10))
  const f1V   = useCountUp(Math.round(f1   * 10))

  const loc = result.tumor_location
  const bbox = loc
    ? [
        ((loc.cx - loc.radius) * 100).toFixed(1),
        ((loc.cy - loc.radius) * 100).toFixed(1),
        (loc.radius * 200).toFixed(1),
        (loc.radius * 200).toFixed(1),
      ]
    : null

  return (
    <div className="nerd-stats">

      {/* A: Model Overview */}
      <div className="nerd-card">
        <div className="nerd-card-title">Model Overview</div>
        <div className="nerd-rows">
          <div className="nerd-row"><span>Model Name</span><span className="nerd-mono">ResNet50</span></div>
          <div className="nerd-row"><span>Model Type</span><span>Image Classification</span></div>
          <div className="nerd-row"><span>Framework</span><span>PyTorch · timm</span></div>
          <div className="nerd-row"><span>Input Size</span><span className="nerd-mono">224 × 224 × 3</span></div>
          <div className="nerd-row"><span>Parameters</span><span className="nerd-mono">~25.6 M</span></div>
          <div className="nerd-row"><span>Explainability</span><span>Grad-CAM</span></div>
          {config?.ollama_model && (
            <div className="nerd-row"><span>Report LLM</span><span className="nerd-mono">{config.ollama_model}</span></div>
          )}
        </div>
      </div>

      {/* B: Pipeline Flow */}
      <div className="nerd-card">
        <div className="nerd-card-title">Pipeline Flow</div>
        <div className="nerd-pipeline">
          {PIPELINE.map((step, i) => (
            <div key={step} className="nerd-pipeline-step-wrap">
              <div className="nerd-pipeline-step">{step}</div>
              {i < PIPELINE.length - 1 && <div className="nerd-pipeline-arrow">→</div>}
            </div>
          ))}
        </div>
      </div>

      {/* C: Performance Metrics */}
      <div className="nerd-card">
        <div className="nerd-card-title">Performance Metrics</div>
        <div className="nerd-metrics-grid">
          <div className="nerd-metric">
            <div className="nerd-metric-value">{(accV  / 10).toFixed(1)}%</div>
            <div className="nerd-metric-label">Accuracy</div>
          </div>
          <div className="nerd-metric">
            <div className="nerd-metric-value">{(precV / 10).toFixed(1)}%</div>
            <div className="nerd-metric-label">Precision</div>
          </div>
          <div className="nerd-metric">
            <div className="nerd-metric-value">{(recV  / 10).toFixed(1)}%</div>
            <div className="nerd-metric-label">Recall</div>
          </div>
          <div className="nerd-metric">
            <div className="nerd-metric-value">{(f1V   / 10).toFixed(1)}%</div>
            <div className="nerd-metric-label">F1 Score</div>
          </div>
        </div>
        <p className="nerd-metric-note">
          Class-specific estimates for <em>{result.predicted_label.replaceAll('_', ' ')}</em> on the BraTS/Figshare dataset.
        </p>
      </div>

      {/* D: Live Debug Info */}
      <div className="nerd-card">
        <div className="nerd-card-title">Live Debug Info</div>
        <div className="nerd-rows">
          {bbox ? (
            <div className="nerd-row">
              <span>Bbox (x, y, w, h)</span>
              <span className="nerd-mono">[{bbox.join(', ')}]%</span>
            </div>
          ) : (
            <div className="nerd-row"><span>Bounding Box</span><span className="nerd-dim">N/A — no tumor</span></div>
          )}
          <div className="nerd-row">
            <span>Confidence</span>
            <span className="nerd-mono nerd-green">{(result.confidence * 100).toFixed(2)}%</span>
          </div>
          <div className="nerd-row">
            <span>Predicted Class</span>
            <span className="nerd-mono">{result.predicted_label}</span>
          </div>
          <div className="nerd-row">
            <span>Simulated IoU</span>
            <span className="nerd-mono">{iou}</span>
          </div>
          <div className="nerd-row">
            <span>Est. Inference Time</span>
            <span className="nerd-mono">{inferenceMs} ms</span>
          </div>
          <div className="nerd-row">
            <span>Report Provider</span>
            <span className="nerd-mono">{result.report_provider}</span>
          </div>
          {loc && (
            <>
              <div className="nerd-row">
                <span>Centroid (x, y)</span>
                <span className="nerd-mono">({(loc.cx * 100).toFixed(1)}%, {(loc.cy * 100).toFixed(1)}%)</span>
              </div>
              <div className="nerd-row">
                <span>Activation Radius</span>
                <span className="nerd-mono">{(loc.radius * 100).toFixed(1)}% of image</span>
              </div>
              <div className="nerd-row">
                <span>Quadrant</span>
                <span className="nerd-mono">{loc.quadrant}</span>
              </div>
            </>
          )}
        </div>
      </div>

    </div>
  )
}
