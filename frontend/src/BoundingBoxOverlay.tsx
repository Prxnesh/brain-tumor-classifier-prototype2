import type { TumorLocation } from './types'

interface BoundingBoxOverlayProps {
  location: TumorLocation
  label: string
  confidence: number
  color?: string
}

export default function BoundingBoxOverlay({
  location,
  label,
  confidence,
  color = '#EA4335',
}: BoundingBoxOverlayProps) {
  const pad = 0.04
  const x = Math.max(0, location.cx - location.radius - pad) * 100
  const y = Math.max(0, location.cy - location.radius - pad) * 100
  const w = (Math.min(1, location.cx + location.radius + pad) - Math.max(0, location.cx - location.radius - pad)) * 100
  const h = (Math.min(1, location.cy + location.radius + pad) - Math.max(0, location.cy - location.radius - pad)) * 100

  return (
    <div
      className="bbox-overlay"
      style={{ left: `${x}%`, top: `${y}%`, width: `${w}%`, height: `${h}%`, borderColor: color }}
      aria-label={`Bounding box: ${label}`}
    >
      <div className="bbox-label" style={{ background: color }}>
        {label.replaceAll('_', ' ')} · {(confidence * 100).toFixed(1)}%
      </div>
    </div>
  )
}
