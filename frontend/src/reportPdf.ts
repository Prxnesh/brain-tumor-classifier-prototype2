import { jsPDF } from 'jspdf'
import type { PredictionResponse, TumorLocation } from './types'

type ExportOptions = {
  generatedAt: Date
  patientName?: string
  patientId?: string
  authorName: string
}

function formatLabel(label: string) {
  return label.replaceAll('_', ' ')
}

function buildFilename(result: PredictionResponse) {
  const safeLabel = result.predicted_label.replaceAll('_', '-')
  return `neurovision-${result.modality}-${safeLabel}-report.pdf`
}

function formatLocation(loc: TumorLocation): string {
  return `${loc.description} (center: ${(loc.cx * 100).toFixed(0)}%, ${(loc.cy * 100).toFixed(0)}%; radius: ${(loc.radius * 100).toFixed(0)}% of image)`
}

export function exportReportPdf(result: PredictionResponse, options: ExportOptions) {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' })
  const pageWidth = doc.internal.pageSize.getWidth()
  const pageHeight = doc.internal.pageSize.getHeight()
  const marginX = 52
  const topY = 52
  const maxTextWidth = pageWidth - marginX * 2
  const lineHeight = 16
  const sectionGap = 18
  let cursorY = topY

  const addFooter = () => {
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(9)
    doc.setTextColor(130, 140, 150)
    doc.text(
      `NeuroVision AI  ·  Generated ${options.generatedAt.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}  ·  Built by ${options.authorName}`,
      marginX,
      pageHeight - 28,
    )
    doc.setDrawColor(218, 220, 224)
    doc.line(marginX, pageHeight - 38, pageWidth - marginX, pageHeight - 38)
  }

  const addPageIfNeeded = (requiredHeight: number) => {
    if (cursorY + requiredHeight <= pageHeight - 52) return
    addFooter()
    doc.addPage()
    cursorY = topY
  }

  const writeWrapped = (text: string, fontSize = 11, color: [number, number, number] = [44, 57, 70], bold = false) => {
    doc.setFont('helvetica', bold ? 'bold' : 'normal')
    doc.setFontSize(fontSize)
    doc.setTextColor(...color)
    const lines = doc.splitTextToSize(text, maxTextWidth)
    addPageIfNeeded(lines.length * lineHeight + 4)
    doc.text(lines, marginX, cursorY)
    cursorY += lines.length * lineHeight
  }

  const writeSection = (title: string, body: string) => {
    addPageIfNeeded(48)
    // Section title bar
    doc.setFillColor(241, 243, 244)
    doc.roundedRect(marginX - 8, cursorY - 12, maxTextWidth + 16, 22, 4, 4, 'F')
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(11)
    doc.setTextColor(32, 33, 36)
    doc.text(title.toUpperCase(), marginX, cursorY + 2)
    cursorY += 20
    writeWrapped(body, 11, [95, 99, 104])
    cursorY += sectionGap
  }

  const rankedProbabilities = [...result.probabilities].sort((a, b) => b.probability - a.probability)
  const patientLabel = options.patientName?.trim() || 'Not provided'
  const patientId = options.patientId?.trim() || '—'
  const isTumor = result.predicted_label !== 'no_tumor'

  // ── Header ───────────────────────────────────────────────────────────────
  doc.setFillColor(26, 115, 232)  // Google Blue
  doc.rect(0, 0, pageWidth, 88, 'F')

  // White stripe at bottom of header
  doc.setFillColor(255, 255, 255)
  doc.rect(0, 82, pageWidth, 6, 'F')

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(20)
  doc.setTextColor(255, 255, 255)
  doc.text('NeuroVision', marginX, 36)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(11)
  doc.setTextColor(210, 227, 252)
  doc.text('AI-Assisted Brain Imaging Report  ·  For decision support only', marginX, 56)

  // Confidence badge on the right
  const confStr = `${(result.confidence * 100).toFixed(1)}%`
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(22)
  doc.setTextColor(255, 255, 255)
  doc.text(confStr, pageWidth - marginX - 10, 40, { align: 'right' })
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(9)
  doc.setTextColor(180, 210, 252)
  doc.text('CONFIDENCE', pageWidth - marginX - 10, 54, { align: 'right' })

  cursorY = 112

  // ── Patient Info block ───────────────────────────────────────────────────
  doc.setFillColor(248, 249, 250)
  doc.roundedRect(marginX - 8, cursorY - 12, maxTextWidth + 16, 78, 6, 6, 'F')
  doc.setDrawColor(218, 220, 224)
  doc.roundedRect(marginX - 8, cursorY - 12, maxTextWidth + 16, 78, 6, 6, 'S')

  const col1 = marginX
  const col2 = marginX + (maxTextWidth + 16) / 2
  const rowH = 18

  const metaField = (label: string, value: string, x: number, y: number) => {
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(8)
    doc.setTextColor(95, 99, 104)
    doc.text(label.toUpperCase(), x, y)
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(11)
    doc.setTextColor(32, 33, 36)
    doc.text(value, x, y + 13)
  }

  metaField('Patient Name', patientLabel, col1, cursorY)
  metaField('Analysis ID', patientId, col2, cursorY)
  cursorY += rowH + 18
  metaField('Modality', result.modality.toUpperCase(), col1, cursorY)
  metaField(
    'Generated',
    options.generatedAt.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }),
    col2,
    cursorY,
  )
  cursorY += rowH + 24

  // ── Primary finding ──────────────────────────────────────────────────────
  const findingColor: [number, number, number] = isTumor ? [197, 34, 31] : [19, 115, 51]
  const findingBg: [number, number, number] = isTumor ? [253, 236, 234] : [230, 244, 234]

  doc.setFillColor(...findingBg)
  doc.roundedRect(marginX - 8, cursorY - 12, maxTextWidth + 16, 36, 6, 6, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(15)
  doc.setTextColor(...findingColor)
  doc.text(`${formatLabel(result.predicted_label).toUpperCase()}`, marginX, cursorY + 5)
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(11)
  doc.setTextColor(95, 99, 104)
  doc.text(`Report engine: ${result.report_provider === 'ollama' ? 'Ollama-enhanced' : 'Built-in template'}`, col2, cursorY + 5)
  cursorY += 40

  // ── Tumor location ───────────────────────────────────────────────────────
  if (result.tumor_location && isTumor) {
    writeSection('Predicted Tumor Location (Grad-CAM)', formatLocation(result.tumor_location))
  }

  // ── Probability Summary ──────────────────────────────────────────────────
  writeSection(
    'Probability Summary',
    rankedProbabilities
      .map((entry, i) => `${i + 1}. ${formatLabel(entry.label)} — ${(entry.probability * 100).toFixed(2)}%`)
      .join('\n'),
  )

  // ── Report sections ──────────────────────────────────────────────────────
  for (const section of result.report) {
    writeSection(section.title, section.body)
  }

  // ── Operational notes ────────────────────────────────────────────────────
  if (result.notes.length) {
    writeSection(
      'Operational Notes',
      result.notes.map((note, i) => `${i + 1}. ${note}`).join('\n'),
    )
  }

  // ── Clinical use statement ───────────────────────────────────────────────
  writeSection(
    'Clinical Use Statement',
    'This document is generated from the NeuroVision AI research prototype for decision support and structured communication only. ' +
      'It does not replace complete multi-sequence imaging review, formal radiology reporting, pathology correlation, or clinician judgment. ' +
      'All findings must be correlated with clinical context and expert review before any diagnostic or treatment decision.',
  )

  addFooter()
  doc.save(buildFilename(result))
}
