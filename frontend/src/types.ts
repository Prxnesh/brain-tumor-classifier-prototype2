export type Modality = 'mri' | 'ct' | 'fusion'
export type Page = 'analysis' | 'history'
export type ThemeMode = 'light' | 'dark'

export interface ClassProbability {
  label: string
  probability: number
}

export interface ReportSection {
  title: string
  body: string
}

export interface TumorLocation {
  cx: number
  cy: number
  radius: number
  quadrant: string
  description: string
}

export interface PredictionResponse {
  modality: Modality
  predicted_label: string
  confidence: number
  probabilities: ClassProbability[]
  gradcam_overlay: string
  original_preview: string
  report: ReportSection[]
  report_provider: 'template' | 'ollama'
  notes: string[]
  tumor_location?: TumorLocation
}

export interface AppConfig {
  supported_modalities: string[]
  available_now: string[]
  pending_datasets: string[]
  ollama_available: boolean
  ollama_model: string | null
}

export interface PatientInfo {
  name: string
  id: string
}

export interface HistoryEntry {
  id: string
  timestamp: string
  patient: PatientInfo
  modality: Modality
  predicted_label: string
  confidence: number
  probabilities: ClassProbability[]
  tumor_location?: TumorLocation
  report_provider: 'template' | 'ollama'
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}
