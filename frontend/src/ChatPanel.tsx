import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import type { AppConfig, ChatMessage, PredictionResponse } from './types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

interface Props {
  isOpen: boolean
  onClose: () => void
  result: PredictionResponse | null
  patientName: string
  config: AppConfig | null
}

function IconClose() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  )
}

function IconSend() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
    </svg>
  )
}

function IconBot() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7H3a7 7 0 0 1 7-7h1V5.73A2 2 0 0 1 10 4a2 2 0 0 1 2-2M5 14v1a7 7 0 0 0 14 0v-1H5m3 4h8a5 5 0 0 1-8 0Z" />
    </svg>
  )
}

function IconUser() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z" />
    </svg>
  )
}

function formatLabel(label: string) {
  return label.replaceAll('_', ' ')
}

function buildWelcomeMessage(result: PredictionResponse | null, patientName: string): string {
  if (!result) {
    return "Hello! I'm the NeuroVision AI Assistant. Run a scan analysis first, and I'll be able to discuss the results with you — including what the prediction means, prognosis considerations, and recommended next steps."
  }

  const label = formatLabel(result.predicted_label)
  const pct = (result.confidence * 100).toFixed(1)
  const patient = patientName ? ` for ${patientName}` : ''
  const location = result.tumor_location ? ` The model's peak activation was in the **${result.tumor_location.quadrant}** region.` : ''

  if (result.predicted_label === 'no_tumor') {
    return `Analysis complete${patient}. The model returned **no tumor** as the leading class at **${pct}% confidence** on the ${result.modality.toUpperCase()} scan.${location} I can discuss what this means, what findings could still be missed, and recommended follow-up steps. What would you like to know?`
  }

  return `Analysis complete${patient}. The model identified a **${label}** pattern at **${pct}% confidence** on the ${result.modality.toUpperCase()} scan.${location} I can discuss what this tumor type generally involves, typical prognosis considerations, treatment approaches, and what the AI findings mean in clinical context. What would you like to explore?`
}

export default function ChatPanel({ isOpen, onClose, result, patientName, config }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const prevResultRef = useRef<PredictionResponse | null>(null)

  // Inject welcome/context message when result changes or panel opens
  useEffect(() => {
    if (!isOpen) return
    if (result !== prevResultRef.current) {
      prevResultRef.current = result
      const welcome: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: buildWelcomeMessage(result, patientName),
        timestamp: new Date().toISOString(),
      }
      setMessages([welcome])
      setError(null)
    }
  }, [isOpen, result, patientName])

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Focus input when panel opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isOpen])

  async function sendMessage() {
    const text = input.trim()
    if (!text || loading) return

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setLoading(true)
    setError(null)

    const context = result
      ? {
          modality: result.modality,
          prediction: result.predicted_label,
          confidence: result.confidence,
          patient: patientName || undefined,
          tumor_location: result.tumor_location,
        }
      : undefined

    const historyForApi = [...messages, userMsg]
      .filter((m) => m.role !== 'assistant' || messages.indexOf(m) > 0)
      .map((m) => ({ role: m.role, content: m.content }))

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: historyForApi, context }),
      })

      if (!response.ok) {
        const payload = (await response.json()) as { detail?: string }
        throw new Error(payload.detail ?? 'Chat request failed.')
      }

      const data = (await response.json()) as { message: string }
      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.message,
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, assistantMsg])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  function onFormSubmit(e: FormEvent) {
    e.preventDefault()
    sendMessage()
  }

  const ollamaReady = config?.ollama_available ?? false

  return (
    <>
      {/* Backdrop */}
      <div
        className={`chat-backdrop ${isOpen ? 'chat-backdrop--open' : ''}`}
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <aside className={`chat-panel ${isOpen ? 'chat-panel--open' : ''}`} aria-label="AI Chat Assistant">
        {/* Header */}
        <div className="chat-header">
          <div className="chat-header-info">
            <div className="chat-avatar">
              <IconBot />
            </div>
            <div>
              <div className="chat-title">NeuroVision Assistant</div>
              <div className="chat-subtitle">
                {ollamaReady
                  ? `Powered by ${config?.ollama_model ?? 'Ollama'}`
                  : 'Ollama not connected'}
              </div>
            </div>
          </div>
          <button className="chat-close-btn" onClick={onClose} aria-label="Close chat">
            <IconClose />
          </button>
        </div>

        {/* Ollama unavailable banner */}
        {!ollamaReady && (
          <div className="chat-unavailable-banner">
            <strong>Ollama is not running.</strong> Start Ollama and load a model (e.g.{' '}
            <code>ollama run llama3.2:3b</code>) to enable the assistant.
          </div>
        )}

        {/* Messages */}
        <div className="chat-messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`chat-bubble-row chat-bubble-row--${msg.role}`}>
              <div className={`chat-avatar-small chat-avatar-small--${msg.role}`}>
                {msg.role === 'assistant' ? <IconBot /> : <IconUser />}
              </div>
              <div className={`chat-bubble chat-bubble--${msg.role}`}>
                <MessageContent content={msg.content} />
              </div>
            </div>
          ))}

          {loading && (
            <div className="chat-bubble-row chat-bubble-row--assistant">
              <div className="chat-avatar-small chat-avatar-small--assistant">
                <IconBot />
              </div>
              <div className="chat-bubble chat-bubble--assistant chat-bubble--typing">
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
              </div>
            </div>
          )}

          {error && (
            <div className="chat-error">
              <strong>Error:</strong> {error}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <form className="chat-input-area" onSubmit={onFormSubmit}>
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={ollamaReady ? 'Ask about the scan, prognosis, treatment...' : 'Start Ollama to enable chat'}
            disabled={!ollamaReady || loading}
            rows={2}
          />
          <button
            className="chat-send-btn"
            type="submit"
            disabled={!ollamaReady || loading || !input.trim()}
            aria-label="Send message"
          >
            <IconSend />
          </button>
        </form>

        <p className="chat-disclaimer">
          AI output is not medical advice. Always consult a qualified clinician.
        </p>
      </aside>
    </>
  )
}

function MessageContent({ content }: { content: string }) {
  // Simple markdown-like rendering for **bold** and newlines
  const parts = content.split(/(\*\*[^*]+\*\*)/g)
  return (
    <p style={{ margin: 0, whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={i}>{part.slice(2, -2)}</strong>
        }
        return <span key={i}>{part}</span>
      })}
    </p>
  )
}
