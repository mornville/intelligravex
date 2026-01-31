import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiGet } from '../api/client'
import {
  ArrowRightIcon,
  CpuChipIcon,
  GlobeAltIcon,
  SparklesIcon,
  ShieldCheckIcon,
  BoltIcon,
  CubeTransparentIcon,
  CodeBracketSquareIcon,
  UserIcon,
} from '@heroicons/react/24/solid'

function RobotIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="7" width="16" height="12" rx="3" />
      <path d="M12 3v4" />
      <path d="M9 13h.01" />
      <path d="M15 13h.01" />
      <path d="M9 16h6" />
    </svg>
  )
}

type GuideMessage = {
  id: string
  role: 'user' | 'assistant'
  text: string
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  return `id_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
}

function wsBase(): string {
  try {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${proto}//${window.location.host}`
  } catch {
    return 'ws://127.0.0.1:8000'
  }
}

function FloatingGuide() {
  const [open, setOpen] = useState(false)
  const [botId, setBotId] = useState<string | null>(null)
  const [messages, setMessages] = useState<GuideMessage[]>([])
  const [input, setInput] = useState('')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [stage, setStage] = useState<'idle' | 'busy' | 'disconnected'>('disconnected')
  const wsRef = useRef<WebSocket | null>(null)
  const draftIdRef = useRef<string | null>(null)
  const pendingRef = useRef<string | null>(null)
  const initRequestedRef = useRef(false)
  const bodyRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let canceled = false
    void (async () => {
      try {
        const res = await apiGet<{ id: string }>('/api/system-bot')
        if (!canceled) setBotId(res.id)
      } catch {
        if (!canceled) setErr('Guide is unavailable.')
      }
    })()
    return () => {
      canceled = true
    }
  }, [])

  useEffect(() => {
    if (!open || !botId) return
    if (wsRef.current) return
    const ws = new WebSocket(`${wsBase()}/ws/bots/${botId}/talk`)
    wsRef.current = ws
    ws.onopen = () => {
      setStage('idle')
      if (!conversationId && !initRequestedRef.current) {
        initRequestedRef.current = true
        ws.send(JSON.stringify({ type: 'init', req_id: makeId(), speak: false, test_flag: false, debug: false }))
      }
    }
    ws.onclose = () => setStage('disconnected')
    ws.onerror = () => setStage('disconnected')
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return
      let msg: any
      try {
        msg = JSON.parse(ev.data)
      } catch {
        return
      }
      if (msg.type === 'status') {
        setStage(msg.stage === 'idle' ? 'idle' : 'busy')
        return
      }
      if (msg.type === 'conversation') {
        const cid = msg.conversation_id || msg.id
        if (cid) setConversationId(String(cid))
        if (pendingRef.current) {
          const text = pendingRef.current
          pendingRef.current = null
          ws.send(
            JSON.stringify({
              type: 'chat',
              req_id: makeId(),
              conversation_id: cid,
              speak: false,
              test_flag: false,
              debug: false,
              text,
            }),
          )
        }
        return
      }
      if (msg.type === 'text_delta') {
        const delta = String(msg.delta || '')
        if (!delta) return
        if (!draftIdRef.current) draftIdRef.current = makeId()
        const draftId = draftIdRef.current
        setMessages((prev) => {
          const has = prev.some((m) => m.id === draftId)
          if (!has) return [...prev, { id: draftId, role: 'assistant', text: delta }]
          return prev.map((m) => (m.id === draftId ? { ...m, text: m.text + delta } : m))
        })
        return
      }
      if (msg.type === 'done') {
        const doneText = String(msg.text || '')
        const draftId = draftIdRef.current
        setMessages((prev) => {
          const has = draftId ? prev.some((m) => m.id === draftId) : false
          if (doneText.trim() && (!draftId || !has)) {
            const newId = makeId()
            draftIdRef.current = newId
            return [...prev, { id: newId, role: 'assistant', text: doneText }]
          }
          if (draftId && has) {
            return prev.map((m) => (m.id === draftId ? { ...m, text: doneText || m.text } : m))
          }
          return prev
        })
        draftIdRef.current = null
      }
      if (msg.type === 'error') {
        setMessages((prev) => [...prev, { id: makeId(), role: 'assistant', text: `Error: ${msg.error || 'Unknown'}` }])
      }
    }
    return () => {
      try {
        ws.close()
      } catch {
        // ignore
      }
      wsRef.current = null
    }
  }, [open, botId, conversationId])

  useEffect(() => {
    const el = bodyRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [messages, stage])

  const starters = useMemo(
    () => [
      'What can GravexStudio do?',
      'How does the Data Agent work?',
      'What tools and scripts can I run?',
      'Do I need Docker to use the Data Agent?',
    ],
    [],
  )

  function send(text: string) {
    const clean = text.trim()
    if (!clean) return
    setErr(null)
    setMessages((prev) => [...prev, { id: makeId(), role: 'user', text: clean }])
    setInput('')
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setErr('Guide is connecting…')
      pendingRef.current = clean
      return
    }
    if (!conversationId) {
      pendingRef.current = clean
      if (!initRequestedRef.current) {
        initRequestedRef.current = true
        wsRef.current.send(JSON.stringify({ type: 'init', req_id: makeId(), speak: false, test_flag: false, debug: false }))
      }
      return
    }
    wsRef.current.send(
      JSON.stringify({
        type: 'chat',
        req_id: makeId(),
        conversation_id: conversationId,
        speak: false,
        test_flag: false,
        debug: false,
        text: clean,
      }),
    )
  }

  return (
    <div className={`floatingGuide ${open ? 'open' : ''}`}>
      <button className="floatingToggle" onClick={() => setOpen((p) => !p)} aria-label="Open guide">
        <SparklesIcon aria-hidden="true" />
      </button>
      {open ? (
        <div className="floatingPanel">
          <div className="floatingHeader">
            <div>
              <div className="floatingTitle">GravexStudio Guide</div>
              <div className="muted">Ask anything about the platform.</div>
            </div>
            <button className="btn iconBtn" onClick={() => setOpen(false)} aria-label="Close">
              ×
            </button>
          </div>
          <div className="floatingBody" ref={bodyRef}>
            {messages.map((m) => (
              <div key={m.id} className={`floatingMsg ${m.role}`}>
                <span className="floatingMsgIcon" aria-hidden="true">
                  {m.role === 'assistant' ? <RobotIcon /> : <UserIcon />}
                </span>
                <span className="floatingMsgText">{m.text}</span>
              </div>
            ))}
            {stage === 'busy' && !draftIdRef.current ? (
              <div className="floatingTyping" aria-label="Assistant is typing">
                <span />
                <span />
                <span />
              </div>
            ) : null}
          </div>
          {err ? <div className="floatingError">{err}</div> : null}
          <div className="floatingStarters">
            {starters.map((s) => (
              <button key={s} className="floatingChip" onClick={() => send(s)}>
                {s}
              </button>
            ))}
          </div>
          <div className="floatingInput">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={stage === 'busy' ? 'Thinking…' : 'Ask a question'}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  send(input)
                }
              }}
            />
            <button className="btn primary" onClick={() => send(input)} disabled={!input.trim()}>
              Send
            </button>
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default function LandingPage() {
  useEffect(() => {
    const cards = Array.from(document.querySelectorAll<HTMLElement>('.featureCard.reveal'))
    if (cards.length === 0) return
    if (typeof window === 'undefined' || !('IntersectionObserver' in window)) {
      cards.forEach((card) => card.classList.add('in'))
      return
    }
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('in')
            observer.unobserve(entry.target)
          }
        })
      },
      { threshold: 0.2 },
    )
    cards.forEach((card) => observer.observe(card))
    return () => observer.disconnect()
  }, [])

  const heroHighlights = [
    {
      title: 'A personal automation studio',
      body: 'Set it up once and let your assistants handle the busywork.',
    },
    {
      title: 'Parallel workspaces per task',
      body: 'Each conversation can run its own isolated workspace for long‑running tasks.',
    },
    {
      title: 'Local‑first control',
      body: 'Files, configs, and context stay on your device unless you choose to share them.',
    },
  ]
  const capabilities = [
    {
      icon: <CpuChipIcon aria-hidden="true" />,
      title: 'Realtime responses',
      body: 'Stream replies, track latency, and keep conversations responsive.',
    },
    {
      icon: <GlobeAltIcon aria-hidden="true" />,
      title: 'Live web search',
      body: 'Ground answers with fresh sources, controlled and auditable.',
    },
    {
      icon: <CubeTransparentIcon aria-hidden="true" />,
      title: 'Tool orchestration',
      body: 'Define tool schemas, map responses, and chain reliable flows.',
    },
    {
      icon: <BoltIcon aria-hidden="true" />,
      title: 'Parallel Data Agent runtime',
      body: 'Spin up a container per conversation with a persistent workspace.',
    },
    {
      icon: <SparklesIcon aria-hidden="true" />,
      title: 'Scripted automations',
      body: 'Use scripts to post‑process large tool responses and save outputs to the workspace.',
    },
    {
      icon: <ShieldCheckIcon aria-hidden="true" />,
      title: 'Local‑first storage',
      body: 'Workspaces and configs live on your machine; you decide what gets sent out.',
    },
    {
      icon: <CodeBracketSquareIcon aria-hidden="true" />,
      title: 'Git + SSH tooling',
      body: 'Connect repos securely for data‑agent workflows.',
    },
  ]
  const useCases = [
    {
      title: 'Personal ops',
      body: 'Run data pulls, automate routines, and summarize updates live.',
    },
    {
      title: 'Research copilots',
      body: 'Blend web search, tools, and notes into a single flow.',
    },
    {
      title: 'Automated errands',
      body: 'Turn repetitive tasks into reliable, scripted workflows.',
    },
    {
      title: 'Creative workflows',
      body: 'Draft, refine, and produce content with tool‑backed context.',
    },
  ]
  const stats = [
    { value: 'Parallel workspaces', label: 'One container per conversation' },
    { value: 'Persistent memory', label: 'Summaries keep context lean' },
    { value: 'Tool‑native', label: 'Web search + APIs + scripts' },
  ]
  const workflow = [
    {
      title: 'Describe what you want',
      body: 'Set goals, guardrails, and response style in one place.',
    },
    {
      title: 'Run tasks in parallel',
      body: 'Let multiple workspaces handle jobs at the same time.',
    },
    {
      title: 'Summaries keep it fast',
      body: 'Long context is automatically summarized over time.',
    },
  ]

  return (
    <div className="landing">
      <section className="landingSection hero" id="top">
        <div className="heroGrid">
          <div className="heroCopy">
            <div className="heroKicker">GravexStudio</div>
            <h1>One studio. Many assistants. Unlimited parallel work.</h1>
            <p className="muted">
              Build your personal automation layer with assistants that spin up isolated workspaces, run tasks in
              parallel, and keep project data anchored on your device.
            </p>
            <div className="heroActions">
              <Link className="btn primary" to="/dashboard">
                Go to dashboard <ArrowRightIcon aria-hidden="true" />
              </Link>
              <a className="btn ghost" href="#capabilities">
                Explore capabilities
              </a>
            </div>
            <div className="heroBadges">
              <span>Parallel workspaces</span>
              <span>Persistent context</span>
              <span>Stored locally</span>
            </div>
          </div>
          <div className="heroVisual">
            <div className="heroOrb" />
            <div className="heroStack">
              {heroHighlights.map((s, idx) => (
                <div key={s.title} className="heroCard" style={{ animationDelay: `${idx * 120}ms` }}>
                  <div className="heroCardTitle">{s.title}</div>
                  <div className="heroCardBody">{s.body}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="scrollHint">Scroll to explore</div>
      </section>

      <section className="landingSection" id="capabilities">
        <div className="sectionHeader">
          <div className="sectionKicker">Capabilities</div>
          <h2>Everything you need to go from idea → shipped assistant.</h2>
        </div>
        <div className="featureGrid">
          {capabilities.map((cap, idx) => (
            <div key={cap.title} className="featureCard reveal" style={{ animationDelay: `${idx * 90}ms` }}>
              {cap.icon}
              <h3>{cap.title}</h3>
              <p>{cap.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="landingSection spotlight">
        <div className="sectionHeader">
          <div className="sectionKicker">Built for outcomes</div>
          <h2>Turn your workload into automated flows.</h2>
        </div>
        <div className="useCaseGrid">
          {useCases.map((useCase) => (
            <div key={useCase.title} className="useCaseCard">
              <h3>{useCase.title}</h3>
              <p>{useCase.body}</p>
            </div>
          ))}
        </div>
        <div className="statGrid">
          {stats.map((stat) => (
            <div key={stat.value} className="statCard">
              <div className="statValue">{stat.value}</div>
              <div className="muted">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="landingSection" id="workflow">
        <div className="sectionHeader">
          <div className="sectionKicker">Workflow</div>
          <h2>Design, test, and ship—without context switching.</h2>
        </div>
        <div className="timeline">
          {workflow.map((step) => (
            <div key={step.title} className="timelineItem">
              <div className="timelineDot" />
              <div>
                <h4>{step.title}</h4>
                <p>{step.body}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <FloatingGuide />
    </div>
  )
}
