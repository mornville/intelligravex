import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiGet } from '../api/client'
import slide1 from '../assets/landing/landing-1.png'
import slide2 from '../assets/landing/landing-2.png'
import slide3 from '../assets/landing/landing-3.png'
import slide4 from '../assets/landing/landing-4.png'
import {
  ArrowRightIcon,
  AdjustmentsHorizontalIcon,
  CameraIcon,
  ChatBubbleBottomCenterTextIcon,
  CommandLineIcon,
  ComputerDesktopIcon,
  CpuChipIcon,
  DeviceTabletIcon,
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
      body: 'Set it up once and let your EMPLOYEES handle the busywork.',
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
      icon: <UserIcon aria-hidden="true" />,
      title: 'Teams (shared chats)',
      body: 'Run shared conversations with multiple agents and route replies by @mention.',
    },
    {
      icon: <GlobeAltIcon aria-hidden="true" />,
      title: 'Live web search',
      body: 'Ground answers with fresh sources, controlled and auditable.',
    },
    {
      icon: <CubeTransparentIcon aria-hidden="true" />,
      title: 'HTTP tools + integrations',
      body: 'Define tool schemas, map responses, and chain reliable flows.',
    },
    {
      icon: <CommandLineIcon aria-hidden="true" />,
      title: 'Host actions',
      body: 'Run local shell commands or AppleScript to get real work done.',
    },
    {
      icon: <CameraIcon aria-hidden="true" />,
      title: 'Screen capture + vision',
      body: 'Capture the screen and summarize what’s happening.',
    },
    {
      icon: <SparklesIcon aria-hidden="true" />,
      title: 'Metadata templating',
      body: 'Use variables in prompts and replies for dynamic, personalized workflows.',
    },
    {
      icon: <AdjustmentsHorizontalIcon aria-hidden="true" />,
      title: 'Multi‑model stack',
      body: 'Tune LLM, ASR, TTS, web search, Codex, and summary models per agent.',
    },
    {
      icon: <CpuChipIcon aria-hidden="true" />,
      title: 'Local LLM runtime',
      body: 'Run GGUF models locally with automatic downloads and no API key.',
    },
    {
      icon: <ChatBubbleBottomCenterTextIcon aria-hidden="true" />,
      title: 'Embeddable widget',
      body: 'Drop a public chat widget anywhere with client keys and WebSocket transport.',
    },
    {
      icon: <BoltIcon aria-hidden="true" />,
      title: 'Optional Data Agent',
      body: 'Run long tasks in a Docker container with a persistent workspace.',
    },
    {
      icon: <ShieldCheckIcon aria-hidden="true" />,
      title: 'Local‑first storage',
      body: 'Workspaces and configs live on your machine; you decide what gets sent out.',
    },
    {
      icon: <CodeBracketSquareIcon aria-hidden="true" />,
      title: 'Codex post‑processing',
      body: 'Extract structured outputs or transform results after tool calls.',
    },
    {
      icon: <DeviceTabletIcon aria-hidden="true" />,
      title: 'Cross‑platform packaging',
      body: 'Ship to macOS, Windows, and Linux with one app.',
    },
  ]
  const featureShowcase = [
    {
      title: 'Host actions',
      short: 'Host actions',
      body: 'Queue local shell or AppleScript tasks for real system work.',
      bullets: ['Great for file ops, settings, and app automation.', 'Approval can be enabled per agent.'],
      icon: <CommandLineIcon aria-hidden="true" />,
      accent: 'rgba(244, 114, 182, 0.9)',
      tags: ['Shell', 'AppleScript'],
    },
    {
      title: 'Teams (shared chats)',
      short: 'Teams',
      body: 'Run shared conversations with multiple agents in one thread.',
      bullets: ['Route replies by @mention.', 'Perfect for multi-agent workflows.'],
      icon: <UserIcon aria-hidden="true" />,
      accent: 'rgba(139, 156, 255, 0.9)',
      tags: ['Groups', 'Routing'],
    },
    {
      title: 'Screen capture + vision',
      short: 'Screen capture',
      body: 'Capture the screen on demand and summarize what’s happening.',
      bullets: ['Perfect for walkthroughs, debugging, and quick audits.', 'Use vision summaries inside workflows.'],
      icon: <CameraIcon aria-hidden="true" />,
      accent: 'rgba(56, 189, 248, 0.9)',
      tags: ['Vision', 'Context'],
    },
    {
      title: 'Live web search',
      short: 'Web search',
      body: 'Fetch fresh sources with searchable citations.',
      bullets: ['Ground answers in current info.', 'Great for research and monitoring.'],
      icon: <GlobeAltIcon aria-hidden="true" />,
      accent: 'rgba(14, 165, 233, 0.9)',
      tags: ['Sources', 'Citations'],
    },
    {
      title: 'HTTP + integration tools',
      short: 'HTTP tools',
      body: 'Call external APIs with schemas and mapped outputs.',
      bullets: ['Validate responses before they hit the model.', 'Map fields into metadata automatically.'],
      icon: <CubeTransparentIcon aria-hidden="true" />,
      accent: 'rgba(167, 139, 250, 0.9)',
      tags: ['Schemas', 'Mapping'],
    },
    {
      title: 'Metadata templating',
      short: 'Templating',
      body: 'Use variables in prompts and replies for dynamic flows.',
      bullets: ['Personalize responses instantly.', 'Store user context across turns.'],
      icon: <SparklesIcon aria-hidden="true" />,
      accent: 'rgba(250, 204, 21, 0.9)',
      tags: ['Dynamic', 'Personalized'],
    },
    {
      title: 'Multi‑model stack',
      short: 'Multi‑model',
      body: 'Tune LLM, ASR, TTS, web search, Codex, and summaries.',
      bullets: ['Mix models per agent.', 'Optimize speed and cost.'],
      icon: <AdjustmentsHorizontalIcon aria-hidden="true" />,
      accent: 'rgba(94, 234, 212, 0.9)',
      tags: ['LLM', 'ASR', 'TTS'],
    },
    {
      title: 'Local LLM runtime',
      short: 'Local LLMs',
      body: 'Run GGUF models locally with automatic downloads and no API key.',
      bullets: ['Tool-call aware model catalog.', 'Great for private, on-device workflows.'],
      icon: <CpuChipIcon aria-hidden="true" />,
      accent: 'rgba(126, 242, 154, 0.9)',
      tags: ['On-device', 'No API key'],
    },
    {
      title: 'Embeddable widget',
      short: 'Widget',
      body: 'Embed a live chat or mic experience on any site.',
      bullets: ['Client keys + WebSocket transport.', 'Customizable experience per agent.'],
      icon: <ChatBubbleBottomCenterTextIcon aria-hidden="true" />,
      accent: 'rgba(59, 130, 246, 0.9)',
      tags: ['Embed', 'Web'],
    },
    {
      title: 'Optional Data Agent',
      short: 'Data Agent',
      body: 'Run long tasks in Docker with a persistent workspace.',
      bullets: ['Parallel containers per conversation.', 'Scripted post‑processing available.'],
      icon: <BoltIcon aria-hidden="true" />,
      accent: 'rgba(129, 140, 248, 0.9)',
      tags: ['Docker', 'Workspace'],
    },
    {
      title: 'Codex post‑processing',
      short: 'Codex',
      body: 'Structured extraction and transformations after tool calls.',
      bullets: ['Great for large or messy responses.', 'Return clean, user‑ready output.'],
      icon: <CodeBracketSquareIcon aria-hidden="true" />,
      accent: 'rgba(52, 211, 153, 0.9)',
      tags: ['Post‑process', 'Structured'],
    },
    {
      title: 'Mic overlay mode',
      short: 'Mic overlay',
      body: 'A floating mic overlay for fast voice workflows.',
      bullets: ['Start/stop sessions instantly.', 'Perfect for hands‑free tasks.'],
      icon: <DeviceTabletIcon aria-hidden="true" />,
      accent: 'rgba(249, 115, 22, 0.9)',
      tags: ['Voice', 'Overlay'],
    },
    {
      title: 'Local‑first storage',
      short: 'Local‑first',
      body: 'Keep configs, workspaces, and context on your device.',
      bullets: ['Share only what you choose.', 'Works offline for local tasks.'],
      icon: <ShieldCheckIcon aria-hidden="true" />,
      accent: 'rgba(74, 222, 128, 0.9)',
      tags: ['Privacy', 'Control'],
    },
    {
      title: 'Cross‑platform app',
      short: 'Cross‑platform',
      body: 'Ship one app across macOS, Windows, and Linux.',
      bullets: ['Consistent experience everywhere.', 'Built for teams and demos.'],
      icon: <ComputerDesktopIcon aria-hidden="true" />,
      accent: 'rgba(196, 181, 253, 0.9)',
      tags: ['macOS', 'Windows', 'Linux'],
    },
  ]
  const [featureIndex, setFeatureIndex] = useState(0)
  const [featurePaused, setFeaturePaused] = useState(false)
  const activeFeature = featureShowcase[featureIndex]
  useEffect(() => {
    if (featurePaused) return
    const id = window.setInterval(() => {
      setFeatureIndex((i) => (i + 1) % featureShowcase.length)
    }, 5200)
    return () => window.clearInterval(id)
  }, [featurePaused, featureShowcase.length])
  const slides = [
    {
      src: slide1,
      title: 'Mission control chat',
      body: 'Keep every employee conversation, tools, and context in one view.',
    },
    {
      src: slide2,
      title: 'Workspace snapshots',
      body: 'See container status and outputs without leaving the flow.',
    },
    {
      src: slide3,
      title: 'Tune employees',
      body: 'Set voice, models, prompts, and behaviors per employee.',
    },
    {
      src: slide4,
      title: 'Data Agent control',
      body: 'Enable long‑running jobs with workspace‑aware tooling.',
    },
  ]
  const [slideIndex, setSlideIndex] = useState(0)
  useEffect(() => {
    const id = window.setInterval(() => {
      setSlideIndex((i) => (i + 1) % slides.length)
    }, 4800)
    return () => window.clearInterval(id)
  }, [slides.length])
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
    { value: 'Local LLM ready', label: 'On-device, tool-call aware' },
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

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <div className="landing">
      <section className="landingSection hero" id="top">
        <div className="heroGrid">
          <div className="heroCopy">
            <div className="heroKicker">GravexStudio</div>
            <h1>One studio. Many EMPLOYEES. Unlimited parallel work.</h1>
            <p className="muted">
              Build your personal automation layer with EMPLOYEES that spin up isolated workspaces, run tasks in
              parallel, and keep project data anchored on your device.
            </p>
            <div className="heroActions">
              <Link className="btn primary" to="/dashboard">
                Go to dashboard <ArrowRightIcon aria-hidden="true" />
              </Link>
              <button className="btn ghost" onClick={() => scrollToSection('capabilities')}>
                Explore capabilities
              </button>
            </div>
            <div className="heroBadges">
              <span>Parallel workspaces</span>
              <span>Persistent context</span>
              <span>Local LLMs</span>
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
      </section>

      <section className="landingSection" id="capabilities">
        <div className="sectionHeader">
          <div className="sectionKicker">Capabilities</div>
          <h2>Everything you need to go from idea → shipped EMPLOYEE.</h2>
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
        <div
          className="featureShowcase"
          onMouseEnter={() => setFeaturePaused(true)}
          onMouseLeave={() => setFeaturePaused(false)}
        >
          <div className="featureLogoRow">
            {featureShowcase.map((feature, idx) => (
              <button
                key={feature.title}
                type="button"
                className={`featureLogo ${idx === featureIndex ? 'active' : ''}`}
                style={
                  {
                    '--accent': feature.accent,
                    animationDelay: `${idx * 120}ms`,
                  } as React.CSSProperties
                }
                onClick={() => setFeatureIndex(idx)}
                onFocus={() => setFeatureIndex(idx)}
                aria-pressed={idx === featureIndex}
              >
                <span className="featureLogoIcon">{feature.icon}</span>
                <span className="featureLogoLabel">{feature.short}</span>
              </button>
            ))}
          </div>
          <div
            key={activeFeature.title}
            className="featurePanel"
            style={{ '--accent': activeFeature.accent } as React.CSSProperties}
          >
            <div className="featurePanelTop">
              <div className="featurePanelIcon">{activeFeature.icon}</div>
              <div>
                <h3>{activeFeature.title}</h3>
                <p>{activeFeature.body}</p>
              </div>
            </div>
            <div className="featurePanelTags">
              {activeFeature.tags.map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
            <ul className="featurePanelList">
              {activeFeature.bullets.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="landingSection" id="studio">
        <div className="sectionHeader">
          <div className="sectionKicker">Studio tour</div>
          <h2>See GravexStudio in action.</h2>
        </div>
        <div className="mediaGrid">
          <div className="mediaSlider">
            <div className="mediaTrack" style={{ transform: `translateX(-${slideIndex * 100}%)` }}>
              {slides.map((s) => (
                <div key={s.title} className="mediaSlide">
                  <img src={s.src} alt={s.title} />
                </div>
              ))}
            </div>
            <div className="mediaDots">
              {slides.map((s, idx) => (
                <button
                  key={s.title}
                  className={`mediaDot ${idx === slideIndex ? 'active' : ''}`}
                  onClick={() => setSlideIndex(idx)}
                  aria-label={`Show ${s.title}`}
                />
              ))}
            </div>
            <div className="mediaCaption">
              <h3>{slides[slideIndex].title}</h3>
              <p>{slides[slideIndex].body}</p>
            </div>
          </div>
          <div className="videoCard">
            <div className="videoFrame">
              <video controls poster={slide1} preload="none">
                <source src="/assets/landing/flow-demo.mp4" type="video/mp4" />
              </video>
              <div className="videoOverlay">
                <div className="videoTitle">Flow walkthrough</div>
                <div className="muted">Drop your demo video at /assets/landing/flow-demo.mp4</div>
              </div>
            </div>
            <div className="mediaCaption">
              <h3>One task, many employees</h3>
              <p>Watch how a request fans out to specialized employees and returns as a finished deliverable.</p>
            </div>
          </div>
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
