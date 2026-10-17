import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowUp, BookOpenText, Check, ChevronDown, Copy, FilePenLine, Square, Zap } from 'lucide-react'
import { api, stream } from '../api'
import Markdown from './Markdown.jsx'

const SUGGESTIONS = [
  'What is a security deposit and when does it have to be returned?',
  'Which clauses should a non-disclosure agreement include?',
  'What formalities make a will valid?',
  'What does a legal notice for a bounced cheque need to contain?',
]

const STAGES = { retrieving: 'Searching the documents', writing: 'Writing the answer' }
const secs = (ms) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`)

function Sources({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources?.length) return null
  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        <BookOpenText size={14} />
        {sources.length} {sources.length === 1 ? 'source' : 'sources'}
        <span className="tabs-inline">
          {[...new Set(sources.map((s) => s.page))].slice(0, 6).map((p) => <code key={p}>p.{p}</code>)}
        </span>
        <ChevronDown size={14} className={open ? 'flip' : ''} />
      </button>
      {open && (
        <ul>
          {sources.map((s, i) => (
            <li key={i}><code>p.{s.page}</code><span>{s.snippet}…</span></li>
          ))}
        </ul>
      )}
    </div>
  )
}

function Answer({ m }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try { await navigator.clipboard.writeText(m.content); setCopied(true); setTimeout(() => setCopied(false), 1400) } catch { /* blocked */ }
  }
  return (
    <article className="a">
      <div className="prose">
        {m.content ? <Markdown>{m.content}</Markdown> : null}
        {m.streaming && (
          <p className="stage"><span className="dots"><i /><i /><i /></span>{STAGES[m.stage] || 'Thinking'}</p>
        )}
      </div>
      {!m.streaming && m.content && (
        <>
          <Sources sources={m.sources} />
          <div className="a-foot">
            {m.meta?.latency_ms != null && (
              <span className="meta">
                {m.meta.cached && <span className="pill fast"><Zap size={12} />from cache</span>}
                answered in {secs(m.meta.latency_ms)}
                {m.meta.hyde && <span className="pill">HyDE</span>}
              </span>
            )}
            <button className="mini text" onClick={copy}>{copied ? <><Check size={13} /> Copied</> : <><Copy size={13} /> Copy</>}</button>
          </div>
        </>
      )}
    </article>
  )
}

export default function Chat({ ws, chatId, onCreated, onChanged, onDraft }) {
  const [cid, setCid] = useState(chatId)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(!!chatId)
  const abortRef = useRef(null)
  const scrollRef = useRef(null)
  const stick = useRef(true)
  const taRef = useRef(null)

  useEffect(() => {
    if (!chatId) return
    let live = true
    api.chat(chatId).then((c) => { if (live) { setMessages(c.messages.map((m) => ({ ...m, meta: m.meta }))); setLoading(false) } })
      .catch(() => live && setLoading(false))
    return () => { live = false }
  }, []) // eslint-disable-line

  useEffect(() => {
    const el = scrollRef.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [messages])

  const onScroll = () => {
    const el = scrollRef.current
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }

  const patchLast = useCallback((fn) => setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m))), [])

  const send = async (raw) => {
    const text = raw.trim()
    if (!text || busy) return
    setBusy(true); setInput(''); stick.current = true
    if (taRef.current) taRef.current.style.height = 'auto'
    setMessages((ms) => [...ms, { role: 'user', content: text }, { role: 'assistant', content: '', streaming: true, stage: 'retrieving', sources: [] }])
    const ctrl = new AbortController()
    abortRef.current = ctrl
    try {
      let id = cid
      if (!id) {
        const c = await api.createChat(ws.id)
        id = c.id; setCid(id); onCreated(id)
      }
      for await (const ev of stream(`/chats/${id}/messages`, { content: text }, ctrl.signal)) {
        if (ev.type === 'status') patchLast((m) => ({ ...m, stage: ev.stage }))
        else if (ev.type === 'sources') patchLast((m) => ({ ...m, sources: ev.sources }))
        else if (ev.type === 'token') patchLast((m) => ({ ...m, content: m.content + ev.text }))
        else if (ev.type === 'replace') patchLast((m) => ({ ...m, content: ev.text }))
        else if (ev.type === 'done') { const { type, ...meta } = ev; patchLast((m) => ({ ...m, meta })) }
      }
    } catch (e) {
      if (e.name !== 'AbortError') patchLast((m) => ({ ...m, content: m.content || `Something went wrong: ${e.message}. Check that the server is running and try again.` }))
    } finally {
      patchLast((m) => ({ ...m, streaming: false }))
      setBusy(false); abortRef.current = null
      onChanged()
    }
  }

  const grow = (e) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px'
  }

  const empty = !loading && messages.length === 0

  return (
    <section className="chat">
      <header className="topbar">
        <div>
          <h2>{ws.name}</h2>
          {ws.description && <p title={ws.description}>{ws.description}</p>}
        </div>
      </header>

      <div className="scroll" ref={scrollRef} onScroll={onScroll}>
        {empty ? (
          <div className="welcome">
            <h1>What are you researching?</h1>
            <p>Ask about the legal documents in this workspace. Answers cite the page they came from.</p>
            <div className="suggest">
              {SUGGESTIONS.map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}
            </div>
            <button className="link" onClick={onDraft}><FilePenLine size={15} /> Or start drafting a document</button>
          </div>
        ) : (
          <div className="thread">
            {messages.map((m, i) => m.role === 'user'
              ? <div className="q" key={i}>{m.content}</div>
              : <Answer m={m} key={i} />)}
          </div>
        )}
      </div>

      <footer className="composer-wrap">
        <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input) }}>
          <textarea ref={taRef} rows={1} value={input} onChange={grow} placeholder="Ask a legal question..."
            aria-label="Your question"
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }} />
          {busy
            ? <button type="button" className="send stop" onClick={() => abortRef.current?.abort()} aria-label="Stop"><Square size={14} fill="currentColor" /></button>
            : <button className="send" disabled={!input.trim()} aria-label="Send"><ArrowUp size={18} /></button>}
        </form>
        <p className="fine">General legal information drawn from your documents, not legal advice.</p>
      </footer>
    </section>
  )
}
