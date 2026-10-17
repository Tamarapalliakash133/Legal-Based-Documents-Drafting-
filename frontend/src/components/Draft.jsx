import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, Copy, Download, Eye, FileUp, Pencil, Save, Sparkles, X } from 'lucide-react'
import { api, stream } from '../api'
import Markdown from './Markdown.jsx'

const secs = (ms) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`)
const STAGES = { researching: 'Checking standard clauses', drafting: 'Drafting' }

export default function Draft({ ws, draftId, docTypes, ocrReady, onSaved }) {
  const [docType, setDocType] = useState('rental_agreement')
  const [details, setDetails] = useState('')
  const [ocrText, setOcrText] = useState('')
  const [ocrName, setOcrName] = useState('')
  const [ocrBusy, setOcrBusy] = useState(false)
  const [ocrErr, setOcrErr] = useState('')
  const [drag, setDrag] = useState(false)
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [mode, setMode] = useState('preview')
  const [busy, setBusy] = useState(false)
  const [stage, setStage] = useState('')
  const [err, setErr] = useState('')
  const [meta, setMeta] = useState(null)
  const [saved, setSaved] = useState(false)
  const [copied, setCopied] = useState(false)
  const abortRef = useRef(null)
  const fileRef = useRef(null)
  const paperRef = useRef(null)

  useEffect(() => {
    if (!draftId) return
    api.draft(draftId).then((d) => {
      setTitle(d.title); setContent(d.content); setDetails(d.details || ''); if (d.doc_type) setDocType(d.doc_type)
      setSaved(true)
    })
  }, []) // eslint-disable-line

  useEffect(() => {
    if (busy && paperRef.current) paperRef.current.scrollTop = paperRef.current.scrollHeight
  }, [content, busy])

  const current = useMemo(() => docTypes.find((d) => d.id === docType), [docTypes, docType])

  const readFile = async (file) => {
    if (!file) return
    setOcrBusy(true); setOcrErr('')
    try {
      const r = await api.ocr(file)
      setOcrText(r.text); setOcrName(file.name)
      if (!r.text) setOcrErr('No text was found in that file. Try a clearer scan.')
      else if (r.warning) setOcrErr(r.warning)
    } catch (e) { setOcrErr(e.message) } finally { setOcrBusy(false) }
  }

  const generate = async () => {
    if (busy) return
    setBusy(true); setErr(''); setMeta(null); setSaved(false); setContent(''); setMode('preview'); setStage('researching')
    const ctrl = new AbortController(); abortRef.current = ctrl
    try {
      for await (const ev of stream('/drafts/generate', { workspace_id: ws.id, doc_type: docType, details, ocr_text: ocrText }, ctrl.signal)) {
        if (ev.type === 'status') setStage(ev.stage)
        else if (ev.type === 'token') setContent((c) => c + ev.text)
        else if (ev.type === 'replace') setContent(ev.text)
        else if (ev.type === 'error') setErr(ev.message)
        else if (ev.type === 'done') setMeta(ev)
      }
    } catch (e) { if (e.name !== 'AbortError') setErr(e.message) } finally { setBusy(false); setStage('') }
  }

  const derivedTitle = () => title.trim() || content.match(/^#\s+(.+)$/m)?.[1]?.trim() || current?.name || 'Untitled draft'

  const save = async () => {
    const t = derivedTitle()
    setTitle(t)
    const body = { title: t, doc_type: docType, details, content }
    const d = draftId ? await api.updateDraft(draftId, body) : await api.createDraft(ws.id, body)
    setSaved(true); onSaved(d.id)
  }

  const download = async () => {
    const blob = await api.pdf(derivedTitle(), content)
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob); a.download = `${derivedTitle().replace(/[^\w\- ]+/g, '').trim().replace(/\s+/g, '_') || 'draft'}.pdf`
    a.click(); URL.revokeObjectURL(a.href)
  }

  const copy = async () => {
    try { await navigator.clipboard.writeText(content); setCopied(true); setTimeout(() => setCopied(false), 1400) } catch { /* blocked */ }
  }

  const canGenerate = (details.trim() || ocrText.trim()) && !busy

  return (
    <section className="draft">
      <aside className="brief">
        <h2>Draft a document</h2>
        <p className="lede">Describe the facts or scan an existing document. Anything missing is left as a blank for you to fill in.</p>

        <label htmlFor="dt">Document type</label>
        <select id="dt" value={docType} onChange={(e) => setDocType(e.target.value)}>
          {docTypes.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
        {current && <p className="hint">{current.hint}</p>}

        <label htmlFor="det">Details</label>
        <textarea id="det" rows={7} value={details} onChange={(e) => setDetails(e.target.value)}
          placeholder="Names of the parties, addresses, amounts, dates, special terms..." />

        <label>Or read from a scan</label>
        {ocrText ? (
          <div className="file-chip">
            <FileUp size={15} />
            <span>{ocrName} <small>{ocrText.length.toLocaleString()} characters read</small></span>
            <button className="mini" onClick={() => { setOcrText(''); setOcrName('') }} aria-label="Remove file"><X size={14} /></button>
          </div>
        ) : (
          <div className={`drop ${drag ? 'over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); readFile(e.dataTransfer.files[0]) }}
            onClick={() => fileRef.current?.click()} role="button" tabIndex={0}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && fileRef.current?.click()}>
            <FileUp size={18} />
            <span>{ocrBusy ? 'Reading the document...' : 'Drop an image or PDF, or click to browse'}</span>
            {!ocrReady && <small>Image scanning needs PaddleOCR on the server. PDFs with selectable text still work.</small>}
            <input ref={fileRef} type="file" hidden accept="image/*,.pdf" onChange={(e) => { readFile(e.target.files[0]); e.target.value = '' }} />
          </div>
        )}
        {ocrErr && <p className="error" role="alert">{ocrErr}</p>}
        {ocrText && (
          <details className="ocr-review">
            <summary>Review extracted text</summary>
            <textarea rows={6} value={ocrText} onChange={(e) => setOcrText(e.target.value)} />
          </details>
        )}

        <div className="row">
          <button className="btn primary wide" disabled={!canGenerate} onClick={generate}>
            <Sparkles size={16} /> {content ? 'Generate again' : 'Generate draft'}
          </button>
          {busy && <button className="btn ghost" onClick={() => abortRef.current?.abort()}>Stop</button>}
        </div>
        {err && <p className="error" role="alert">{err}</p>}
      </aside>

      <div className="desk">
        <div className="desk-bar">
          <input className="title-input" value={title} onChange={(e) => { setTitle(e.target.value); setSaved(false) }}
            placeholder={current?.name || 'Untitled draft'} aria-label="Draft title" />
          <div className="seg" role="group" aria-label="View">
            <button className={mode === 'preview' ? 'on' : ''} onClick={() => setMode('preview')}><Eye size={14} /> Preview</button>
            <button className={mode === 'edit' ? 'on' : ''} onClick={() => setMode('edit')} disabled={busy}><Pencil size={14} /> Edit</button>
          </div>
          <button className="btn small" onClick={copy} disabled={!content}>{copied ? <><Check size={14} /> Copied</> : <><Copy size={14} /> Copy</>}</button>
          <button className="btn small" onClick={download} disabled={!content || busy}><Download size={14} /> PDF</button>
          <button className="btn small primary" onClick={save} disabled={!content || busy || saved}><Save size={14} /> {saved ? 'Saved' : 'Save'}</button>
        </div>

        <div className="desk-scroll" ref={paperRef}>
          <div className="paper">
            {!content && !busy ? (
              <div className="paper-empty">
                <p>Your draft will appear here.</p>
                <span /><span /><span className="short" />
              </div>
            ) : mode === 'edit' ? (
              <textarea className="paper-edit" value={content} spellCheck
                onChange={(e) => { setContent(e.target.value); setSaved(false) }} aria-label="Draft text" />
            ) : (
              <div className="doc">
                <Markdown cite={false}>{content}</Markdown>
                {busy && <p className="stage"><span className="dots"><i /><i /><i /></span>{STAGES[stage] || 'Drafting'}</p>}
              </div>
            )}
          </div>
        </div>

        <p className="fine desk-fine">
          {meta?.latency_ms != null && <>{meta.cached ? 'From cache - ' : ''}drafted in {secs(meta.latency_ms)}. </>}
          AI-generated draft. Have a qualified advocate review it before use.
        </p>
      </div>
    </section>
  )
}
