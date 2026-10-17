import { useState } from 'react'
import { ArrowRight, Scale } from 'lucide-react'

export default function Gate({ first, onCreate, onCancel }) {
  const [name, setName] = useState('')
  const [desc, setDesc] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    if (!name.trim()) return
    setBusy(true); setErr('')
    try { await onCreate(name.trim(), desc.trim()) } catch (x) { setErr(x.message); setBusy(false) }
  }

  return (
    <div className="gate">
      <aside className="gate-art" aria-hidden="true">
        <div className="gate-mark"><Scale size={26} strokeWidth={1.6} /></div>
        <p className="gate-quote">Research the law, then draft from what you found - in one place per matter.</p>
        <ul className="gate-points">
          <li>Answers cite the page they came from</li>
          <li>Drafts start from your facts or a scanned document</li>
          <li>Each workspace keeps its own chats and drafts</li>
        </ul>
      </aside>

      <form className="gate-form" onSubmit={submit}>
        <h1>{first ? 'Create your first workspace' : 'New workspace'}</h1>
        <p className="lede">
          A workspace is one matter or client. Its chat history and drafts live together, and the
          details you add here are used as background when drafting.
        </p>

        <label htmlFor="ws-name">Workspace name</label>
        <input id="ws-name" autoFocus value={name} onChange={(e) => setName(e.target.value)}
          placeholder="Sharma v. Rao - tenancy dispute" maxLength={80} />

        <label htmlFor="ws-desc">Matter details <span className="opt">optional</span></label>
        <textarea id="ws-desc" rows={5} value={desc} onChange={(e) => setDesc(e.target.value)}
          placeholder="Parties, jurisdiction, what the matter is about. For example: Landlord Mr. Rao, tenant Ms. Sharma, Pune, 11-month leave and licence, deposit dispute." />

        {err && <p className="error" role="alert">{err}</p>}
        <div className="row">
          <button className="btn primary" disabled={!name.trim() || busy}>
            Create workspace <ArrowRight size={16} />
          </button>
          {onCancel && <button type="button" className="btn ghost" onClick={onCancel}>Cancel</button>}
        </div>
      </form>
    </div>
  )
}
