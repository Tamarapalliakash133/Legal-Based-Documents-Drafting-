import { useEffect, useRef, useState } from 'react'
import { Check, ChevronsUpDown, FilePenLine, MessageSquareText, Pencil, Plus, Trash2, X } from 'lucide-react'

function bucket(ts) {
  const d = new Date(ts * 1000), now = new Date()
  const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime()
  const diff = Math.round((day(now) - day(d)) / 86400000)
  if (diff <= 0) return 'Today'
  if (diff === 1) return 'Yesterday'
  if (diff < 8) return 'Previous 7 days'
  return 'Older'
}

function group(items) {
  const out = []
  for (const it of items) {
    const b = bucket(it.updated_at)
    const last = out[out.length - 1]
    if (last && last.label === b) last.items.push(it)
    else out.push({ label: b, items: [it] })
  }
  return out
}

function Row({ active, icon, title, onClick, onDelete, onRename }) {
  const [editing, setEditing] = useState(false)
  const [val, setVal] = useState(title)
  const [confirm, setConfirm] = useState(false)

  if (editing) {
    return (
      <form className="row-item editing" onSubmit={(e) => { e.preventDefault(); onRename(val); setEditing(false) }}>
        <input autoFocus value={val} onChange={(e) => setVal(e.target.value)} onBlur={() => setEditing(false)} maxLength={80} />
      </form>
    )
  }
  return (
    <div className={`row-item ${active ? 'active' : ''}`}>
      <button className="row-main" onClick={onClick} title={title}>
        {icon}<span>{title}</span>
      </button>
      <span className="row-actions">
        {confirm ? (
          <>
            <button className="mini danger" onClick={onDelete} aria-label="Confirm delete"><Check size={14} /></button>
            <button className="mini" onClick={() => setConfirm(false)} aria-label="Cancel"><X size={14} /></button>
          </>
        ) : (
          <>
            {onRename && <button className="mini" onClick={() => { setVal(title); setEditing(true) }} aria-label="Rename"><Pencil size={13} /></button>}
            <button className="mini" onClick={() => setConfirm(true)} aria-label="Delete"><Trash2 size={13} /></button>
          </>
        )}
      </span>
    </div>
  )
}

export default function Sidebar({
  open, onClose, workspaces, ws, onPickWorkspace, onNewWorkspace, onDeleteWorkspace,
  chats, drafts, view, go, onDeleteChat, onDeleteDraft, onRenameChat,
}) {
  const [tab, setTab] = useState(view.type === 'draft' ? 'drafts' : 'chats')
  const [menu, setMenu] = useState(false)
  const [confirmWs, setConfirmWs] = useState(false)
  const menuRef = useRef(null)

  useEffect(() => { setTab(view.type === 'draft' ? 'drafts' : 'chats') }, [view.type])
  useEffect(() => {
    const h = (e) => { if (menuRef.current && !menuRef.current.contains(e.target)) { setMenu(false); setConfirmWs(false) } }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [])

  return (
    <>
      {open && <div className="scrim" onClick={onClose} />}
      <aside className={`sidebar ${open ? 'open' : ''}`}>
        <div className="brand">
          <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
            <path d="M16 5v21M9 10h14M9 10l-3.5 8h7zM23 10l-3.5 8h7zM11 26h10" stroke="#C99A3B" strokeWidth="1.9" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span>Chambers</span>
        </div>

        <div className="ws-switch" ref={menuRef}>
          <button className="ws-current" onClick={() => setMenu(!menu)} aria-expanded={menu}>
            <span className="ws-initial">{ws.name.trim()[0]?.toUpperCase()}</span>
            <span className="ws-name">{ws.name}</span>
            <ChevronsUpDown size={15} />
          </button>
          {menu && (
            <div className="menu" role="menu">
              {workspaces.map((w) => (
                <button key={w.id} role="menuitem" className={w.id === ws.id ? 'on' : ''} onClick={() => { setMenu(false); onPickWorkspace(w) }}>
                  <span className="ws-initial sm">{w.name.trim()[0]?.toUpperCase()}</span>
                  <span className="grow">{w.name}</span>
                  <small>{w.chat_count} chats</small>
                </button>
              ))}
              <hr />
              <button role="menuitem" onClick={() => { setMenu(false); onNewWorkspace() }}><Plus size={15} /> New workspace</button>
              {confirmWs ? (
                <button role="menuitem" className="danger" onClick={() => { setMenu(false); setConfirmWs(false); onDeleteWorkspace(ws.id) }}>
                  <Trash2 size={15} /> Delete "{ws.name}" and everything in it
                </button>
              ) : (
                <button role="menuitem" className="danger-soft" onClick={() => setConfirmWs(true)}><Trash2 size={15} /> Delete this workspace</button>
              )}
            </div>
          )}
        </div>

        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'chats'} className={tab === 'chats' ? 'on' : ''} onClick={() => setTab('chats')}>
            <MessageSquareText size={15} /> Chats
          </button>
          <button role="tab" aria-selected={tab === 'drafts'} className={tab === 'drafts' ? 'on' : ''} onClick={() => setTab('drafts')}>
            <FilePenLine size={15} /> Drafts
          </button>
        </div>

        <button className="new-btn" onClick={() => go(tab === 'chats' ? { type: 'chat', id: null } : { type: 'draft', id: null })}>
          <Plus size={16} /> {tab === 'chats' ? 'New chat' : 'New draft'}
        </button>

        <nav className="history">
          {tab === 'chats' ? (
            chats.length === 0 ? (
              <p className="empty-note">Your conversations will appear here.</p>
            ) : (
              group(chats).map((g) => (
                <section key={g.label}>
                  <h3>{g.label}</h3>
                  {g.items.map((c) => (
                    <Row key={c.id} active={view.type === 'chat' && view.id === c.id}
                      icon={<MessageSquareText size={14} />} title={c.title}
                      onClick={() => go({ type: 'chat', id: c.id })}
                      onDelete={() => onDeleteChat(c.id)}
                      onRename={(t) => onRenameChat(c.id, t)} />
                  ))}
                </section>
              ))
            )
          ) : drafts.length === 0 ? (
            <p className="empty-note">Saved drafts will appear here.</p>
          ) : (
            group(drafts).map((g) => (
              <section key={g.label}>
                <h3>{g.label}</h3>
                {g.items.map((d) => (
                  <Row key={d.id} active={view.type === 'draft' && view.id === d.id}
                    icon={<FilePenLine size={14} />} title={d.title}
                    onClick={() => go({ type: 'draft', id: d.id })}
                    onDelete={() => onDeleteDraft(d.id)} />
                ))}
              </section>
            ))
          )}
        </nav>
      </aside>
    </>
  )
}
