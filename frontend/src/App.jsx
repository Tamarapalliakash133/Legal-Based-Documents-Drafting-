import { useCallback, useEffect, useState } from 'react'
import { Menu } from 'lucide-react'
import { api } from './api'
import Gate from './components/Gate.jsx'
import Sidebar from './components/Sidebar.jsx'
import Chat from './components/Chat.jsx'
import Draft from './components/Draft.jsx'

const LS = 'chambers.workspace'

export default function App() {
  const [workspaces, setWorkspaces] = useState(null)
  const [ws, setWs] = useState(null)
  const [chats, setChats] = useState([])
  const [drafts, setDrafts] = useState([])
  const [view, setView] = useState({ type: 'chat', id: null, key: 'new-0' })
  const [navOpen, setNavOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const [meta, setMeta] = useState({ doc_types: [], ocr: false })

  const loadWorkspaces = useCallback(async () => {
    const list = await api.workspaces()
    setWorkspaces(list)
    return list
  }, [])

  useEffect(() => {
    api.meta().then(setMeta).catch(() => {})
    loadWorkspaces().then((list) => {
      const saved = (() => { try { return localStorage.getItem(LS) } catch { return null } })()
      const pick = list.find((w) => w.id === saved) || list[0] || null
      setWs(pick)
    }).catch(() => setWorkspaces([]))
  }, [loadWorkspaces])

  const refreshLists = useCallback(async (w = ws) => {
    if (!w) return
    const [c, d] = await Promise.all([api.chats(w.id), api.drafts(w.id)])
    setChats(c); setDrafts(d)
  }, [ws])

  useEffect(() => {
    if (!ws) return
    try { localStorage.setItem(LS, ws.id) } catch { /* private mode */ }
    setView({ type: 'chat', id: null, key: 'new-' + Date.now() })
    refreshLists(ws)
  }, [ws?.id]) // eslint-disable-line

  const createWorkspace = async (name, description) => {
    const w = await api.createWorkspace({ name, description })
    const list = await loadWorkspaces()
    setWs(list.find((x) => x.id === w.id) || w)
    setCreating(false)
  }

  const removeWorkspace = async (id) => {
    await api.deleteWorkspace(id)
    const list = await loadWorkspaces()
    setWs(list[0] || null)
  }

  const go = (v) => { setView({ ...v, key: v.id ? v.id : `new-${v.type}-${Date.now()}` }); setNavOpen(false) }

  if (workspaces === null) return <div className="boot"><span className="dots"><i /><i /><i /></span></div>

  if (!ws || creating) {
    return (
      <Gate
        first={!workspaces.length}
        onCreate={createWorkspace}
        onCancel={workspaces.length ? () => setCreating(false) : null}
      />
    )
  }

  return (
    <div className="shell">
      <Sidebar
        open={navOpen}
        onClose={() => setNavOpen(false)}
        workspaces={workspaces}
        ws={ws}
        onPickWorkspace={(w) => { setWs(w); setNavOpen(false) }}
        onNewWorkspace={() => setCreating(true)}
        onDeleteWorkspace={removeWorkspace}
        chats={chats}
        drafts={drafts}
        view={view}
        go={go}
        onDeleteChat={async (id) => {
          await api.deleteChat(id)
          if (view.type === 'chat' && view.id === id) go({ type: 'chat', id: null })
          refreshLists()
        }}
        onDeleteDraft={async (id) => {
          await api.deleteDraft(id)
          if (view.type === 'draft' && view.id === id) go({ type: 'draft', id: null })
          refreshLists()
        }}
        onRenameChat={async (id, t) => { await api.renameChat(id, t); refreshLists() }}
      />

      <main className="main">
        <button className="burger" onClick={() => setNavOpen(true)} aria-label="Open menu"><Menu size={20} /></button>
        {view.type === 'chat' ? (
          <Chat
            key={view.key}
            ws={ws}
            chatId={view.id}
            onCreated={(id) => setView((v) => ({ type: 'chat', id, key: v.key }))}
            onChanged={refreshLists}
            onDraft={() => go({ type: 'draft', id: null })}
          />
        ) : (
          <Draft
            key={view.key}
            ws={ws}
            draftId={view.id}
            docTypes={meta.doc_types}
            ocrReady={meta.ocr}
            onSaved={(id) => { setView({ type: 'draft', id, key: id }); refreshLists() }}
          />
        )}
      </main>
    </div>
  )
}
