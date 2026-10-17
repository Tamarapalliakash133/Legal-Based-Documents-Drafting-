const base = '/api'

async function request(url, opts = {}) {
  const res = await fetch(base + url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body && typeof opts.body !== 'string' && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  meta: () => request('/meta'),
  health: () => request('/health'),
  workspaces: () => request('/workspaces'),
  createWorkspace: (b) => request('/workspaces', { method: 'POST', body: b }),
  updateWorkspace: (id, b) => request(`/workspaces/${id}`, { method: 'PATCH', body: b }),
  deleteWorkspace: (id) => request(`/workspaces/${id}`, { method: 'DELETE' }),
  chats: (wid) => request(`/workspaces/${wid}/chats`),
  createChat: (wid) => request(`/workspaces/${wid}/chats`, { method: 'POST', body: {} }),
  chat: (id) => request(`/chats/${id}`),
  renameChat: (id, title) => request(`/chats/${id}`, { method: 'PATCH', body: { title } }),
  deleteChat: (id) => request(`/chats/${id}`, { method: 'DELETE' }),
  drafts: (wid) => request(`/workspaces/${wid}/drafts`),
  draft: (id) => request(`/drafts/${id}`),
  createDraft: (wid, b) => request(`/workspaces/${wid}/drafts`, { method: 'POST', body: b }),
  updateDraft: (id, b) => request(`/drafts/${id}`, { method: 'PATCH', body: b }),
  deleteDraft: (id) => request(`/drafts/${id}`, { method: 'DELETE' }),
  ocr: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch(base + '/ocr', { method: 'POST', body: fd }).then(async (r) => {
      const j = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(j.detail || 'Could not read that file')
      return j
    })
  },
  pdf: async (title, content) => {
    const r = await fetch(base + '/pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, content }),
    })
    if (!r.ok) throw new Error('PDF export failed')
    return r.blob()
  },
}

// POST + Server-Sent Events -> async iterator of parsed events
export async function* stream(url, body, signal) {
  const res = await fetch(base + url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Request failed')
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let i
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, i)
      buf = buf.slice(i + 2)
      if (frame.startsWith('data: ')) {
        try { yield JSON.parse(frame.slice(6)) } catch { /* ignore partial */ }
      }
    }
  }
}
