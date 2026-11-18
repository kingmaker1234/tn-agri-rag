import { useEffect, useMemo, useState } from 'react'
import { Search, ExternalLink, Globe } from 'lucide-react'
import { api, type SourceRow } from '../api'
import { Tag, TypeIcon, btn, input, ErrorNote, withAdmin } from '../ui'
import type { Page } from '../App'

export default function Sources({ onGo }: { onGo: (p: Page) => void }) {
  const [rows, setRows] = useState<SourceRow[] | null>(null)
  const [q, setQ] = useState('')
  const [cat, setCat] = useState('')
  const [type, setType] = useState('')
  const [status, setStatus] = useState<'all' | 'pending' | 'indexed'>('all')
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [err, setErr] = useState<string | null>(null)
  const [msg, setMsg] = useState<string | null>(null)

  useEffect(() => { api.sources().then(r => setRows(r.items)).catch(e => setErr(e.message)) }, [])
  const cats = useMemo(() => [...new Set((rows ?? []).map(r => r.category))].sort(), [rows])
  const types = useMemo(() => [...new Set((rows ?? []).map(r => r.expected_type))].sort(), [rows])
  const shown = useMemo(() => (rows ?? []).filter(r =>
    (!q || (r.title + ' ' + r.url).toLowerCase().includes(q.toLowerCase())) && (!cat || r.category === cat) &&
    (!type || r.expected_type === type) && (status === 'all' || (status === 'indexed' ? r.indexed : !r.indexed))), [rows, q, cat, type, status])
  const indexed = (rows ?? []).filter(r => r.indexed).length

  const togglePick = (id: string) => setPicked(p => { const n = new Set(p); if (n.has(id)) n.delete(id); else n.add(id); return n })
  const pickAllShown = () => setPicked(new Set(shown.filter(r => !r.indexed).map(r => r.id)))

  async function fetchPicked() {
    setErr(null)
    try {
      const r = await withAdmin(() => api.ingestManifest({ relevance: 'all', ids: [...picked] }))
      setMsg(`Started fetching ${r.selected} source(s).`)
      setPicked(new Set())
      setTimeout(() => onGo('ingest'), 600)
    } catch (e) { setErr((e as Error).message) }
  }

  const label = (r: SourceRow) => r.title || decodeURIComponent(r.url.split('/').pop() || r.url).replace(/\.[a-z]+$/i, '').replace(/[_+]/g, ' ')

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="max-w-3xl">
          <h1 className="text-3xl font-extrabold tracking-tight text-field">Web sources</h1>
          <p className="mt-1 text-muted">The research list behind the knowledge base: official Tamil Nadu and Union government documents, TNAU Agritech archives, district pages and recent news on the 2025-26 and 2026-27 budgets. Pick any to download and index.</p>
        </div>
        {rows && <p className="tnum text-muted"><span className="text-2xl font-extrabold text-ink">{indexed}</span> of {rows.length} indexed</p>}
      </header>

      <div className="mt-6 flex flex-wrap gap-2">
        <div className="relative min-w-60 flex-1">
          <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-faint" />
          <input className={input + ' pl-9'} placeholder="Filter by title or link" value={q} onChange={e => setQ(e.target.value)} aria-label="Filter sources" />
        </div>
        <select className="rounded-lg border border-line bg-surface px-3" value={cat} onChange={e => setCat(e.target.value)} aria-label="Category">
          <option value="">All categories</option>{cats.map(c => <option key={c} value={c}>{c.replace(/_/g, ' ')}</option>)}
        </select>
        <select className="rounded-lg border border-line bg-surface px-3" value={type} onChange={e => setType(e.target.value)} aria-label="File type">
          <option value="">All file types</option>{types.map(t => <option key={t}>{t}</option>)}
        </select>
        <select className="rounded-lg border border-line bg-surface px-3" value={status} onChange={e => setStatus(e.target.value as typeof status)} aria-label="Status">
          <option value="all">Any status</option><option value="pending">Not indexed</option><option value="indexed">Indexed</option>
        </select>
      </div>

      <div className="sticky top-0 z-10 mt-3 flex flex-wrap items-center gap-3 bg-paper py-2 text-sm">
        <span className="tnum text-muted">{shown.length} shown</span>
        <button className={btn.ghost} onClick={pickAllShown}>Select all not indexed</button>
        {picked.size > 0 && <button className={btn.ghost} onClick={() => setPicked(new Set())}>Clear selection</button>}
        <button className={btn.primary + ' ml-auto'} disabled={!picked.size} onClick={fetchPicked}><Globe size={16} /> Fetch and index {picked.size || ''}</button>
      </div>
      {err && <ErrorNote>{err}</ErrorNote>}
      {msg && <p className="text-field-3">{msg}</p>}

      <ul className="mt-2 divide-y divide-line-soft rounded-xl border border-line bg-surface">
        {shown.slice(0, 600).map(r => (
          <li key={r.id} className="flex items-center gap-3 px-3 py-2">
            <input type="checkbox" className="h-4 w-4 accent-[var(--color-field)]" checked={picked.has(r.id)} disabled={r.indexed}
              onChange={() => togglePick(r.id)} aria-label={`Select ${label(r)}`} />
            <TypeIcon type={r.expected_type} />
            <div className="min-w-0 flex-1">
              <div className="truncate font-medium" title={label(r)}>{label(r)}</div>
              <div className="truncate text-xs text-faint">{r.url}</div>
            </div>
            <span className="hidden capitalize text-muted sm:block">{r.category.replace(/_/g, ' ')}</span>
            {r.relevance === 'supporting' && <Tag>supporting</Tag>}
            {r.indexed ? <Tag tone="green">indexed</Tag> : r.downloaded ? <Tag tone="turmeric">downloaded</Tag> : null}
            <a href={r.url} target="_blank" rel="noreferrer" className="text-faint hover:text-field" aria-label="Open source"><ExternalLink size={15} /></a>
          </li>
        ))}
        {rows && !shown.length && <li className="px-4 py-8 text-center text-muted">No sources match these filters.</li>}
        {!rows && !err && <li className="px-4 py-8 text-center text-muted">Loading…</li>}
      </ul>
    </div>
  )
}
