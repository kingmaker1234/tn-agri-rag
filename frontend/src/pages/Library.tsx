import { useCallback, useEffect, useState } from 'react'
import { Search, Trash2, ExternalLink, Download, ChevronLeft, ChevronRight } from 'lucide-react'
import { api, fmtBytes, fmtUsd, LANG_LABEL, locLabel, type ChunkRow, type Doc } from '../api'
import { Drawer, Tag, TypeIcon, btn, input, ErrorNote, Skeleton, withAdmin, toast } from '../ui'

const PAGE = 25

export default function Library({ onChanged }: { onChanged: () => void }) {
  const [q, setQ] = useState('')
  const [type, setType] = useState('')
  const [lang, setLang] = useState('')
  const [sort, setSort] = useState('added_at')
  const [offset, setOffset] = useState(0)
  const [data, setData] = useState<{ total: number; items: Doc[] } | null>(null)
  const [open, setOpen] = useState<Doc | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const load = useCallback(() => {
    api.documents({ q, file_type: type, language: lang, sort, limit: PAGE, offset }).then(setData).catch(e => setErr(e.message))
  }, [q, type, lang, sort, offset])
  useEffect(() => { const t = setTimeout(load, 200); return () => clearTimeout(t) }, [load])

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-field">Library</h1>
          <p className="mt-1 text-muted">Every document in the index, with the chunks that were stored for it.</p>
        </div>
        {data && <p className="tnum text-muted"><span className="text-2xl font-extrabold text-ink">{data.total.toLocaleString()}</span> documents</p>}
      </header>

      <div className="mt-6 flex flex-wrap gap-2">
        <div className="relative min-w-60 flex-1">
          <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-faint" />
          <input className={input + ' pl-9'} placeholder="Search titles, file names, links" value={q} onChange={e => { setQ(e.target.value); setOffset(0) }} aria-label="Search documents" />
        </div>
        <select className="rounded-lg border border-line bg-surface px-3" value={type} onChange={e => { setType(e.target.value); setOffset(0) }} aria-label="File type">
          <option value="">All types</option>
          {['pdf', 'docx', 'doc', 'xlsx', 'xls', 'csv', 'pptx', 'html', 'txt', 'md', 'json', 'xml'].map(t => <option key={t}>{t}</option>)}
        </select>
        <select className="rounded-lg border border-line bg-surface px-3" value={lang} onChange={e => { setLang(e.target.value); setOffset(0) }} aria-label="Language">
          <option value="">All languages</option>
          {['en', 'ta', 'mixed'].map(l => <option key={l} value={l}>{LANG_LABEL[l]}</option>)}
        </select>
        <select className="rounded-lg border border-line bg-surface px-3" value={sort} onChange={e => setSort(e.target.value)} aria-label="Sort">
          <option value="added_at">Newest first</option><option value="year">Year</option><option value="title">Title</option>
          <option value="chunks">Most chunks</option><option value="size">Largest</option>
        </select>
      </div>
      {err && <div className="mt-4"><ErrorNote>{err}</ErrorNote></div>}

      <div className="mt-4 overflow-x-auto rounded-xl border border-line bg-surface">
        <table className="w-full text-sm">
          <thead className="bg-paper text-left text-muted">
            <tr>
              <th className="px-4 py-2 font-medium">Document</th><th className="px-3 py-2 font-medium">Year</th><th className="px-3 py-2 font-medium">Type</th>
              <th className="px-3 py-2 font-medium">Language</th><th className="px-3 py-2 text-right font-medium">Pages</th>
              <th className="px-3 py-2 text-right font-medium">Chunks</th><th className="px-3 py-2 text-right font-medium">Size</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line-soft">
            {data?.items.map(d => (
              <tr key={d.doc_id} className="cursor-pointer hover:bg-paper" onClick={() => setOpen(d)} tabIndex={0} onKeyDown={e => e.key === 'Enter' && setOpen(d)}>
                <td className="max-w-0 px-4 py-2.5">
                  <div className="flex items-center gap-2"><TypeIcon type={d.file_type} /><span className="truncate font-medium" title={d.title}>{d.title || d.filename}</span></div>
                  <div className="truncate pl-6 text-xs text-faint">{d.summary || (d.source_url ? new URL(d.source_url).host : d.filename)}</div>
                </td>
                <td className="tnum px-3 py-2.5">{d.year || '—'}</td>
                <td className="px-3 py-2.5 capitalize text-muted">{d.doc_type || (d.category ?? '').replace(/_/g, ' ')}</td>
                <td className="px-3 py-2.5">{LANG_LABEL[d.language] ?? d.language}</td>
                <td className="tnum px-3 py-2.5 text-right">{d.n_pages || '—'}</td>
                <td className="tnum px-3 py-2.5 text-right font-semibold">{d.n_chunks}</td>
                <td className="tnum px-3 py-2.5 text-right text-muted">{fmtBytes(d.size_bytes)}</td>
              </tr>
            ))}
            {!data && [0, 1, 2, 3, 4].map(i => (
              <tr key={i}><td colSpan={7} className="px-4 py-3"><Skeleton className="h-5 w-full" /></td></tr>
            ))}
            {data && !data.items.length && <tr><td colSpan={7} className="px-4 py-10 text-center text-muted">No documents match. Add documents from the “Add documents” page.</td></tr>}
          </tbody>
        </table>
      </div>
      {data && data.total > PAGE && (
        <div className="mt-3 flex items-center justify-end gap-2 text-sm">
          <span className="tnum text-muted">{offset + 1}–{Math.min(offset + PAGE, data.total)} of {data.total}</span>
          <button className={btn.ghost} disabled={offset === 0} onClick={() => setOffset(o => Math.max(0, o - PAGE))} aria-label="Previous page"><ChevronLeft size={16} /></button>
          <button className={btn.ghost} disabled={offset + PAGE >= data.total} onClick={() => setOffset(o => o + PAGE)} aria-label="Next page"><ChevronRight size={16} /></button>
        </div>
      )}

      <Drawer open={!!open} onClose={() => setOpen(null)} wide title={open && (
        <div>
          <div className="flex items-center gap-2"><TypeIcon type={open.file_type} size={18} /><h2 className="text-lg font-bold leading-snug">{open.title || open.filename}</h2></div>
          <p className="mt-0.5 text-sm text-muted">{open.filename}</p>
        </div>)}>
        {open && <DocDetail doc={open} onDeleted={() => { setOpen(null); load(); onChanged() }} />}
      </Drawer>
    </div>
  )
}

function DocDetail({ doc, onDeleted }: { doc: Doc; onDeleted: () => void }) {
  const [chunks, setChunks] = useState<{ total: number; items: ChunkRow[] } | null>(null)
  const [offset, setOffset] = useState(0)
  const [q, setQ] = useState('')
  const [confirm, setConfirm] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    const t = setTimeout(() => api.document(doc.doc_id, 10, offset, q).then(r => setChunks(r.chunks)).catch(e => setErr(e.message)), 150)
    return () => clearTimeout(t)
  }, [doc.doc_id, offset, q])

  const facts: [string, string][] = [
    ['Type', doc.file_type.toUpperCase()], ['Language', LANG_LABEL[doc.language] ?? doc.language],
    ['Pages', doc.n_pages ? String(doc.n_pages) : '—'], ['Chunks', String(doc.n_chunks)], ['Size', fmtBytes(doc.size_bytes)],
    ['Year', doc.year || '—'], ['Department', doc.department || '—'], ['Text from', doc.extraction],
    ['Category', doc.category], ['Added', new Date(doc.added_at).toLocaleString()], ['Index cost', fmtUsd(doc.cost_usd ?? 0)],
  ]
  return (
    <div>
      {doc.summary && <p className="mb-4 max-w-[70ch] text-[0.98rem]">{doc.summary}</p>}
      {(doc.schemes?.length ?? 0) > 0 && <div className="mb-4 flex flex-wrap gap-1.5">{doc.schemes!.map(sc => <Tag key={sc} tone="green">{sc}</Tag>)}</div>}
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
        {facts.map(([k, v]) => <div key={k}><dt className="text-faint">{k}</dt><dd className="tnum font-medium capitalize">{v}</dd></div>)}
      </dl>
      {doc.notes?.length > 0 && <div className="mt-3 flex flex-wrap gap-1.5">{doc.notes.map(n => <Tag key={n} tone="turmeric">{n}</Tag>)}</div>}
      <div className="mt-4 flex flex-wrap gap-2">
        {doc.source_url && <a href={doc.source_url} target="_blank" rel="noreferrer" className={btn.secondary}><ExternalLink size={15} /> Source page</a>}
        <a href={api.fileUrl(doc.doc_id)} target="_blank" rel="noreferrer" className={btn.secondary}><Download size={15} /> Original file</a>
        {!confirm
          ? <button className={btn.danger + ' ml-auto'} onClick={() => setConfirm(true)}><Trash2 size={15} /> Remove from index</button>
          : <span className="ml-auto flex items-center gap-2 text-sm">Remove its {doc.n_chunks} vectors?
              <button className={btn.danger} onClick={() => withAdmin(() => api.deleteDocument(doc.doc_id)).then(() => { toast('Removed from the index.'); onDeleted() }).catch(e => setErr(e.message))}>Remove</button>
              <button className={btn.ghost} onClick={() => setConfirm(false)}>Keep</button></span>}
      </div>
      {err && <div className="mt-3"><ErrorNote>{err}</ErrorNote></div>}

      <h3 className="mt-8 font-bold">Stored chunks</h3>
      <input className={input + ' mt-2'} placeholder="Find text inside this document" value={q} onChange={e => { setQ(e.target.value); setOffset(0) }} aria-label="Search chunks" />
      <ol className="mt-3 space-y-3">
        {chunks?.items.map(c => (
          <li key={c.id} className="rounded-xl border border-line p-3">
            <div className="mb-1.5 flex flex-wrap gap-1.5 text-xs">
              <Tag tone="green">chunk {c.chunk_index + 1}</Tag>
              {locLabel(c.meta) && <Tag>{locLabel(c.meta)}</Tag>}
              <Tag>{LANG_LABEL[c.language] ?? c.language}</Tag>
              <Tag>{c.n_chars} characters</Tag>
            </div>
            <p className={`whitespace-pre-line text-[0.93rem] ${c.language !== 'en' ? 'ta' : ''}`}>{c.text}</p>
          </li>
        ))}
      </ol>
      {chunks && chunks.total > 10 && (
        <div className="mt-3 flex items-center justify-end gap-2 text-sm">
          <span className="tnum text-muted">{offset + 1}–{Math.min(offset + 10, chunks.total)} of {chunks.total}</span>
          <button className={btn.ghost} disabled={offset === 0} onClick={() => setOffset(o => Math.max(0, o - 10))} aria-label="Previous chunks"><ChevronLeft size={16} /></button>
          <button className={btn.ghost} disabled={offset + 10 >= chunks.total} onClick={() => setOffset(o => o + 10)} aria-label="Next chunks"><ChevronRight size={16} /></button>
        </div>
      )}
    </div>
  )
}
