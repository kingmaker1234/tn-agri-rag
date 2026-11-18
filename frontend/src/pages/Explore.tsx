import { useMemo, useState } from 'react'
import { Search, ExternalLink, Download } from 'lucide-react'
import { api, fmtUsd, locLabel, LANG_LABEL, type Health, type Hit, type Mode, type SearchResponse, type UsageSummary } from '../api'
import { Highlighted, ScoreLadder, Segmented, Tag, Thinking, Toggle, TypeIcon, btn, ErrorNote, input } from '../ui'

type View = 'list' | 'compare'
const COLS: { key: 'bm25' | 'dense' | 'hybrid' | 'hybrid_rerank'; label: string; note: string }[] = [
  { key: 'bm25', label: 'Keyword (BM25)', note: 'exact words, scheme names, G.O. numbers' },
  { key: 'dense', label: 'Dense (FAISS)', note: 'meaning, paraphrase, Tamil ↔ English' },
  { key: 'hybrid', label: 'Hybrid (RRF)', note: 'both lists fused by rank' },
  { key: 'hybrid_rerank', label: 'Hybrid + rerank', note: 'reranker re-orders the top 30' },
]

export default function Explore({ health }: { health: Health | null }) {
  const [q, setQ] = useState('')
  const [mode, setMode] = useState<Mode>('hybrid')
  const [rerank, setRerank] = useState(true)
  const [rewrite, setRewrite] = useState(true)
  const [topK, setTopK] = useState(10)
  const [view, setView] = useState<View>('list')
  const [types, setTypes] = useState<string[]>([])
  const [langs, setLangs] = useState<string[]>([])
  const [res, setRes] = useState<SearchResponse | null>(null)
  const [cmp, setCmp] = useState<(Record<'bm25' | 'dense' | 'hybrid' | 'hybrid_rerank', SearchResponse> & { usage: UsageSummary }) | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [hoverChunk, setHoverChunk] = useState<number | null>(null)

  const typeOptions = Object.keys(health?.store.by_type ?? {})
  const langOptions = Object.keys(health?.store.by_language ?? {})
  const filters = { file_types: types, languages: langs }

  async function run(e?: React.FormEvent) {
    e?.preventDefault()
    if (!q.trim()) return
    setLoading(true); setErr(null)
    try {
      if (view === 'list') { setRes(await api.search({ query: q, mode, rerank, rewrite, top_k: topK, max_per_doc: 5, filters })); setCmp(null) }
      else { setCmp(await api.compare({ query: q, mode, rerank: true, rewrite, top_k: 8, max_per_doc: 5, filters })); setRes(null) }
    } catch (ex) { setErr((ex as Error).message) } finally { setLoading(false) }
  }

  const toggle = (arr: string[], v: string, set: (a: string[]) => void) => set(arr.includes(v) ? arr.filter(x => x !== v) : [...arr, v])

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <header className="max-w-3xl">
        <h1 className="text-3xl font-extrabold tracking-tight text-field">Explore retrieval</h1>
        <p className="mt-1 text-muted">See exactly which passages the vector database returns for a query, and how keyword search, vector search, fusion and the reranker each ranked them. No answer is written here, so you can judge retrieval on its own.</p>
      </header>

      <form onSubmit={run} className="mt-6 space-y-3">
        <div className="flex gap-2">
          <label htmlFor="explore-q" className="sr-only">Query</label>
          <input id="explore-q" value={q} onChange={e => setQ(e.target.value)} className={input + ' text-[1.02rem]'}
            placeholder="e.g. farm pond 100% subsidy, PMFBY samba enrolment, நுண்ணீர் பாசன மானியம்" />
          <button className={btn.primary} disabled={!q.trim() || loading}><Search size={17} /> Search</button>
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
          <Segmented label="View" value={view} onChange={v => { setView(v); setRes(null); setCmp(null) }} options={[
            { value: 'list', label: 'Ranked list' }, { value: 'compare', label: 'Compare 4 ways' }]} />
          {view === 'compare' && <Toggle checked={rewrite} onChange={setRewrite} label="Query rewrite" />}
          {view === 'list' && (
            <>
              <Segmented label="Mode" value={mode} onChange={setMode} options={[
                { value: 'hybrid', label: 'Hybrid' }, { value: 'dense', label: 'Dense' }, { value: 'bm25', label: 'Keyword' }]} />
              <Toggle checked={rewrite} onChange={setRewrite} label="Query rewrite" hint="Search with English + Tamil rewrites and exact keywords" />
              <Toggle checked={rerank} onChange={setRerank} label="Rerank" />
              <label className="flex items-center gap-2">Show
                <select value={topK} onChange={e => setTopK(Number(e.target.value))} className="rounded-md border border-line bg-surface px-2 py-1">
                  {[5, 10, 20, 30].map(n => <option key={n}>{n}</option>)}
                </select>
              </label>
            </>
          )}
        </div>
        {(typeOptions.length > 1 || langOptions.length > 1) && (
          <div className="flex flex-wrap items-center gap-1.5 text-sm">
            <span className="mr-1 text-muted">Limit to</span>
            {typeOptions.map(t => (
              <button type="button" key={t} onClick={() => toggle(types, t, setTypes)} aria-pressed={types.includes(t)}
                className={`rounded-full border px-2.5 py-0.5 ${types.includes(t) ? 'border-field bg-field text-paper' : 'border-line bg-surface text-muted hover:text-ink'}`}>
                {t}
              </button>
            ))}
            <span className="mx-1 h-4 w-px bg-line" />
            {langOptions.map(l => (
              <button type="button" key={l} onClick={() => toggle(langs, l, setLangs)} aria-pressed={langs.includes(l)}
                className={`rounded-full border px-2.5 py-0.5 ${langs.includes(l) ? 'border-field bg-field text-paper' : 'border-line bg-surface text-muted hover:text-ink'}`}>
                {LANG_LABEL[l] ?? l}
              </button>
            ))}
          </div>
        )}
      </form>

      <div className="mt-8">
        {loading && <Thinking label="Querying FAISS and BM25…" />}
        {err && <ErrorNote>{err}</ErrorNote>}
        {res && !loading && <RankedList res={res} />}
        {cmp && !loading && (
          <div>
            <Variants res={cmp.hybrid} />
            <p className="mb-4 text-sm text-muted">Point at a passage to see where the same chunk landed in the other columns.{cmp.usage?.cost_usd ? ` Cost of this comparison: ${fmtUsd(cmp.usage.cost_usd)}.` : ''}</p>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {COLS.map(c => (
                <section key={c.key} className="min-w-0">
                  <h2 className="font-bold">{c.label}</h2>
                  <p className="mb-2 min-h-[2.5em] text-xs text-muted">{c.note}{c.key === 'hybrid_rerank' && !cmp[c.key].reranked ? ' (reranker not loaded)' : ''}</p>
                  <ol className="space-y-2">
                    {cmp[c.key].results.map(h => (
                      <li key={h.chunk_id} onMouseEnter={() => setHoverChunk(h.chunk_id)} onMouseLeave={() => setHoverChunk(null)}
                        className={`rounded-lg border p-2.5 text-sm transition-colors ${hoverChunk === h.chunk_id ? 'border-turmeric bg-turmeric-soft/50' : 'border-line bg-surface'}`}>
                        <div className="flex gap-2">
                          <span className="tnum font-bold text-field-3">{h.rank}</span>
                          <div className="min-w-0">
                            <div className="truncate font-semibold" title={h.title}>{h.title}</div>
                            <div className="text-xs text-muted">{locLabel(h.location)}</div>
                            <p className="mt-1 line-clamp-3 text-[0.85rem] text-ink/80"><Highlighted text={h.text} terms={h.matched_terms} max={220} /></p>
                          </div>
                        </div>
                      </li>
                    ))}
                    {!cmp[c.key].results.length && <li className="text-sm text-muted">No matches.</li>}
                  </ol>
                </section>
              ))}
            </div>
          </div>
        )}
        {!res && !cmp && !loading && !err && (
          <div className="kolam-grid rounded-2xl border border-line-soft px-6 py-14 text-center text-muted">
            Type a question or keyword to see the raw retrieval results.
          </div>
        )}
      </div>
    </div>
  )
}

function Variants({ res }: { res: SearchResponse }) {
  const v = (res.variants ?? []).filter(x => x.label !== 'original')
  const kw = res.plan?.keywords ?? []
  if (!v.length && !kw.length) return null
  return (
    <div className="mb-4 rounded-xl border border-line bg-surface p-3 text-sm">
      <div className="mb-1.5 font-semibold">The question was also searched as</div>
      <ul className="space-y-1">
        {v.map(x => <li key={x.label} className="flex gap-2"><span className="w-24 shrink-0 capitalize text-muted">{x.label}</span><span className={x.label === 'tamil' ? 'ta' : ''}>{x.text}</span></li>)}
        {kw.length > 0 && <li className="flex gap-2"><span className="w-24 shrink-0 text-muted">Keywords</span><span>{kw.join(', ')}</span></li>}
      </ul>
    </div>
  )
}

function RankedList({ res }: { res: SearchResponse }) {
  const t = res.timings
  const summary = useMemo(() => {
    const parts = []
    if (t.plan_ms != null) parts.push(`query rewrite ${Math.round(t.plan_ms)} ms`)
    if (t.dense_ms != null) parts.push(`vector search ${Math.round(t.dense_ms)} ms`)
    if (t.bm25_ms != null) parts.push(`keyword search ${Math.round(t.bm25_ms)} ms`)
    if (t.rerank_ms != null) parts.push(`rerank ${Math.round(t.rerank_ms)} ms`)
    return parts.join(', ')
  }, [t])
  return (
    <div>
      <Variants res={res} />
      <p className="mb-4 text-sm text-muted">
        <span className="tnum">{res.candidates.dense}</span> vector and <span className="tnum">{res.candidates.bm25}</span> keyword
        candidates{res.mode === 'hybrid' ? <> fused into <span className="tnum">{res.candidates.fused}</span></> : null}
        {res.reranked ? ', then reranked' : ''}. {summary}{summary ? '.' : ''}{res.usage?.cost_usd ? ` Cost ${fmtUsd(res.usage.cost_usd)}.` : ''}
      </p>
      {res.results.length === 0 && <p className="text-muted">No passages matched. Try other words, or remove the filters.</p>}
      <ol className="divide-y divide-line border-y border-line">
        {res.results.map(h => <Row key={h.chunk_id} h={h} reranked={res.reranked} />)}
      </ol>
    </div>
  )
}

function Row({ h, reranked }: { h: Hit; reranked: boolean }) {
  const [open, setOpen] = useState(false)
  return (
    <li className="grid grid-cols-[2.5rem_1fr] gap-3 py-4">
      <div className="tnum pt-0.5 text-2xl font-extrabold text-field-3">{h.rank}</div>
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <TypeIcon type={h.file_type} />
          <h3 className="font-semibold">{h.title}</h3>
          {h.year && <Tag tone="indigo">{h.year}</Tag>}
          {locLabel(h.location) && <Tag>{locLabel(h.location)}</Tag>}
          <Tag>{LANG_LABEL[h.language] ?? h.language}</Tag>
          <Tag tone="green">chunk {h.chunk_index + 1} of {h.n_chunks}</Tag>
        </div>
        <p className={`mt-2 max-w-[75ch] whitespace-pre-line text-[0.95rem] ${h.language !== 'en' ? 'ta' : ''}`}>
          <Highlighted text={h.text} terms={h.matched_terms} max={open ? undefined : 480} />
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-1">
          {h.text.length > 480 && <button className={btn.ghost} onClick={() => setOpen(o => !o)}>{open ? 'Show less' : 'Show full chunk'}</button>}
          {h.source_url && <a className={btn.ghost} href={h.source_url} target="_blank" rel="noreferrer"><ExternalLink size={14} /> Source page</a>}
          <a className={btn.ghost} href={api.fileUrl(h.doc_id)} target="_blank" rel="noreferrer"><Download size={14} /> Original file</a>
          {h.matched_terms.length > 0 && <span className="text-xs text-muted">matched: {h.matched_terms.join(', ')}</span>}
        </div>
        <div className="mt-3 max-w-xl"><ScoreLadder s={h.scores} reranked={reranked} /></div>
      </div>
    </li>
  )
}
