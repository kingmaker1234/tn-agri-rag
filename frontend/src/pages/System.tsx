import { useEffect, useState } from 'react'
import { RefreshCw, PlugZap, ThumbsUp, ThumbsDown } from 'lucide-react'
import { api, fmtBytes, fmtUsd, LANG_LABEL, type Health, type UsageTotals } from '../api'
import { btn, ErrorNote, toast, withAdmin } from '../ui'

function Bars({ data, label }: { data: Record<string, number>; label: (k: string) => string }) {
  const entries = Object.entries(data)
  const max = Math.max(1, ...entries.map(e => e[1]))
  if (!entries.length) return <p className="text-sm text-muted">Nothing indexed yet.</p>
  return (
    <ul className="space-y-1.5">
      {entries.map(([k, v]) => (
        <li key={k} className="grid grid-cols-[7rem_1fr_3.5rem] items-center gap-3 text-sm">
          <span className="truncate capitalize text-muted">{label(k)}</span>
          <span className="h-2 rounded-full bg-line-soft"><span className="block h-full rounded-full bg-field-3" style={{ width: `${(v / max) * 100}%` }} /></span>
          <span className="tnum text-right font-semibold">{v.toLocaleString()}</span>
        </li>
      ))}
    </ul>
  )
}

function Component({ name, state, model, detail }: { name: string; state: 'ok' | 'warn' | 'off'; model?: string | null; detail?: string }) {
  const dot = state === 'ok' ? 'bg-field-3' : state === 'warn' ? 'bg-turmeric' : 'bg-faint'
  return (
    <div className="border-t border-line py-3 first:border-t-0">
      <div className="flex items-center gap-2"><span className={`h-2.5 w-2.5 rounded-full ${dot}`} /><span className="font-semibold">{name}</span></div>
      {model && <div className="mt-0.5 break-all pl-[18px] text-sm">{model}</div>}
      {detail && <div className="pl-[18px] text-sm text-muted">{detail}</div>}
    </div>
  )
}

export default function System({ health, onRefresh }: { health: Health | null; onRefresh: () => void }) {
  const [confirm, setConfirm] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [usage, setUsage] = useState<UsageTotals | null>(null)
  const [ping, setPing] = useState<Awaited<ReturnType<typeof api.ping>> | null>(null)
  const [pinging, setPinging] = useState(false)
  useEffect(() => { api.usage().then(setUsage).catch(() => {}) }, [health?.store.updated_at])
  if (!health) return <div className="px-6 py-8 text-muted">Connecting to the backend…</div>
  const s = health.store
  const st = health.settings as Record<string, string | number | boolean>

  async function test() {
    setPinging(true); setPing(null)
    try { const r = await withAdmin(() => api.ping()); setPing(r); const good = r.embeddings.startsWith('ok') && r.fast_model.startsWith('ok'); toast(good ? 'OpenAI connection works.' : 'OpenAI answered with an error — see System.', good ? 'ok' : 'error'); onRefresh() }
    catch (e) { setPing({ base_url: '', api_style: '', embeddings: `error: ${(e as Error).message}`, fast_model: '—', ms: 0, cost_usd: 0 }) }
    finally { setPinging(false) }
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <header className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-field">System</h1>
          <p className="mt-1 text-muted">Models, spend, index size and retrieval settings. Settings come from <code>backend/.env</code>.</p>
        </div>
        <button className={btn.secondary} onClick={onRefresh}><RefreshCw size={15} /> Refresh</button>
      </header>

      <div className="mt-8 grid gap-10 lg:grid-cols-3">
        <section>
          <h2 className="mb-2 font-bold">Components</h2>
          <Component name="Answer model" state={health.llm.provider === 'none' ? 'off' : health.llm.ready && !health.llm.fallback_active ? 'ok' : 'warn'}
            model={health.llm.provider === 'none' ? 'none' : `${health.llm.provider}: ${health.llm.model}`}
            detail={health.llm.fallback_active
              ? `OpenAI no longer serves the configured model, so answers use the fallback ${String(st.fallback_model)}. Set OPENAI_CHAT_MODEL in backend/.env to make it permanent.`
              : `${health.llm.detail}${st.fallback_model && health.llm.provider === 'openai' ? ` · fallback if retired: ${String(st.fallback_model)}` : ''}`} />
          <Component name="Helper model" state={st.query_rewrite ? 'ok' : 'off'} model={health.llm.fast_model}
            detail={st.query_rewrite ? 'query rewrite, reranking, document summaries, follow-ups' : 'off (needs an OpenAI key)'} />
          <Component name="Embeddings" state={health.embedder.error ? 'warn' : 'ok'} model={String(st.embedding_model)}
            detail={health.embedder.error ? `Error: ${health.embedder.error}` : `${health.embedder.dim} dimensions, ${health.embedder.device ?? 'local'}`} />
          <Component name="Reranker" state={!health.reranker.enabled ? 'off' : health.reranker.error ? 'warn' : 'ok'}
            model={health.reranker.enabled ? `${health.reranker.kind}: ${health.reranker.name}` : null}
            detail={!health.reranker.enabled ? health.reranker.detail : health.reranker.error ? `Error: ${health.reranker.error}` : 'scores the top 30 candidates'} />
          <div className="mt-3 rounded-xl border border-line bg-surface p-3">
            <button className={btn.secondary + ' w-full'} onClick={test} disabled={pinging}><PlugZap size={16} /> {pinging ? 'Testing…' : 'Test OpenAI connection'}</button>
            {ping && (
              <dl className="mt-3 space-y-1 text-sm">
                <div className="flex justify-between gap-3"><dt className="text-muted">Embeddings</dt><dd className={ping.embeddings.startsWith('ok') ? 'text-field-3' : 'text-danger'}>{ping.embeddings}</dd></div>
                <div className="flex justify-between gap-3"><dt className="text-muted">Helper model</dt><dd className={ping.fast_model.startsWith('ok') ? 'text-field-3' : 'break-all text-danger'}>{ping.fast_model}</dd></div>
                {ping.api_style && <div className="flex justify-between gap-3"><dt className="text-muted">API</dt><dd>{ping.api_style}, {ping.ms} ms</dd></div>}
              </dl>
            )}
          </div>
        </section>

        <section>
          <h2 className="mb-3 font-bold">Index</h2>
          <dl className="tnum grid grid-cols-2 gap-y-3">
            <div><dt className="text-sm text-muted">Documents</dt><dd className="text-2xl font-extrabold">{s.documents.toLocaleString()}</dd></div>
            <div><dt className="text-sm text-muted">Vectors</dt><dd className="text-2xl font-extrabold">{s.vectors.toLocaleString()}</dd></div>
            <div><dt className="text-sm text-muted">index.faiss</dt><dd className="font-semibold">{fmtBytes(s.index_bytes)}</dd></div>
            <div><dt className="text-sm text-muted">Source files</dt><dd className="font-semibold">{fmtBytes(s.total_bytes)}</dd></div>
            <div><dt className="text-sm text-muted">Keyword terms</dt><dd className="font-semibold">{s.bm25_terms.toLocaleString()}</dd></div>
            <div><dt className="text-sm text-muted">Indexing cost</dt><dd className="font-semibold">{fmtUsd(s.ingest_cost_usd ?? 0)}</dd></div>
          </dl>
          {s.store_dir && (
            <div className="mt-4 rounded-xl border border-line bg-surface p-3 text-sm">
              <div className="font-semibold">Saved on this computer</div>
              <code className="mt-1 block break-all text-muted">{s.store_dir}</code>
              <p className="mt-1 text-xs text-faint">index.faiss + meta.sqlite. Change the folder with VECTOR_STORE_DIR in backend/.env.</p>
            </div>
          )}
          <h3 className="mb-2 mt-6 text-sm font-semibold text-muted">Documents by file type</h3>
          <Bars data={s.by_type} label={k => k.toUpperCase()} />
          {s.by_year && Object.keys(s.by_year).length > 0 && <>
            <h3 className="mb-2 mt-6 text-sm font-semibold text-muted">Documents by year</h3>
            <Bars data={s.by_year} label={k => k} />
          </>}
          <h3 className="mb-2 mt-6 text-sm font-semibold text-muted">Chunks by language</h3>
          <Bars data={s.by_language} label={k => LANG_LABEL[k] ?? k} />
        </section>

        <section>
          <h2 className="mb-3 font-bold">Spend and feedback</h2>
          {usage ? (
            <>
              <dl className="tnum grid grid-cols-2 gap-y-3">
                <div><dt className="text-sm text-muted">Total OpenAI spend</dt><dd className="text-2xl font-extrabold">{fmtUsd(usage.cost_usd)}</dd></div>
                <div><dt className="text-sm text-muted">Per question (avg)</dt><dd className="text-2xl font-extrabold">{fmtUsd(usage.avg_cost_per_question)}</dd></div>
                <div><dt className="text-sm text-muted">Questions</dt><dd className="font-semibold">{usage.questions.toLocaleString()}</dd></div>
                <div><dt className="text-sm text-muted">Tokens</dt><dd className="font-semibold">{(usage.input_tokens + usage.output_tokens).toLocaleString()}</dd></div>
              </dl>
              {Object.keys(usage.by_model).length > 0 && (
                <table className="mt-4 w-full text-sm">
                  <thead className="text-left text-muted"><tr><th className="py-1 font-medium">Model</th><th className="py-1 text-right font-medium">Tokens</th><th className="py-1 text-right font-medium">Cost</th></tr></thead>
                  <tbody className="tnum divide-y divide-line-soft">
                    {Object.entries(usage.by_model).map(([m, v]) => (
                      <tr key={m}><td className="py-1">{m}</td><td className="py-1 text-right">{(v.input_tokens + v.output_tokens).toLocaleString()}</td><td className="py-1 text-right">{fmtUsd(v.cost_usd)}</td></tr>
                    ))}
                  </tbody>
                </table>
              )}
              {usage.feedback && (
                <div className="mt-5 flex items-center gap-4 text-sm">
                  <span className="flex items-center gap-1.5"><ThumbsUp size={15} className="text-field-3" /> <b className="tnum">{usage.feedback.up}</b> helpful</span>
                  <span className="flex items-center gap-1.5"><ThumbsDown size={15} className="text-danger" /> <b className="tnum">{usage.feedback.down}</b> not helpful</span>
                </div>
              )}
              <p className="mt-2 text-xs text-faint">Costs are estimates from OpenAI's published per-token prices.</p>
            </>
          ) : <p className="text-sm text-muted">No usage recorded yet.</p>}

          <h2 className="mb-3 mt-8 font-bold">How a question is answered</h2>
          <ol className="list-decimal space-y-2 pl-5 text-sm">
            <li><b>Understand</b>: {String(st.fast_model)} rewrites the question (Tanglish, follow-ups) into English and Tamil queries plus exact keywords.</li>
            <li><b>Search</b>: every variant runs through BM25 and FAISS ({String(st.embedding_model)}).</li>
            <li><b>Fuse</b>: all ranked lists merge with reciprocal rank fusion (k={String(st.rrf_k)}).</li>
            <li><b>Rerank</b>: the top {String(st.rerank_candidates)} are scored for relevance; at most {String(st.max_chunks_per_doc)} passages per document are kept.</li>
            <li><b>Answer</b>: the top {String(st.top_k)} go to {String(st.llm_model)}, which must cite every fact.</li>
          </ol>
          <p className="mt-3 text-sm text-muted">Chunks are {String(st.chunk_size)} characters with {String(st.chunk_overlap)} overlap, and each carries its document's title, year and summary.</p>

          <div className="mt-8 rounded-xl border border-danger/30 p-4">
            <h3 className="font-semibold text-danger">Rebuild from scratch</h3>
            <p className="mt-1 text-sm text-muted">Deletes every vector and chunk (original files stay on disk). Needed after changing the embedding model.</p>
            <div className="mt-3 flex gap-2">
              <input className="w-32 rounded-lg border border-line bg-surface px-2 py-1 text-sm" placeholder="Type RESET" value={confirm} onChange={e => setConfirm(e.target.value)} aria-label="Type RESET to confirm" />
              <button className={btn.danger} disabled={confirm !== 'RESET'} onClick={() => withAdmin(() => api.resetIndex()).then(() => { setConfirm(''); toast('Index cleared.'); onRefresh() }).catch(e => setErr(e.message))}>Clear index</button>
            </div>
            {err && <div className="mt-2"><ErrorNote>{err}</ErrorNote></div>}
          </div>
        </section>
      </div>
    </div>
  )
}
