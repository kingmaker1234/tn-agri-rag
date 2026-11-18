import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { SendHorizontal, Square, ExternalLink, Download, Copy, Check, SlidersHorizontal, RotateCcw, ThumbsUp, ThumbsDown,
  FileDown, ChevronDown, CircleCheck, CircleAlert } from 'lucide-react'
import { ask, api, fmtUsd, locLabel, LANG_LABEL, STEP_LABEL, type Health, type Hit, type Mode, type QueryPlan,
  type TraceStep, type UsageSummary } from '../api'
import { Highlighted, ScoreLadder, Segmented, Tag, Thinking, Toggle, TypeIcon, btn, ErrorNote, KolamMark, Drawer, toast } from '../ui'
import type { Page } from '../App'

interface Turn {
  id: number; question: string; answer: string; sources: Hit[]; reranked: boolean; mode: Mode
  status: 'retrieving' | 'answering' | 'done' | 'error'; notice?: string; error?: string
  plan?: QueryPlan; variants?: { label: string; text: string }[]; followups?: string[]
  timings?: Record<string, number | null>; trace?: TraceStep[]; usage?: UsageSummary; model?: string | null
  rating?: 1 | -1
}

const EXAMPLES = [
  'What subsidy do small farmers get for drip irrigation?',
  'நுண்ணீர் பாசனத்திற்கு எவ்வளவு மானியம் கிடைக்கும்?',
  "What's new in the 2026-27 agriculture budget?",
  'PM-KUSUM solar pump ku yaaru apply pannalam?',
  'உழவர் சந்தை எப்போது தொடங்கப்பட்டது?',
  'Who is eligible for the 2026 crop loan waiver?',
]

const withCiteLinks = (md: string) => md.replace(/\[(\d{1,2})\](?!\()/g, '[$1](#cite-$1)')

function exportMarkdown(turns: Turn[]) {
  const lines = ['# Uzhavan Assist — conversation', '', `_Exported ${new Date().toLocaleString()}_`, '']
  turns.filter(t => t.answer).forEach(t => {
    lines.push(`## ${t.question}`, '', t.answer, '', '**Sources**', '')
    t.sources.forEach((s, i) => lines.push(`${i + 1}. ${s.title}${locLabel(s.location) ? ` (${locLabel(s.location)})` : ''}${s.source_url ? ` — ${s.source_url}` : ''}`))
    lines.push('')
  })
  const blob = new Blob([lines.join('\n')], { type: 'text/markdown' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `uzhavan-assist-${new Date().toISOString().slice(0, 10)}.md`
  a.click()
  URL.revokeObjectURL(a.href)
}

export default function Ask({ health, onGo }: { health: Health | null; onGo: (p: Page) => void }) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [q, setQ] = useState('')
  const [mode, setMode] = useState<Mode>('hybrid')
  const [rerank, setRerank] = useState(true)
  const [rewrite, setRewrite] = useState(true)
  const [topK, setTopK] = useState(6)
  const [showSettings, setShowSettings] = useState(false)
  const [activeTurn, setActiveTurn] = useState<number | null>(null)
  const [focusCite, setFocusCite] = useState<number | null>(null)
  const [evidenceOpen, setEvidenceOpen] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const busy = turns.some(t => t.status === 'retrieving' || t.status === 'answering')
  const needsSetup = health && (health.store.documents === 0 || !health.llm.ready)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [turns.length])

  const update = (id: number, fn: (t: Turn) => Turn) => setTurns(ts => ts.map(t => (t.id === id ? fn(t) : t)))

  async function submit(text?: string) {
    const question = (text ?? q).trim()
    if (!question || busy) return
    setQ('')
    const id = Date.now()
    const history = turns.filter(t => t.status === 'done' && t.answer).slice(-2)
      .flatMap(t => [{ role: 'user', content: t.question }, { role: 'assistant', content: t.answer }])
    setTurns(ts => [...ts, { id, question, answer: '', sources: [], reranked: false, mode, status: 'retrieving' }])
    setActiveTurn(id)
    setFocusCite(null)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    try {
      await ask({ query: question, mode, rerank, rewrite, top_k: topK, history }, e => {
        if (e.type === 'plan') update(id, t => ({ ...t, plan: e.plan, variants: e.variants }))
        else if (e.type === 'sources') update(id, t => ({ ...t, sources: e.results, reranked: e.reranked, status: 'answering' }))
        else if (e.type === 'token') update(id, t => ({ ...t, answer: t.answer + e.text }))
        else if (e.type === 'followups') update(id, t => ({ ...t, followups: e.questions }))
        else if (e.type === 'notice') update(id, t => ({ ...t, notice: e.text }))
        else if (e.type === 'error') update(id, t => ({ ...t, error: e.message }))
        else if (e.type === 'done') update(id, t => ({ ...t, status: 'done', timings: e.timings, trace: e.trace, usage: e.usage, model: e.model }))
      }, ctrl.signal)
      update(id, t => (t.status === 'done' ? t : { ...t, status: 'done' }))
    } catch (err) {
      if (ctrl.signal.aborted) update(id, t => ({ ...t, status: 'done', notice: 'Stopped.' }))
      else update(id, t => ({ ...t, status: 'error', error: (err as Error).message }))
    }
  }

  const current = turns.find(t => t.id === activeTurn) ?? turns[turns.length - 1]

  function openCite(turnId: number, n: number) {
    setActiveTurn(turnId)
    setFocusCite(n)
    setEvidenceOpen(true)
    setTimeout(() => document.getElementById(`ev-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50)
  }

  async function rate(t: Turn, rating: 1 | -1) {
    update(t.id, x => ({ ...x, rating }))
    try { await api.feedback({ question: t.question, answer: t.answer, rating, sources: t.sources }); toast('Thanks — feedback saved.') }
    catch { toast('Could not save feedback.', 'error') }
  }

  return (
    <div className="flex h-full">
      <section className="flex min-w-0 flex-1 flex-col">
        {turns.length > 0 && (
          <div className="flex items-center justify-end gap-1 border-b border-line-soft px-4 py-1.5">
            <button className={btn.ghost} onClick={() => exportMarkdown(turns)} disabled={!turns.some(t => t.answer)}><FileDown size={15} /> Export</button>
            <button className={btn.ghost} onClick={() => { setTurns([]); setActiveTurn(null) }} disabled={busy}><RotateCcw size={14} /> New conversation</button>
          </div>
        )}
        <div className="flex-1 overflow-y-auto">
          {turns.length === 0 ? (
            <div className="kolam-grid flex min-h-full flex-col justify-center py-12">
              <div className="mx-auto w-full max-w-3xl px-6">
                <h1 className="text-4xl font-extrabold leading-[1.1] tracking-tight text-field md:text-5xl">
                  Ask about Tamil Nadu farm schemes
                </h1>
                <p lang="ta" className="ta mt-3 text-lg text-muted">
                  மானியம், தகுதி, விண்ணப்ப முறை — தமிழ், English அல்லது Tanglish-ல் கேளுங்கள். ஒவ்வொரு பதிலும் அரசு ஆவணங்களிலிருந்து மேற்கோளுடன்.
                </p>
                {needsSetup ? <SetupChecklist health={health!} onGo={onGo} /> : (
                  <div className="mt-8">
                    <p className="mb-2 text-sm text-muted">Try one of these</p>
                    <div className="flex flex-wrap gap-2">
                      {EXAMPLES.map(e => (
                        <button key={e} onClick={() => submit(e)}
                          className="rounded-xl border border-line bg-surface px-3.5 py-1.5 text-left text-[0.95rem] hover:border-field-3 hover:text-field">
                          {e}
                        </button>
                      ))}
                    </div>
                    <p className="mt-6 text-sm text-faint">Press <kbd className="rounded border border-line bg-surface px-1.5 text-xs">/</kbd> anywhere to ask.</p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-12 px-6 py-10">
              {turns.map(t => (
                <article key={t.id} onClick={() => setActiveTurn(t.id)}>
                  <h2 className="text-xl font-bold leading-snug text-field">{t.question}</h2>
                  <Understood t={t} />
                  <div className="mt-3">
                    {t.status === 'retrieving' && <Thinking label={rewrite ? 'Understanding the question and searching…' : 'Searching the documents…'} />}
                    {t.status === 'answering' && !t.answer && !t.notice && !t.error && <Thinking label={`Found ${t.sources.length} passages — writing the answer…`} />}
                    {t.answer && (
                      <div className="prose-answer text-[1.02rem]">
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
                          a: ({ href, children }) => {
                            if (href?.startsWith('#cite-')) {
                              const n = Number(href.slice(6))
                              const hit = t.sources[n - 1]
                              return (
                                <button onClick={ev => { ev.stopPropagation(); openCite(t.id, n) }}
                                  title={hit ? `${hit.title}${locLabel(hit.location) ? ', ' + locLabel(hit.location) : ''}` : 'source'}
                                  className="mx-0.5 inline-flex h-5 min-w-5 -translate-y-px items-center justify-center rounded bg-turmeric-soft px-1 align-middle text-xs font-bold text-turmeric-deep hover:bg-turmeric hover:text-white">
                                  {children}
                                </button>
                              )
                            }
                            return <a href={href} target="_blank" rel="noreferrer" className="text-field-3 underline">{children}</a>
                          },
                        }}>{withCiteLinks(t.answer)}</ReactMarkdown>
                      </div>
                    )}
                    {t.notice && <p className="mt-2 text-muted">{t.notice}</p>}
                    {t.error && <div className="mt-3"><ErrorNote>{t.error}</ErrorNote></div>}
                    {t.status === 'done' && !t.answer && t.sources.length > 0 && (
                      <ol className="mt-3 space-y-3">
                        {t.sources.slice(0, 3).map(s => (
                          <li key={s.chunk_id} className="border-l-2 border-turmeric pl-3">
                            <div className="text-sm font-semibold">{s.title} <span className="font-normal text-muted">{locLabel(s.location)}</span></div>
                            <p className="mt-1 text-[0.95rem]"><Highlighted text={s.text} terms={s.matched_terms} max={420} /></p>
                          </li>
                        ))}
                      </ol>
                    )}
                    {t.followups && t.followups.length > 0 && t.status === 'done' && (
                      <div className="mt-4 flex flex-wrap gap-2">
                        {t.followups.map(f => (
                          <button key={f} onClick={e => { e.stopPropagation(); submit(f) }} disabled={busy}
                            className="rounded-xl border border-dashed border-field-3/60 px-3 py-1 text-left text-sm text-field hover:border-solid hover:bg-surface">
                            {f}
                          </button>
                        ))}
                      </div>
                    )}
                    {(t.status === 'done' || t.status === 'error') && (
                      <TurnFooter t={t} onEvidence={() => { setActiveTurn(t.id); setEvidenceOpen(true) }} onRate={r => rate(t, r)} />
                    )}
                  </div>
                </article>
              ))}
              <div ref={endRef} />
            </div>
          )}
        </div>

        {/* composer */}
        <div className="border-t border-line bg-paper px-4 py-3 md:px-6">
          <div className="mx-auto max-w-3xl">
            <form onSubmit={e => { e.preventDefault(); submit() }}
              className="flex items-end gap-2 rounded-2xl border border-line bg-surface p-2 focus-within:border-field-3">
              <label htmlFor="ask-input" className="sr-only">Your question</label>
              <textarea id="ask-input" rows={1} value={q} onChange={e => {
                setQ(e.target.value); e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
              }}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }}
                placeholder="Ask in Tamil, English or Tanglish — e.g. drip irrigation ku evlo subsidy?"
                className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-[1.02rem] placeholder:text-faint focus:outline-none focus-visible:outline-none" />
              <button type="button" className={btn.ghost} onClick={() => setShowSettings(s => !s)} aria-expanded={showSettings} title="Retrieval settings">
                <SlidersHorizontal size={18} />
              </button>
              {busy ? (
                <button type="button" onClick={() => abortRef.current?.abort()} className={btn.secondary} aria-label="Stop"><Square size={16} /> Stop</button>
              ) : (
                <button type="submit" disabled={!q.trim()} className={btn.primary} aria-label="Send"><SendHorizontal size={18} /></button>
              )}
            </form>
            {showSettings && (
              <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-2 px-1 text-sm">
                <Segmented label="Retrieval mode" value={mode} onChange={setMode} options={[
                  { value: 'hybrid', label: 'Hybrid', hint: 'BM25 + vectors fused with RRF (recommended)' },
                  { value: 'dense', label: 'Vector', hint: 'FAISS similarity only' },
                  { value: 'bm25', label: 'Keyword', hint: 'BM25 only' },
                ]} />
                <Toggle checked={rewrite} onChange={setRewrite} label="Query rewrite" hint="Rewrites Tanglish / follow-ups into English + Tamil search queries" />
                <Toggle checked={rerank} onChange={setRerank} label="Rerank" hint="Re-orders the top 30 candidates by relevance" />
                <label className="flex items-center gap-2">Passages
                  <select value={topK} onChange={e => setTopK(Number(e.target.value))} className="rounded-md border border-line bg-surface px-2 py-1">
                    {[3, 4, 6, 8, 10].map(n => <option key={n}>{n}</option>)}
                  </select>
                </label>
              </div>
            )}
          </div>
        </div>
      </section>

      {current && current.sources.length > 0 && (
        <aside className="hidden w-[400px] shrink-0 overflow-y-auto border-l border-line bg-surface xl:block 2xl:w-[460px]" aria-label="Evidence">
          <EvidenceList turn={current} focus={focusCite} />
        </aside>
      )}
      <div className="xl:hidden">
        <Drawer open={evidenceOpen && !!current} onClose={() => setEvidenceOpen(false)} title={<span className="font-bold">Evidence</span>}>
          {current && <EvidenceList turn={current} focus={focusCite} bare />}
        </Drawer>
      </div>
    </div>
  )
}

function SetupChecklist({ health, onGo }: { health: Health; onGo: (p: Page) => void }) {
  const items = [
    { ok: health.llm.ready, label: health.llm.ready ? `Answer model ready (${health.llm.model})` : 'Add your OpenAI API key',
      hint: health.llm.ready ? '' : 'Put OPENAI_API_KEY in backend/.env and restart the server.', action: () => onGo('system'), cta: 'Open System' },
    { ok: health.store.documents > 0, label: health.store.documents > 0 ? `${health.store.documents} documents indexed` : 'Index some documents',
      hint: health.store.documents > 0 ? '' : 'Index the starter pack (seconds) or fetch the 569 researched sources.', action: () => onGo('ingest'), cta: 'Add documents' },
  ]
  return (
    <div className="mt-8 max-w-xl rounded-2xl border border-line bg-surface p-5">
      <h2 className="font-bold">Two steps before the first question</h2>
      <ul className="mt-3 space-y-3">
        {items.map(i => (
          <li key={i.label} className="flex items-start gap-3">
            {i.ok ? <CircleCheck size={20} className="mt-0.5 shrink-0 text-field-3" /> : <CircleAlert size={20} className="mt-0.5 shrink-0 text-turmeric" />}
            <div className="min-w-0 flex-1">
              <div className="font-medium">{i.label}</div>
              {i.hint && <div className="text-sm text-muted">{i.hint}</div>}
            </div>
            {!i.ok && <button className={btn.secondary + ' py-1 text-sm'} onClick={i.action}>{i.cta}</button>}
          </li>
        ))}
      </ul>
    </div>
  )
}

function Understood({ t }: { t: Turn }) {
  const p = t.plan
  if (!p || p.source === 'fallback' || !t.variants || t.variants.length < 2) return null
  const other = t.variants.filter(v => v.label !== 'original')
  return (
    <div className="mt-1.5 flex flex-wrap items-baseline gap-x-2 gap-y-1 text-sm text-muted">
      <span>Searched as</span>
      {other.map(v => (
        <span key={v.label} className={`rounded bg-line-soft px-1.5 py-px text-ink/80 ${v.label === 'tamil' ? 'ta' : ''}`} title={v.label}>{v.text}</span>
      ))}
      {p.keywords.length > 0 && <span className="text-faint">with keywords {p.keywords.join(', ')}</span>}
      {p.language === 'tanglish' && <Tag tone="turmeric">Tanglish understood</Tag>}
    </div>
  )
}

function TurnFooter({ t, onEvidence, onRate }: { t: Turn; onEvidence: () => void; onRate: (r: 1 | -1) => void }) {
  const [copied, setCopied] = useState(false)
  const [showTrace, setShowTrace] = useState(false)
  const secs = t.timings?.total_ms ? (Number(t.timings.total_ms) / 1000).toFixed(1) : null
  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted">
        <button onClick={onEvidence} className="hover:text-ink xl:pointer-events-none">
          {t.sources.length} passages{t.reranked ? ', reranked' : ''}
        </button>
        {secs && <span className="tnum">{secs} s</span>}
        {t.usage && t.usage.cost_usd > 0 && <span className="tnum" title={`${t.usage.input_tokens.toLocaleString()} input + ${t.usage.output_tokens.toLocaleString()} output tokens`}>{fmtUsd(t.usage.cost_usd)}</span>}
        {t.trace && <button className={btn.ghost} onClick={() => setShowTrace(s => !s)} aria-expanded={showTrace}>
          How this answer was built <ChevronDown size={14} className={showTrace ? 'rotate-180' : ''} /></button>}
        <span className="ml-auto flex items-center gap-0.5">
          {t.answer && (
            <button className={btn.ghost} onClick={() => { navigator.clipboard.writeText(t.answer); setCopied(true); toast('Answer copied.'); setTimeout(() => setCopied(false), 1500) }} aria-label="Copy answer">
              {copied ? <Check size={15} /> : <Copy size={15} />}
            </button>
          )}
          {t.answer && <>
            <button className={`${btn.ghost} ${t.rating === 1 ? 'text-field' : ''}`} onClick={() => onRate(1)} aria-pressed={t.rating === 1} aria-label="Helpful"><ThumbsUp size={15} /></button>
            <button className={`${btn.ghost} ${t.rating === -1 ? 'text-danger' : ''}`} onClick={() => onRate(-1)} aria-pressed={t.rating === -1} aria-label="Not helpful"><ThumbsDown size={15} /></button>
          </>}
        </span>
      </div>
      {showTrace && t.trace && <TracePanel t={t} />}
    </div>
  )
}

function TracePanel({ t }: { t: Turn }) {
  const usageBy = (step: string) => t.usage?.steps.filter(s => s.step === step || (step === 'plan' && s.step === 'plan_query')
    || (step === 'dense' && s.step === 'embed_query')) ?? []
  const detail = (s: TraceStep) => {
    if (s.step === 'plan') return s.source === 'llm' ? `language: ${LANG_LABEL[s.language ?? ''] ?? s.language}` : s.source === 'cache' ? 'cached' : 'no LLM, used the question as typed'
    if (s.step === 'dense' || s.step === 'bm25') return `${s.queries} quer${s.queries === 1 ? 'y' : 'ies'}, ${s.hits} hits`
    if (s.step === 'fuse') return `${s.candidates} unique candidates`
    if (s.step === 'rerank') return s.skipped ? `skipped: ${s.skipped}` : `${s.candidates} candidates by ${s.model}`
    if (s.step === 'answer') return `${s.model}${s.first_token_ms ? `, first words after ${(s.first_token_ms / 1000).toFixed(1)} s` : ''}`
    return ''
  }
  return (
    <div className="mt-3 rounded-xl border border-line bg-surface p-4">
      <ol className="space-y-2.5">
        {t.trace!.map((s, i) => {
          const u = usageBy(s.step)
          const cost = u.reduce((a, x) => a + x.cost_usd, 0)
          const toks = u.reduce((a, x) => a + x.input_tokens + x.output_tokens, 0)
          return (
            <li key={s.step + i} className="grid grid-cols-[1.5rem_1fr_auto] items-baseline gap-2 text-sm">
              <span className="tnum text-faint">{i + 1}</span>
              <div className="min-w-0">
                <div className="font-semibold">{STEP_LABEL[s.step] ?? s.step}</div>
                <div className="truncate text-muted">{detail(s)}</div>
              </div>
              <div className="tnum text-right text-muted">
                {s.ms != null && <div>{s.ms < 1000 ? `${Math.round(s.ms)} ms` : `${(s.ms / 1000).toFixed(1)} s`}</div>}
                {toks > 0 && <div className="text-xs">{toks.toLocaleString()} tok, {fmtUsd(cost)}</div>}
              </div>
            </li>
          )
        })}
      </ol>
      {t.usage && (
        <div className="tnum mt-3 border-t border-line-soft pt-2 text-right text-sm">
          Total <b>{fmtUsd(t.usage.cost_usd)}</b> <span className="text-muted">({t.usage.input_tokens.toLocaleString()} in, {t.usage.output_tokens.toLocaleString()} out tokens)</span>
        </div>
      )}
    </div>
  )
}

function EvidenceList({ turn, focus, bare = false }: { turn: Turn; focus: number | null; bare?: boolean }) {
  return (
    <div className={bare ? '' : 'px-5 py-6'}>
      {!bare && (
        <div className="mb-4">
          <h2 className="flex items-center gap-2 font-bold"><KolamMark size={20} className="text-field" /> Evidence</h2>
          <p className="mt-0.5 line-clamp-2 text-sm text-muted">Passages used for “{turn.question}”</p>
        </div>
      )}
      <ol className="space-y-4">
        {turn.sources.map((s, i) => <Evidence key={s.chunk_id} n={i + 1} s={s} reranked={turn.reranked} focused={focus === i + 1} />)}
      </ol>
    </div>
  )
}

function Evidence({ n, s, reranked, focused }: { n: number; s: Hit; reranked: boolean; focused: boolean }) {
  const [open, setOpen] = useState(false)
  useEffect(() => { if (focused) setOpen(true) }, [focused])
  const loc = locLabel(s.location) === s.title ? '' : locLabel(s.location)
  return (
    <li id={`ev-${n}`} className={`rounded-xl border p-3.5 transition-colors ${focused ? 'border-turmeric bg-turmeric-soft/40' : 'border-line'}`}>
      <div className="flex items-start gap-2.5">
        <span className="tnum mt-0.5 inline-flex h-6 min-w-6 items-center justify-center rounded bg-turmeric-soft text-sm font-bold text-turmeric-deep">{n}</span>
        <div className="min-w-0 flex-1">
          <div className="flex items-start gap-1.5">
            <TypeIcon type={s.file_type} />
            <h3 className="min-w-0 text-[0.95rem] font-semibold leading-snug">{s.title}</h3>
          </div>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {s.year && <Tag tone="indigo">{s.year}</Tag>}
            {loc && <Tag>{loc}</Tag>}
            <Tag>{LANG_LABEL[s.language] ?? s.language}</Tag>
            {s.doc_type && <Tag tone="green">{s.doc_type}</Tag>}
          </div>
        </div>
      </div>
      <p className={`mt-2.5 whitespace-pre-line text-[0.93rem] leading-relaxed ${s.language !== 'en' ? 'ta' : ''}`}>
        <Highlighted text={s.text} terms={s.matched_terms} max={open ? undefined : 360} />
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-1">
        {s.text.length > 360 && <button className={btn.ghost} onClick={() => setOpen(o => !o)}>{open ? 'Show less' : 'Show full passage'}</button>}
        {s.source_url && <a className={btn.ghost} href={s.source_url} target="_blank" rel="noreferrer"><ExternalLink size={14} /> Source</a>}
        <a className={btn.ghost} href={api.fileUrl(s.doc_id)} target="_blank" rel="noreferrer"><Download size={14} /> File</a>
      </div>
      {open && <div className="mt-2"><ScoreLadder s={s.scores} reranked={reranked} /></div>}
    </li>
  )
}
