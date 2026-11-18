import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Upload, FolderOpen, X, Link2, Globe, Check, CircleX, Square, Database } from 'lucide-react'
import { api, fmtBytes, fmtUsd, uploadFiles, type Health, type Job, type SourceRow } from '../api'
import { Segmented, Tag, Toggle, TypeIcon, btn, ErrorNote, input, KolamMark, withAdmin, toast } from '../ui'
import type { Page } from '../App'

type Tab = 'upload' | 'links' | 'research' | 'folder'
const ACCEPT = '.pdf,.docx,.doc,.odt,.rtf,.pptx,.ppt,.odp,.xlsx,.xlsm,.xls,.ods,.csv,.tsv,.html,.htm,.txt,.md,.json,.xml,.zip,.png,.jpg,.jpeg,.tif,.tiff,.webp'

export default function Ingest({ health, onChanged, onGo }: { health: Health | null; onChanged: () => void; onGo: (p: Page) => void }) {
  const [tab, setTab] = useState<Tab>('upload')
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [history, setHistory] = useState<Job[]>([])
  const [uploadPct, setUploadPct] = useState<number | null>(null)
  const [err, setErr] = useState<string | null>(null)

  // pick up a running job (e.g. after a page reload)
  useEffect(() => {
    api.jobs().then(js => {
      setHistory(js)
      if (!jobId && js.length) setJobId((js.find(j => j.status === 'running' || j.status === 'queued') ?? js[0]).id)
    }).catch(() => {})
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!jobId) return
    let stop = false
    const tick = async () => {
      try {
        const j = await api.job(jobId)
        if (stop) return
        setJob(j)
        if (j.status === 'running' || j.status === 'queued') setTimeout(tick, 900)
        else { onChanged(); api.jobs().then(setHistory).catch(() => {}) }
      } catch { if (!stop) setTimeout(tick, 2500) }
    }
    tick()
    return () => { stop = true }
  }, [jobId, onChanged])

  const started = useCallback((id: string) => { setJobId(id); setErr(null); setUploadPct(null) }, [])
  const running = job && (job.status === 'running' || job.status === 'queued')

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <header className="max-w-3xl">
        <h1 className="text-3xl font-extrabold tracking-tight text-field">Add documents</h1>
        <p className="mt-1 text-muted">Every file goes through the same pipeline: load the text, split it into overlapping chunks, add a short document summary to each chunk, embed with OpenAI, and store the vectors in the FAISS index on disk. Files already in the index are skipped automatically.</p>
      </header>

      <div className="mt-6 grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <section aria-label="Choose documents">
          <Segmented label="Source" value={tab} onChange={setTab} options={[
            { value: 'upload', label: 'Upload files' }, { value: 'links', label: 'From links' },
            { value: 'research', label: 'Researched sources' }, { value: 'folder', label: 'Server folder' }]} />
          <div className="mt-4">
            {err && <div className="mb-3"><ErrorNote>{err}</ErrorNote></div>}
            {tab === 'upload' && <UploadPanel busy={!!running} uploadPct={uploadPct} setUploadPct={setUploadPct} onStarted={started} onError={setErr} exts={health?.supported_extensions ?? []} />}
            {tab === 'links' && <LinksPanel busy={!!running} onStarted={started} onError={setErr} />}
            {tab === 'research' && <ResearchPanel busy={!!running} onStarted={started} onError={setErr} onGo={onGo} />}
            {tab === 'folder' && <FolderPanel busy={!!running} onStarted={started} onError={setErr} />}
          </div>
        </section>

        <section aria-label="Pipeline progress" className="min-w-0">
          {job ? <JobView job={job} uploadPct={uploadPct} /> : (
            <div className="kolam-grid flex h-full min-h-64 items-center justify-center rounded-2xl border border-line-soft p-8 text-center text-muted">
              Progress for each pipeline stage will appear here once you start indexing.
            </div>
          )}
          {history.length > 1 && (
            <div className="mt-8">
              <h2 className="mb-2 font-bold">Earlier runs</h2>
              <ul className="divide-y divide-line rounded-xl border border-line bg-surface text-sm">
                {history.slice(0, 8).map(h => (
                  <li key={h.id}>
                    <button onClick={() => setJobId(h.id)} className={`flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-paper ${h.id === jobId ? 'bg-paper' : ''}`}>
                      <StatusDot status={h.status} />
                      <span className="w-24 shrink-0 font-medium capitalize">{h.kind}</span>
                      <span className="tnum flex-1 text-muted">{h.files_indexed} indexed, {h.files_skipped} skipped, {h.files_failed} failed</span>
                      <span className="tnum text-xs text-faint">{new Date(h.created_at).toLocaleString()}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

function StatusDot({ status }: { status: string }) {
  const c = status === 'done' ? 'bg-field-3' : status === 'failed' ? 'bg-danger' : status === 'cancelled' ? 'bg-faint' : 'bg-turmeric'
  return <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${c}`} aria-label={status} />
}

/* ------------------------------------------------------------------ upload */
function UploadPanel({ busy, uploadPct, setUploadPct, onStarted, onError, exts }:
  { busy: boolean; uploadPct: number | null; setUploadPct: (n: number | null) => void; onStarted: (id: string) => void; onError: (e: string | null) => void; exts: string[] }) {
  const [files, setFiles] = useState<File[]>([])
  const [drag, setDrag] = useState(false)
  const [category, setCategory] = useState('uploaded')
  const [advanced, setAdvanced] = useState(false)
  const [chunkSize, setChunkSize] = useState(1000)
  const [overlap, setOverlap] = useState(150)
  const fileRef = useRef<HTMLInputElement>(null)
  const dirRef = useRef<HTMLInputElement>(null)

  const add = (list: FileList | File[]) => {
    const arr = Array.from(list).filter(f => !f.name.startsWith('.'))
    setFiles(prev => {
      const keys = new Set(prev.map(f => f.name + f.size))
      return [...prev, ...arr.filter(f => !keys.has(f.name + f.size))]
    })
  }
  const total = files.reduce((s, f) => s + f.size, 0)
  const unsupported = files.filter(f => exts.length && !exts.includes('.' + (f.name.split('.').pop() ?? '').toLowerCase()))

  async function start() {
    onError(null)
    setUploadPct(0)
    try {
      const r = await withAdmin(() => uploadFiles(files, { category, chunk_size: advanced ? chunkSize : undefined, chunk_overlap: advanced ? overlap : undefined }, setUploadPct))
      if (r.rejected.length) onError(`Skipped ${r.rejected.length} file(s): ${r.rejected.map(x => `${x.name} (${x.reason})`).join(', ')}`)
      setFiles([])
      onStarted(r.job_id)
    } catch (e) { onError((e as Error).message); setUploadPct(null) }
  }

  return (
    <div>
      <div onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files) }}
        className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors ${drag ? 'border-turmeric bg-turmeric-soft/40' : 'border-line bg-surface'}`}>
        <Upload size={28} className="text-field-3" />
        <p className="mt-3 text-lg font-semibold">Drop files or a whole folder here</p>
        <p className="mt-1 max-w-md text-sm text-muted">PDF, Word, Excel, PowerPoint, CSV, HTML, text, JSON, XML, images (OCR) and ZIP archives. Tamil and English both work.</p>
        <div className="mt-4 flex flex-wrap justify-center gap-2">
          <button type="button" className={btn.secondary} onClick={() => fileRef.current?.click()}>Choose files</button>
          <button type="button" className={btn.secondary} onClick={() => dirRef.current?.click()}><FolderOpen size={16} /> Choose folder</button>
        </div>
        <input ref={fileRef} type="file" multiple accept={ACCEPT} className="hidden" onChange={e => e.target.files && add(e.target.files)} />
        <input ref={dirRef} type="file" multiple className="hidden" onChange={e => e.target.files && add(e.target.files)}
          {...({ webkitdirectory: '', directory: '' } as Record<string, string>)} />
      </div>

      {files.length > 0 && (
        <div className="mt-4">
          <div className="mb-2 flex items-baseline justify-between">
            <h3 className="font-bold"><span className="tnum">{files.length}</span> file{files.length > 1 ? 's' : ''} selected <span className="tnum font-normal text-muted">({fmtBytes(total)})</span></h3>
            <button className={btn.ghost} onClick={() => setFiles([])}>Clear</button>
          </div>
          <ul className="max-h-56 divide-y divide-line-soft overflow-y-auto rounded-xl border border-line bg-surface text-sm">
            {files.map((f, i) => (
              <li key={f.name + f.size} className="flex items-center gap-2 px-3 py-1.5">
                <TypeIcon type={f.name.split('.').pop() ?? ''} />
                <span className="min-w-0 flex-1 truncate">{f.name}</span>
                <span className="tnum text-xs text-muted">{fmtBytes(f.size)}</span>
                <button onClick={() => setFiles(fs => fs.filter((_, j) => j !== i))} className="text-faint hover:text-danger" aria-label={`Remove ${f.name}`}><X size={14} /></button>
              </li>
            ))}
          </ul>
          {unsupported.length > 0 && <p className="mt-2 text-sm text-turmeric-deep">{unsupported.length} file(s) have an unusual extension; the server will try to detect their type.</p>}
        </div>
      )}

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <label className="text-sm">
          <span className="mb-1 block font-medium">Category label</span>
          <input className={input} value={category} onChange={e => setCategory(e.target.value)} placeholder="e.g. guidelines, go_notification" />
        </label>
        <div className="flex items-end">
          <Toggle checked={advanced} onChange={setAdvanced} label="Custom chunk size" />
        </div>
        {advanced && (
          <>
            <label className="text-sm"><span className="mb-1 block font-medium">Chunk size (characters)</span>
              <input type="number" min={200} max={4000} step={50} className={input} value={chunkSize} onChange={e => setChunkSize(Number(e.target.value))} /></label>
            <label className="text-sm"><span className="mb-1 block font-medium">Overlap (characters)</span>
              <input type="number" min={0} max={1000} step={10} className={input} value={overlap} onChange={e => setOverlap(Number(e.target.value))} /></label>
          </>
        )}
      </div>

      <button className={btn.primary + ' mt-5 w-full py-2.5'} disabled={!files.length || busy || uploadPct !== null} onClick={start}>
        {uploadPct !== null ? `Uploading… ${uploadPct}%` : busy ? 'Wait for the current run to finish' : `Index ${files.length || ''} file${files.length === 1 ? '' : 's'}`}
      </button>
    </div>
  )
}

/* ------------------------------------------------------------------ links */
function LinksPanel({ busy, onStarted, onError }: { busy: boolean; onStarted: (id: string) => void; onError: (e: string | null) => void }) {
  const [text, setText] = useState('')
  const [discover, setDiscover] = useState(false)
  const [category, setCategory] = useState('web')
  const urls = text.split(/\s+/).filter(u => /^https?:\/\//.test(u))
  return (
    <div className="space-y-4">
      <label className="block text-sm">
        <span className="mb-1 block font-medium">Links, one per line</span>
        <textarea rows={8} className={input + ' font-[inherit]'} value={text} onChange={e => setText(e.target.value)}
          placeholder={'https://agritech.tnau.ac.in/pdf/agri_t_cc_2026_27.pdf\nhttps://aed.tn.gov.in/en/individual-based-subsidy-schemes'} />
      </label>
      <Toggle checked={discover} onChange={setDiscover} label="Also follow document links found on these pages" hint="Works on hub pages and Apache directory listings" />
      <label className="block text-sm"><span className="mb-1 block font-medium">Category label</span>
        <input className={input} value={category} onChange={e => setCategory(e.target.value)} /></label>
      <button className={btn.primary + ' w-full py-2.5'} disabled={!urls.length || busy}
        onClick={() => withAdmin(() => api.ingestUrls({ urls, category, discover })).then(r => { setText(''); onStarted(r.job_id) }).catch(e => onError(e.message))}>
        <Link2 size={17} /> Download and index {urls.length || ''} link{urls.length === 1 ? '' : 's'}
      </button>
    </div>
  )
}

/* ------------------------------------------------------------------ research manifest */
function ResearchPanel({ busy, onStarted, onError, onGo }: { busy: boolean; onStarted: (id: string) => void; onError: (e: string | null) => void; onGo: (p: Page) => void }) {
  const [rows, setRows] = useState<SourceRow[] | null>(null)
  const [relevance, setRelevance] = useState<'core' | 'all'>('core')
  const [discover, setDiscover] = useState(false)
  useEffect(() => { api.sources().then(r => setRows(r.items)).catch(e => onError(e.message)) }, [onError])
  const sel = useMemo(() => (rows ?? []).filter(r => relevance === 'all' || r.relevance === 'core'), [rows, relevance])
  const byCat = useMemo(() => {
    const m = new Map<string, number>()
    sel.forEach(r => m.set(r.category, (m.get(r.category) ?? 0) + 1))
    return [...m.entries()].sort((a, b) => b[1] - a[1])
  }, [sel])
  const pending = sel.filter(r => !r.indexed).length
  if (!rows) return <p className="text-muted">Loading the source list…</p>
  return (
    <div className="space-y-4">
      <p>
        <span className="tnum text-3xl font-extrabold text-field">{rows.length}</span>{' '}
        <span className="text-muted">government and institutional sources were found by web research: policy notes 2009–2025, budget speeches up to 2026-27, citizen charters, G.O.s, scheme guidelines, district plans and scheme pages.</span>
      </p>
      <Segmented label="Which sources" value={relevance} onChange={setRelevance} options={[
        { value: 'core', label: `Core (${rows.filter(r => r.relevance === 'core').length})` },
        { value: 'all', label: `Core + supporting (${rows.length})` }]} />
      <ul className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-3">
        {byCat.map(([c, n]) => <li key={c} className="flex justify-between gap-2"><span className="capitalize text-muted">{c.replace(/_/g, ' ')}</span><span className="tnum font-semibold">{n}</span></li>)}
      </ul>
      <Toggle checked={discover} onChange={setDiscover} label="Also crawl the seed directories for more documents" hint="Walks TNAU directory listings and hub pages; adds up to 300 extra files" />
      <div className="rounded-xl border border-line bg-surface p-3 text-sm text-muted">
        <span className="tnum font-semibold text-ink">{pending}</span> of {sel.length} not yet indexed. Downloads run 4 at a time with a polite delay per site; large runs take a while, and you can keep using the app meanwhile.
      </div>
      <div className="flex flex-wrap gap-2">
        <button className={btn.primary + ' flex-1 py-2.5'} disabled={busy || (!pending && !discover)}
          onClick={() => withAdmin(() => api.ingestManifest({ relevance, include_discovered: discover })).then(r => onStarted(r.job_id)).catch(e => onError(e.message))}>
          <Globe size={17} /> Fetch and index {pending} sources
        </button>
        <button className={btn.secondary} onClick={() => onGo('sources')}>Pick individually</button>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ folder */
function FolderPanel({ busy, onStarted, onError }: { busy: boolean; onStarted: (id: string) => void; onError: (e: string | null) => void }) {
  const [path, setPath] = useState('seed')
  const [category, setCategory] = useState('local')
  return (
    <div className="space-y-4">
      <p className="text-muted">Index files that are already on the server, inside <code>backend/data/</code>. Use <b>seed</b> for the starter pack compiled from this research, or <b>raw</b> for files downloaded by <code>scripts/crawl.py</code>.</p>
      <label className="block text-sm"><span className="mb-1 block font-medium">Folder inside backend/data</span>
        <input className={input} value={path} onChange={e => setPath(e.target.value)} /></label>
      <label className="block text-sm"><span className="mb-1 block font-medium">Category label</span>
        <input className={input} value={category} onChange={e => setCategory(e.target.value)} /></label>
      <button className={btn.primary + ' w-full py-2.5'} disabled={busy || !path}
        onClick={() => withAdmin(() => api.ingestFolder({ path, category })).then(r => onStarted(r.job_id)).catch(e => onError(e.message))}>
        <Database size={17} /> Index folder
      </button>
    </div>
  )
}

/* ------------------------------------------------------------------ job view */
function JobView({ job, uploadPct }: { job: Job; uploadPct: number | null }) {
  const [tab, setTab] = useState<'files' | 'log'>('files')
  const isDl = job.kind === 'urls' || job.kind === 'manifest'
  const processed = job.files_indexed + job.files_skipped + job.files_failed
  const stageIdx = ['queued', 'downloading', 'loading', 'splitting', 'embedding', 'indexing', 'finalizing', 'done'].indexOf(job.stage)
  const done = job.status === 'done'
  const steps = [
    { name: isDl ? 'Download' : job.kind === 'folder' || job.kind === 'cli' ? 'Collect files' : 'Upload', detail: isDl ? `${job.downloads_done} of ${job.downloads_total} fetched${job.downloads_failed ? `, ${job.downloads_failed} failed` : ''}` : (uploadPct !== null && uploadPct < 100 ? `${uploadPct}%` : `${job.files_total} file${job.files_total === 1 ? '' : 's'} ready`),
      frac: isDl ? (job.downloads_total ? (job.downloads_done + job.downloads_failed) / job.downloads_total : stageIdx > 1 ? 1 : 0) : 1 },
    { name: 'Load text', detail: `${job.files_loaded} of ${job.files_total} files read`, frac: job.files_total ? job.files_loaded / job.files_total : 0 },
    { name: 'Split into chunks', detail: `${job.chunks_total.toLocaleString()} chunks from ${job.files_split} files`, frac: job.files_total ? job.files_split / job.files_total : 0 },
    { name: 'Summarise + embed', detail: `${job.chunks_embedded.toLocaleString()} of ${job.chunks_total.toLocaleString()} chunks`, frac: job.chunks_total ? job.chunks_embedded / job.chunks_total : processed && job.files_total === processed ? 1 : 0 },
    { name: 'Store in FAISS', detail: `${job.files_indexed} indexed, ${job.files_skipped} duplicates skipped, ${job.files_failed} failed`, frac: job.files_total ? processed / job.files_total : 0 },
    { name: 'Save index and keyword table', detail: done ? 'index.faiss written, BM25 rebuilt' : job.stage === 'finalizing' ? 'writing…' : 'after all files', frac: done ? 1 : job.stage === 'finalizing' ? 0.5 : 0 },
  ]
  const running = job.status === 'running' || job.status === 'queued'
  return (
    <div>
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-xl font-bold">
            {running ? <KolamMark size={22} animate className="text-field" /> : <StatusDot status={job.status} />}
            {running ? 'Indexing' : job.status === 'done' ? 'Finished' : job.status === 'failed' ? 'Failed' : 'Cancelled'}
          </h2>
          <p className="tnum text-sm text-muted">
            {job.current_file ? <>Now: <span className="text-ink">{job.current_file}</span></> : `Run ${job.id}`}
            {job.elapsed_s ? `, ${Math.round(job.elapsed_s)} s elapsed` : ''}
            {job.cost_usd ? `, OpenAI cost ${fmtUsd(job.cost_usd)}` : ''}
          </p>
        </div>
        {running && <button className={btn.danger} onClick={() => withAdmin(() => api.cancelJob(job.id)).then(() => toast('Stopping after the current file…', 'info')).catch(e => toast(e.message, 'error'))}><Square size={14} /> Stop</button>}
      </div>
      {job.error && <div className="mt-3"><ErrorNote>{job.error}</ErrorNote></div>}

      <ol className="mt-5 space-y-3">
        {steps.map((s, i) => {
          const complete = s.frac >= 1 && (done || i < steps.length - 1)
          return (
            <li key={s.name} className="grid grid-cols-[1.75rem_1fr] gap-3">
              <span className={`tnum flex h-7 w-7 items-center justify-center rounded-full text-sm font-bold ${complete ? 'bg-field text-paper' : s.frac > 0 ? 'bg-turmeric text-white' : 'bg-line-soft text-faint'}`}>
                {complete ? <Check size={15} /> : i + 1}
              </span>
              <div className="min-w-0">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-semibold">{s.name}</span>
                  <span className="tnum truncate text-sm text-muted">{s.detail}</span>
                </div>
                <div className="mt-1 h-1.5 rounded-full bg-line-soft" aria-hidden="true">
                  <div className={`h-full rounded-full transition-[width] duration-500 ${complete ? 'bg-field-3' : 'bg-turmeric'}`} style={{ width: `${Math.min(100, Math.round(s.frac * 100))}%` }} />
                </div>
              </div>
            </li>
          )
        })}
      </ol>

      <div className="mt-6">
        <Segmented label="Details" value={tab} onChange={setTab} options={[{ value: 'files', label: `Files (${job.files.length})` }, { value: 'log', label: 'Log' }]} />
        {tab === 'files' ? (
          <div className="mt-3 max-h-80 overflow-y-auto rounded-xl border border-line bg-surface">
            <table className="w-full table-fixed text-sm">
              <colgroup><col /><col className="w-28" /><col className="w-20" /></colgroup>
              <thead className="sticky top-0 bg-paper text-left text-muted">
                <tr><th className="px-3 py-1.5 font-medium">File</th><th className="px-2 py-1.5 font-medium">Status</th><th className="px-3 py-1.5 text-right font-medium">Chunks</th></tr>
              </thead>
              <tbody className="divide-y divide-line-soft">
                {job.files.slice().reverse().map((f, i) => (
                  <tr key={f.name + i}>
                    <td className="px-3 py-1.5">
                      <div className="flex min-w-0 items-center gap-2"><TypeIcon type={f.type ?? f.name.split('.').pop() ?? ''} /><span className="truncate" title={f.name}>{f.title || f.name}</span></div>
                      {f.error && <div className="truncate pl-6 text-xs text-muted" title={f.error}>{f.error}</div>}
                    </td>
                    <td className="px-2 py-1.5">
                      {f.status === 'indexed' ? <Tag tone="green"><Check size={12} /> indexed</Tag>
                        : f.status === 'failed' ? <Tag tone="danger"><CircleX size={12} /> failed</Tag>
                        : f.status === 'skipped' ? <Tag>duplicate</Tag> : <Tag tone="turmeric">{f.status}</Tag>}
                    </td>
                    <td className="tnum px-3 py-1.5 text-right">{f.chunks || '—'}</td>
                  </tr>
                ))}
                {!job.files.length && <tr><td colSpan={3} className="px-3 py-4 text-center text-muted">{isDl ? 'Downloading…' : 'Waiting…'}</td></tr>}
              </tbody>
            </table>
          </div>
        ) : (
          <pre className="mt-3 max-h-80 overflow-y-auto whitespace-pre-wrap rounded-xl bg-rail p-3 text-xs leading-relaxed text-rail-ink/85">
            {(job.logs ?? []).map(l => `${l.t.slice(11, 19)}  ${l.msg}`).join('\n') || 'No log lines yet.'}
          </pre>
        )}
      </div>
    </div>
  )
}
