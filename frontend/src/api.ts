// Typed client for the FastAPI backend.
export type Mode = 'hybrid' | 'dense' | 'bm25'

export interface Scores {
  dense: number | null; dense_rank: number | null
  bm25: number | null; bm25_rank: number | null
  fused: number | null; fused_rank: number | null
  rerank: number | null
}
export interface Hit {
  rank: number; chunk_id: number; doc_id: string; chunk_index: number; n_chunks: number
  title: string; filename: string; file_type: string; category: string; source_url: string | null
  language: string; location: Record<string, string | number>; text: string; matched_terms: string[]
  scores: Scores; year?: string | null; doc_type?: string | null
}
export interface QueryPlan {
  language: 'ta' | 'en' | 'tanglish' | 'mixed'; standalone_question: string; english_query: string
  tamil_query: string; keywords: string[]; years: string[]; source: 'llm' | 'cache' | 'fallback'; error?: string
}
export interface TraceStep { step: string; ms?: number; model?: string; kind?: string; source?: string; language?: string
  queries?: number; hits?: number; candidates?: number; skipped?: string; first_token_ms?: number | null }
export interface UsageStep { step: string; model: string; input_tokens: number; output_tokens: number; cost_usd: number; ms?: number }
export interface UsageSummary { input_tokens: number; output_tokens: number; cost_usd: number; steps: UsageStep[] }
export interface SearchResponse {
  query: string; mode: Mode; reranked: boolean; results: Hit[]
  candidates: { dense: number; bm25: number; fused: number }
  timings: Record<string, number>; plan?: QueryPlan; variants?: { label: string; text: string }[]
  trace?: TraceStep[]; usage?: UsageSummary
}
export interface SearchParams {
  query: string; mode: Mode; top_k: number; rerank: boolean
  candidate_k?: number; rerank_candidates?: number; max_per_doc?: number; rewrite?: boolean
  filters?: { file_types?: string[]; categories?: string[]; languages?: string[] }
}
export interface JobFile { name: string; status: string; chunks: number; error: string | null; type?: string; size?: number; doc_id?: string; language?: string; title?: string; notes?: string[] }
export interface Job {
  id: string; kind: string; status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled'; stage: string
  created_at: string; started_at: string | null; finished_at: string | null; error: string | null
  downloads_total: number; downloads_done: number; downloads_failed: number
  files_total: number; files_loaded: number; files_split: number; files_indexed: number
  files_skipped: number; files_failed: number; chunks_total: number; chunks_embedded: number
  current_file: string | null; elapsed_s: number | null; files: JobFile[]; cost_usd?: number; tokens?: number
  logs?: { t: string; level: string; msg: string }[]; params: Record<string, unknown>
}
export interface Doc {
  doc_id: string; filename: string; title: string; file_type: string; category: string; language: string
  source_url: string | null; source_id: string | null; sha256: string; size_bytes: number; n_pages: number
  n_chunks: number; extraction: string; notes: string[]; added_at: string; job_id: string
  summary?: string | null; year?: string | null; doc_type?: string | null; department?: string | null
  schemes?: string[]; cost_usd?: number | null
}
export interface ChunkRow { id: number; chunk_index: number; text: string; meta: Record<string, string | number>; language: string; n_chars: number }
export interface Stats {
  store_dir?: string; documents: number; chunks: number; vectors: number; bm25_terms: number; index_bytes: number
  embedding_model: string | null; dim: number | null; updated_at: string | null
  by_type: Record<string, number>; by_category: Record<string, number>; by_language: Record<string, number>
  total_bytes: number; by_year?: Record<string, number>; ingest_cost_usd?: number
}
export interface UsageTotals {
  cost_usd: number; input_tokens: number; output_tokens: number; questions: number; avg_cost_per_question: number
  by_kind: Record<string, { cost_usd: number; calls: number }>
  by_model: Record<string, { cost_usd: number; input_tokens: number; output_tokens: number }>
  feedback?: { up: number; down: number; recent: { ts: string; question: string; rating: number; comment: string }[] }
}
export interface Health {
  status: string
  embedder: { name: string; dim: number; ready: boolean; device?: string; error?: string | null }
  reranker: { name?: string; ready?: boolean; enabled: boolean; error?: string | null; device?: string; kind?: string; detail?: string }
  reranker_usable: boolean
  llm: { provider: string; model: string | null; fast_model?: string; ready: boolean; detail: string; fallback_active?: boolean }
  usage?: UsageTotals
  store: Stats; active_job: Job | null; settings: Record<string, unknown>; supported_extensions: string[]
}
export interface SourceRow {
  id: string; url: string; title: string; category: string; relevance: string; language: string
  expected_type: string; origin: string; downloaded: boolean; indexed: boolean; size: number | null
}

// ---- admin token (only needed when the server sets ADMIN_TOKEN) ----
const TOKEN_KEY = 'uzhavan.adminToken'
export const adminToken = {
  get: () => { try { return localStorage.getItem(TOKEN_KEY) ?? '' } catch { return '' } },
  set: (t: string) => { try { localStorage.setItem(TOKEN_KEY, t) } catch { /* private mode */ } },
}
const authHeaders = (): Record<string, string> => (adminToken.get() ? { 'X-Admin-Token': adminToken.get() } : {})
export class AuthError extends Error {}

async function j<T>(res: Response): Promise<T> {
  if (res.status === 401) throw new AuthError('This server needs an admin token for that action.')
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      msg = typeof body.detail === 'string' ? body.detail : body.detail?.message ?? JSON.stringify(body.detail ?? body)
    } catch { /* not json */ }
    throw new Error(msg)
  }
  return res.json() as Promise<T>
}
const post = (url: string, body: unknown) =>
  fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders() }, body: JSON.stringify(body) })

export const api = {
  health: () => fetch('/api/health').then(j<Health>),
  search: (p: SearchParams) => post('/api/search', p).then(j<SearchResponse>),
  compare: (p: SearchParams) => post('/api/search/compare', p).then(j<Record<'bm25' | 'dense' | 'hybrid' | 'hybrid_rerank', SearchResponse> & { usage: UsageSummary }>),
  jobs: () => fetch('/api/jobs').then(j<Job[]>),
  job: (id: string) => fetch(`/api/jobs/${id}`).then(j<Job>),
  cancelJob: (id: string) => post(`/api/jobs/${id}/cancel`, {}).then(j<{ ok: boolean }>),
  documents: (q: Record<string, string | number>) =>
    fetch('/api/documents?' + new URLSearchParams(Object.entries(q).map(([k, v]) => [k, String(v)]))).then(j<{ total: number; items: Doc[] }>),
  document: (id: string, limit = 20, offset = 0, q = '') =>
    fetch(`/api/documents/${id}?limit=${limit}&offset=${offset}&q=${encodeURIComponent(q)}`).then(j<{ document: Doc; chunks: { total: number; items: ChunkRow[] } }>),
  deleteDocument: (id: string) => fetch(`/api/documents/${id}`, { method: 'DELETE', headers: authHeaders() }).then(j<{ ok: boolean }>),
  sources: () => fetch('/api/sources').then(j<{ total: number; items: SourceRow[] }>),
  ingestManifest: (body: { relevance: 'core' | 'all'; categories?: string[]; ids?: string[]; limit?: number; include_discovered?: boolean }) =>
    post('/api/ingest/manifest', body).then(j<{ job_id: string; selected: number }>),
  ingestUrls: (body: { urls: string[]; category: string; discover: boolean }) => post('/api/ingest/urls', body).then(j<{ job_id: string }>),
  ingestFolder: (body: { path: string; category: string }) => post('/api/ingest/folder', body).then(j<{ job_id: string; files: number }>),
  resetIndex: () => fetch('/api/index/reset?confirm=RESET', { method: 'POST', headers: authHeaders() }).then(j<{ ok: boolean }>),
  usage: () => fetch('/api/usage').then(j<UsageTotals>),
  ping: () => post('/api/llm/ping', {}).then(j<{ base_url: string; api_style: string; embeddings: string; fast_model: string; ms: number; cost_usd: number; answer?: string; fallback_active?: boolean }>),
  feedback: (b: { question: string; answer: string; rating: 1 | -1; comment?: string; sources: Hit[] }) =>
    post('/api/feedback', b).then(j<{ id: number }>),
  fileUrl: (docId: string) => `/api/documents/${docId}/file`,
}

/** Upload with progress (fetch has no upload progress, so XHR). */
export function uploadFiles(files: File[], opts: { category: string; chunk_size?: number; chunk_overlap?: number },
  onProgress: (pct: number) => void): Promise<{ job_id: string; accepted: number; rejected: { name: string; reason: string }[] }> {
  return new Promise((resolve, reject) => {
    const fd = new FormData()
    files.forEach(f => fd.append('files', f, (f as File & { webkitRelativePath?: string }).name))
    fd.append('category', opts.category)
    if (opts.chunk_size) fd.append('chunk_size', String(opts.chunk_size))
    if (opts.chunk_overlap) fd.append('chunk_overlap', String(opts.chunk_overlap))
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/ingest/upload')
    Object.entries(authHeaders()).forEach(([k, v]) => xhr.setRequestHeader(k, v))
    xhr.upload.onprogress = e => e.lengthComputable && onProgress(Math.round((e.loaded / e.total) * 100))
    xhr.onload = () => {
      if (xhr.status === 401) { reject(new AuthError('This server needs an admin token to upload.')); return }
      try {
        const body = JSON.parse(xhr.responseText)
        if (xhr.status >= 400) reject(new Error(typeof body.detail === 'string' ? body.detail : body.detail?.message ?? 'Upload failed'))
        else resolve(body)
      } catch { reject(new Error(`Upload failed (${xhr.status})`)) }
    }
    xhr.onerror = () => reject(new Error('Upload failed — is the backend running on port 8000?'))
    xhr.send(fd)
  })
}

export type AskEvent =
  | { type: 'plan'; plan: QueryPlan; variants: { label: string; text: string }[] }
  | { type: 'sources'; results: Hit[]; reranked: boolean; timings: Record<string, number>; mode: Mode; trace: TraceStep[]; candidates: { dense: number; bm25: number; fused: number } }
  | { type: 'followups'; questions: string[] }
  | { type: 'token'; text: string }
  | { type: 'notice'; text: string }
  | { type: 'error'; message: string }
  | { type: 'done'; timings: Record<string, number | null>; trace: TraceStep[]; usage: UsageSummary; model: string | null }

/** Streams NDJSON events from /api/ask. */
export async function ask(body: SearchParams & { history: { role: string; content: string }[] },
  onEvent: (e: AskEvent) => void, signal?: AbortSignal) {
  const res = await post('/api/ask', body)
  if (res.status === 429) throw new Error('Too many questions in a minute — please wait a moment and try again.')
  if (!res.ok || !res.body) throw new Error(`Ask failed (${res.status})`)
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  signal?.addEventListener('abort', () => reader.cancel())
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let nl
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim()
      buf = buf.slice(nl + 1)
      if (line) onEvent(JSON.parse(line) as AskEvent)
    }
  }
}

export const fmtBytes = (n: number) =>
  n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : n < 1073741824 ? `${(n / 1048576).toFixed(1)} MB` : `${(n / 1073741824).toFixed(2)} GB`

export const locLabel = (loc: Record<string, string | number>) => {
  if (loc.page) return loc.page_end && loc.page_end !== loc.page ? `pages ${loc.page}–${loc.page_end}` : `page ${loc.page}`
  if (loc.slide) return loc.slide_end ? `slides ${loc.slide}–${loc.slide_end}` : `slide ${loc.slide}`
  if (loc.sheet) return `sheet “${loc.sheet}”`
  if (loc.heading) return String(loc.heading)
  return ''
}

export const LANG_LABEL: Record<string, string> = { ta: 'Tamil', en: 'English', mixed: 'Tamil + English', unknown: '—' }

export const fmtUsd = (n: number | undefined | null) =>
  n == null ? '—' : n === 0 ? '$0' : n < 0.0001 ? '<$0.0001' : n < 0.01 ? `$${n.toFixed(4)}` : `$${n.toFixed(n < 1 ? 3 : 2)}`

export const STEP_LABEL: Record<string, string> = {
  plan: 'Understand the question', dense: 'Vector search (FAISS)', bm25: 'Keyword search (BM25)', fuse: 'Fuse rankings (RRF)',
  rerank: 'Rerank candidates', answer: 'Write the answer', followups: 'Suggest follow-ups',
}
