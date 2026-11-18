import { useEffect, useState, type ReactNode } from 'react'
import { FileText, FileSpreadsheet, Presentation, FileCode, FileImage, FileArchive, Globe, X, TriangleAlert } from 'lucide-react'
import { AuthError, adminToken, type Scores } from './api'

/* ------------------------------------------------------------------ kolam */
// 3x3 pulli (dot) kolam: one continuous line of eight petals, each looping around a dot.
const KOLAM_PATH = (() => {
  const petals: string[] = []
  for (let i = 0; i < 8; i++) {
    const diag = i % 2 === 1
    const r = diag ? 21.5 : 16.5, w = diag ? 6.5 : 7
    const a = (i * Math.PI) / 4
    const rot = (x: number, y: number) => {
      const X = 20 + x * Math.cos(a) - y * Math.sin(a), Y = 20 + x * Math.sin(a) + y * Math.cos(a)
      return `${X.toFixed(2)} ${Y.toFixed(2)}`
    }
    petals.push(`C${rot(-w, -r * 0.5)} ${rot(-w * 0.95, -r)} ${rot(0, -r)} C${rot(w * 0.95, -r)} ${rot(w, -r * 0.5)} ${rot(0, 0)}`)
  }
  return 'M20 20 ' + petals.join(' ')
})()

export function KolamMark({ size = 28, animate = false, className = '' }: { size?: number; animate?: boolean; className?: string }) {
  const dots = [8, 20, 32].flatMap(x => [8, 20, 32].map(y => [x, y]))
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" className={className} aria-hidden="true">
      {dots.map(([x, y]) => <circle key={`${x}-${y}`} cx={x} cy={y} r="1.9" fill="currentColor" opacity=".55" />)}
      <path d={KOLAM_PATH} fill="none" stroke="var(--color-turmeric)" strokeWidth="2" strokeLinecap="round"
        className={animate ? 'kolam-path' : ''} />
    </svg>
  )
}

export function Thinking({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 text-muted" role="status" aria-live="polite">
      <KolamMark size={30} animate className="text-field" />
      <span>{label}</span>
    </div>
  )
}

/* ------------------------------------------------------------------ controls */
export function Segmented<T extends string>({ value, options, onChange, label }:
  { value: T; options: { value: T; label: string; hint?: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-line bg-surface p-0.5">
      {options.map(o => (
        <button key={o.value} role="radio" aria-checked={value === o.value} title={o.hint}
          onClick={() => onChange(o.value)}
          className={`px-3 py-1 text-sm rounded-md transition-colors ${value === o.value ? 'bg-field text-paper font-semibold' : 'text-muted hover:text-ink'}`}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <label className="inline-flex items-center gap-2 text-sm cursor-pointer select-none" title={hint}>
      <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}
        className={`relative h-5 w-9 rounded-full transition-colors ${checked ? 'bg-field' : 'bg-line'}`}>
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${checked ? 'left-[18px]' : 'left-0.5'}`} />
      </button>
      <span>{label}</span>
    </label>
  )
}

export const btn = {
  primary: 'inline-flex items-center justify-center gap-2 rounded-lg bg-field px-4 py-2 font-semibold text-paper hover:bg-field-2 disabled:opacity-50 disabled:cursor-not-allowed',
  secondary: 'inline-flex items-center justify-center gap-2 rounded-lg border border-line bg-surface px-3.5 py-2 font-medium text-ink hover:border-field-3 disabled:opacity-50',
  ghost: 'inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-sm text-muted hover:text-ink hover:bg-line-soft',
  danger: 'inline-flex items-center gap-2 rounded-lg border border-danger/40 px-3 py-1.5 text-sm font-medium text-danger hover:bg-danger-soft',
}
export const input = 'w-full rounded-lg border border-line bg-surface px-3 py-2 text-ink placeholder:text-faint focus:border-field-3 focus:outline-none'

export function TypeIcon({ type, size = 16 }: { type: string; size?: number }) {
  const t = type.toLowerCase()
  const Icon = ['xlsx', 'xls', 'csv', 'tsv', 'ods'].includes(t) ? FileSpreadsheet
    : ['pptx', 'ppt', 'odp'].includes(t) ? Presentation
    : ['html', 'htm', 'asp', 'aspx', 'php'].includes(t) ? Globe
    : ['json', 'xml'].includes(t) ? FileCode
    : ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'webp', 'bmp'].includes(t) ? FileImage
    : t === 'zip' ? FileArchive : FileText
  return <Icon size={size} className="shrink-0 text-field-3" aria-hidden="true" />
}

export function Tag({ children, tone = 'plain' }: { children: ReactNode; tone?: 'plain' | 'turmeric' | 'indigo' | 'green' | 'danger' }) {
  const tones = {
    plain: 'bg-line-soft text-muted', turmeric: 'bg-turmeric-soft text-turmeric-deep', indigo: 'bg-indigo-soft text-indigo',
    green: 'bg-field/10 text-field', danger: 'bg-danger-soft text-danger',
  }
  return <span className={`inline-flex items-center gap-1 rounded px-1.5 py-px text-xs font-medium ${tones[tone]}`}>{children}</span>
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-danger/30 bg-danger-soft px-3 py-2 text-sm text-danger" role="alert">
      <TriangleAlert size={16} className="mt-0.5 shrink-0" /> <div>{children}</div>
    </div>
  )
}

/* ------------------------------------------------------------------ drawer */
export function Drawer({ open, onClose, title, children, wide = false }:
  { open: boolean; onClose: () => void; title: ReactNode; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    if (!open) return
    const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [open, onClose])
  if (!open) return null
  return (
    <div className="fixed inset-0 z-40 flex justify-end" role="dialog" aria-modal="true">
      <button className="absolute inset-0 bg-ink/30" aria-label="Close panel" onClick={onClose} />
      <aside className={`relative h-full w-full ${wide ? 'max-w-3xl' : 'max-w-xl'} overflow-y-auto bg-surface shadow-2xl`}>
        <header className="sticky top-0 z-10 flex items-start justify-between gap-4 border-b border-line bg-surface px-6 py-4">
          <div className="min-w-0">{title}</div>
          <button onClick={onClose} className={btn.ghost} aria-label="Close"><X size={18} /></button>
        </header>
        <div className="px-6 py-5">{children}</div>
      </aside>
    </div>
  )
}

/* ------------------------------------------------------------------ text highlight */
export function Highlighted({ text, terms, max }: { text: string; terms: string[]; max?: number }) {
  const t = max && text.length > max ? text.slice(0, max).trimEnd() + '…' : text
  const words = terms.filter(w => w.length > 1).map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  if (!words.length) return <>{t}</>
  // extend each match through Tamil vowel signs / virama so a highlight never splits a letter
  const re = new RegExp(`((?:${words.join('|')})[\u0B82\u0BBE-\u0BCD\u0BD7]*)`, 'gi')
  const parts = t.split(re)
  return <>{parts.map((p, i) => (i % 2 ? <mark key={i} className="hit">{p}</mark> : <span key={i}>{p}</span>))}</>
}

/* ------------------------------------------------------------------ score ladder */
// Shows where a passage ranked in each retriever and after fusion / reranking.
export function ScoreLadder({ s, reranked }: { s: Scores; reranked: boolean }) {
  const cell = (label: string, rank: number | null, score: number | null, color: string, fmt: (n: number) => string) => (
    <div className="min-w-0">
      <div className="text-xs text-faint">{label}</div>
      <div className="tnum flex items-baseline gap-1.5">
        <span className="font-bold" style={{ color: rank ? color : 'var(--color-faint)' }}>{rank ? `#${rank}` : '—'}</span>
        {score != null && <span className="text-xs text-muted">{fmt(score)}</span>}
      </div>
    </div>
  )
  return (
    <div className="grid grid-cols-4 gap-3 rounded-lg bg-paper px-3 py-2">
      {cell('BM25', s.bm25_rank, s.bm25, 'var(--color-indigo)', n => n.toFixed(2))}
      {cell('Dense', s.dense_rank, s.dense, 'var(--color-field-3)', n => n.toFixed(3))}
      {cell('Fused (RRF)', s.fused_rank, s.fused, 'var(--color-ink)', n => n.toFixed(4))}
      {reranked
        ? (
          <div>
            <div className="text-xs text-faint">Reranker</div>
            <div className="flex items-center gap-2" title="relevance judged by the reranker">
              <div className="h-1.5 w-full max-w-16 rounded-full bg-line" aria-hidden="true">
                <div className="h-full rounded-full bg-turmeric" style={{ width: `${Math.round((s.rerank ?? 0) * 100)}%` }} />
              </div>
              <span className="tnum text-xs font-semibold">{((s.rerank ?? 0) * 100).toFixed(0)}%</span>
            </div>
          </div>)
        : <div><div className="text-xs text-faint">Reranker</div><div className="text-sm text-faint">off</div></div>}
    </div>
  )
}

/* ------------------------------------------------------------------ toasts */
type ToastTone = 'ok' | 'error' | 'info'
interface ToastMsg { id: number; text: string; tone: ToastTone }
export function toast(text: string, tone: ToastTone = 'ok') {
  window.dispatchEvent(new CustomEvent<ToastMsg>('uz-toast', { detail: { id: Date.now() + Math.random(), text, tone } }))
}
export function Toaster() {
  const [items, setItems] = useState<ToastMsg[]>([])
  useEffect(() => {
    const on = (e: Event) => {
      const t = (e as CustomEvent<ToastMsg>).detail
      setItems(xs => [...xs.slice(-3), t])
      setTimeout(() => setItems(xs => xs.filter(x => x.id !== t.id)), t.tone === 'error' ? 6000 : 3200)
    }
    window.addEventListener('uz-toast', on)
    return () => window.removeEventListener('uz-toast', on)
  }, [])
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex flex-col gap-2" role="status" aria-live="polite">
      {items.map(t => (
        <div key={t.id} className={`toast-in pointer-events-auto max-w-sm rounded-lg px-4 py-2.5 text-sm font-medium shadow-lg ${
          t.tone === 'error' ? 'bg-danger text-white' : t.tone === 'info' ? 'bg-surface text-ink border border-line' : 'bg-rail text-rail-ink'}`}>
          {t.text}
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ admin token */
let pendingToken: ((ok: boolean) => void) | null = null
function requestAdminToken(): Promise<boolean> {
  return new Promise(resolve => { pendingToken = resolve; window.dispatchEvent(new Event('uz-admin')) })
}
/** Runs an admin action; if the server asks for a token, prompts once and retries. */
export async function withAdmin<T>(fn: () => Promise<T>): Promise<T> {
  try { return await fn() } catch (e) {
    if (e instanceof AuthError && await requestAdminToken()) return fn()
    throw e
  }
}
export function AdminTokenDialog() {
  const [open, setOpen] = useState(false)
  const [val, setVal] = useState('')
  useEffect(() => {
    const on = () => { setVal(adminToken.get()); setOpen(true) }
    window.addEventListener('uz-admin', on)
    return () => window.removeEventListener('uz-admin', on)
  }, [])
  const close = (ok: boolean) => { if (ok) adminToken.set(val.trim()); setOpen(false); pendingToken?.(ok); pendingToken = null }
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4" role="dialog" aria-modal="true" aria-labelledby="admin-title">
      <form onSubmit={e => { e.preventDefault(); close(true) }} className="w-full max-w-sm rounded-2xl bg-surface p-6 shadow-2xl">
        <h2 id="admin-title" className="text-lg font-bold">Admin token needed</h2>
        <p className="mt-1 text-sm text-muted">This server protects uploads and deletions. Enter the <code>ADMIN_TOKEN</code> from its <code>.env</code>; it is kept in this browser only.</p>
        <input autoFocus type="password" className={input + ' mt-4'} value={val} onChange={e => setVal(e.target.value)} aria-label="Admin token" />
        <div className="mt-4 flex justify-end gap-2">
          <button type="button" className={btn.secondary} onClick={() => close(false)}>Cancel</button>
          <button type="submit" className={btn.primary} disabled={!val.trim()}>Continue</button>
        </div>
      </form>
    </div>
  )
}

/* ------------------------------------------------------------------ theme */
export function useTheme(): ['light' | 'dark', () => void] {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    try {
      const saved = localStorage.getItem('uzhavan.theme')
      if (saved === 'light' || saved === 'dark') return saved
    } catch { /* ignore */ }
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('uzhavan.theme', theme) } catch { /* ignore */ }
  }, [theme])
  return [theme, () => setTheme(t => (t === 'dark' ? 'light' : 'dark'))]
}

export function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />
}
