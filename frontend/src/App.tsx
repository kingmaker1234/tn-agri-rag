import { useCallback, useEffect, useRef, useState } from 'react'
import { MessageSquareText, ScanSearch, Upload, Library as LibraryIcon, Globe, Activity, Moon, Sun } from 'lucide-react'
import { api, type Health } from './api'
import { AdminTokenDialog, KolamMark, Toaster, toast, useTheme } from './ui'
import Ask from './pages/Ask'
import Explore from './pages/Explore'
import Ingest from './pages/Ingest'
import Library from './pages/Library'
import Sources from './pages/Sources'
import System from './pages/System'

export type Page = 'ask' | 'explore' | 'ingest' | 'library' | 'sources' | 'system'

const NAV: { id: Page; label: string; tamil: string; icon: typeof Upload }[] = [
  { id: 'ask', label: 'Ask', tamil: 'கேள்', icon: MessageSquareText },
  { id: 'explore', label: 'Explore retrieval', tamil: 'தேடல் ஆய்வு', icon: ScanSearch },
  { id: 'ingest', label: 'Add documents', tamil: 'ஆவணம் சேர்', icon: Upload },
  { id: 'library', label: 'Library', tamil: 'நூலகம்', icon: LibraryIcon },
  { id: 'sources', label: 'Web sources', tamil: 'இணைய ஆதாரங்கள்', icon: Globe },
  { id: 'system', label: 'System', tamil: 'அமைப்பு', icon: Activity },
]

const fromHash = (): Page => {
  const h = window.location.hash.replace('#/', '') as Page
  return NAV.some(n => n.id === h) ? h : 'ask'
}

export default function App() {
  const [page, setPage] = useState<Page>(fromHash)
  const [health, setHealth] = useState<Health | null>(null)
  const [offline, setOffline] = useState(false)
  const [theme, toggleTheme] = useTheme()
  const prevJob = useRef<string | null>(null)

  const go = useCallback((p: Page) => { window.location.hash = `/${p}`; setPage(p) }, [])
  useEffect(() => {
    const on = () => setPage(fromHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  const refresh = useCallback(() => {
    api.health().then(h => { setHealth(h); setOffline(false) }).catch(() => setOffline(true))
  }, [])
  useEffect(() => {
    refresh()
    const t = setInterval(refresh, health?.active_job ? 3000 : 12000)
    return () => clearInterval(t)
  }, [refresh, health?.active_job])

  // tell the user when a background indexing run finishes, wherever they are in the app
  useEffect(() => {
    const id = health?.active_job?.id ?? null
    if (prevJob.current && !id) toast('Indexing finished. The new documents are searchable now.')
    prevJob.current = id
  }, [health?.active_job?.id])

  // "/" focuses the question box from anywhere
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
        e.preventDefault()
        if (page !== 'ask') go('ask')
        setTimeout(() => document.getElementById('ask-input')?.focus(), 50)
      }
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [page, go])

  const s = health?.store
  return (
    <div className="flex h-full">
      <nav className="flex w-16 shrink-0 flex-col bg-rail text-rail-ink md:w-60" aria-label="Main">
        <div className="flex items-center gap-2.5 px-4 pb-6 pt-5">
          <KolamMark size={30} className="text-rail-ink" />
          <div className="hidden leading-tight md:block">
            <div className="text-[1.05rem] font-extrabold tracking-tight">Uzhavan Assist</div>
            <div className="text-xs text-rail-ink/60">உழவர் திட்ட உதவியாளர்</div>
          </div>
        </div>
        <ul className="flex-1 space-y-0.5 px-2">
          {NAV.map(({ id, label, tamil, icon: Icon }) => (
            <li key={id}>
              <button onClick={() => go(id)} aria-current={page === id ? 'page' : undefined} title={label}
                className={`group flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left transition-colors ${page === id ? 'bg-paper text-field' : 'text-rail-ink/80 hover:bg-rail-2 hover:text-rail-ink'}`}>
                <Icon size={18} className="shrink-0" />
                <span className="hidden min-w-0 md:block">
                  <span className="block text-[0.95rem] font-semibold leading-tight">{label}</span>
                  <span className={`block text-xs leading-tight ${page === id ? 'text-field/70' : 'text-rail-ink/45'}`}>{tamil}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
        <div className="px-2 pb-2">
          <button onClick={toggleTheme} className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-rail-ink/70 hover:bg-rail-2 hover:text-rail-ink"
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} title="Theme">
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            <span className="hidden text-sm md:block">{theme === 'dark' ? 'Light theme' : 'Dark theme'}</span>
          </button>
        </div>
        <div className="hidden border-t border-rail-ink/10 px-4 py-4 text-sm md:block">
          {offline ? (
            <p className="text-turmeric">Backend not reachable on port 8000. Start it with <code className="text-rail-ink">uvicorn app.main:app</code>.</p>
          ) : s ? (
            <>
              <div className="tnum"><span className="text-xl font-extrabold">{s.documents.toLocaleString()}</span> documents</div>
              <div className="tnum text-rail-ink/60">{s.vectors.toLocaleString()} vectors in FAISS</div>
              {health?.llm && (
                <div className="mt-1.5 flex items-center gap-1.5 text-rail-ink/60">
                  <span className={`h-2 w-2 rounded-full ${health.llm.ready ? 'bg-[#5fb68a]' : 'bg-turmeric'}`} />
                  {health.llm.ready ? health.llm.model : 'LLM not configured'}
                </div>
              )}
              {health?.active_job && (
                <button onClick={() => go('ingest')} className="mt-2 flex items-center gap-2 text-turmeric hover:underline">
                  <KolamMark size={16} animate className="text-rail-ink" /> Indexing in progress
                </button>
              )}
            </>
          ) : <div className="text-rail-ink/50">Connecting…</div>}
        </div>
      </nav>

      <main className="min-w-0 flex-1 overflow-y-auto">
        {page === 'ask' && <Ask health={health} onGo={go} />}
        {page === 'explore' && <Explore health={health} />}
        {page === 'ingest' && <Ingest health={health} onChanged={refresh} onGo={go} />}
        {page === 'library' && <Library onChanged={refresh} />}
        {page === 'sources' && <Sources onGo={go} />}
        {page === 'system' && <System health={health} onRefresh={refresh} />}
      </main>
      <Toaster />
      <AdminTokenDialog />
    </div>
  )
}
