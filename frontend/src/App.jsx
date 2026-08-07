import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import FormShowcase from './FormShowcase.jsx'

const POLL_MS = 1200
const PILL = { high: 'High', low: 'Review' }
const MAX_PAGES = 9

const isEmpty = (v) => v === null || v === undefined || v === '' || v === 'null'
/* "1, 2, 5" but "2-5" when they run on - a selection reads better collapsed. */
function listPages(ps) {
  if (!ps || !ps.length) return ''
  const runs = []
  for (const p of [...ps].sort((a, b) => a - b)) {
    const last = runs[runs.length - 1]
    if (last && p === last[1] + 1) last[1] = p
    else runs.push([p, p])
  }
  return runs.map(([a, b]) => (a === b ? `${a}` : `${a}–${b}`)).join(', ')
}
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

function download(filename, text, type) {
  const blob = new Blob([text], { type })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

/* ============================ results ============================ */

function ConfPill({ conf }) {
  if (!PILL[conf]) return null
  return <span className={`pill ${conf}`}>{PILL[conf]}</span>
}

/* One field row. Clicking it reveals the exact patch of the scan the value came
   from — the fastest way to settle whether a flagged value is actually wrong. */
function FieldRow({ jobId, rec: incoming, onEdited }) {
  const [open, setOpen] = useState(false)
  const [rec, setRec] = useState(incoming)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [editErr, setEditErr] = useState(null)
  // a fresh extraction replaces the record underneath us
  useEffect(() => { setRec(incoming) }, [incoming])

  const showable = rec.kind !== 'checkbox'
  const isBox = rec.kind === 'checkbox'

  async function save(value) {
    setBusy(true); setEditErr(null)
    try {
      const r = await fetch(`/api/jobs/${jobId}/fields/${rec.field}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value }),
      })
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`)
      const { field } = await r.json()
      setRec(field); onEdited?.(field)
    } catch (e) { setEditErr(e.message) } finally { setBusy(false) }
  }

  async function revert() {
    setBusy(true); setEditErr(null)
    try {
      const r = await fetch(`/api/jobs/${jobId}/fields/${rec.field}/edit`, { method: 'DELETE' })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const { field } = await r.json()
      setRec(field); setDraft(field.value == null ? '' : String(field.value)); onEdited?.(field)
    } catch (e) { setEditErr(e.message) } finally { setBusy(false) }
  }

  function toggleOpen() {
    if (!showable) return
    if (!open) setDraft(rec.value == null ? '' : String(rec.value))
    setOpen((o) => !o)
  }
  // a wrapped second line: one answer written across two boxes on the page, so it is
  // shown tucked under its first line rather than as another field of the same name
  const cont = Boolean(rec.continuation_of)
  const val = rec.kind === 'checkbox'
    ? (rec.value ? 'Ticked' : '—')
    : rec.value

  return (
    <div className={`frow ${cont ? 'cont' : ''} ${rec.needs_review ? 'flagged' : ''} ${rec.edited ? 'edited' : ''} ${open ? 'open' : ''}`}>
      <button
        className="frow-main"
        onClick={toggleOpen}
        aria-expanded={open}
        disabled={!showable}
      >
        <span className="frow-label">
          {cont && <span className="frow-cont" aria-hidden="true">↳</span>}
          {cont ? 'continued' : rec.label}
          {rec.group && rec.kind === 'checkbox' && <span className="frow-group">{rec.group}</span>}
        </span>
        <span className="frow-value">
          {isEmpty(val)
            ? <span className="value empty">Not filled</span>
            : <span className="value">{String(val)}</span>}
          {rec.edited && <span className="frow-edited" title="corrected by hand">edited</span>}
          <ConfPill conf={rec.confidence} />
          {showable && <span className="frow-chev" aria-hidden="true" />}
        </span>
      </button>
      {rec.note && <div className="frow-note">{rec.note}</div>}
      {isBox && (
        <div className="frow-boxedit">
          <button className="btn btn-tiny" disabled={busy} onClick={() => save(!rec.value)}>
            {rec.value ? 'Mark as not ticked' : 'Mark as ticked'}
          </button>
          {rec.edited && (
            <button className="btn btn-tiny ghost" disabled={busy} onClick={revert}>
              Undo
            </button>
          )}
        </div>
      )}
      {open && showable && (
        <div className="frow-crop">
          <img
            src={`/api/jobs/${jobId}/fields/${rec.field}/crop.png`}
            alt={`Scan of ${rec.label}`}
            loading="lazy"
          />
          {/* Correcting a value is the point of the review list, so the box to type in sits
              directly beneath the handwriting it is being checked against. */}
          <div className="frow-edit">
            <label className="frow-edit-lab" htmlFor={`ed-${rec.field}`}>Correct this value</label>
            <div className="frow-edit-row">
              <input
                id={`ed-${rec.field}`}
                className="frow-input"
                value={draft}
                placeholder="Type the value as written"
                disabled={busy}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') save(draft)
                  if (e.key === 'Escape') setDraft(rec.value == null ? '' : String(rec.value))
                }}
              />
              <button
                className="btn btn-tiny"
                disabled={busy || draft === (rec.value == null ? '' : String(rec.value))}
                onClick={() => save(draft)}
              >
                {busy ? 'Saving…' : 'Save'}
              </button>
              {rec.edited && (
                <button className="btn btn-tiny ghost" disabled={busy} onClick={revert}>
                  Undo
                </button>
              )}
            </div>
            {rec.edited && (
              <div className="frow-was">
                model read <span className="frow-was-v">{isEmpty(rec.edited_from)
                  ? '(nothing)' : String(rec.edited_from)}</span>
              </div>
            )}
            {editErr && <div className="frow-editerr">{editErr}</div>}
          </div>
          <div className="frow-meta">
            <span>{rec.type}</span>
            <span>page {rec.page}</span>
            <span>{rec.source}</span>
          </div>
        </div>
      )}
    </div>
  )
}

/* Checkbox questions read better as one answer per question than as N booleans. */
function questionRows(records) {
  const groups = new Map()
  for (const r of records) {
    if (r.kind !== 'checkbox') continue
    const key = r.group || r.label
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key).push(r)
  }
  const out = []
  for (const [question, members] of groups) {
    const picked = members.filter((m) => m.value)
    out.push({
      question,
      picked,
      answer: picked.map((p) => p.label).join(', '),
      needs_review: picked.some((p) => p.needs_review) || (picked.length > 1 &&
        members.some((m) => m.select_mode === 'one')),
      page: members[0].page,
    })
  }
  return out
}

function Section({ jobId, name, fields, questions }) {
  const [open, setOpen] = useState(true)
  const flagged = fields.filter((f) => f.needs_review).length +
    questions.filter((q) => q.needs_review).length
  const filled = fields.filter((f) => !isEmpty(f.value)).length + questions.filter((q) => q.picked.length).length

  return (
    <div className="sect">
      <button className="sect-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className={`sect-caret ${open ? 'on' : ''}`} aria-hidden="true" />
        <span className="sect-name">{name}</span>
        <span className="sect-count">{filled} filled</span>
        {flagged > 0 && <span className="sect-flag">{flagged} to review</span>}
      </button>
      {open && (
        <div className="sect-body">
          {questions.map((q) => (
            <div key={q.question} className={`frow choice ${q.needs_review ? 'flagged' : ''}`}>
              <div className="frow-main static">
                <span className="frow-label">{q.question}</span>
                <span className="frow-value">
                  {q.picked.length
                    ? <span className="value">{q.answer}</span>
                    : <span className="value empty">Nothing ticked</span>}
                  {q.picked.length > 1 && <span className="pill low">Multiple</span>}
                </span>
              </div>
            </div>
          ))}
          {fields.map((f) => <FieldRow key={f.field} jobId={jobId} rec={f} />)}
        </div>
      )}
    </div>
  )
}

function Results({ job, onReset }) {
  const data = job.result
  const [onlyFlagged, setOnlyFlagged] = useState(false)
  const jobId = job.job_id

  const { sections, stats } = useMemo(() => {
    const recs = data.fields || []
    const readable = recs.filter((r) => r.kind !== 'checkbox')
    const qs = questionRows(recs)
    const order = []
    const map = new Map()
    for (const r of recs) {
      const s = r.section || 'Other details'
      if (!map.has(s)) { map.set(s, { fields: [], questions: [] }); order.push(s) }
    }
    for (const r of readable) map.get(r.section || 'Other details').fields.push(r)
    for (const q of qs) {
      const home = recs.find((r) => r.kind === 'checkbox' && (r.group || r.label) === q.question)
      map.get((home?.section) || 'Other details').questions.push(q)
    }
    return {
      sections: order.map((s) => ({ name: s, ...map.get(s) })),
      stats: {
        total: recs.length,
        filled: readable.filter((r) => !isEmpty(r.value)).length,
        ticked: recs.filter((r) => r.kind === 'checkbox' && r.value).length,
        review: recs.filter((r) => r.needs_review).length,
        invalid: readable.filter((r) => !r.valid).length,
      },
    }
  }, [data])

  const shown = useMemo(() => sections
    .map((s) => onlyFlagged
      ? { ...s, fields: s.fields.filter((f) => f.needs_review), questions: s.questions.filter((q) => q.needs_review) }
      : s)
    .filter((s) => s.fields.length || s.questions.length), [sections, onlyFlagged])

  const badPages = (data.alignment || []).filter((p) => !p.ok)
  const base = (data.source_file || 'extraction').replace(/\.[^.]+$/, '')

  return (
    <div className="results" >
      <div className={`banner ${stats.review ? 'review' : 'ok'}`}>
        <span className="dot" />
        {stats.review ? `${stats.review} field${stats.review === 1 ? '' : 's'} need review` : 'All fields passed their checks'}
        <span className="note">
          &nbsp;&middot;&nbsp;{stats.filled} values read, {stats.ticked} boxes ticked,
          &nbsp;{Array.isArray(data.pages_scanned)
            ? `page${data.pages_scanned.length === 1 ? '' : 's'} ${listPages(data.pages_scanned)} scanned`
            : `${data.pages_scanned} pages scanned`}
        </span>
      </div>

      {badPages.length > 0 && (
        <div className="banner error">
          <span className="dot" style={{ background: 'var(--red)' }} />
          Alignment problems on page{badPages.length > 1 ? 's' : ''}{' '}
          {badPages.map((p) => p.template_page).join(', ')}
          <span className="note">&nbsp;&middot;&nbsp;values from {badPages.length > 1 ? 'these pages' : 'this page'} may be unreliable</span>
        </div>
      )}

      {(data.cross_check || []).length > 0 && (
        <div className="card xcheck">
          <div className="card-title">Cross-field checks</div>
          <ul>{data.cross_check.map((m) => <li key={m}>{m}</li>)}</ul>
        </div>
      )}

      <div className="res-bar">
        <div className="stat-row">
          <div className="stat"><b>{stats.total}</b><span>fields</span></div>
          <div className="stat"><b>{stats.filled}</b><span>values</span></div>
          <div className="stat"><b>{stats.ticked}</b><span>ticked</span></div>
          <div className={`stat ${stats.review ? 'warn' : ''}`}><b>{stats.review}</b><span>to review</span></div>
        </div>
        <label className="toggle">
          <input type="checkbox" checked={onlyFlagged} onChange={(e) => setOnlyFlagged(e.target.checked)} />
          <span>Show only fields to review</span>
        </label>
      </div>

      <div className="sections">
        {shown.length === 0
          ? <div className="empty-note">Nothing flagged — every field passed its checks.</div>
          : shown.map((s) => <Section key={s.name} jobId={jobId} {...s} />)}
      </div>

      <div className="result-actions">
        <button className="btn btn-ghost" onClick={() => window.open(`/api/jobs/${jobId}/download.json`, '_blank')}>Download JSON</button>
        <button className="btn btn-ghost" onClick={() => window.open(`/api/jobs/${jobId}/download.csv`, '_blank')}>Download CSV</button>
        <button className="btn btn-ghost" onClick={() => download(`${base}-answers.json`, JSON.stringify(data.answers, null, 2), 'application/json')}>Download summary</button>
        <button className="btn btn-primary" onClick={onReset}>Extract another form</button>
      </div>
    </div>
  )
}

/* ============================ processing ============================ */

function CheckIcon() {
  return (
    <svg className="ck" viewBox="0 0 24 24" fill="none" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 13l4 4L19 7" />
    </svg>
  )
}

function Pipeline({ job }) {
  const stages = job.stages || []
  const idx = job.stage_index ?? 0
  return (
    <div className="pipeline-wrap">
      <div className="pl-head"><span className="live" />Extracting {job.filename}</div>

      <div className="pl-pages">
        {(job.pages || []).map((p, i) => (
          <img
            key={p}
            className={`pl-page ${idx > 0 ? 'ready' : ''}`}
            src={`/api/jobs/${job.job_id}/pages/${p}/preview.png?width=320`}
            alt={`Page ${p}`}
            onError={(e) => { e.currentTarget.style.visibility = 'hidden' }}
          />
        ))}
      </div>

      <div className="pipeline">
        {stages.map((s, i) => {
          const state = i < idx ? 'done' : i === idx ? 'active' : 'pending'
          return (
            <div key={s.key} className={`pl-item ${state}`} style={{ animationDelay: `${0.06 * i}s` }}>
              <div className="pl-rail">
                <div className="pl-node">
                  {state === 'done' ? <CheckIcon />
                    : state === 'active' ? <span className="pl-spin" />
                      : <span className="pl-num">{i + 1}</span>}
                </div>
                {i < stages.length - 1 && (
                  <div className="pl-line">
                    <span className="pl-fill" />
                    <span className="pl-flow" />
                  </div>
                )}
              </div>
              <div className="pl-body">
                <div className="pl-title">{s.label}</div>
                <div className="pl-status">
                  {state === 'done' ? 'Completed' : state === 'active' ? 'In progress' : 'Pending'}
                </div>
                <div className="pl-desc">{state === 'active' ? job.message : s.description}</div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="proc-track">
        <div className="proc-bar" style={{ width: `${Math.max(2, job.progress * 100)}%` }} />
      </div>
      <div className="proc-time">
        {Math.round(job.progress * 100)}% &middot; {fmtTime(job.elapsed || 0)} elapsed &middot; running locally
      </div>
    </div>
  )
}

/* Types out a heading segment-by-segment with a blinking caret. */
function Typewriter({ segments, speed = 46, startDelay = 350 }) {
  const total = segments.reduce((a, s) => a + (s.text ? s.text.length : 0), 0)
  const [n, setN] = useState(0)
  useEffect(() => {
    let i = 0
    let timer
    const start = setTimeout(function tick() {
      i += 1
      setN(i)
      if (i < total) timer = setTimeout(tick, speed)
    }, startDelay)
    return () => { clearTimeout(start); clearTimeout(timer) }
  }, [total, speed, startDelay])

  let used = 0
  const done = n >= total
  return (
    <>
      {segments.map((s, idx) => {
        if (s.br) return n >= used ? <br key={idx} /> : null
        const take = Math.max(0, Math.min(s.text.length, n - used))
        used += s.text.length
        return <span key={idx} className={s.className || ''}>{s.text.slice(0, take)}</span>
      })}
      <span className={`caret ${done ? 'done' : ''}`}>&nbsp;</span>
    </>
  )
}

/* ============================ app ============================ */

export default function App() {
  const [file, setFile] = useState(null)
  const [pageCount, setPageCount] = useState(null)
  const [pages, setPages] = useState([1])
  const [job, setJob] = useState(null)
  const [status, setStatus] = useState('idle')   // idle | processing | done | error
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)
  const [drag, setDrag] = useState(false)
  const inputRef = useRef(null)
  const workRef = useRef(null)
  const resultsRef = useRef(null)

  useEffect(() => {
    fetch('/api/health').then((r) => r.json()).then(setHealth).catch(() => setHealth(null))
  }, [])

  const pickFile = useCallback(async (f) => {
    if (!f) return
    setFile(f)
    setJob(null)
    setStatus('idle')
    setError(null)
    setPages([1])
    setPageCount(null)
    // Read the page count in the browser so the page selector knows its range
    // before anything is uploaded.
    if (f.type === 'application/pdf' || /\.pdf$/i.test(f.name)) {
      try {
        const buf = await f.arrayBuffer()
        const text = new TextDecoder('latin1').decode(buf)
        const m = text.match(/\/Type\s*\/Page[^s]/g)
        setPageCount(m ? m.length : null)
      } catch { setPageCount(null) }
    } else {
      setPageCount(1)
    }
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDrag(false)
    const f = e.dataTransfer.files?.[0]
    if (f) pickFile(f)
  }, [pickFile])

  // Poll the job while it runs.
  useEffect(() => {
    if (status !== 'processing' || !job?.job_id) return
    let alive = true
    const tick = async () => {
      try {
        const res = await fetch(`/api/jobs/${job.job_id}`)
        if (!res.ok) throw new Error(`Server error (${res.status})`)
        const j = await res.json()
        if (!alive) return
        setJob(j)
        if (j.status === 'done') setStatus('done')
        else if (j.status === 'error') { setError(j.error || 'Extraction failed'); setStatus('error') }
      } catch (err) {
        if (alive) { setError(err.message); setStatus('error') }
      }
    }
    const t = setInterval(tick, POLL_MS)
    tick()
    return () => { alive = false; clearInterval(t) }
  }, [status, job?.job_id])

  useEffect(() => {
    if (status === 'processing') {
      requestAnimationFrame(() => workRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' }))
    } else if (status === 'done') {
      const t = setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120)
      return () => clearTimeout(t)
    }
  }, [status])

  const extract = useCallback(async () => {
    if (!file) return
    setStatus('processing')
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      // a selection, not a count: '2' means page 2 alone
      form.append('pages', pages.join(','))
      const res = await fetch('/api/extract', { method: 'POST', body: form })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}))
        throw new Error(detail.detail || `Server error (${res.status})`)
      }
      setJob(await res.json())
    } catch (err) {
      setError(err.message || 'Extraction failed')
      setStatus('error')
    }
  }, [file, pages])

  const reset = useCallback(() => {
    setFile(null)
    setJob(null)
    setStatus('idle')
    setError(null)
    setPageCount(null)
    setPages([1])
  }, [])

  const maxPages = Math.min(pageCount || MAX_PAGES, MAX_PAGES)

  // Selecting pages, not a page count. Kept sorted so "pages 2, 5" always reads in order.
  const togglePage = (n) => setPages((prev) => prev.includes(n)
    ? prev.filter((p) => p !== n)
    : [...prev, n].sort((a, b) => a - b))
  const modelDown = health && !health.model_ready
  // a missing OCR reader does not stop a run, it quietly costs accuracy on the digits -
  // which is exactly the kind of thing that should not be silent
  const digitReader = health?.readers?.find((r) => r.role === 'digits')
  const degraded = Boolean(health?.model_ready && health?.degraded)

  return (
    <div className="app">
      <nav className="nav">
        <div className="nav-inner">
          <div className="nav-logo">
            <span className="logo-badge">HBL</span>
            <span className="logo-ring" />
          </div>
          <div className="nav-lockup">
            <span className="nav-title">Form Intelligence</span>
            <span className="nav-desc">Account Opening &middot; Customer Information</span>
          </div>
          <div className="nav-right">
            <span className="nav-chip"><span className="live" />On-premise</span>
            <span className="nav-chip"><span className="lock-dot" />Secure</span>
          </div>
        </div>
        <span className="nav-accent" />
      </nav>

      <div className="container">
        <section className={`hero-grid ${file ? 'compact' : ''}`}>
          <header className="hero-copy">
            <div className="eyebrow">Intelligent Document Processing</div>
            <h1 className="hero-title">
              <Typewriter
                segments={[
                  { text: 'Account opening forms,' },
                  { br: true },
                  { text: 'read on-device', className: 'accent' },
                ]}
              />
            </h1>
            <p>
              Upload a scanned Customer Information Form and extract every field &mdash; names,
              dates, CNIC, IBAN, addresses and every ticked box. Processed entirely on this machine.
            </p>
            <div className="steps">
              <div className="step"><span className="step-n">1</span>Upload form</div>
              <span className="step-sep">&middot;</span>
              <div className="step"><span className="step-n">2</span>Choose pages</div>
              <span className="step-sep">&middot;</span>
              <div className="step"><span className="step-n">3</span>Review &amp; export</div>
            </div>
            {!file && (
              <div className="hero-actions">
                <button className="btn btn-primary" onClick={() => inputRef.current?.click()}>
                  Upload a form
                </button>
                <span className="hero-note"><span className="live" />No data leaves this machine</span>
              </div>
            )}
            {health?.schema_ready && (
              <div className="hero-facts">
                <span><b>{health.schema_fields}</b> fields mapped</span>
                {(health.readers || [{ model: health.model, ready: health.model_ready, role: 'words' }])
                  .map((r) => (
                    <span key={r.model} title={r.reads}>
                      <b>{r.model}</b>
                      {r.ready ? '' : r.required ? ' (offline)' : ' (missing)'}
                    </span>
                  ))}
              </div>
            )}
          </header>
          {!file && (
            <div className="hero-visual">
              <FormShowcase />
            </div>
          )}
        </section>

        <main className="section" ref={workRef}>
          {modelDown && status === 'idle' && (
            <div className="banner error alert-wide">
              <span className="dot" style={{ background: 'var(--red)' }} />
              The local vision model is not reachable
              <span className="note">
                &nbsp;&middot;&nbsp;start Ollama and pull <code>{health.model}</code>
              </span>
            </div>
          )}

          {degraded && status === 'idle' && (
            <div className="banner review alert-wide">
              <span className="dot" />
              Digit reader missing &mdash; numbers will be less reliable
              <span className="note">
                &nbsp;&middot;&nbsp;pull <code>{digitReader?.model}</code> so dates, CNICs,
                phone numbers and amounts are read by the OCR model.
                Without it {health.model} reads them, and its digit mistakes are
                well-formed &mdash; they pass every format check.
              </span>
            </div>
          )}

          {!file && (
            <div
              className={`dropzone ${drag ? 'drag' : ''}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
              onDragLeave={() => setDrag(false)}
              onDrop={onDrop}
            >
              <div className="dz-icon">
                <svg viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 16V4M12 4l-4 4M12 4l4 4" />
                  <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
                </svg>
              </div>
              <div className="dz-title">Drop a scanned form here, or click to browse</div>
              <div className="dz-sub">PDF (up to 9 pages) or a single page image &middot; PNG, JPG, TIFF</div>
              <input
                ref={inputRef}
                type="file"
                accept="application/pdf,image/png,image/jpeg,image/webp,image/bmp,image/tiff"
                hidden
                onChange={(e) => pickFile(e.target.files?.[0])}
              />
            </div>
          )}

          {file && status === 'idle' && (
            <div className="setup">
              <div className="card">
                <div className="card-title">Document</div>
                <div className="file-line">
                  <span className="file-ico" aria-hidden="true" />
                  <span className="file-name">{file.name}</span>
                  <span className="file-meta">
                    {(file.size / 1024 / 1024).toFixed(1)} MB
                    {pageCount ? ` · ${pageCount} page${pageCount === 1 ? '' : 's'}` : ''}
                  </span>
                </div>
              </div>

              <div className="card">
                <div className="card-title">Pages to scan</div>
                <p className="card-sub">
                  Pick any pages you like &mdash; they need not start at page 1, and page 2
                  on its own is a valid choice. Fewer pages means a faster run.
                </p>
                <div className="page-picker">
                  {Array.from({ length: maxPages }, (_, i) => i + 1).map((n) => (
                    <button
                      key={n}
                      className={`page-chip ${pages.includes(n) ? 'on' : ''}`}
                      onClick={() => togglePage(n)}
                      title={pages.includes(n) ? `Skip page ${n}` : `Also scan page ${n}`}
                    >
                      {n}
                    </button>
                  ))}
                  {maxPages > 1 && (
                    <button
                      className={`page-chip all ${pages.length === maxPages ? 'on' : ''}`}
                      onClick={() => setPages(pages.length === maxPages
                        ? [1]
                        : Array.from({ length: maxPages }, (_, i) => i + 1))}
                    >
                      {pages.length === maxPages ? 'Clear' : `All ${maxPages}`}
                    </button>
                  )}
                </div>
                <div className="page-summary">
                  {pages.length === 0
                    ? 'Select at least one page'
                    : `Scanning ${pages.length === 1 ? 'page' : 'pages'} ${listPages(pages)} of ${pageCount || '?'}`}
                </div>
              </div>

              <div className="preview-actions">
                <button className="btn btn-ghost" onClick={reset}>Choose another</button>
                <button className="btn btn-primary" onClick={extract}
                        disabled={pages.length === 0}>Extract fields</button>
              </div>
              <p className="hint">Runs locally &middot; a few minutes per page on CPU, far quicker on a GPU</p>
            </div>
          )}

          {status === 'processing' && job && <Pipeline job={job} />}

          {status === 'error' && (
            <div className="setup">
              <div className="banner error">
                <span className="dot" style={{ background: 'var(--red)' }} />{error}
              </div>
              <div className="preview-actions">
                <button className="btn btn-ghost" onClick={reset}>Start over</button>
                <button className="btn btn-primary" onClick={extract}>Try again</button>
              </div>
            </div>
          )}

          {status === 'done' && job?.result && (
            <div ref={resultsRef}>
              <Results job={job} onReset={reset} />
            </div>
          )}
        </main>
      </div>

      <footer className="footer">
        <div className="footer-inner">
          <div><span className="brand">HBL</span>&nbsp;&nbsp;Account Opening Form Extraction</div>
          <div className="fnote">Processed locally via Ollama &middot; no data leaves this machine</div>
        </div>
      </footer>
    </div>
  )
}
