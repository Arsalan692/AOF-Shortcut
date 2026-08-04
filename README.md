# HBL Account Opening Form — field extraction

Extracts every field and its value from scanned, handwritten HBL Consumer Products /
Customer Information Forms.

**All processing is local.** The forms contain customer PII, so nothing is sent to a
cloud API. Handwriting is read by a local vision model served by Ollama.

---

## The idea

`Forms/Account_Opening_Form_blank template.pdf` is a **true vector PDF with a live text
layer** — not a scan. It therefore tells us, exactly and for free:

* where every fillable area is (each one is a white-filled rectangle, `fill == (1,1,1)`;
  printed rules, shading and the Urdu — which is drawn as vector outlines, not text —
  use other colours, so one colour test separates input from printing)
* what each one is called (the printed label beside it)
* which tick boxes belong to which question, and whether it is pick-one or pick-many
* the expected shape of structured values (date cells literally contain `D D M M Y Y Y Y`)

So instead of OCR-ing a page and trying to work out which value belongs to which label —
the thing that goes wrong with whole-page OCR — each scan is **aligned onto the template**
and every field is read from a known rectangle.

---

## Pipeline

| step | script | output |
|---|---|---|
| 1. derive the field schema from the blank template | `src/build_schema.py` | `build/template_schema.json` (431 fields, 0 unlabelled) |
| 2. align each scan page to the template, isolate pen ink | `src/register.py` | `build/registered/<doc>/{reg,ink}_p*.png`, `alignment.json` |
| 3. read tick boxes by measuring ink | `src/read_checkboxes.py` | in-memory (used by step 6) |
| 4. cut one image per field | `src/crop_fields.py` | `build/crops/<doc>/`, `index.json` |
| 5. transcribe handwriting with the local VLM | `src/read_text.py` | `build/reads/<doc>.json` |
| 6. validate, cross-check and assemble | `src/extract.py` | `build/output/<doc>.{json,csv}` |

Supporting tools: `src/validate.py` (types, format checks, repairs), `src/qa_schema.py`
(draws the schema over the template), `src/diagnose_alignment.py` (per-page scanner
distortion report), `src/evaluate.py` (scores output against `tests/ground_truth_*.json`).

### Web layer

`backend/jobs.py` runs the pipeline on a worker thread — extraction takes minutes, so an
HTTP request cannot wait for it. Each upload becomes a job whose id is also its document
name, so all of its artefacts stay isolated from every other job's. Progress is real, not
a timer: each stage reports through a callback, weighted by where the time actually goes.

`backend/app.py` exposes it, and serves `frontend/dist` at `/` when that exists.

| method | path | purpose |
|---|---|---|
| POST | `/api/extract` | upload a form (`file`, `pages`), returns a job id |
| GET | `/api/jobs/{id}` | progress while running, the result when done |
| GET | `/api/jobs/{id}/pages/{n}/preview.png` | the aligned page |
| GET | `/api/jobs/{id}/fields/{fid}/crop.png` | the exact patch a value was read from |
| GET | `/api/jobs/{id}/download.{json,csv}` | the assembled output |
| DELETE | `/api/jobs/{id}` | delete the upload and its artefacts |
| GET | `/api/health` | schema + model readiness |

`pages` selects how many pages to read **from the start** of the document, and is the
main lever on runtime — a one-page run costs roughly a ninth of a full form.

## Run the web app

One-time setup:

```bash
pip install -r requirements.txt
python src/build_schema.py            # derives the 431-field schema from the blank template
cd frontend && npm install && npm run build && cd ..
```

Then, from the **`backend/` directory**:

```bash
cd backend
python -m uvicorn app:app --port 8000
```

Open <http://localhost:8000>. FastAPI serves the built UI itself, so this one command is
the whole app.

Working on the frontend? Run `npm run dev` in `frontend/` as well and use
<http://localhost:5173> instead — Vite proxies `/api` to port 8000 and hot-reloads.

> Run uvicorn from `backend/`, or pass `--app-dir backend` from the project root.
> `app:app` has to be importable, and it imports `jobs` from the same folder.

Ollama must be running with `qwen2.5vl:7b` pulled. `GET /api/health` reports whether the
model and schema are ready, and the UI shows a banner if the model is unreachable.
Override with the `HBL_VLM` and `OLLAMA_HOST` environment variables.

## Run the pipeline from the command line

```bash
python src/register.py Forms/Cif-form.pdf
python src/crop_fields.py Cif-form
python src/read_text.py  Cif-form           # slow: see performance below
python src/extract.py    Cif-form
python src/evaluate.py   Cif-form --page 1  # optional accuracy check
```

`read_text.py` caches per field, so an interrupted run resumes. Useful flags:
`--pages 1 2`, `--batch N`, `--limit N` (first N batches only), `--force`, `--no-retry`.

---

## Things that turned out to matter

**Handwriting leaves its box.** Measured on a real form: overflow reaches 26pt to the
right and 34pt vertically against a 12pt-tall box, and 23.5% of all ink sits wholly
outside every box. A fixed pad clips 102 of 139 filled fields, and cannot be enlarged
because form rows are only ~16pt apart. Crops are therefore sized from the ink itself —
a nearest-box ownership partition, a proximity gate, then connected-component
completion — and other fields' strokes inside the grown rectangle are painted out so
each crop shows exactly one answer.

**Tick boxes are decided per question, not per box.** Ticks are drawn far larger than the
11pt box and spill across neighbours, so an absolute threshold marks several options per
question. Each question is resolved competitively instead: most ink wins, provided it
holds ≥55% of the group's total. If any option was struck through its interior, interiors
decide and the halo measurements are discarded — otherwise the halos of *losing* options
pick up bleed from nearby handwriting and out-vote a clear winner.

**Blank detection is deliberately conservative.** Because writing overflows, a box can
contain real ink belonging to its neighbour (on page 3 the ID Number digits spill into the
CIF Number cells), so "has ink" and "is filled" are different questions. Against ground
truth no pixel statistic separates them — stroke thickness overlaps (filled min 2.87 vs
empty max 3.82), as does eroded area (41 vs 251). Only zero-ink fields skip the model; the
rest are read and may come back `EMPTY`.

**A bad page must fail loudly.** A misregistered page does not produce slightly wrong
data, it reads ink from the wrong boxes. Every page is gated on coverage (≥0.80 of
template printing found underneath) and residual offset (≤4px). Note that the obvious
metrics are the wrong ones: IoU reads 0.18 on a page aligned to 0.15px, and inlier count
says how confident the fit was, not how correct — page 2 of the sample has 29 inliers and
aligns to 1.0px.

**Page order is verified, not assumed.** Each page is matched against the template it is
expected to be, and only if that fails is the full search run. Scoring candidates by raw
descriptor matches does *not* work (it assigned scan page 2 to template page 6); the score
must be RANSAC-verified inliers. Tested against a scan with pages 1–2 swapped, page 9
removed and page 4 rotated 180°: the swap was detected and recovered, the missing page
flagged, and the upside-down page silently corrected — ORB is rotation-invariant, so the
homography absorbs the flip.

**Batching is the only performance lever.** On CPU (no GPU) a distinct image costs
~120s almost regardless of size, and the 3B model is no faster than the 7B. Ten fields
stacked into one numbered montage costs ~19s/field — about an hour for a full form.
The JSON **schema** on the Ollama request is required, not cosmetic: with a bare
`format: "json"` the model emitted two keys and stopped.

---

## Output

`build/output/<doc>.json` carries one record per field with `value`, `raw`, `type`,
`source` (`checkbox-cv` / `vlm-batch` / `vlm-single` / `ink-gate`), `confidence`,
`valid`, `note` and `needs_review`, plus an `answers` view grouped by form section, the
per-page alignment verdicts, and any cross-field contradictions. `<doc>.csv` is the same
records flattened for review in Excel.

Nothing is silently dropped or silently corrected. Repairs are one-directional — a
letter/digit substitution is accepted only when it turns an invalid value into a valid
one — and every change is recorded in `note`.
