# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Extracts every field and value from scanned, handwritten HBL Consumer Account Opening /
Customer Information Forms. **All processing is local** — the forms hold customer PII, so
nothing goes to a cloud API. Handwriting is read by vision models served by Ollama at
`localhost:11434`. Do not introduce a hosted-API reader.

`README.md` is not a summary — it is the design record, and documents *why* each threshold
and rule is what it is, with the measurements behind them. Read the relevant section before
changing a constant.

## Commands

Setup (once):

```bash
pip install -r requirements.txt
python src/build_schema.py               # REQUIRED: writes build/template_schema.json
cd frontend && npm install && npm run build && cd ..
```

Nothing downstream works without `build/template_schema.json`; `build/` is gitignored and
absent on a fresh clone, so rebuild it first.

Web app — run uvicorn **from `backend/`** (`app.py` imports `jobs` as a sibling):

```bash
cd backend && python -m uvicorn app:app --port 8000     # serves frontend/dist at /
cd frontend && npm run dev                              # optional: :5173, proxies /api to :8000
cd frontend && npm run lint                             # oxlint
```

Pipeline from the CLI (`<doc>` is the PDF basename):

```bash
python src/register.py Forms/Cif-form.pdf 1   # trailing ints = scan pages; omit for all
python src/fields.py     Cif-form
python src/read_page.py  Cif-form --pages 1 --save-images
python src/extract.py    Cif-form
python src/evaluate.py   Cif-form --page 1    # scores against tests/ground_truth_*
```

`read_page.py` caches per field in `build/reads/<doc>.json`, so an interrupted run resumes;
`--force` ignores the cache, `--no-retry` skips the re-read pass, and `--save-images` writes
the marked-up page actually sent to the model to `build/qa/marked_<doc>_p<N>.png` — the first
thing to look at when a page reads badly. Ollama needs `qwen2.5vl:7b` pulled, and `glm-ocr`
for the re-read pass (`HBL_VLM`, `HBL_VLM_NUM`, `OLLAMA_HOST`, `HBL_PAGE_PX`, `HBL_NUM_CTX`
override).

There is no pytest suite. Checks available: `python src/validate.py` runs a table of
format-rule cases; `python src/evaluate.py <doc> --page N` scores real output against
hand-transcribed ground truth (pages 1 and 2 of `Cif-form` exist); `python src/qa_schema.py`
and `src/diagnose_alignment.py` produce visual/diagnostic output under `build/qa/`.

## Architecture

The central idea: `Forms/Account_Opening_Form_blank template.pdf` is a **vector PDF with a
live text layer**, so the blank form itself tells us where every field is, what it is called,
and what shape its value takes. Each scan is aligned onto that template and every field is
read from a known rectangle — instead of OCR-ing a page and guessing which value belongs to
which label.

Stages, each writing a file the next one reads:

| module | reads | writes |
|---|---|---|
| `src/build_schema.py` | blank template PDF | `build/template_schema.json` (431 fields) |
| `src/register.py` | scan + template | `build/registered/<doc>/{reg,ink}_p*.png`, `alignment.json` |
| `src/fields.py` | schema + registered/ink | `build/fields/<doc>.json` |
| `src/read_checkboxes.py` | ink layer | in-memory (called by `extract`) |
| `src/read_page.py` | field index + registered page | `build/reads/<doc>.json` |
| `src/prompts.py` | — | the per-field format instructions `read_page` sends |
| `src/extract.py` | all of the above | `build/output/<doc>.{json,csv}` |

**Reading is page-at-a-time, one model call per page**, then two cheap correction passes:

1. **Page pass** (`qwen2.5vl:7b`) — the page image carries a numbered marker on every answer
   box (drawn from the template's exact rectangles), and the prompt is the matching numbered
   list with the format each box must hold. That pairing is what makes whole-page reading
   safe: the model never has to work out which value belongs to which label.
2. **OCR sweep** (`glm-ocr`) — every structured field (`validate.OCR_TYPES`) is read again
   from its own crop. At a page resolution cheap enough to be practical, digits suffer, and
   a wrong digit yields a *valid* value that no rule can catch. Measured: this took page 1
   from 89% to 98%. Where the two readers disagree and both are well formed, the OCR value
   is taken and the field is flagged `needs_review` — nothing downstream can settle it.
3. **Re-read** — anything that fails its format check, or whose value contradicts the ink
   measurement, is re-read from an enlarged crop.

Accuracy moves a point or two between runs (one image is a shared canvas), so quote ranges:
page 1 has measured 96–98%, page 2 95%, checkboxes 100% throughout.

Cross-cutting invariants worth knowing before editing:

- **`<doc>` is the namespace.** Every artefact path contains it, which is why the web layer
  uses the job id as the document name — jobs never collide and need no locking.
- **Files are named by TEMPLATE page, not scan page.** `register.py` identifies which
  template page each scan page is (RANSAC-verified inliers, not raw match counts), so a
  shuffled or rotated scan lands in the right place automatically.
- **`validate.infer_type()` is the single source of type truth.** It drives *both* format
  validation and what `prompts.spec()` tells the model to expect, so what a field is asked
  for and what it is checked against cannot drift apart. `DESCRIPTIVE_RE` is tested before
  `LABEL_TYPES`, and the address pattern before the `name` pattern — both orderings are
  load-bearing.
- **`prompts.py` resolves a field's instruction most-specific-first**: `FIELD_SPECS` by id,
  then `LABEL_SPECS` by wording, then `type_spec()`. The label rules are consulted *only* for
  the two weak types (`text`, `name`) — a date is 8 digits whatever its label says, but
  "Branch Name" and "Dealer Name" are not people.
- **Nothing is silently corrected.** `validate.repair()` accepts a substitution only when it
  turns an invalid value into a valid one, and records it in `note`. Every record carries
  `source`, `model`, `confidence`, `valid`, `note`, `needs_review`.
- **A wrapped answer is one value across two boxes.** Fields carry `continuation_of`; the
  page prompt tells the model which box holds the second line, `extract.answers_view()`
  rejoins them, and `evaluate.py` folds them before scoring *only* when the continuation came
  back empty (i.e. the model merged the lines itself).
- **Ownership, not padding, decides what belongs to a field.** `fields.py` partitions the
  page by nearest box, then completes each claimed stroke's connected component. This is what
  handles handwriting that leaves its box — 23.5% of ink on this form sits wholly outside
  every box. It supplies the ink gate (zero ink ⇒ never sent to a model), the grown rectangle
  the page markers are drawn against, the `[INK PRESENT]` / `[NO INK]` notes in the prompt,
  and the crops used for re-reads and review.
- **Prefill dominates the cost, not the call count.** Measured on page 1: 5495 prompt tokens
  in 950s versus 534 output tokens in 257s. So the page image (`HBL_PAGE_PX`) and the prompt
  length are the levers, and both trade against accuracy — every call logs the split.

### Web layer

`backend/jobs.py` inserts `src/` on `sys.path` and imports the pipeline modules flat, then
runs them on a worker thread with weighted per-stage progress callbacks (`STAGES`). Job state
is an in-memory `Registry`; the durable record is the files under `build/`.
`backend/app.py` is the FastAPI surface and mounts `frontend/dist` at `/` when it exists — it
must import `jobs` *before* anything from `src/`, since that import is what puts `src/` on the
path.

Two behaviours that are deliberate and easy to break:

- **`pages` is a selection, not a count** (`jobs.parse_pages`): `2` means page 2 alone, and
  page identification still uses each page's real position in the scan.
- **Edits are recorded, never written over the extraction.** `PATCH` stores the correction in
  `build/output/<job>.edits.json` with the value it replaced; `apply_edits()` overlays it and
  re-runs `validate.assess()`, so a hand-typed value is checked exactly like a read one.

`GET /api/health` reports the two readers separately on purpose: a missing word reader blocks
upload, a missing digit reader only degrades (the VLM's digit errors are well-formed and pass
every format check).

### Frontend

React 19 + Vite, no router, no state library. `src/App.jsx` holds essentially the whole UI —
upload, `Pipeline` progress, `Results`/`Section`/`FieldRow` review list — polling
`/api/jobs/{id}` every 1.2s. Clicking a field row fetches its crop from the backend so a
reviewer can see the handwriting a value came from.

## Repository conventions

- `Forms/*.pdf` is tracked **on purpose** — the schema cannot be rebuilt without the blank
  template. Never add a blanket `*.pdf` rule to `.gitignore` (it says so at the top).
- `build/` is entirely derived, and `build/uploads/` holds customer PII. It must stay
  ignored.
- Python here is stdlib + pymupdf/opencv/numpy/fastapi only; the modules use module-level
  tuning constants with a comment giving the measurement that justifies each one. Match that
  style rather than introducing config files or new dependencies.
