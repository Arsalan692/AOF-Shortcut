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
| 5. transcribe handwriting locally — digits by OCR model, words by VLM | `src/read_text.py` | `build/reads/<doc>.json` |
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
| PATCH | `/api/jobs/{id}/fields/{fid}` | correct one value by hand — `{"value": …}` |
| DELETE | `/api/jobs/{id}/fields/{fid}/edit` | undo that correction |
| GET | `/api/jobs/{id}/download.{json,csv}` | the assembled output |
| DELETE | `/api/jobs/{id}` | delete the upload and its artefacts |
| GET | `/api/health` | schema + model readiness |

`pages` is a **selection, not a count**: `2` reads page 2 on its own, `1,3` reads those two,
`2-4` a range, and `all` (or omitting it) the whole document. It is the main lever on
runtime — a one-page run costs roughly a ninth of a full form. A selection is not the same
as a prefix, which matters for page identification: each page is matched using its real
position in the scan, so asking for page 2 alone tries it against template page 2 rather
than page 1. `src/register.py form.pdf 2` does the same from the command line.

**A correction is recorded, not written over the extraction.** `PATCH` stores the new value
in `build/output/<job>.edits.json` alongside the value it replaced, then re-assembles the
downloads. The pipeline output therefore stays exactly what the models produced, every
correction can be undone, and `edited_from` keeps the original visible in the record.

Crucially a hand-typed value is **re-validated, not trusted**: it goes through the same
format rules as a read one, so typing `0399-1234567` into a mobile field comes back flagged
with *"0399 is not a Pakistani mobile prefix"*. A reviewer can make a typo too.

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

Ollama must be running with **both** `qwen2.5vl:7b` and `glm-ocr` pulled — the first reads
words, the second reads digits (see *Two readers* below). Override with `HBL_VLM`,
`HBL_VLM_NUM` and `OLLAMA_HOST`.

`GET /api/health` reports each reader separately, because the two failures are not
equivalent and should not look alike:

| state | `ok` | `degraded` | what the UI shows |
|---|---|---|---|
| both readers pulled | `true` | `false` | — |
| digit reader missing | **`true`** | **`true`** | amber banner; upload still allowed |
| Ollama unreachable | `false` | `false` | red banner; upload blocked |

Without the word reader nothing can be read at all, so that blocks the upload. Without the
digit reader the run still finishes — the VLM covers the numeric fields — so it is a warning
rather than a wall. It is a warning worth making loud, though: the VLM's digit mistakes are
*well-formed*, so a wrong CNIC is still a valid CNIC and every format check passes it. Being
quietly less accurate is the failure mode that most deserves saying out loud.

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

**A section heading is text on the form's own heading strip.** Styling does not identify
one: requiring teal *and* bold left 24 fields unsectioned, because "Residential Address",
"Work Address" and "Permanent Address" are bold but grey while "Financial Supporter Details
for Housewives" is teal but not bold. Relaxing to a size threshold instead loses "For Bank
Use Only" at 10pt and promotes the two document-title lines into sections. What actually
separates a heading from the title is the pale strip printed behind it — and the title has
none. The strip test needs one guard: input grids sit on the same strips and print
placeholder glyphs inside their cells, so the IBAN's `P K` and `H A B B 0` become sections
and displace the real ones unless headings are also required to start at the page margin
(x0 27–41; the glyphs are indented to 72–320). Page 4 still needs the teal-and-bold test as
well — its headings carry no strip, and `UNDERTAKING` is centred. Both tests together,
unsectioned 24 → 8.

Sectioning is not cosmetic. Page 2 repeats a whole address block for the work address, so
until the two blocks were named, six pairs of fields — `City`, `Country`, `Street No./Name`,
`Area/District`, `Post/Zip Code`, `Nearest Landmark` — were indistinguishable.

**The national formats are the strongest validation available.** These are Pakistani
formats as issued, not patterns copied off the sample form, and they reject values that are
well-formed in general but impossible here:

| field | rule |
|---|---|
| CNIC / SNIC / NICOP | 13 digits as `XXXXX-XXXXXXX-X`; first digit is the issuing region, **1–8**, never 0 |
| Mobile | 11 digits starting `03`; live ranges are `0300`–`0349` plus SCO `0355` |
| Landline | area code + subscriber. **Karachi 021 and Lahore 042 issue 8-digit subscribers (11 total); every other city 7 (10 total)** |
| IBAN | `PK` + 2 check digits + 4-**letter** bank code + 16 alphanumerics = 24, and the bank code **must be HBL's `HABB`** — the form's only IBAN field is the HBL account being opened, so anyone else's code there is a misread |
| Passport | 2 capital letters then 7 digits |
| Post code | exactly 5 digits |
| NTN | 7 digits, 8 with a check digit, or the holder's 13-digit CNIC |

The landline rule is the one that pays. A misread came back as `09136109237` — eleven digits
on `091`, which is Peshawar and issues seven-digit subscriber numbers. The old rule accepted
any 9–15 digits and waved it through; the real rule proves it wrong and flags it. That is the
first of these single-character misreads anything has ever caught, after both a second model
and heavier prompting failed to.

Length is not the only signal the template gives. A CNIC grid is **15** cells for 13 digits
and two dashes, a mobile **12** for 11 digits and one, an IBAN **24** for 24 characters — so
the cell count confirms the format including its separators, and is quoted to the reader as
something it can count against its own answer.

Two rules span fields rather than validating one. The last digit of a CNIC is the holder's
gender — odd male, even female — so it must agree with the Gender tick, which is a real check
on a value no format rule can question, because a wrong digit still yields a well-formed
CNIC. Phone numbers are also normalised back to the dashed form Pakistan writes them in
(`0320-5120612`, `021-36102837`) rather than flattened to bare digits.

**A label that asks for a description is not a number.** The money and duration words sit
inside descriptive labels: "Other Source of Income (Please specify)", "Line of
Business/Industry/Source of Income", "…the nature and expected months of high turnover".
Since the type now decides *which model reads the field*, mis-typing one is no longer just a
failed format check — it routes free text to the OCR reader. `DESCRIPTIVE_RE` is tested
before `LABEL_TYPES` for that reason; without it six page-2 fields went to the wrong reader.

**Erasing a neighbour's stroke must cover its soft edge too.** Crops are grown to chase
overflowing handwriting, so they overlap the next row, and the strokes this field does not
own are filled with the local paper level. Filling only the ink mask was not enough: the
mask is a threshold, so every erased stroke kept an anti-aliased rim a pixel or two wide
that fell below it. The dark core became paper *brighter* than the surrounding CamScanner
band while the rim survived — leaving a glowing hollow outline of the letter that was
supposed to be gone. `GHOST_K` widens the fill to swallow that rim.

It cannot widen blindly: an overflowing neighbour physically crosses the answer, so a plain
dilation erases part of the value being read. Owned ink is protected with a halo of its own,
which leaves a short stub of ghost at each crossing — the price of not damaging the answer.
Measured per crop: 9–11k pixels of ghost removed, against 0.06–0.25% of the answer's own
ink touched. Ink statistics are unaffected, because they are measured from the ink layer
rather than the painted patch.

**Tick boxes are decided per question, not per box.** Ticks are drawn far larger than the
11pt box and spill across neighbours, so an absolute threshold marks several options per
question. Each question is resolved competitively instead: most ink wins, provided it
holds ≥55% of the group's total. If any option was struck through its interior, interiors
decide and the halo measurements are discarded — otherwise the halos of *losing* options
pick up bleed from nearby handwriting and out-vote a clear winner.

**A wrapped line is one answer, not two.** The Next of Kin address is two white boxes,
one per line, and the label probe finds the printed word "Address" for both — so the form
reported two fields of the same name. A second line is told apart from a stacked table row
by its **left edge**: a wrapped line starts at the page margin, left of its parent, because
no printed label precedes it, whereas two rows of one table column share an x0 exactly.
That distinction keeps page 2's `Amount (PKR)` and `No. of Transactions` pairs and page 9's
two `Employee No.` boxes as the separate answers they really are — getting it wrong in that
direction would silently move one field's value into another, and both would still look
valid. Records stay one-per-box so each line keeps its own reviewable crop; the reader shows
the lines joined (see *A wrapped answer* below) and the `answers` view rejoins the values.

**Blank detection is deliberately conservative.** Because writing overflows, a box can
contain real ink belonging to its neighbour (on page 3 the ID Number digits spill into the
CIF Number cells), so "has ink" and "is filled" are different questions. Against ground
truth no pixel statistic separates them — stroke thickness overlaps (filled min 2.87 vs
empty max 3.82), as does eroded area (41 vs 251). Only zero-ink fields skip the model; the
rest are read and may come back `EMPTY`.

**But a trace of ink is not evidence of an answer.** "Ink here, nothing transcribed" is
worth reviewing only above a floor (`INK_FLOOR_PX`): page 1's three genuinely-empty
*Other (specify)* boxes hold 186, 328 and 373px of a neighbour's overflow, while the
smallest real answer on the page — the single digit in *Number of Dependents* — holds 827.
This suppresses the flag only; the field is still cropped and still read.

Date grids stay unreliable here and are left flagged: the printed `D D M M Y Y Y Y`
placeholders leave a residue the ink layer cannot fully subtract, and subtracting the
template's own ink (`baseline_px`) does not separate them — empty *HBL Customer Since*
carries 1342px above its baseline, filled *CIF Opening Date* carries 1358.

**A traced rule is not part of the value.** A reader that follows the printed line into its
answer returns `-Karachi`. Removing a *leading* dash or underscore is exempt from the
invalid→valid repair rule below, because it removes the box rather than correcting the
writer. Leading side only: `Shabn.` and `H. No.` end in a full stop the writer made.

**Address lines are not names.** The form labels them `House/Appt. No./Appt. Name`,
`Street No./Name` and `Office No./Office Name`, so a rule matching *name* claims them and
then rejects `H. No. 43/2` for containing digits and a slash. Address vocabulary is matched
first — but after `email`, since `Email Address` contains "address".

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

Measured again later, the per-image cost is what dominates: five page-1 montages cost
224/186/189/186/180s, and the 5-strip half-empty one cost the same as the full ones — the
noise between two identical requests (186 vs 224s) exceeds the difference between a full
image and a half-empty one. Packing more strips per image by trimming the 60% of each
montage that is white padding did cut 5 images to 3, but corrupted 3 of 26 fields and
manufactured exactly the format-valid single-character errors that are hardest to detect.
Not worth it; the montage is left alone.

**Two readers, routed by field type.** A general VLM reads cursive well *because* it knows
the language — shown a blurry scrawl it recovers "Fountain". That same prior is what makes
it invent a digit: a CNIC has no vocabulary to fall back on, so `35810` came back as
`35820`, well-formed and therefore invisible to every format check. An OCR model has no
such prior and simply copies characters. Measured on page 1:

| | structured (16) | words (28) |
|---|---|---|
| `qwen2.5vl:7b` (8.3B) | 14 (87%) | **27 (96%)** |
| `glm-ocr` (1.1B) | **16 (100%)** | 22 (79%) |

So numeric fields go to the OCR model and words to the VLM — 43/46 → 44/46, and 18.2 min
→ 11.2, because the small model is 2.3x faster on the fields it owns. The routing is free:
the template established every field's type at schema time.

Prompting cannot substitute for this. The expected shape of every value is already in the
prompt (`describe()`), and pushing harder — numeric-only batches, a prompt forbidding
"plausible" answers, double row height — scored **12/16** and began attaching values to the
wrong strip. The failure was never classification; the model knew it was reading a CNIC.
It could not see the digit.

**A montage is a shared canvas, so reads are not independent.** Changing which fields share
an image changes answers for fields whose own pixels did not move: routing the numeric
fields away re-rolled the text reads, costing `Father's Name` in one run, `Mother's Maiden
Name` in another, and nothing in a third. Nothing about a strip's own resolution changed in
any of them. This is the main reason a score moves by a field or two between runs, and it is
the cost of batching — which is not optional on CPU.

**A wrapped answer is read as one strip, on purpose.** Left as two adjacent strips, the two
Next-of-Kin address lines were read as a single value: the whole address came back in the
first box and the second came back empty, which then flagged a box holding 19,092px of ink
as "ink present but read as empty". The crops were not at fault — they do not overlap by a
single point — the two lines simply look like one line of handwriting.

Splitting the pair across separate images fixed the empty box but cost accuracy: line 1
alone reads `Block B`, whereas with `Gulshan-e-Iqbal` visible after it the same pixels read
`Block 6` — following a street name, a block *number* is what fits. So the context is worth
having, and `field_strip()` supplies it deliberately: a continuation is stacked beneath its
parent, separated by white rather than the montage's black rule, and the prompt says
"written across 2 lines - read them as one value". The continuation is then not a batch item
at all, so it costs no extra call and no extra image, and it is recorded as
`read-with-parent` rather than as a filled box that came back empty.

Ground truth was transcribed box-by-box, so `evaluate.py` joins a parent and its
continuation before scoring; otherwise both halves would count as misses purely because of
where the line break happened to be recorded.

---

## Output

`build/output/<doc>.json` carries one record per field with `value`, `raw`, `type`,
`source` (`checkbox-cv` / `vlm-batch` / `vlm-recheck` / `read-with-parent` / `ink-gate`),
`model` (which reader produced it), `confidence`, `valid`, `note` and `needs_review`,
plus an `answers` view
grouped by form section, the per-page alignment verdicts, and any cross-field
contradictions. `<doc>.csv` is the same records flattened for review in Excel.

Nothing is silently dropped or silently corrected. Repairs are one-directional — a
letter/digit substitution is accepted only when it turns an invalid value into a valid
one — and every change is recorded in `note`.
