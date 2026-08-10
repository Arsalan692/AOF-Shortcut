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

So the thing that goes wrong with whole-page OCR — working out which value belongs to which
label — never has to be guessed at. Each scan is **aligned onto the template**, and the page
handed to the model arrives with every answer box already outlined and numbered, beside a
list saying what number 21 is called and that it must hold a 13-digit CNIC starting 1–8.
The model is asked to read handwriting, and nothing else.

---

## Pipeline

| step | script | output |
|---|---|---|
| 1. derive the field schema from the blank template | `src/build_schema.py` | `build/template_schema.json` (431 fields, 0 unlabelled) |
| 2. align each scan page to the template, isolate pen ink | `src/register.py` | `build/registered/<doc>/{reg,ink}_p*.png`, `alignment.json` |
| 3. read tick boxes by measuring ink | `src/read_checkboxes.py` | in-memory (used by step 6) |
| 4. locate every answer box and measure the ink in it | `src/fields.py` | `build/fields/<doc>.json` |
| 5. transcribe each page in one call to a local VLM, then second-opinion every structured field with the OCR model | `src/read_page.py` | `build/reads/<doc>.json` |
| 6. validate, cross-check and assemble | `src/extract.py` | `build/output/<doc>.{json,csv}` |

A nine-page form costs **five model calls for the pages themselves** — page 5 carries no
handwriting at all and is skipped, and 60 of the 243 answer boxes have no ink and never
reach a model — plus a short OCR verification and re-read pass per page.

Supporting tools: `src/prompts.py` (what the model is told each field must look like),
`src/validate.py` (types, format checks, repairs), `src/qa_schema.py` (draws the schema over
the template), `src/diagnose_alignment.py` (per-page scanner distortion report),
`src/evaluate.py` (scores output against `tests/ground_truth_*.json`).

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

One-time setup on a fresh machine:

```bash
python -m venv venv                   # Python 3.11+ (developed on 3.13)
venv\Scripts\activate                 # Windows;  source venv/bin/activate elsewhere
pip install -r requirements.txt

ollama pull qwen2.5vl:7b              # reads the pages   (~6 GB)
ollama pull glm-ocr                   # checks the numbers (~2 GB)

python src/build_schema.py            # derives the 435-field schema from the blank template
cd frontend && npm install && npm run build && cd ..
```

`build/` is gitignored and absent on a fresh clone, so `build_schema.py` is not optional —
nothing downstream runs without `build/template_schema.json`. Likewise `frontend/dist` is
built, not committed: skip `npm run build` and `/` serves JSON instead of the app.

Check it before uploading anything: `GET /api/health` should report `schema_ready` with 435
fields and both readers `ready`.

**Hardware is the thing that decides whether this is usable.** Measured on a 4-core 15 W
laptop CPU with no usable GPU, one page costs ~14 minutes, of which 76% is prefill. The same
code and models on a machine with a CUDA GPU do a page in well under a minute. A GPU
workstation is still local, so nothing about the privacy constraint changes.

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

Ollama must be running with `qwen2.5vl:7b` pulled, which reads the pages, and `glm-ocr` for
the re-read pass (see *Two readers* below). Override with `HBL_VLM`, `HBL_VLM_NUM` and
`OLLAMA_HOST`; `HBL_PAGE_PX` and `HBL_NUM_CTX` tune the page image and the context window.

`GET /api/health` reports each reader separately, because the two failures are not
equivalent and should not look alike:

| state | `ok` | `degraded` | what the UI shows |
|---|---|---|---|
| both readers pulled | `true` | `false` | — |
| re-read model missing | **`true`** | **`true`** | amber banner; upload still allowed |
| Ollama unreachable | `false` | `false` | red banner; upload blocked |

Without the page reader nothing can be read at all, so that blocks the upload. Without the
re-read model the run still finishes — the VLM re-reads its own misses — so it is a warning
rather than a wall. It is a warning worth making loud, though: the VLM's digit mistakes are
*well-formed*, so a wrong CNIC is still a valid CNIC and every format check passes it. Being
quietly less accurate is the failure mode that most deserves saying out loud.

## Run the pipeline from the command line

```bash
python src/register.py  Forms/Cif-form.pdf
python src/fields.py    Cif-form
python src/read_page.py Cif-form            # one model call per page
python src/extract.py   Cif-form
python src/evaluate.py  Cif-form --page 1   # optional accuracy check
```

`read_page.py` caches per field, so an interrupted run resumes. Useful flags:
`--pages 1 2`, `--force`, `--no-retry`, and `--save-images`, which writes the marked-up page
actually sent to the model into `build/qa/`. That image is the first thing to look at when a
page reads badly: if a number tag has landed on the handwriting it was pointing at, or an
outline has grown across two answers, it shows there rather than in the values.

---

## Things that turned out to matter

**Handwriting leaves its box.** Measured on a real form: overflow reaches 26pt to the
right and 34pt vertically against a 12pt-tall box, and 23.5% of all ink sits wholly
outside every box. A fixed pad clips 102 of 139 filled fields, and cannot be enlarged
because form rows are only ~16pt apart. Each field's extent is therefore sized from the ink
itself — a nearest-box ownership partition, a proximity gate, then connected-component
completion (`src/fields.py`).

That partition is doing three jobs at once. It decides which boxes hold no ink and so are
never shown to a model; it gives the page image's outlines their shape, so an outline
contains the whole of an answer that has wandered out of its printed box; and when a value
has to be re-read on its own, it is what lets the neighbours' strokes inside the grown
rectangle be painted out, so the re-read image shows exactly one answer.

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

**Erasing a neighbour's stroke must cover its soft edge too.** This applies to the re-read
crops. They are grown to chase overflowing handwriting, so they overlap the next row, and
the strokes this field does not own are filled with the local paper level. Filling only the ink mask was not enough: the
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
This suppresses the flag only; the field is still read.

**Reading a page in one pass, the model skims — so tell it what the pixels already know.**
Shown 45 numbered boxes at once it answered EMPTY for a Passport box holding 7,961px of
pen, and copied a ticked option's printed label into an *Other (specify)* box holding 186px
of a neighbour's bleed. Neither is a guess made blindly; it simply is not looking hard at
one box out of 45, and no amount of "be careful" in the prompt fixed either (both survived
an explicit rule against them).

But blankness is not something the model has to judge: the ink layer and the ownership
partition have already measured it. So each field carries its verdict into the prompt —
`[INK PRESENT: this box IS filled - read it, never answer EMPTY]` or `[NO INK: blank box]`.
It is measurement handed over, not another instruction to try harder.

The claim has to be one the measurement can actually support, which is why it is stated
only at the extremes (`≤500px`, or `≥4000px` **and** at least 3x the template's own printed
baseline). That last guard is what stands every date grid down: the faint printed
`D D M M Y Y Y Y` survives subtraction, and an empty *HBL Customer Since* carries 2,989px
against a filled *Date of Birth*'s 4,326 — too close to assert anything about either.

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

**One image per page cuts the call count, not the clock.** This is worth being exact about,
because the intuition is wrong. Cutting fields out and reading them in batches of ten cost
five model calls for page 1; the whole page costs one. On CPU that is *not* five times
faster — page 1 measured **1105s as a single call** against roughly 950s as five montages.
The call count was never the cost.

What the cost actually is, measured on that call:

```
prefill 5495 tok in 950s (6 tok/s)     <- 76% of the call
output   534 tok in 257s (2.1 tok/s)
```

So the levers are the two things that make up those 5495 tokens: the page image
(~2550 tokens at `HBL_PAGE_PX`=2.0 MP) and the prompt (~2900). Neither is free, and both
buy accuracy — dropping the image resolution is what makes a reader insert a digit into a
P.A. number. Every call logs this breakdown, so a page that reads slowly says which half
to attack rather than leaving it to guesswork.

The prompt half is why a format used more than once on a page is defined once in a glossary
and referenced by tag: page 1 alone holds six dates and five person-names, and stating the
date rule six times cost ~275 tokens for no extra information.

The JSON **schema** on the Ollama request is required, not cosmetic: with a bare
`format: "json"` the model emitted two keys and stopped. It matters more now than it did
per-batch, because one truncated reply loses a whole page rather than ten fields — which is
also why `num_predict` is set well above what the page should need, and why `num_ctx` is set
explicitly (Ollama's 4096 default truncates a page prompt and silently loses the fields at
the bottom).

**Two readers, but only one of them reads pages.** A general VLM reads cursive well
*because* it knows the language — shown a blurry scrawl it recovers "Fountain". That same
prior is what makes it invent a digit: a CNIC has no vocabulary to fall back on, so `35810`
came back as `35820`, well-formed and therefore invisible to every format check. An OCR
model has no such prior and simply copies characters. Measured on page 1:

| | structured (16) | words (28) |
|---|---|---|
| `qwen2.5vl:7b` (8.3B) | 14 (87%) | **27 (96%)** |
| `glm-ocr` (1.1B) | **16 (100%)** | 22 (79%) |

A page mixes both kinds, so the page pass has to be the VLM — a 1.1B OCR model cannot hold
45 numbered boxes and their formats in one image. The OCR model keeps the job it is best at,
on the passes where a field is alone in its own strip again.

**That is now a routine second opinion, not only a rescue.** Reading a page cheaply enough
to be practical means reading it at a resolution where digits suffer: at 1.2 Mpx page 1 came
back with a branch code of `1093` for `1023`, a customer number of `395678` for `345678`, a
P.A. number of `034165` for `03465` and a residence date of `10102026` for `10102016`. Every
one is the right length and the right character class, so every format rule passes it — the
errors are invisible. So each structured field is read a second time from its own crop by
the OCR model, whose reading wins where it is well formed. It corrected all four and touched
nothing else, taking page 1 from 89% to 98%.

It is affordable for the same reason it is accurate: 1.1B parameters run prefill at ~45 tok/s
here against the VLM's 12, so a page's worth of structured fields costs well under a minute.

**When the two readers disagree and both are well formed, the field goes to review.** That is
the one case nothing downstream can settle — "both valid" is exactly why the disagreement
survived the format rules — and it is also the case that otherwise ships silently, because a
misread digit in a CNIC still yields a valid CNIC. The OCR reading is taken and the
disagreement is recorded, so a human sees a short list of the digits that were actually at
risk instead of trusting a number no rule ever questioned.

Prompting cannot substitute for this. The expected shape of every value is already in the
prompt (`prompts.spec()`), and pushing harder — numeric-only batches, a prompt forbidding
"plausible" answers, double row height — scored **12/16** and began attaching values to the
wrong strip. The failure was never classification; the model knew it was reading a CNIC.
It could not see the digit.

**One image means the reads are not independent, and that cuts both ways.** With fields
batched into montages, changing which fields shared an image changed answers for fields
whose own pixels had not moved — routing the numeric fields into their own batches cost
`Father's Name` in one run and `Mother's Maiden Name` in another. A page is a shared canvas
too, so a score can still move by a field or two between runs.

The difference is that on a page the shared context is the real document rather than an
arbitrary grouping. The neighbours a field is read alongside are the ones the writer
actually wrote it beside, which is why a wrapped address line reads correctly (below) and
why `Karachi East` in Area/District is not confused with the `Karachi` in City underneath
it. A montage's neighbours were an artefact of how the batch happened to be packed.

**A wrapped answer needs the line after it to be read correctly.** The Next-of-Kin address
runs across two boxes, and read from separate images line 1 alone comes back `Block B` —
whereas with `Gulshan-e-Iqbal` visible after it the same pixels read `Block 6`, because
following a street name a block *number* is what fits. Cutting fields apart destroyed that
context and had to have it deliberately restored by stacking the two lines back together.

Reading the whole page removes the problem rather than working around it: both lines are in
the image anyway, and the prompt names the relationship — field 35 is told its answer may
run on into field 36, and field 36 is told it holds the second line of field 35. The pair
stays two records, so each line keeps its own reviewable crop.

Ground truth was transcribed box-by-box, so `evaluate.py` joins a parent and its
continuation before scoring — but only when the continuation came back empty, meaning the
model merged the lines itself. Folding unconditionally would score a correctly-split first
line against the text of both lines.

---

## Output

`build/output/<doc>.json` carries one record per field with `value`, `raw`, `type`,
`source` (`checkbox-cv` / `vlm-page` / `vlm-recheck` / `ink-gate` / `reviewer`),
`model` (which reader produced it), `confidence`, `valid`, `note` and `needs_review`,
plus an `answers` view
grouped by form section, the per-page alignment verdicts, and any cross-field
contradictions. `<doc>.csv` is the same records flattened for review in Excel.

Nothing is silently dropped or silently corrected. Repairs are one-directional — a
letter/digit substitution is accepted only when it turns an invalid value into a valid
one — and every change is recorded in `note`.
