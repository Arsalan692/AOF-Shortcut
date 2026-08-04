"""
Phase A - Template schema harvest.

Reads the BLANK template PDF (a true digital/vector PDF) and derives the complete
field schema automatically: every input box, its type, its geometry in PDF points,
and its label - taken from the template's own text + vector layers.

Why this works: in the blank template every fillable area is drawn as a
white-filled rectangle, fill == (1,1,1). Printed chrome (rules, shading, the Urdu
glyph outlines that masquerade as tiny boxes) is drawn in other colours, so a
single colour test separates "places a human writes" from "printing".

Output: build/template_schema.json
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLANK = os.path.join(ROOT, "Forms", "Account_Opening_Form_blank template.pdf")
OUT = os.path.join(ROOT, "build", "template_schema.json")

# ---------------------------------------------------------------- tuning knobs
WHITE_TOL = 0.02        # how close fill must be to pure white
MIN_W, MIN_H = 4.0, 6.0  # ignore vector dust
MAX_H = 45.0             # tallest input area (signature / stamp blocks reach ~35)
BLOCK_MIN_H = 20.0       # at/above this a box is a signature-or-stamp sized block
CELL_MAX_W = 21.0        # widest char-grid cell (Reference Name cells are 19.5)
CHECKBOX_MAX_W = 14.0    # a lone box this narrow is a tick box, not an entry box
ROW_TOL = 3.0            # y0 within this => same visual row
PITCH_TOL = 1.8          # spacing regularity for a char-grid run
DEDUPE_TOL = 1.5         # near-identical rects are the same box

# Teal used for section headings, RGB approx (0.0, 0.662, 0.591)
TEAL = (0.0, 0.662, 0.591)


def is_white(fill) -> bool:
    if fill is None:
        return False
    try:
        return all(abs(c - 1.0) <= WHITE_TOL for c in fill[:3])
    except TypeError:
        return False


def is_teal(color) -> bool:
    if color is None:
        return False
    try:
        return all(abs(c - t) < 0.12 for c, t in zip(color[:3], TEAL))
    except TypeError:
        return False


# ------------------------------------------------------------------ data model
@dataclass
class Box:
    """One physical white rectangle on the page."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    def as_list(self):
        return [round(self.x0, 2), round(self.y0, 2), round(self.x1, 2), round(self.y1, 2)]


@dataclass
class Field:
    id: str
    page: int
    kind: str                    # text | grid | checkbox
    label: str
    rect: list                   # union bbox
    section: str = ""
    cells: list = field(default_factory=list)   # grid only: per-character rects
    grid_hint: str = ""          # grid only: e.g. "DDMMYYYY" from template glyphs
    group: str = ""              # checkbox only: the question it belongs to
    table: str = ""              # table cells only: the table's caption
    mirror: bool = False         # a right-hand Urdu-column duplicate of another field
    select_mode: str = ""        # checkbox only: "one" | "many" (from the form's own wording)
    row_y: float = 0.0


# ------------------------------------------------------------- vector harvest
def harvest_white_rects(page) -> list[Box]:
    out: list[Box] = []
    for item in page.get_drawings():
        if not is_white(item.get("fill")):
            continue
        r = item["rect"]
        if r.width < MIN_W or r.height < MIN_H or r.height > MAX_H:
            continue
        out.append(Box(r.x0, r.y0, r.x1, r.y1))

    # drop near-duplicates (fill + stroke drawn as two paths)
    out.sort(key=lambda b: (round(b.y0, 1), b.x0))
    kept: list[Box] = []
    for b in out:
        dup = False
        for k in kept:
            if (abs(b.x0 - k.x0) < DEDUPE_TOL and abs(b.y0 - k.y0) < DEDUPE_TOL
                    and abs(b.x1 - k.x1) < DEDUPE_TOL and abs(b.y1 - k.y1) < DEDUPE_TOL):
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def group_rows(boxes: list[Box]) -> list[list[Box]]:
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: (b.y0, b.x0)):
        for row in rows:
            if abs(row[0].y0 - b.y0) <= ROW_TOL:
                row.append(b)
                break
        else:
            rows.append([b])
    for row in rows:
        row.sort(key=lambda b: b.x0)
    return rows


def split_runs(row: list[Box]) -> list[list[Box]]:
    """Split a row into evenly-pitched runs of same-width narrow cells (char grids)
    versus standalone boxes."""
    runs: list[list[Box]] = []
    cur: list[Box] = [row[0]]
    for prev, b in zip(row, row[1:]):
        gap = b.x0 - prev.x1
        same_w = abs(b.w - prev.w) < 1.5
        narrow = b.w <= CELL_MAX_W and prev.w <= CELL_MAX_W
        contiguous = -1.0 <= gap <= 3.0          # cells touch or nearly touch
        if same_w and narrow and contiguous:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    return runs


# --------------------------------------------------------------- text harvest
def page_words(page):
    """(x0,y0,x1,y1,text) for every word, PDF points."""
    return [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]


def page_sections(page) -> list[tuple[float, str]]:
    """(y, heading) for teal section headings, top to bottom.

    Headings are teal AND bold. Size alone is not enough to separate them from body
    text: "Personal Information" is 12pt but "For Bank Use Only" is only 10pt, and a
    threshold set above 10 silently leaves that whole band unsectioned.
    """
    heads = []
    for blk in page.get_text("dict")["blocks"]:
        for line in blk.get("lines", []):
            txt, size, teal, bold = "", 0.0, False, False
            for sp in line["spans"]:
                txt += sp["text"]
                size = max(size, sp["size"])
                if "bold" in sp.get("font", "").lower():
                    bold = True
                # PyMuPDF gives span colour as packed sRGB int
                c = sp.get("color", 0)
                rgb = ((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255
                if is_teal(rgb):
                    teal = True
            txt = clean_label(txt.strip())
            if teal and bold and size >= 9.5 and len(txt) > 2 and not txt.startswith("Page"):
                heads.append((line["bbox"][1], txt))
    heads.sort()
    return heads


def section_for(y: float, heads: list[tuple[float, str]]) -> str:
    cur = ""
    for hy, name in heads:
        if hy <= y + 2:
            cur = name
        else:
            break
    return cur


GRID_GLYPH = re.compile(r"^[DMY]$")


def grid_hint(cells: list[Box], words) -> str:
    """If the template prints D/M/Y placeholders inside the cells, recover them -
    that identifies the grid as a date and fixes the component order."""
    hint = []
    for c in cells:
        found = ""
        for x0, y0, x1, y1, t in words:
            if not GRID_GLYPH.match(t):
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if c.x0 - 1 <= cx <= c.x1 + 1 and c.y0 - 3 <= cy <= c.y1 + 3:
                found = t
                break
        hint.append(found)
    joined = "".join(hint)
    return joined if joined.strip() else ""


# ----------------------------------------------------------- label assignment
STOP = {"", "-", "/", ":", "|"}


def text_in(words, x0, y0, x1, y1) -> str:
    """Words whose centre falls inside the probe window, left to right."""
    hits = []
    for wx0, wy0, wx1, wy1, t in words:
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1 and t not in STOP:
            hits.append((wx0, t))
    hits.sort()
    return " ".join(t for _, t in hits)


# This template sets text with typographic ligatures and curly quotes, so the text
# layer yields "Veriﬁed" (U+FB01) and "Father’s". Left alone they leak into field names
# and break plain-ASCII consumers such as the Windows console and most CSV readers.
TYPOGRAPHY = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st", "‘": "'", "’": "'",
    "“": '"', "”": '"', "–": "-", "—": "-", " ": " ",
})


def clean_label(s: str) -> str:
    s = s.translate(TYPOGRAPHY)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*\((?:Please select|Select one|Select all that apply|if applicable|If applicable)\)\s*",
               " ", s, flags=re.I)
    # Urdu is drawn as vector outlines, so a bilingual label often leaves the
    # English acronym stranded twice: "CIF Opening Date CIF" -> "CIF Opening Date"
    toks = s.split()
    if len(toks) > 1 and toks[-1] in toks[:-1]:
        toks = toks[:-1]
    return " ".join(toks).strip(" :,-")


LEFT_LIMIT = 300.0
RIGHT_LIMIT = 190.0
MAX_GAP = 11.0       # gap that ends a checkbox's option label
# The left probe needs a far looser gap: Urdu on this form is drawn as vector
# outlines, not text, so a bilingual label like "CIF Opening Date <urdu> CIF"
# leaves a wide hole in the text layer that a tight rule would cut at.
MAX_GAP_LEFT = 45.0
LINE_TOL = 4.0       # y-centre spread within one printed line


def lines_in(words, x0, y0, x1, y1) -> list[list[tuple[float, float, str]]]:
    """Words whose centre falls in the window, grouped into printed lines and
    returned top-to-bottom, each line ordered left-to-right.

    Reading order matters: labels here routinely wrap over two or three lines
    ("Passport No. (Foreign nationals holding a valid Pakistani visa only)"), and
    sorting by x alone shuffles them into nonsense.
    """
    buckets: list[tuple[float, list]] = []
    for wx0, wy0, wx1, wy1, t in words:
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1) or t in STOP:
            continue
        for cy0, bucket in buckets:
            if abs(cy0 - cy) <= LINE_TOL:
                bucket.append((wx0, wx1, t))
                break
        else:
            buckets.append((cy, [(wx0, wx1, t)]))
    buckets.sort(key=lambda z: z[0])
    for _, bucket in buckets:
        bucket.sort()
    return [b for _, b in buckets]


def probe_right(words, b: Box, stop_x: float):
    """Option label printed to the right of a checkbox.

    Bounded three ways: by the next box on the row, by RIGHT_LIMIT, and - the one
    that actually matters - by the first wide gap. Without the gap rule the last
    checkbox of a row swallows the following field's label, turning "No" into
    "No HBL Customer Since".
    """
    # Cap the vertical reach. Signature and stamp blocks are ~31pt tall, and a pad
    # proportional to that spans whole neighbouring rows - which is how "Approved By"
    # became "Yes Approved By Date Received D D M M Y Y Y Y P.A. No.".
    pad = min(b.h * 0.55, 7.0)
    hi = min(b.x1 + RIGHT_LIMIT, stop_x)
    out, end, anchor = [], b.x1, None
    for line in lines_in(words, b.x1 + 0.5, b.y0 - pad, hi, b.y1 + pad):
        # a continuation line must start roughly under the first line's start
        start = b.x1 + 8.0 if anchor is None else anchor + 15.0
        taken, prev = [], None
        for wx0, wx1, t in line:
            if prev is None:
                if wx0 > start:
                    break
            elif wx0 - prev > MAX_GAP:
                break
            taken.append(t)
            prev = wx1
            end = max(end, wx1)
        if not taken:
            continue
        if anchor is None:
            anchor = line[0][0]
        out.append(" ".join(taken))
    return clean_label(" ".join(out)), end


def probe_left(words, b: Box, start_x: float) -> tuple[str, str]:
    """Label printed to the left of a text/grid field. Same gap rule, walked
    right-to-left away from the box.

    Returns (cleaned, raw). The raw form is kept because the parentheticals that
    clean_label strips - "(Select one)", "(Select all that apply)" - are exactly
    what tells us whether a checkbox group is radio or multi-select.
    """
    # Cap the vertical reach. Signature and stamp blocks are ~31pt tall, and a pad
    # proportional to that spans whole neighbouring rows - which is how "Approved By"
    # became "Yes Approved By Date Received D D M M Y Y Y Y P.A. No.".
    pad = min(b.h * 0.55, 7.0)
    lo = max(b.x0 - LEFT_LIMIT, start_x)
    out = []
    for line in lines_in(words, lo, b.y0 - pad, b.x0 - 1.0, b.y1 + pad):
        taken, prev = [], None
        for wx0, wx1, t in reversed(line):
            if prev is not None and prev - wx1 > MAX_GAP_LEFT:
                break
            taken.append(t)
            prev = wx0
        if taken:
            out.append(" ".join(reversed(taken)))
    raw = " ".join(out)
    return clean_label(raw), raw


def select_mode_of(raw: str) -> str:
    low = raw.lower()
    if "all that apply" in low:
        return "many"
    if "select one" in low or "please select" in low:
        return "one"
    return ""


def build_page(page, pno: int, carry: dict) -> list[Field]:
    """carry: mutable dict holding the last checkbox group seen, so that groups can
    span rows (Education, Property Status, Source of Income ...)."""
    words = page_words(page)
    heads = page_sections(page)
    boxes = harvest_white_rects(page)
    fields: list[Field] = []

    for row in group_rows(boxes):
        # Collapse the row into units: char-grid runs stay together, everything
        # else is a single box. Units are ordered left to right.
        units: list[tuple[str, list[Box]]] = []
        for run in split_runs(row):
            if len(run) >= 3 and all(b.w <= CELL_MAX_W for b in run):
                units.append(("grid", run))
            else:
                for b in run:
                    if b.h >= BLOCK_MIN_H and b.w > CHECKBOX_MAX_W:
                        kind = "block"      # signature, stamp, multi-line entry
                    elif b.w <= CHECKBOX_MAX_W:
                        kind = "checkbox"
                    else:
                        kind = "text"
                    units.append((kind, [b]))

        # frontier = x beyond which text is unclaimed by anything to our left
        frontier = 0.0
        for idx, (kind, run) in enumerate(units):
            nxt = units[idx + 1][1][0].x0 if idx + 1 < len(units) else 1e9
            bbox = Box(min(b.x0 for b in run), min(b.y0 for b in run),
                       max(b.x1 for b in run), max(b.y1 for b in run))
            sect = section_for(bbox.y0, heads)

            if kind == "checkbox":
                # text sitting between the frontier and this box is a NEW question
                q, q_raw = probe_left(words, bbox, frontier)
                if q:
                    carry["group"] = q
                    carry["mode"] = select_mode_of(q_raw)
                opt, end = probe_right(words, bbox, nxt)
                fields.append(Field(
                    id="", page=pno, kind="checkbox", label=opt, rect=bbox.as_list(),
                    section=sect, group=carry.get("group", ""),
                    select_mode=carry.get("mode", ""), row_y=round(bbox.y0, 1),
                ))
                frontier = max(end, bbox.x1)
            else:
                # A signature/stamp block is tall enough to sit beside two or three
                # printed rows; its own caption is the topmost one, so probe only the
                # upper slice instead of the whole height.
                probe_box = bbox
                if kind == "block":
                    probe_box = Box(bbox.x0, bbox.y0, bbox.x1,
                                    bbox.y0 + bbox.h * 0.45)
                lab = (probe_left(words, probe_box, frontier)[0]
                       or probe_above(words, probe_box))
                fields.append(Field(
                    id="", page=pno, kind=kind, label=lab, rect=bbox.as_list(),
                    cells=[b.as_list() for b in run] if kind == "grid" else [],
                    grid_hint=grid_hint(run, words) if kind == "grid" else "",
                    section=sect, row_y=round(bbox.y0, 1),
                ))
                frontier = bbox.x1
                carry["group"] = ""   # a text field ends any checkbox group

    fields.extend(harvest_tables(page, pno, heads))
    return fields


# The SBP Annexure CF-1 exposure tables on page 4 are drawn differently from every
# other field: one near-white outer rect per table, with the cell grid formed by
# stroked rules inside it. They need their own harvester.
TABLE_FILL = (0.961, 0.984, 0.980)


def is_table_fill(fill) -> bool:
    if fill is None:
        return False
    try:
        return all(abs(c - t) < 0.02 for c, t in zip(fill[:3], TABLE_FILL))
    except TypeError:
        return False


def harvest_tables(page, pno: int, heads) -> list[Field]:
    draws = page.get_drawings()
    words = page_words(page)
    outers = [it["rect"] for it in draws
              if it["type"] == "f" and is_table_fill(it.get("fill"))
              and it["rect"].width > 120 and it["rect"].height > 30]

    out: list[Field] = []
    for T in sorted(outers, key=lambda r: (round(r.y0, 1), r.x0)):
        hs, vs = set(), set()
        for it in draws:
            if it["type"] != "s":
                continue
            r = it["rect"]
            if r.x0 < T.x0 - 2 or r.x1 > T.x1 + 2 or r.y0 < T.y0 - 2 or r.y1 > T.y1 + 2:
                continue
            if r.height < 0.6 and r.width > T.width * 0.8:
                hs.add(round(r.y0, 1))
            if r.width < 0.6 and r.height > T.height * 0.8:
                vs.add(round(r.x0, 1))
        ys = sorted({round(T.y0, 1)} | hs | {round(T.y1, 1)})
        xs = sorted({round(T.x0, 1)} | vs | {round(T.x1, 1)})
        if len(ys) < 3 or len(xs) < 2:
            continue

        cap_lines = lines_in(words, T.x0 - 4, T.y0 - 24, T.x1 + 6, T.y0 - 1.5)
        caption = clean_label(" ".join(" ".join(t for _, _, t in ln) for ln in cap_lines))

        # row 0 of the grid holds the column headers
        headers = []
        for i in range(len(xs) - 1):
            hl = lines_in(words, xs[i] + 1, ys[0] - 1, xs[i + 1] - 1, ys[1] + 1)
            headers.append(clean_label(" ".join(" ".join(t for _, _, t in ln) for ln in hl))
                           or f"column {i + 1}")

        for ri in range(1, len(ys) - 1):
            for ci in range(len(xs) - 1):
                b = Box(xs[ci], ys[ri], xs[ci + 1], ys[ri + 1])
                out.append(Field(
                    id="", page=pno, kind="text",
                    label=f"{headers[ci]} (row {ri})", rect=b.as_list(),
                    section=section_for(T.y0, heads), table=caption,
                    row_y=round(b.y0, 1),
                ))
    return out


def probe_above(words, b: Box, limit=26.0) -> str:
    """Header printed above the box. This is how the SBP Annexure CF-1 tables on
    page 4 are labelled - column header on top, nothing to the left."""
    lines = lines_in(words, b.x0 - 4.0, b.y0 - limit, b.x1 + 4.0, b.y0 - 1.0)
    return clean_label(" ".join(" ".join(t for _, _, t in ln) for ln in lines))


def fill_table_columns(fields: list[Field]) -> None:
    """Second and third rows of a ruled table sit under another cell, not under the
    header, so they inherit the column label and get a row index."""
    for f in fields:
        if f.label or f.kind != "text":
            continue
        col = [g for g in fields
               if g.page == f.page and g.kind == "text" and g.label
               and abs(g.rect[0] - f.rect[0]) < 6 and abs((g.rect[2] - g.rect[0]) - (f.rect[2] - f.rect[0])) < 6
               and g.rect[1] < f.rect[1]]
        if not col:
            continue
        head = max(col, key=lambda g: g.rect[1])
        base = re.sub(r"\s*\(row \d+\)$", "", head.label)
        n = sum(1 for g in fields if g.page == f.page and g.label.startswith(base + " (row"))
        f.label = f"{base} (row {n + 2})"
        f.section = head.section


OTHERISH = re.compile(r"^(other|others)\b", re.I)


def merge_grid_segments(fields: list[Field]) -> list[Field]:
    """Some numbers are printed as several cell runs with a visible gutter between
    them (Direct Debit A/c No. is 4 + 4 + 16 cells). They are one value, so stitch
    same-row grids separated by only a gutter back together."""
    out, drop = [], set()
    for i, f in enumerate(fields):
        if i in drop or f.kind != "grid":
            continue
        cells = list(f.cells)
        rect = list(f.rect)
        for j, g in enumerate(fields):
            if j <= i or j in drop or g.kind != "grid" or g.page != f.page:
                continue
            if abs(g.rect[1] - rect[1]) <= ROW_TOL and 0 <= g.rect[0] - rect[2] <= 12:
                cells += g.cells
                rect[2], rect[3] = g.rect[2], max(rect[3], g.rect[3])
                drop.add(j)
        f.cells, f.rect = cells, [round(v, 2) for v in rect]
    return [f for i, f in enumerate(fields) if i not in drop]


def flag_urdu_mirrors(fields: list[Field]) -> None:
    """This form prints an Urdu column on the right that repeats some tick boxes and
    slots verbatim. They are the same datum drawn twice; flag the right-hand copy so
    extraction reads each value once."""
    for f in fields:
        if f.label or f.mirror:
            continue
        fw = f.rect[2] - f.rect[0]
        for g in fields:
            if g is f or g.page != f.page or g.kind != f.kind or not g.label or g.mirror:
                continue
            if abs(g.rect[1] - f.rect[1]) <= 2.5 and abs((g.rect[2] - g.rect[0]) - fw) <= 1.5 \
                    and g.rect[2] < f.rect[0]:
                f.label = f"{g.label} (Urdu column)"
                f.group = g.group
                f.section = g.section
                f.mirror = True
                break


def fill_specify_boxes(fields: list[Field]) -> None:
    """A text box sitting immediately right of an "Other" checkbox is that option's
    free-text slot - carry the group in so it is self-describing.

    This overrides any label already guessed: the header-above fallback tends to
    grab whatever paragraph happens to sit over these inline slots.
    """
    for f in fields:
        if f.kind != "text" or f.table:
            continue
        for g in fields:
            if g.page != f.page or g.kind != "checkbox" or not OTHERISH.match(g.label or ""):
                continue
            if abs(g.rect[1] - f.rect[1]) <= ROW_TOL and 0 < f.rect[0] - g.rect[2] < 190:
                stem = g.group or g.section
                f.label = f"{stem} - Other (specify)" if stem else "Other (specify)"
                f.section = g.section
                break


def fill_continuations(fields: list[Field]) -> None:
    """A wide unlabelled text box directly under a labelled one of similar width is
    that field's second line (Next of Kin address, permanent address...).

    The width guard matters: without it, narrow boxes that merely happen to sit
    below a long field - "Other (specify)" slots, Number of Dependents - get
    mislabelled as continuations of it.
    """
    for f in fields:
        if f.label or f.kind != "text":
            continue
        fw = f.rect[2] - f.rect[0]
        best, bestdy = None, 1e9
        for g in fields:
            if g.page != f.page or not g.label or g.kind != "text":
                continue
            gw = g.rect[2] - g.rect[0]
            dy = f.rect[1] - g.rect[3]
            if 0 <= dy < 14 and abs(g.rect[0] - f.rect[0]) < 60 and dy < bestdy \
                    and min(fw, gw) / max(fw, gw) > 0.7:
                best, bestdy = g, dy
        if best is not None:
            f.label = f"{best.label} (cont.)"
            f.section = best.section


def slug(s: str, fallback: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or fallback


def assign_ids(fields: list[Field]) -> None:
    seen = defaultdict(int)
    for f in fields:
        # a checkbox is identified by group + option, so Yes/No pairs stay distinct
        if f.kind == "checkbox" and f.group:
            name = f"{f.group}_{f.label}"
        elif f.table:
            name = f"{f.table[:40]}_{f.label}"
        else:
            name = f.label
        base = f"p{f.page}_{f.kind[:3]}_{slug(name, 'unlabelled')}"
        seen[base] += 1
        f.id = base if seen[base] == 1 else f"{base}__{seen[base]}"


def main():
    doc = fitz.open(BLANK)
    all_fields: list[Field] = []
    for pno, page in enumerate(doc, start=1):
        all_fields.extend(build_page(page, pno, {"group": ""}))
    all_fields = merge_grid_segments(all_fields)
    fill_specify_boxes(all_fields)
    fill_continuations(all_fields)
    fill_table_columns(all_fields)
    flag_urdu_mirrors(all_fields)
    assign_ids(all_fields)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = {
        "source": os.path.basename(BLANK),
        "page_size": [round(doc[0].rect.width, 2), round(doc[0].rect.height, 2)],
        "pages": len(doc),
        "fields": [asdict(f) for f in all_fields],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)

    # ------------------------------------------------------------- summary
    by_page = defaultdict(lambda: defaultdict(int))
    for f in all_fields:
        by_page[f.page][f.kind] += 1
    kinds = ["text", "grid", "checkbox", "block"]
    print(f"wrote {OUT}")
    print("  page " + "".join(f"{k:>9}" for k in kinds) + f"{'total':>8}")
    tot = defaultdict(int)
    for p in sorted(by_page):
        k = by_page[p]
        print(f"{p:>6} " + "".join(f"{k[x]:>9}" for x in kinds) + f"{sum(k.values()):>8}")
        for kk, vv in k.items():
            tot[kk] += vv
    print(f"{'ALL':>6} " + "".join(f"{tot[x]:>9}" for x in kinds) + f"{sum(tot.values()):>8}")
    unl = [f for f in all_fields if not f.label]
    print(f"\nunlabelled: {len(unl)} / {len(all_fields)}")
    for f in unl:
        print(f"   p{f.page} {f.kind:9} {f.rect}")


if __name__ == "__main__":
    main()
