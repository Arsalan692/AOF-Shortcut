"""
Phase C3 - read a whole page of handwriting in one call to a LOCAL vision model (Ollama).

Nothing leaves the machine: the model runs at http://localhost:11434.

One image, one call, one page. The alternative - cutting every field out and sending the
crops in batches - cost ~50 calls for a nine-page form, because on CPU a distinct image
costs roughly the same however small it is. Whole pages cost 6 calls for the same form.

What makes a whole page readable at all is that we already know, exactly, where every
answer box is: the blank template is a vector PDF, so each field's rectangle is known
before the scan is ever opened. The page sent to the model therefore carries a numbered
marker on every box, and the prompt is a numbered list of those same boxes with the format
each one must hold (see prompts.py). The model is never asked to work out which value
belongs to which label, which is the thing whole-page OCR gets wrong.

Two passes:

  page pass   every field with ink on it, numbered, in one image
  re-read     only the fields whose value failed its format check, cut out and stacked
              at ~4x the height they had on the page. Usually a handful of fields and one
              extra call, and it is where the digits in a CNIC or a phone number are won.

Results are cached per field, so an interrupted run resumes instead of restarting.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

import fields as fieldmod
from prompts import ink_verdict, page_prompt, retry_prompt
from validate import OCR_TYPES, assess, infer_type

# field labels carry non-ASCII typography; never let printing them kill a long run
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# One model reads the pages. A page mixes names, addresses, dates and account numbers, so
# it has to be the general VLM: an OCR model has no language prior and loses the cursive.
MODEL = os.environ.get("HBL_VLM", "qwen2.5vl:7b")
# The re-read is the one place where a second reader still pays. Measured on page 1, the
# 1.1B OCR model beat the 8.3B VLM on every structured field (16/16 vs 14/16) precisely
# because it has no vocabulary to fall back on: a CNIC is not a word, so a prior that
# recovers "Fountain" from a scrawl is the same prior that turns 35810 into 35820. By the
# re-read the field is alone in its own strip, which is the situation that model handles.
MODEL_NUM = os.environ.get("HBL_VLM_NUM", "glm-ocr:latest")

# Ceiling on the page image. Qwen2.5-VL caps its input pixels and silently downscales
# anything larger, which would shrink the handwriting without saying so - better to do it
# here, where the marker text is drawn afterwards and therefore stays legible.
# Measured on page 1, trimmed: 2.0 Mpx scored 93% in 1139s, 1.2 Mpx 89% in 567s - and
# 1.2 Mpx with the structured fields verified by the OCR reader scored 98% in 635+379s.
# Paying for pixels across the whole page to protect a dozen digits is the worse deal;
# the vision encoder's cost grows faster than linearly, so halving the image halves the
# call while the sweep buys the digits back for a fraction of it.
PAGE_MAX_PX = int(os.environ.get("HBL_PAGE_PX", 1_200_000))
TRIM_MARGIN = 18.0        # pt of page kept around the outermost field being read
# Ollama's default 4096-token window truncates a page prompt, and a truncated reply loses
# the fields at the bottom of the page. The window is therefore set per call, from what the
# call actually needs: over-allocating is not free either, since llama.cpp reserves the KV
# cache for the whole window up front and this machine has no memory to spare.
NUM_CTX = int(os.environ.get("HBL_NUM_CTX", 0))       # 0 = size it per call
NUM_CTX_MIN, NUM_CTX_MAX = 4096, 32768
TOK_PER_CHAR = 1 / 3.5    # rough, and rounded up to a 1024 boundary afterwards
TIMEOUT = 3600

SWEEP = os.environ.get("HBL_SWEEP", "1") != "0"
SWEEP_BATCH = 6       # structured fields per verification image
RETRY_BATCH = 6       # fields per re-read image...
RETRY_ROW_H = 168     # ...each given this much height...
RETRY_MAX_W = 2000    # ...and this much width, which is what the long fields need: an
                      # address crop is ~4400x224px, so width binds first
RETRY_MAX_PX = 1_400_000

# --- marker drawing ---------------------------------------------------------
# Blue outline, red number tag. Both are colours no pen on these forms uses, so the model
# can be told to ignore them without being able to confuse them for ink.
BOX_BGR = (200, 60, 0)
TAG_BGR = (0, 0, 210)
TAG_TEXT = (255, 255, 255)
BOX_ALPHA = 0.55      # the outline is blended, so a stroke crossing it stays visible
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.42
FONT_THICK = 1
TAG_PAD = 3


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _tag_spot(x0, y0, x1, y1, tw, th, shape, taken) -> tuple[int, int]:
    """Where to put a number tag for the box (x0,y0,x1,y1). Returns its top-left.

    Left of the box first, which lands in the printed label beside it. Covering printed
    text costs nothing here - the prompt already carries every label, so the model never
    needs to read one off the page - whereas covering handwriting would destroy the answer
    the tag is pointing at. Above-left is the fallback when the box starts at the page
    margin, and a tag that would land on another tag is stepped further out.
    """
    h, w = shape[:2]
    cy = (y0 + y1) // 2
    spots = [(x0 - tw - 2, cy - th // 2),          # beside it, in the label column
             (x0, y0 - th - 2),                    # just above it
             (x0 - tw - 2, y0 - th - 2),           # above and out
             (x1 + 2, cy - th // 2)]               # after it, for boxes hard against the margin
    for sx, sy in spots:
        sx = max(1, min(sx, w - tw - 1))
        sy = max(1, min(sy, h - th - 1))
        rect = (sx, sy, sx + tw, sy + th)
        if not any(_overlaps(rect, t) for t in taken):
            return sx, sy
    return max(1, min(x0 - tw - 2, w - tw - 1)), max(1, min(cy - th // 2, h - th - 1))


def trim_box(reg: np.ndarray, items: list[dict], margin: float = TRIM_MARGIN):
    """The part of the page worth sending: the fields being read, plus a margin.

    Every pixel costs. A page's answer boxes never reach its edges - the title block at the
    top of page 1 and the whole lower half of page 4's mostly-blank tables carry nothing
    this call is asking about - and trimming that away buys either a cheaper call or a
    sharper one at the same price. The margin is generous because handwriting overflows,
    and the extents here are already grown to the ink.
    """
    h, w = reg.shape[:2]
    rects = [it.get("crop_rect") or it["rect"] for it in items]
    x0 = max(0, int((min(r[0] for r in rects) - margin) * fieldmod.S))
    y0 = max(0, int((min(r[1] for r in rects) - margin) * fieldmod.S))
    x1 = min(w, int((max(r[2] for r in rects) + margin) * fieldmod.S))
    y1 = min(h, int((max(r[3] for r in rects) + margin) * fieldmod.S))
    if x1 - x0 < 64 or y1 - y0 < 64:        # degenerate: keep the whole page
        return 0, 0, w, h
    return x0, y0, x1, y1


def marked_page(reg: np.ndarray, items: list[dict],
                max_px: int = PAGE_MAX_PX) -> np.ndarray:
    """The page region holding the fields to read, scaled to fit the model's pixel budget,
    with every box outlined and numbered 1..N in the order the prompt lists them.

    The image is trimmed and scaled BEFORE the markers are drawn, so the tags are sized in
    final pixels and cannot be shrunk into illegibility by a later resize.
    """
    ox, oy, x1, y1 = trim_box(reg, items)
    reg = reg[oy:y1, ox:x1]
    h, w = reg.shape[:2]
    f = min(1.0, (max_px / float(h * w)) ** 0.5)
    if f < 1.0:
        reg = cv2.resize(reg, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(reg, cv2.COLOR_GRAY2BGR)

    def to_px(v_pt: float, origin: int) -> int:
        """PDF point -> pixel in the trimmed, scaled image."""
        return int((v_pt * fieldmod.S - origin) * f)

    boxes = []
    for it in items:
        r = it.get("crop_rect") or it["rect"]
        boxes.append((max(0, to_px(r[0], ox)), max(0, to_px(r[1], oy)),
                      min(img.shape[1] - 1, to_px(r[2], ox)),
                      min(img.shape[0] - 1, to_px(r[3], oy))))

    overlay = img.copy()
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), BOX_BGR, 1, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, BOX_ALPHA, img, 1 - BOX_ALPHA, 0)

    taken: list[tuple] = []
    for n, (x0, y0, x1, y1) in enumerate(boxes, 1):
        text = str(n)
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
        bw, bh = tw + TAG_PAD * 2, th + TAG_PAD * 2
        sx, sy = _tag_spot(x0, y0, x1, y1, bw, bh, img.shape, taken)
        taken.append((sx, sy, sx + bw, sy + bh))
        cv2.rectangle(img, (sx, sy), (sx + bw, sy + bh), TAG_BGR, -1)
        cv2.putText(img, text, (sx + TAG_PAD, sy + bh - TAG_PAD - 1),
                    FONT, FONT_SCALE, TAG_TEXT, FONT_THICK, cv2.LINE_AA)
    return img


def stacked(images: list[np.ndarray], row_h: int, max_w: int,
            max_px: int = RETRY_MAX_PX) -> np.ndarray:
    """Numbered strips stacked into one image, for the re-read and verification passes.

    The canvas is only as wide as the widest strip actually in it, not a fixed max_w. A
    batch of dates and branch codes is a few hundred pixels wide; padding each row out to
    2000 made four fifths of the image white, and white costs exactly as many tokens as
    handwriting does.

    Note this is not the packing change the README warns off: every strip keeps its own
    scale and its own row, and nothing is repacked to fit more per image. Only the dead
    margin to the right of the strips goes.
    """
    num_w = 74
    scaled = []
    for im in images:
        h, w = im.shape[:2]
        s = min(row_h / h, max_w / w)
        if abs(s - 1.0) > 0.01:
            im = cv2.resize(im, None, fx=s, fy=s,
                            interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        scaled.append(im)
    width = max(im.shape[1] for im in scaled)

    tiles = []
    for n, im in enumerate(scaled, 1):
        tile = np.full((im.shape[0], num_w + width), 255, np.uint8)
        tile[:, num_w:num_w + im.shape[1]] = im
        cv2.putText(tile, str(n), (8, im.shape[0] // 2 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, 0, 2, cv2.LINE_AA)
        tiles.append(tile)
        tiles.append(np.zeros((3, num_w + width), np.uint8))
    sheet = np.vstack(tiles)
    px = sheet.shape[0] * sheet.shape[1]
    if px > max_px:
        s = (max_px / px) ** 0.5
        sheet = cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return sheet


# --------------------------------------------------------------------- ollama
def keys_schema(n: int) -> dict:
    """A JSON schema requiring exactly keys "1".."n", all strings.

    Necessary, not decorative. With a bare format:"json" the model is free to emit any
    valid object, and it reliably stopped after a couple of keys, mirroring the shape of
    the example in the prompt. Requiring the keys makes a short reply impossible - which
    matters far more now that one reply carries a whole page.
    """
    keys = [str(i) for i in range(1, n + 1)]
    return {"type": "object",
            "properties": {k: {"type": "string"} for k in keys},
            "required": keys}


def context_for(prompt: str, img: np.ndarray, num_predict: int) -> int:
    """A context window big enough for this call and no bigger.

    Qwen2.5-VL turns each 28x28 block of pixels into one token, so an image's cost is
    predictable from its size. A quarter of headroom covers the estimate being rough.
    """
    visual = (img.shape[0] * img.shape[1]) / (28 * 28)
    need = (len(prompt) * TOK_PER_CHAR + visual + num_predict) * 1.25
    return max(NUM_CTX_MIN, min(NUM_CTX_MAX, int((need // 1024 + 1) * 1024)))


def generate(prompt: str, img: np.ndarray, fmt, num_predict: int,
             model: str) -> tuple[str, str]:
    """-> (response text, a one-line cost breakdown).

    The breakdown is worth having on every call. On CPU the two halves behave completely
    differently - prefill scales with the image and the prompt, generation with how many
    fields the page holds - and only splitting them says which one to attack when a page
    reads too slowly.
    """
    body = {"model": model, "prompt": prompt, "images": [png_b64(img)],
            "stream": False, "keep_alive": "1h",     # never reload 6GB between pages
            "options": {"temperature": 0, "num_predict": num_predict,
                        "num_ctx": NUM_CTX or context_for(prompt, img, num_predict)}}
    if fmt:
        body["format"] = fmt
    req = urllib.request.Request(f"{OLLAMA}/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.loads(r.read())

    def secs(k):
        return d.get(k, 0) / 1e9

    pin, pout = d.get("prompt_eval_count", 0), d.get("eval_count", 0)
    ps, es = secs("prompt_eval_duration"), secs("eval_duration")
    cost = (f"prefill {pin} tok in {ps:.0f}s ({pin / ps:.0f}/s), "
            f"output {pout} tok in {es:.0f}s ({pout / max(es, 0.001):.1f}/s)"
            if pin and ps else f"total {secs('total_duration'):.0f}s")
    return d.get("response", ""), cost


def first_line(v: str) -> str:
    """glm-ocr answers correctly and then keeps talking - '1023\\n```markdown\\n1023'.
    The first non-empty line is the value. Scoped to the OCR reader: qwen returns clean
    strings, and a blanket first-line rule would silently truncate anything multi-line."""
    for ln in str(v).splitlines():
        ln = ln.strip().strip("`").strip()
        if ln:
            return ln
    return ""


def _parse(raw: str, items: list[dict], model: str) -> dict[str, str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for i, it in enumerate(items, 1):
        v = data.get(str(i), data.get(i))
        if isinstance(v, (str, int, float)):
            out[it["field"]] = first_line(v) if model == MODEL_NUM else str(v)
    return out


def read_one_page(pg, items: list[dict], page_no: int,
                  model: str = "") -> tuple[dict[str, str], str]:
    """Send one marked-up page and return ({field id: value}, cost breakdown)."""
    model = model or MODEL
    img = marked_page(pg.reg, items)
    # Generous, because running out mid-reply is not a partial result: the schema requires
    # every key, so a truncated object fails to parse and the whole page is lost. An
    # unused allowance costs nothing.
    raw, cost = generate(page_prompt(page_no, items), img, keys_schema(len(items)),
                         200 + 64 * len(items), model)
    return _parse(raw, items, model), cost


def read_retry(pg, group: list[tuple[dict, str]], model: str) -> tuple[dict[str, str], str]:
    """Re-read a handful of failed fields from their own enlarged crops."""
    items = [r for r, _ in group]
    img = stacked([pg.crop(r) for r in items], RETRY_ROW_H, RETRY_MAX_W)
    raw, cost = generate(retry_prompt(items, {r["field"]: why for r, why in group}),
                         img, keys_schema(len(items)), 80 + 40 * len(items), model)
    return _parse(raw, items, model), cost


def available_models() -> set[str]:
    """What Ollama actually has pulled. Returns an empty set if it cannot be asked, which
    the caller reads as "assume it is there" - a health probe must not abort a real run."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=15) as r:
            return {m["name"] for m in json.loads(r.read()).get("models", [])}
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return set()


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def chunk(seq: list, size: int) -> list[list]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


# ----------------------------------------------------------------------- main
def run(doc: str, pages=None, retry: bool = True, force: bool = False,
        save_images: bool = False, progress=None, sweep: bool = SWEEP):
    """Read every requested page of `doc`. progress: callable(done, total, message)."""
    index = fieldmod.load_index(doc)
    cache_path = os.path.join(ROOT, "build", "reads", f"{doc}.json")
    cache = load_cache(cache_path)
    if force:
        # --force means "ignore what you cached for the pages I am asking you to read", and
        # nothing more. Emptying the whole cache instead destroys the pages this run is not
        # touching: `--pages 4 --force` silently erased pages 1, 2, 3 and 9 and cost an hour
        # of re-reading, because the cache is then saved back over the file.
        page_of = {r["field"]: r["page"] for r in index}
        cache = {fid: v for fid, v in cache.items()
                 if pages is not None and page_of.get(fid) not in pages}

    # A field with no ink at all is settled without spending a call on it, and is left out
    # of the page image so the model is never shown a box it could invent a value into.
    for r in index:
        if not r["needs_model"] and r["field"] not in cache:
            cache[r["field"]] = {"value": "", "source": "ink-gate", "seconds": 0.0}

    todo = [r for r in index
            if r["needs_model"] and (pages is None or r["page"] in pages)
            and r["field"] not in cache]
    by_page: dict[int, list] = {}
    for r in todo:
        by_page.setdefault(r["page"], []).append(r)

    avail = available_models()
    if avail and not any(m.split(":")[0] == MODEL.split(":")[0] for m in avail):
        raise RuntimeError(f"{MODEL} is not pulled in Ollama at {OLLAMA}")

    print(f"{len(todo)} fields with ink across {len(by_page)} page(s) at {OLLAMA}")
    t0 = time.time()
    done = 0
    for pno in sorted(by_page):
        items = by_page[pno]
        pg = fieldmod.load_page(doc, pno)
        if pg is None:
            print(f"  page {pno}: not registered, skipped")
            continue
        if progress:
            progress(done, len(todo), f"Reading page {pno} ({len(items)} fields)")
        if save_images:
            qa = os.path.join(ROOT, "build", "qa", f"marked_{doc}_p{pno}.png")
            os.makedirs(os.path.dirname(qa), exist_ok=True)
            cv2.imwrite(qa, marked_page(pg.reg, items))
            print(f"  page {pno}: marked image -> {os.path.relpath(qa, ROOT)}")

        t = time.time()
        cost = ""
        try:
            got, cost = read_one_page(pg, items, pno)
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            print(f"  page {pno} FAILED: {e}")
            got = {}
        dt = time.time() - t
        for it in items:
            if it["field"] in got:
                cache[it["field"]] = {"value": got[it["field"]], "source": "page",
                                      "model": MODEL,
                                      "seconds": round(dt / max(len(items), 1), 2)}
        miss = [it for it in items if it["field"] not in got]
        done += len(items)
        print(f"  page {pno}: {len(items) - len(miss)}/{len(items)} fields in {dt:.0f}s "
              f"({dt / max(len(items), 1):.1f}s/field)"
              f"{f' | {len(miss)} unparsed' if miss else ''}")
        if cost:
            print(f"    {cost}")
        sys.stdout.flush()
        save_cache(cache_path, cache)

        # The sweep runs first so that the re-read below sees corrected values and only
        # chases what is still wrong.
        if sweep:
            _sweep_structured(pg, items, cache, avail, progress, done, len(todo))
            save_cache(cache_path, cache)
        if retry:
            _retry_page(doc, pg, items, cache, avail, progress, done, len(todo))
            save_cache(cache_path, cache)

    if progress:
        progress(len(todo), len(todo), "Handwriting read")
    print(f"\ntotal {time.time() - t0:.0f}s -> {os.path.relpath(cache_path, ROOT)}")
    return cache, {r["field"]: r for r in index}


def _sweep_structured(pg, items, cache, avail, progress, done, total) -> None:
    """Second-opinion every structured field on this page with the OCR reader.

    A page read cheaply enough to be practical is read at a resolution where digits get
    hurt: at 1.2 Mpx page 1 came back with a branch code of 1093 for 1023 and a customer
    number of 395678 for 345678. Nothing can catch those. They are the right length, the
    right character class, and a valid date is still a valid date - so every format rule
    passes them and the reviewer sees nothing to look at.

    The fix is not more pixels for the whole page, it is more pixels for the fields that
    need them. Measured on page 1 (see README), the 1.1B OCR model reads 16/16 structured
    fields against the VLM's 14/16, precisely because it has no vocabulary to fall back on
    and simply copies characters. So every CNIC, date, phone, IBAN and amount is read a
    second time from its own crop by that model, and its answer wins where it is well
    formed. It is affordable for the same reason it is accurate: 1.1B parameters, ~45 tok/s
    of prefill here against the VLM's 12, so a whole page of structured fields costs less
    than a minute.
    """
    if not avail or not any(m.split(":")[0] == MODEL_NUM.split(":")[0] for m in avail):
        return
    todo = [r for r in items
            if infer_type(r) in OCR_TYPES and ink_verdict(r) != "blank"
            and (cache.get(r["field"]) or {}).get("source") == "page"]
    if not todo:
        return

    print(f"  verifying {len(todo)} structured field(s) on page {pg.page} with {MODEL_NUM}:")
    for gi, group in enumerate(chunk(todo, SWEEP_BATCH), 1):
        if progress:
            progress(done, total, f"Double-checking numbers on page {pg.page}")
        t = time.time()
        try:
            got, cost = read_retry(pg, [(r, "read again from its own image, character by "
                                            "character") for r in group], MODEL_NUM)
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            print(f"    sweep {gi} FAILED: {e}")
            continue
        for r in group:
            v = got.get(r["field"])
            if v is None:
                continue
            was = (cache.get(r["field"]) or {}).get("value", "")
            a, b = assess(v, r), assess(was, r)
            # Only overrule the page on a well-formed reading, and never trade a real value
            # for a blank one: an OCR model that sees nothing is likelier to be wrong about
            # a filled box than the VLM was, which is the one thing it is worse at.
            if not a["valid"] or (a["empty"] and not b["empty"]):
                continue
            if str(a["value"] or "") != str(b["value"] or ""):
                print(f"    {r['label'][:32]:<34} {was!r} -> {v!r}")
                # Keep what the page said. Both readings are well formed - that is why the
                # disagreement survived the format rules - so nothing downstream can prove
                # which is right, and the honest thing is to record that two readers looked
                # at this box and saw different digits. extract.py turns it into a review
                # flag, which is the only way these ever reach a human: a wrong CNIC that
                # is still a valid CNIC is invisible to every check we have.
                cache[r["field"]] = {"value": v, "source": "ocr-sweep",
                                     "model": MODEL_NUM, "was": was,
                                     "disputed": bool(was) and not b["empty"],
                                     "seconds": round((time.time() - t) / len(group), 2)}
        print(f"    sweep {gi}/{len(chunk(todo, SWEEP_BATCH))}: {time.time() - t:.0f}s"
              f"{f' | {cost}' if cost else ''}")
        sys.stdout.flush()


def _retry_page(doc, pg, items, cache, avail, progress, done, total) -> None:
    """Re-read the fields on this page whose first value cannot be right.

    Two kinds of "cannot be right". A value that fails its format check is the obvious one.
    The other is a value that contradicts the pixels: reading a whole page in one pass, the
    model skims, and it answered EMPTY for a Passport box holding 7,961px of pen while
    copying a ticked option's label into an "Other" box holding 186px of a neighbour's
    bleed. Both are individually well-formed, so a format check can never catch either -
    but the ink measurement disagrees with both, and disagreement is exactly what the
    re-read exists to settle. On its own, enlarged, the field is a much easier read.
    """
    bad = []
    for r in items:
        c = cache.get(r["field"])
        if c is None:
            bad.append((r, "nothing was returned"))
        elif c["source"] == "page":
            a = assess(c["value"], r)
            verdict = ink_verdict(r)
            if not a["valid"]:
                bad.append((r, a["note"]))
            elif verdict == "filled" and a["empty"]:
                bad.append((r, "the scan shows clear ink in this box but nothing was read"))
            elif verdict == "blank" and not a["empty"]:
                bad.append((r, "the scan shows no ink of its own in this box, yet a value "
                               "was read - it may have been copied from a neighbour"))
    if not bad:
        return

    # A structured field goes back to the OCR reader if it is installed, because that is the
    # measured split: no language prior, no invented digit. Everything else stays with the
    # VLM that read the page.
    ocr_ok = (not avail) or any(m.split(":")[0] == MODEL_NUM.split(":")[0] for m in avail)
    per_model: dict[str, list] = {}
    for r, why in bad:
        m = MODEL_NUM if (ocr_ok and infer_type(r) in OCR_TYPES) else MODEL
        per_model.setdefault(m, []).append((r, why))

    groups = [(m, g) for m, gs in per_model.items() for g in chunk(gs, RETRY_BATCH)]
    print(f"  re-reading {len(bad)} field(s) on page {pg.page} in {len(groups)} batch(es):")
    for gi, (model, group) in enumerate(groups, 1):
        if progress:
            progress(done, total,
                     f"Re-checking {len(bad)} uncertain field(s) on page {pg.page}")
        t = time.time()
        try:
            got, cost = read_retry(pg, group, model)
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            print(f"    re-read {gi}/{len(groups)} FAILED: {e}")
            continue
        dt = time.time() - t
        per = round(dt / max(len(group), 1), 2)
        for r, why in group:
            v = got.get(r["field"])
            if v is None:
                print(f"    {r['label'][:32]:<34} ({why}) -> nothing returned, keeping "
                      f"the page value")
                continue
            # A second reading only wins if it is well-formed: never trade a valid value
            # for an invalid one just because it came later.
            keep = assess(v, r)["valid"] or cache.get(r["field"]) is None
            print(f"    {r['label'][:32]:<34} ({why}) -> {v!r} "
                  f"{'accepted' if keep else 'rejected, keeping the page value'}")
            if keep:
                cache[r["field"]] = {"value": v, "source": "retry", "model": model,
                                     "seconds": per}
        print(f"    re-read {gi}/{len(groups)} with {model}: {dt:.0f}s ({per:.1f}s/field)"
              f"{f' | {cost}' if cost else ''}")
        sys.stdout.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Read a scanned form one page per model call")
    ap.add_argument("doc", nargs="?", default="Cif-form")
    ap.add_argument("--pages", type=int, nargs="*")
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the OCR second opinion on structured fields")
    ap.add_argument("--force", action="store_true", help="ignore cached reads")
    ap.add_argument("--save-images", action="store_true",
                    help="also write each marked page to build/qa/ for inspection")
    a = ap.parse_args()
    run(a.doc, a.pages, not a.no_retry, a.force, a.save_images,
        sweep=SWEEP and not a.no_sweep)
