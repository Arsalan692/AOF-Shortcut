"""
Phase B - put a scanned form into template coordinates, then isolate the ink the
customer added.

Two products per page:

  registered  - the scan warped so that template PDF point (x,y) is always at
                pixel (x*S, y*S). Every rect in template_schema.json therefore
                addresses the right patch of the scan, whatever the scanner did.
  ink         - a mask of strokes present in the scan but NOT in the blank
                template, i.e. handwriting and tick marks with the printed form
                subtracted away.

Registration is feature based (ORB + RANSAC homography) against the blank template
render. It corrects rotation, scale, translation and mild perspective at once, which
is what a phone scan of a paper form actually needs.
"""
from __future__ import annotations

import json
import os

import cv2
import fitz
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLANK = os.path.join(ROOT, "Forms", "Account_Opening_Form_blank template.pdf")
DPI = 300
S = DPI / 72.0

MIN_INLIERS = 25
MATCH_DPI = 150          # features are matched at half scale, then H is rescaled;
                         # 300 dpi matching is ~6x slower for no accuracy gain
# At 300 dpi a pen stroke fragment is hundreds of pixels; anything under this is
# scanner noise or a sliver of printed text left behind by the subtraction.
MIN_STROKE_AREA = 40
MIN_INK_AREA = 60

# --- alignment acceptance -----------------------------------------------------
# A misregistered page does not produce slightly-wrong data, it silently reads ink
# from the wrong boxes, so every page has to earn its way through a quality gate.
COVER_RADIUS = 3      # px of slack when asking "is the printing underneath?"
MIN_COVER = 0.80      # share of template printing that must be found under the scan
MAX_OFFSET_PX = 4.0   # ~1pt of residual global shift
IDENT_MARGIN = 1.25   # best page match must beat the runner-up by this factor


# --------------------------------------------------------------------- helpers
def render_gray(page, dpi=DPI) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def flatten(gray: np.ndarray, k: int = 41) -> np.ndarray:
    """Divide out the illumination gradient. CamScanner output has broad grey bands
    that wreck a global threshold; this removes them without touching stroke detail."""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.divide(gray, bg, scale=255)


def drop_specks(bw: np.ndarray, min_area: int) -> np.ndarray:
    """Remove components smaller than min_area. Table-driven, so it costs one pass
    over the label image rather than one pass per component."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False                       # component 0 is the background
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def binarize(gray: np.ndarray, min_area: int = MIN_STROKE_AREA) -> np.ndarray:
    """Ink = 255, paper = 0.

    The median blur is essential, not cosmetic: these scans are grey-banded JPEGs,
    and an adaptive threshold run straight on them turns compression noise into a
    dense field of speckles that swamps the ink measurements downstream.
    """
    flat = cv2.medianBlur(flatten(gray), 3)
    bw = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 15)
    return drop_specks(bw, min_area)


def strip_rules(ink: np.ndarray, min_len: int = 80, max_thick: int = 5) -> np.ndarray:
    """Remove long, thin, straight runs from the ink layer.

    Printed rules and box borders survive the template subtraction as 1px slivers
    wherever registration is off by a pixel. A table rule 116pt wide leaves ~480 such
    pixels - more than the real handwriting in a small field - so a blank cell reads
    as filled. Handwriting contains no 80px-long run that is only a few pixels thick,
    which makes the two separable on shape alone.
    """
    out = ink
    for size, horizontal in (((min_len, 1), True), ((1, min_len), False)):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        cand = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
        if not cand.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
        w = stats[:, cv2.CC_STAT_WIDTH]
        h = stats[:, cv2.CC_STAT_HEIGHT]
        keep = ((h <= max_thick) & (w >= min_len)) if horizontal \
            else ((w <= max_thick) & (h >= min_len))
        keep[0] = False
        lines = np.where(keep[lab], 255, 0).astype(np.uint8)
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        out = cv2.bitwise_and(out, cv2.bitwise_not(lines))
    return out


def homography(tpl_gray: np.ndarray, scan_gray: np.ndarray):
    t, s = flatten(tpl_gray), flatten(scan_gray)
    orb = cv2.ORB_create(nfeatures=12000, scaleFactor=1.2, nlevels=8)
    k1, d1 = orb.detectAndCompute(t, None)
    k2, d2 = orb.detectAndCompute(s, None)
    if d1 is None or d2 is None:
        return None, 0
    good = [m for m, n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
            if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return H, int(mask.sum()) if mask is not None else 0


def alignment_quality(tpl_gray: np.ndarray, warped: np.ndarray) -> tuple[float, float]:
    """(coverage, offset_px) - how well the warped scan sits on the template.

    Coverage = share of template ink that finds scan ink within COVER_RADIUS.
    Deliberately asymmetric and position-only: plain IoU is useless here because a
    crisp vector render and a blotchy B&W scan never overlap well even when perfectly
    aligned - it reads 0.18 on a page whose true residual is 0.15px. Inlier count is
    no good either; it says how confident the fit was, not how correct (page 2 of the
    sample scan has just 29 inliers and aligns to 1.0px).
    """
    t = binarize(tpl_gray, min_area=1)
    s = binarize(warped, min_area=1)
    near = cv2.dilate(s, np.ones((2 * COVER_RADIUS + 1,) * 2, np.uint8))
    cover = np.count_nonzero(cv2.bitwise_and(t, near)) / max(np.count_nonzero(t), 1)
    (ox, oy), _ = cv2.phaseCorrelate(flatten(tpl_gray).astype(np.float64),
                                     flatten(warped).astype(np.float64))
    return float(cover), float(np.hypot(ox, oy))


def orb_features(gray: np.ndarray, nfeatures=12000):
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8)
    return orb.detectAndCompute(flatten(gray), None)


def fit(tpl_kd, scan_kd):
    """Homography mapping scan -> template, with its RANSAC inlier count.

    The inlier count is a *geometrically verified* score. Raw descriptor-match counts
    are not usable for telling pages apart: two dense pages of the same form share
    plenty of similar-looking text features, and scoring page 2 of this scan that way
    picked template page 6. Requiring the matches to agree on one transform removes
    almost all of that confusion.
    """
    k1, d1 = tpl_kd
    k2, d2 = scan_kd
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, 0
    good = [m for m, n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, len(good)
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return H, int(mask.sum()) if mask is not None else 0


def resolve_template_page(tpl_kds, scan_kd, prefer: int):
    """Which template page is this scan page? Returns (index, inliers, searched).

    Page order is trusted first and only checked - a scan is normally in order, and
    verifying one candidate costs one fit instead of nine. The full search runs only
    when the expected page fails verification, which is exactly the shuffled / missing
    / rotated-page case that needs it.
    """
    H, inl = fit(tpl_kds[prefer], scan_kd)
    if H is not None and inl >= MIN_INLIERS:
        return prefer, inl, False

    best, best_inl = prefer, inl
    for ti in range(len(tpl_kds)):
        if ti == prefer:
            continue
        _, cand = fit(tpl_kds[ti], scan_kd)
        if cand > best_inl:
            best, best_inl = ti, cand
    return best, best_inl, True


def upscale_H(H: np.ndarray) -> np.ndarray:
    """Lift a homography fitted at MATCH_DPI to full DPI: scale(1/r) . H . scale(r)."""
    r = MATCH_DPI / DPI
    up = np.array([[1 / r, 0, 0], [0, 1 / r, 0], [0, 0, 1]], np.float64)
    down = np.array([[r, 0, 0], [0, r, 0], [0, 0, 1]], np.float64)
    return up @ H @ down


def register_page(tpl_page, scan_page, H=None, inl=0):
    """-> (registered_gray, ink_mask, inliers). registered_gray is in template space.

    H may be passed in already fitted at MATCH_DPI - the caller normally has it from
    page resolution, so re-fitting here would just repeat that work.
    """
    tpl = render_gray(tpl_page)
    scan = render_gray(scan_page)
    h, w = tpl.shape

    if H is None:
        H, inl = fit(orb_features(render_gray(tpl_page, MATCH_DPI)),
                     orb_features(render_gray(scan_page, MATCH_DPI)))
    if H is not None:
        H = upscale_H(H)

    if H is None:
        # fall back to a plain resize; the caller sees inliers==0 and can flag it
        reg = cv2.resize(scan, (w, h), interpolation=cv2.INTER_AREA)
    else:
        reg = cv2.warpPerspective(scan, H, (w, h), flags=cv2.INTER_LINEAR,
                                  borderValue=255)

    # keep the template layer complete (min_area 1) - every printed mark must be
    # available to subtract, including the small ones
    tpl_ink = binarize(tpl, min_area=1)
    scan_ink = binarize(reg)
    # dilate the printed layer so that residual misalignment of a printed rule does
    # not survive the subtraction and masquerade as a pen stroke
    chrome = cv2.dilate(tpl_ink, np.ones((7, 7), np.uint8), iterations=1)
    ink = cv2.bitwise_and(scan_ink, cv2.bitwise_not(chrome))
    ink = strip_rules(ink)
    # a real stroke survives subtraction as a large blob; printed glyphs survive as
    # thin slivers, so a second, stricter area filter separates them
    return reg, drop_specks(ink, MIN_INK_AREA), inl


def run(scan_pdf: str, outdir: str, identify: bool = True,
        max_pages: int | None = None, pages: list[int] | None = None,
        progress=None) -> list[dict]:
    """pages: the 1-based scan pages to process, in any order - [2] reads page 2 alone.
    max_pages: the older "first N pages" form, kept because it is the simpler ask.
    Pass one or the other; `pages` wins if both are given.
    progress: optional callable(done, total, message) for UI reporting.

    A selection is not the same as a prefix. Page identification still uses each page's
    real position in the scan to decide which template page to try first, so asking for
    page 2 alone matches it against template page 2 rather than page 1.
    """
    os.makedirs(outdir, exist_ok=True)
    tpl_doc, scan_doc = fitz.open(BLANK), fitz.open(scan_pdf)

    # template features are reused for every scan page, so compute them once
    tpl_kds = [orb_features(render_gray(p, MATCH_DPI)) for p in tpl_doc]

    if pages:
        want = sorted({p for p in pages if 1 <= p <= len(scan_doc)})
        if not want:
            raise ValueError(f"none of pages {sorted(set(pages))} exist in a "
                             f"{len(scan_doc)}-page document")
    elif max_pages is not None:
        want = list(range(1, min(len(scan_doc), max(1, max_pages)) + 1))
    else:
        want = list(range(1, len(scan_doc) + 1))
    n_scan = len(want)

    claimed: dict[int, int] = {}
    report = []
    for done, page_no in enumerate(want):
        si = page_no - 1
        if progress:
            progress(done, n_scan, f"Aligning page {page_no}"
                                   f"{f' ({done + 1} of {n_scan})' if n_scan > 1 else ''}")
        scan_kd = orb_features(render_gray(scan_doc[si], MATCH_DPI))
        prefer = min(si, len(tpl_kds) - 1)
        if identify:
            ti, inl, searched = resolve_template_page(tpl_kds, scan_kd, prefer)
        else:
            _, inl = fit(tpl_kds[prefer], scan_kd)
            ti, searched = prefer, False
        tp = ti + 1

        H, inl = fit(tpl_kds[ti], scan_kd)
        reg, ink, inl = register_page(tpl_doc[ti], scan_doc[si], H, inl)
        cover, off = alignment_quality(render_gray(tpl_doc[ti]), reg)

        problems = []
        if inl < MIN_INLIERS:
            problems.append(f"few_inliers({inl})")
        if cover < MIN_COVER:
            problems.append(f"low_coverage({cover:.2f})")
        if off > MAX_OFFSET_PX:
            problems.append(f"offset({off:.1f}px)")
        if searched:
            problems.append("expected_page_failed_verification")
        if tp in claimed:
            problems.append(f"duplicate_of_scan_page({claimed[tp] + 1})")
        else:
            claimed[tp] = si
        if tp != si + 1:
            problems.append(f"out_of_order(scan p{si + 1} -> template p{tp})")

        # Files are named by TEMPLATE page, because the field schema is keyed that
        # way. A shuffled scan therefore lands in the right place automatically.
        cv2.imwrite(os.path.join(outdir, f"reg_p{tp}.png"), reg)
        cv2.imwrite(os.path.join(outdir, f"ink_p{tp}.png"), ink)

        rec = {"scan_page": si + 1, "template_page": tp, "inliers": inl,
               "coverage": round(cover, 3), "offset_px": round(off, 2),
               "searched": searched,
               "ok": not problems, "problems": problems}
        report.append(rec)
        print(f"  scan p{si + 1} -> tpl p{tp}: inliers={inl:<4} cover={cover:.3f} "
              f"off={off:.2f}px  {'OK' if not problems else 'REVIEW: ' + ', '.join(problems)}")

    # Only complain about template pages the run was actually asked to cover - the rest
    # are absent on purpose, which is the whole point of selecting pages.
    expected = {p for p in want if p <= len(tpl_doc)}
    missing = sorted(expected - set(claimed))
    if missing:
        print(f"  template pages with no scan: {missing}")
    if progress:
        progress(n_scan, n_scan, "Alignment complete")
    with open(os.path.join(outdir, "alignment.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": os.path.basename(scan_pdf), "pages": report,
                   "scanned_pages": n_scan, "requested_pages": want,
                   "missing_template_pages": missing}, fh, indent=1)
    return report


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Forms", "Cif-form.pdf")
    # any further arguments are the pages to register: `register.py form.pdf 2` does p2 only
    sel = [int(a) for a in sys.argv[2:] if a.isdigit()] or None
    name = os.path.splitext(os.path.basename(pdf))[0]
    out = os.path.join(ROOT, "build", "registered", name)
    print(f"registering {os.path.basename(pdf)}"
          f"{f' pages {sel}' if sel else ' (all pages)'} -> {out}")
    run(pdf, out, pages=sel)
