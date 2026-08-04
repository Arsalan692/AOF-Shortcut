"""
Report what the scanner actually did to each page, and how well we undid it.

Decomposes each page's homography into rotation / scale / shear / perspective, then
measures how well the warped scan lines up with the template afterwards. The residual
is the number that matters: inlier counts say how confident the fit was, not how right
it is.
"""
from __future__ import annotations

import os
import sys

import cv2
import fitz
import numpy as np

from register import (BLANK, DPI, MATCH_DPI, MIN_INLIERS, binarize, flatten,
                      homography, render_gray)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def decompose(H: np.ndarray) -> dict:
    """Pull human-readable scanner distortions out of a homography."""
    A = H[:2, :2]
    # polar decomposition: A = R @ S, R rotation, S symmetric (scale + shear)
    U, sv, Vt = np.linalg.svd(A)
    R = U @ Vt
    angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return {
        "rotation_deg": angle,
        "scale": float(np.sqrt(sv[0] * sv[1])),
        "aspect": float(sv[0] / sv[1]),          # 1.0 = no anisotropic stretch
        "shear": float(abs(sv[0] - sv[1]) / sv[1]),
        "dx": float(H[0, 2] / (DPI / 72.0)),     # in PDF points
        "dy": float(H[1, 2] / (DPI / 72.0)),
        "persp": float(max(abs(H[2, 0]), abs(H[2, 1])) * 1e4),
    }


COVER_RADIUS = 3     # px of slack allowed when asking "is the printing on top of it"
MIN_COVER = 0.80     # share of template printing that must be found under the scan
MAX_OFFSET_PX = 4.0  # ~1pt of residual global shift


def residual(tpl_gray: np.ndarray, warped: np.ndarray) -> tuple[float, float]:
    """(coverage, offset_px) between template printing and the warped scan.

    Coverage = the share of template ink that finds scan ink within a few pixels.
    Deliberately asymmetric: plain IoU is useless here because a crisp vector render
    and a blotchy B&W scan never overlap well even when perfectly aligned - it reads
    0.18 on a page whose true offset is 0.15px. Coverage compares position only, not
    stroke weight, which is the thing we actually need to know.
    """
    t = binarize(tpl_gray, min_area=1)
    s = binarize(warped, min_area=1)
    near = cv2.dilate(s, np.ones((2 * COVER_RADIUS + 1,) * 2, np.uint8))
    tpl_px = np.count_nonzero(t)
    cover = np.count_nonzero(cv2.bitwise_and(t, near)) / max(tpl_px, 1)
    (ox, oy), _ = cv2.phaseCorrelate(flatten(tpl_gray).astype(np.float64),
                                     flatten(warped).astype(np.float64))
    return cover, float(np.hypot(ox, oy))


def main(scan_pdf: str):
    tpl_doc, scan_doc = fitz.open(BLANK), fitz.open(scan_pdf)
    n = min(len(tpl_doc), len(scan_doc))
    r = MATCH_DPI / DPI
    up = np.array([[1 / r, 0, 0], [0, 1 / r, 0], [0, 0, 1]], np.float64)
    down = np.array([[r, 0, 0], [0, r, 0], [0, 0, 1]], np.float64)

    print(f"{os.path.basename(scan_pdf)}  ({n} pages, matched at {MATCH_DPI} dpi)\n")
    print(f"{'pg':>3} {'inl':>5} {'rot deg':>9} {'scale':>7} {'aspect':>7} "
          f"{'shear':>7} {'dx pt':>7} {'dy pt':>7} {'persp':>7} {'cover':>6} {'off px':>7}")
    bad = []
    for i in range(n):
        tpl = render_gray(tpl_doc[i])
        H, inl = homography(render_gray(tpl_doc[i], MATCH_DPI),
                           render_gray(scan_doc[i], MATCH_DPI))
        if H is None:
            print(f"{i+1:>3} {inl:>5}   FAILED - not enough matches")
            bad.append(i + 1)
            continue
        H = up @ H @ down
        d = decompose(H)
        warped = cv2.warpPerspective(render_gray(scan_doc[i]), H,
                                     (tpl.shape[1], tpl.shape[0]), borderValue=255)
        cover, off = residual(tpl, warped)
        flag = "" if (inl >= MIN_INLIERS and cover >= MIN_COVER
                      and off <= MAX_OFFSET_PX) else "  <-- REVIEW"
        print(f"{i+1:>3} {inl:>5} {d['rotation_deg']:>9.3f} {d['scale']:>7.4f} "
              f"{d['aspect']:>7.4f} {d['shear']:>7.4f} {d['dx']:>7.1f} {d['dy']:>7.1f} "
              f"{d['persp']:>7.2f} {cover:>6.3f} {off:>7.2f}{flag}")
        if flag:
            bad.append(i + 1)
    print(f"\npages needing review: {bad if bad else 'none'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Forms", "Cif-form.pdf"))
