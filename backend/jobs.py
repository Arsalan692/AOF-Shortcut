"""
Job runner for the extraction pipeline.

Extraction takes minutes, not milliseconds, so an HTTP request cannot wait for it.
Uploads therefore create a job that runs on a worker thread while the browser polls for
progress. Each job gets its own document name, so every artefact it writes
(build/registered/<job>, build/crops/<job>, build/reads/<job>.json, ...) is naturally
isolated from every other job - no locking needed.

Progress is real, not a timer: each stage reports through a callback, and the stage
weights below reflect where the time actually goes (the vision model dominates).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

BUILD = os.path.join(ROOT, "build")
UPLOADS = os.path.join(BUILD, "uploads")

import crop_fields                      # noqa: E402
import extract as extract_mod           # noqa: E402
import read_checkboxes                  # noqa: E402
import read_text                        # noqa: E402
import register                         # noqa: E402
from validate import assess             # noqa: E402

# (key, label, description, weight). Weights are the share of wall-clock each stage
# takes; "Reading handwriting" is by far the longest because it is the only stage that
# runs a model.
STAGES = [
    ("align", "Aligning pages",
     "Matching each scanned page onto the blank form", 0.14),
    ("locate", "Locating fields",
     "Cutting one image per field from the aligned scan", 0.05),
    ("boxes", "Reading checkboxes",
     "Measuring ink in every tick box", 0.05),
    ("read", "Reading handwriting",
     "Transcribing handwritten values with the local vision model", 0.68),
    ("verify", "Validating",
     "Checking formats, repairing slips and cross-checking fields", 0.08),
]
STAGE_KEYS = [s[0] for s in STAGES]


@dataclass
class Job:
    id: str
    filename: str
    pages: list[int]                # the 1-based scan pages this job covers
    total_pages: int
    status: str = "queued"          # queued | running | done | error
    stage: str = "align"
    message: str = "Queued"
    progress: float = 0.0           # 0..1 overall
    stage_progress: float = 0.0     # 0..1 within the current stage
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    result: dict | None = None

    @property
    def stage_index(self) -> int:
        return STAGE_KEYS.index(self.stage) if self.stage in STAGE_KEYS else 0

    def public(self) -> dict:
        now = time.time()
        elapsed = 0.0
        if self.started:
            elapsed = (self.finished or now) - self.started
        d = {
            "job_id": self.id, "filename": self.filename,
            "pages": self.pages, "page_count": len(self.pages),
            "total_pages": self.total_pages,
            "status": self.status, "stage": self.stage,
            "stage_index": self.stage_index, "stage_count": len(STAGES),
            "message": self.message,
            "progress": round(self.progress, 4),
            "stage_progress": round(self.stage_progress, 4),
            "elapsed": round(elapsed, 1),
            "error": self.error,
            "stages": [{"key": k, "label": lab, "description": desc}
                       for k, lab, desc, _ in STAGES],
        }
        if self.status == "done":
            d["result"] = self.result
        return d


class Registry:
    """Thread-safe in-memory job store. Deliberately not a database: jobs are
    per-session work, and the artefacts on disk are the durable record."""

    def __init__(self, keep: int = 40):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._keep = keep

    def create(self, filename: str, pages: list[int], total_pages: int) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=filename,
                  pages=pages, total_pages=total_pages)
        with self._lock:
            self._jobs[job.id] = job
            if len(self._jobs) > self._keep:
                for old in sorted(self._jobs.values(), key=lambda j: j.created)[:-self._keep]:
                    if old.status in ("done", "error"):
                        self._jobs.pop(old.id, None)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)


REGISTRY = Registry()


def _weights() -> dict[str, tuple[float, float]]:
    """stage key -> (offset, weight) so a stage fraction maps onto overall progress."""
    out, acc = {}, 0.0
    for key, _, _, w in STAGES:
        out[key] = (acc, w)
        acc += w
    return out


WEIGHTS = _weights()


def _set(job: Job, stage: str, frac: float, message: str) -> None:
    off, w = WEIGHTS[stage]
    job.stage = stage
    job.stage_progress = max(0.0, min(1.0, frac))
    job.progress = min(0.999, off + w * job.stage_progress)
    job.message = message


def page_count(path: str) -> int:
    with fitz.open(path) as d:
        return len(d)


def run_job(job: Job, pdf_path: str) -> None:
    """Execute the pipeline for one upload. Runs on a worker thread."""
    job.status = "running"
    job.started = time.time()
    doc = job.id
    try:
        pages = list(job.pages)

        _set(job, "align", 0.0, "Aligning pages")
        register.run(
            pdf_path, os.path.join(BUILD, "registered", doc),
            identify=True, pages=pages,
            progress=lambda d, t, m: _set(job, "align", d / max(t, 1), m),
        )

        _set(job, "locate", 0.2, "Locating fields")
        index = crop_fields.run(doc)
        n_read = sum(1 for r in index if r["needs_model"] and r["page"] in pages)
        _set(job, "locate", 1.0, f"{len(index)} fields located, {n_read} to transcribe")

        _set(job, "boxes", 0.3, "Reading checkboxes")
        read_checkboxes.run(os.path.join(BUILD, "registered", doc))
        _set(job, "boxes", 1.0, "Checkboxes read")

        _set(job, "read", 0.0, f"Transcribing {n_read} handwritten fields")
        read_text.run(
            doc, pages=pages,
            progress=lambda d, t, m: _set(job, "read", d / max(t, 1), m),
        )

        _set(job, "verify", 0.3, "Validating and cross-checking")
        data = extract_mod.build(doc)
        extract_mod.write_outputs(doc, data)
        data["answers"] = extract_mod.answers_view(data["fields"])
        data["source_file"] = job.filename
        data["pages_scanned"] = job.pages

        job.result = data
        job.progress = 1.0
        job.stage_progress = 1.0
        job.message = "Extraction complete"
        job.status = "done"
    except Exception as exc:                                  # noqa: BLE001
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "Extraction failed"
        traceback.print_exc()
    finally:
        job.finished = time.time()


def parse_pages(spec: str | None, total: int) -> list[int]:
    """Turn a page selection into page numbers. "" or "all" means every page.

    Accepts a list, ranges, or both: "2", "1,3", "2-4", "1,4-6". A bare number is a
    single page, NOT the first N - selecting page 2 must not drag page 1 along with it.
    """
    if spec is None or not str(spec).strip() or str(spec).strip().lower() == "all":
        return list(range(1, total + 1))
    want: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part[1:]:                       # "2-4"; the [1:] allows a stray leading -
            a, _, b = part.partition("-")
            if not (a.isdigit() and b.isdigit()):
                raise ValueError(f"{part!r} is not a page range like 2-4")
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            want.update(range(lo, hi + 1))
        elif part.isdigit():
            want.add(int(part))
        else:
            raise ValueError(f"{part!r} is not a page number")
    inside = sorted(p for p in want if 1 <= p <= total)
    if not inside:
        raise ValueError(f"none of those pages exist in a {total}-page document")
    return inside


def start(filename: str, data: bytes, pages: str | None) -> Job:
    """Persist the upload, resolve which pages to scan, and launch the job."""
    os.makedirs(UPLOADS, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(filename)[1].lower() or ".pdf"
    path = os.path.join(UPLOADS, f"{job_id}{ext}")
    with open(path, "wb") as fh:
        fh.write(data)

    total = page_count(path)
    want = parse_pages(pages, total)

    job = Job(id=job_id, filename=filename, pages=want, total_pages=total)
    with REGISTRY._lock:                                      # noqa: SLF001
        REGISTRY._jobs[job.id] = job                          # noqa: SLF001

    threading.Thread(target=run_job, args=(job, path), daemon=True,
                     name=f"extract-{job_id}").start()
    return job


# --------------------------------------------------------------- reviewer edits
# A reviewer correcting a value is the point of the review list, so the correction has to
# survive into the downloads. Edits are kept in their own file rather than written over the
# extraction: the pipeline output stays exactly what the models produced, every edit keeps
# the value it replaced, and a correction can be undone.
def edits_path(job_id: str) -> str:
    return os.path.join(BUILD, "output", f"{job_id}.edits.json")


def load_edits(job_id: str) -> dict:
    p = edits_path(job_id)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_edits(job_id: str, edits: dict) -> None:
    os.makedirs(os.path.join(BUILD, "output"), exist_ok=True)
    tmp = edits_path(job_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(edits, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, edits_path(job_id))


def _crop_record(job_id: str, field_id: str) -> dict | None:
    p = os.path.join(BUILD, "crops", job_id, "index.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return next((r for r in json.load(fh) if r["field"] == field_id), None)


def apply_edits(job_id: str, data: dict) -> dict:
    """Overlay stored edits on a result, re-validating each edited value.

    Re-validation matters: a hand-typed CNIC is no more trustworthy than a read one, so it
    goes through the same format rules and can still come back flagged.
    """
    edits = load_edits(job_id)
    if not edits:
        return data
    out = {k: (list(v) if k == "fields" else v) for k, v in data.items()}
    fields = []
    for rec in out["fields"]:
        e = edits.get(rec["field"])
        if not e:
            fields.append(rec)
            continue
        r = dict(rec)
        new = e.get("value", "")
        if r["kind"] == "checkbox":
            r["value"] = bool(new) and str(new).lower() not in ("false", "0", "", "no")
            r["note"] = "corrected by reviewer"
        else:
            crop = _crop_record(job_id, r["field"]) or {"label": r["label"],
                                                        "grid_hint": "", "cells": 0,
                                                        "kind": r["kind"]}
            a = assess(str(new), crop)
            r.update(type=a["type"], value=a["value"], valid=a["valid"])
            r["note"] = a["note"] or "corrected by reviewer"
        r["edited"] = True
        r["edited_from"] = e.get("original")
        r["source"] = "reviewer"
        r["confidence"] = "high" if r.get("valid", True) else "low"
        r["needs_review"] = not r.get("valid", True)
        fields.append(r)
    out["fields"] = fields
    out["answers"] = extract_mod.answers_view(fields)
    out["edit_count"] = len(edits)
    return out


def set_edit(job_id: str, field_id: str, value, revert: bool = False) -> dict:
    """Record (or undo) one correction and rewrite the downloadable output."""
    job = REGISTRY.get(job_id)
    if job is None or job.result is None:
        raise KeyError("no finished result for this job")
    rec = next((r for r in job.result["fields"] if r["field"] == field_id), None)
    if rec is None:
        raise KeyError(f"unknown field {field_id}")

    edits = load_edits(job_id)
    if revert:
        edits.pop(field_id, None)
    else:
        edits[field_id] = {
            "value": value,
            # the value first recorded is the model's, not a previous edit's
            "original": edits.get(field_id, {}).get("original", rec["value"]),
            "at": time.time(),
        }
    _save_edits(job_id, edits)

    merged = apply_edits(job_id, job.result)
    extract_mod.write_outputs(job_id, merged)      # keep json/csv in step with the edits
    return merged


def cleanup(job_id: str) -> None:
    """Drop a job's artefacts. Uploads hold customer PII, so make removal explicit
    and complete rather than leaving files behind after a review is finished."""
    for sub in ("registered", "crops"):
        shutil.rmtree(os.path.join(BUILD, sub, job_id), ignore_errors=True)
    for f in (os.path.join(BUILD, "reads", f"{job_id}.json"),
              os.path.join(BUILD, "output", f"{job_id}.json"),
              os.path.join(BUILD, "output", f"{job_id}.csv"),
              os.path.join(BUILD, "output", f"{job_id}.edits.json")):
        try:
            os.remove(f)
        except OSError:
            pass
    for ext in (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"):
        try:
            os.remove(os.path.join(UPLOADS, f"{job_id}{ext}"))
        except OSError:
            pass
