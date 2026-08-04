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
    pages: int
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
            "pages": self.pages, "total_pages": self.total_pages,
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

    def create(self, filename: str, pages: int, total_pages: int) -> Job:
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
        pages = list(range(1, job.pages + 1))

        _set(job, "align", 0.0, "Aligning pages")
        register.run(
            pdf_path, os.path.join(BUILD, "registered", doc),
            identify=True, max_pages=job.pages,
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


def start(filename: str, data: bytes, pages: int | None) -> Job:
    """Persist the upload, work out how many pages to scan, and launch the job."""
    os.makedirs(UPLOADS, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(filename)[1].lower() or ".pdf"
    path = os.path.join(UPLOADS, f"{job_id}{ext}")
    with open(path, "wb") as fh:
        fh.write(data)

    total = page_count(path)
    want = total if not pages else max(1, min(int(pages), total))

    job = Job(id=job_id, filename=filename, pages=want, total_pages=total)
    with REGISTRY._lock:                                      # noqa: SLF001
        REGISTRY._jobs[job.id] = job                          # noqa: SLF001

    threading.Thread(target=run_job, args=(job, path), daemon=True,
                     name=f"extract-{job_id}").start()
    return job


def cleanup(job_id: str) -> None:
    """Drop a job's artefacts. Uploads hold customer PII, so make removal explicit
    and complete rather than leaving files behind after a review is finished."""
    for sub in ("registered", "crops"):
        shutil.rmtree(os.path.join(BUILD, sub, job_id), ignore_errors=True)
    for f in (os.path.join(BUILD, "reads", f"{job_id}.json"),
              os.path.join(BUILD, "output", f"{job_id}.json"),
              os.path.join(BUILD, "output", f"{job_id}.csv")):
        try:
            os.remove(f)
        except OSError:
            pass
    for ext in (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"):
        try:
            os.remove(os.path.join(UPLOADS, f"{job_id}{ext}"))
        except OSError:
            pass
