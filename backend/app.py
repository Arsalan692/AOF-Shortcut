"""
FastAPI service for the Account Opening Form extractor.

Endpoints
  POST   /api/extract                          upload a form, start a job
  PATCH  /api/jobs/{id}/fields/{fid}           correct one value by hand
  DELETE /api/jobs/{id}/fields/{fid}/edit      undo that correction
  GET    /api/jobs/{id}                        progress, then the result
  GET    /api/jobs/{id}/pages/{n}/preview.png  the aligned page, for review
  GET    /api/jobs/{id}/fields/{fid}/crop.png  the exact patch a value was read from
  GET    /api/jobs/{id}/download.{json,csv}    the assembled output
  DELETE /api/jobs/{id}                        remove the upload and its artefacts
  GET    /api/health                           model/schema readiness

If the frontend has been built (frontend/dist), it is served from / so that running the
backend alone gives a working page.
"""
from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request

import cv2
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from jobs import (BUILD, REGISTRY, ROOT, STAGES, apply_edits, cleanup,
                  field_record, set_edit, start)

# jobs puts src/ on sys.path as a side effect of importing, so anything from there has to
# come after it - including this, which is why it is not up with the other imports.
import fields as fieldmod              # noqa: E402

DIST = os.path.join(ROOT, "frontend", "dist")
SCHEMA = os.path.join(BUILD, "template_schema.json")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("HBL_VLM", "qwen2.5vl:7b")            # reads whole pages
MODEL_NUM = os.environ.get("HBL_VLM_NUM", "glm-ocr:latest")  # re-reads structured fields

MAX_UPLOAD = 64 * 1024 * 1024
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

app = FastAPI(title="HBL Account Opening Form Extraction", version="1.0")

# Vite dev server runs on another port; in production the app is same-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)


# ------------------------------------------------------------------- helpers
def _job_or_404(job_id: str):
    job = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


def _png(img) -> Response:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "Could not encode image")
    return Response(buf.tobytes(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


# -------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    """Everything the UI needs to know before accepting an upload.

    Two readers have to be reported, not one. The VLM reads every page; the OCR model only
    re-reads structured fields whose first value failed its format check. The two failures
    are not equivalent: without the VLM nothing can be read at all, whereas without the OCR
    model the run still completes with the VLM re-reading its own misses - at the cost of
    digit errors no format check can see. So a missing OCR reader is a warning about
    accuracy, not a blocked upload.
    """
    models: list[str] = []
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=4) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        pass

    def installed(name: str) -> bool:
        # tags carry a :tag suffix that the configured name may omit
        return any(m.split(":")[0] == name.split(":")[0] for m in models)

    text_ready, num_ready = installed(MODEL), installed(MODEL_NUM)

    schema_fields = 0
    if os.path.exists(SCHEMA):
        with open(SCHEMA, encoding="utf-8") as fh:
            schema_fields = len(json.load(fh).get("fields", []))

    warnings = []
    if text_ready and not num_ready:
        warnings.append(
            f"{MODEL_NUM} is not pulled, so a date, CNIC, phone number or amount that "
            f"fails its format check will be re-read by {MODEL} instead. It is measurably "
            f"weaker on digits and its mistakes are well-formed, so they pass every "
            f"format check."
        )

    return {
        # blocked only by the things that stop a run outright
        "ok": schema_fields > 0 and text_ready,
        "degraded": bool(warnings),
        "warnings": warnings,
        "schema_fields": schema_fields,
        "schema_ready": schema_fields > 0,
        # kept for callers written against the single-model shape
        "model": MODEL, "model_ready": text_ready,
        "readers": [
            {"role": "pages", "model": MODEL, "ready": text_ready, "required": True,
             "reads": "every field on a page, in one pass"},
            {"role": "re-reads", "model": MODEL_NUM, "ready": num_ready, "required": False,
             "reads": "checks every date, CNIC, phone, IBAN and amount a second time"},
        ],
        "models_available": models,
        "ollama": OLLAMA,
        "stages": [{"key": k, "label": lab, "description": d} for k, lab, d, _ in STAGES],
    }


@app.post("/api/extract")
async def extract(file: UploadFile = File(...), pages: str | None = Form(None)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type {ext or '(none)'}. "
                                 f"Upload a PDF or an image.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File is larger than 64 MB")
    if not os.path.exists(SCHEMA):
        raise HTTPException(503, "Field schema missing - run src/build_schema.py first")

    # `pages` is a selection, not a count: "2" reads page 2 on its own, and "1,4-6" reads
    # exactly those. Validation happens inside start(), which knows the document length.
    try:
        job = start(file.filename or "form.pdf", data, pages)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(400, f"Could not read the document: {exc}") from None
    return JSONResponse(job.public(), status_code=202)


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": [{k: v for k, v in j.public().items() if k != "result"}
                     for j in REGISTRY.all()]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _job_or_404(job_id)
    d = job.public()
    if d.get("result"):
        d["result"] = apply_edits(job_id, d["result"])
    return d


@app.patch("/api/jobs/{job_id}/fields/{field_id}")
def edit_field(job_id: str, field_id: str, body: dict = Body(...)):
    """Correct one value by hand.

    The extraction itself is never overwritten - the correction is stored separately, keeps
    the value it replaced, and is re-validated, so a hand-typed CNIC is checked exactly as
    a read one would be and can still come back flagged.
    """
    job = _job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, "Extraction has not finished")
    if "value" not in body:
        raise HTTPException(400, "Body must be {\"value\": ...}")
    try:
        merged = set_edit(job_id, field_id, body["value"])
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    rec = next((r for r in merged["fields"] if r["field"] == field_id), None)
    return {"field": rec, "edit_count": merged.get("edit_count", 0)}


@app.delete("/api/jobs/{job_id}/fields/{field_id}/edit")
def revert_field(job_id: str, field_id: str):
    """Undo a correction and put the model's own value back."""
    job = _job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, "Extraction has not finished")
    try:
        merged = set_edit(job_id, field_id, None, revert=True)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    rec = next((r for r in merged["fields"] if r["field"] == field_id), None)
    return {"field": rec, "edit_count": merged.get("edit_count", 0)}


@app.delete("/api/jobs/{job_id}")
def job_delete(job_id: str):
    job = _job_or_404(job_id)
    if job.status == "running":
        raise HTTPException(409, "Job is still running")
    cleanup(job_id)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/pages/{page}/preview.png")
def page_preview(job_id: str, page: int, width: int = 1100):
    """The aligned page as the pipeline saw it."""
    _job_or_404(job_id)
    path = os.path.join(BUILD, "registered", job_id, f"reg_p{page}.png")
    if not os.path.exists(path):
        raise HTTPException(404, "No such page for this job")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(500, "Could not read the page image")
    if 200 <= width < img.shape[1]:
        f = width / img.shape[1]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return _png(img)


@app.get("/api/jobs/{job_id}/fields/{field_id}/crop.png")
def field_crop(job_id: str, field_id: str, scale: float = 1.0):
    """The exact patch a value was read from.

    This is the review feature that matters: a reviewer can see the handwriting behind any
    value without hunting for it on the page. Cut from the aligned page when asked for,
    rather than pre-generated - the extent each answer occupies is already in the field
    index, so this costs one file read.
    """
    _job_or_404(job_id)
    if "/" in field_id or "\\" in field_id or ".." in field_id:
        raise HTTPException(400, "Bad field id")
    rec = field_record(job_id, field_id)
    if rec is None:
        raise HTTPException(404, "Unknown field for this job")
    img = fieldmod.review_crop(job_id, rec, scale)
    if img is None:
        raise HTTPException(404, "The aligned page for this field is missing")
    return _png(img)


@app.get("/api/jobs/{job_id}/download.{fmt}")
def download(job_id: str, fmt: str):
    job = _job_or_404(job_id)
    if fmt not in ("json", "csv"):
        raise HTTPException(400, "Format must be json or csv")
    if job.status != "done":
        raise HTTPException(409, "Extraction has not finished")
    path = os.path.join(BUILD, "output", f"{job_id}.{fmt}")
    if not os.path.exists(path):
        raise HTTPException(404, "Output file missing")
    base = os.path.splitext(job.filename)[0] or "extraction"
    return FileResponse(path, filename=f"{base}.{fmt}",
                        media_type="application/json" if fmt == "json" else "text/csv")


# ------------------------------------------------- serve the built frontend
if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
else:
    @app.get("/")
    def no_ui():
        return {
            "message": "API is running. The frontend has not been built.",
            "hint": "Either build it once (cd frontend && npm install && npm run build) "
                    "and restart this server, or run `npm run dev` in frontend/ and "
                    "use http://localhost:5173 - Vite proxies /api here.",
            "api": "/docs",
        }
