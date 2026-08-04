"""
FastAPI service for the Account Opening Form extractor.

Endpoints
  POST   /api/extract                          upload a form, start a job
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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from jobs import BUILD, REGISTRY, ROOT, STAGES, cleanup, start

DIST = os.path.join(ROOT, "frontend", "dist")
SCHEMA = os.path.join(BUILD, "template_schema.json")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("HBL_VLM", "qwen2.5vl:7b")

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
    """Everything the UI needs to know before accepting an upload."""
    model_ready, models = False, []
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=4) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        model_ready = any(m.split(":")[0] == MODEL.split(":")[0] for m in models)
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        pass

    schema_fields = 0
    if os.path.exists(SCHEMA):
        with open(SCHEMA, encoding="utf-8") as fh:
            schema_fields = len(json.load(fh).get("fields", []))

    return {
        "ok": schema_fields > 0 and model_ready,
        "schema_fields": schema_fields,
        "schema_ready": schema_fields > 0,
        "model": MODEL, "model_ready": model_ready,
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

    n = None
    if pages not in (None, "", "all"):
        try:
            n = int(pages)
        except ValueError:
            raise HTTPException(400, "pages must be a whole number, or 'all'") from None
        if n < 1:
            raise HTTPException(400, "pages must be at least 1")

    try:
        job = start(file.filename or "form.pdf", data, n)
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(400, f"Could not read the document: {exc}") from None
    return JSONResponse(job.public(), status_code=202)


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": [{k: v for k, v in j.public().items() if k != "result"}
                     for j in REGISTRY.all()]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job_or_404(job_id).public()


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

    This is the review feature that matters: a reviewer can see the handwriting behind
    any value without hunting for it on the page.
    """
    _job_or_404(job_id)
    if "/" in field_id or "\\" in field_id or ".." in field_id:
        raise HTTPException(400, "Bad field id")
    idx_path = os.path.join(BUILD, "crops", job_id, "index.json")
    if not os.path.exists(idx_path):
        raise HTTPException(404, "No crops for this job yet")
    with open(idx_path, encoding="utf-8") as fh:
        rec = next((r for r in json.load(fh) if r["field"] == field_id), None)
    if rec is None:
        raise HTTPException(404, "Unknown field for this job")
    img = cv2.imread(os.path.join(ROOT, rec["crop"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(404, "Crop image missing")
    if scale and scale != 1.0 and 0.2 <= scale <= 4.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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
