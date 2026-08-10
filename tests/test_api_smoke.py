"""
Smoke test for the review API, without spending an extraction on it.

Every endpoint a reviewer touches after a run finishes - the page preview, the crop behind a
value, a correction, undoing it, the downloads - reads artefacts that a completed job left
on disk. So this stands a finished job up from the artefacts already in build/ and exercises
the endpoints against it. It costs a second and covers the paths that a model run would take
twenty minutes to reach.

Needs a document that has been through the pipeline (build/registered/<doc>,
build/fields/<doc>.json, build/output/<doc>.json).

Run: python tests/test_api_smoke.py [doc]
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from fastapi.testclient import TestClient      # noqa: E402

import app as api                              # noqa: E402  (also puts src/ on the path)
from jobs import REGISTRY, Job                 # noqa: E402


def stand_up_job(doc: str) -> Job:
    """Register a finished job whose id is `doc`, so it finds that doc's artefacts."""
    with open(os.path.join(ROOT, "build", "output", f"{doc}.json"), encoding="utf-8") as fh:
        result = json.load(fh)
    pages = sorted({r["page"] for r in result["fields"]})
    job = Job(id=doc, filename=f"{doc}.pdf", pages=pages, total_pages=len(pages))
    job.status, job.progress, job.result = "done", 1.0, result
    job.started = job.finished = time.time()
    REGISTRY._jobs[doc] = job                                  # noqa: SLF001
    return job


def main(doc: str = "Cif-form") -> int:
    job = stand_up_job(doc)
    client = TestClient(api.app)
    fields = job.result["fields"]
    # a text field with a value, so the crop and the edit both have something to act on
    rec = next(r for r in fields
               if r["kind"] != "checkbox" and r["value"] not in (None, ""))
    page = rec["page"]
    fid = rec["field"]

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    r = client.get("/api/health")
    h = r.json()
    check("health", r.status_code == 200 and h["schema_ready"],
          f"{h.get('schema_fields')} fields, ok={h.get('ok')}")
    check("health names both readers", len(h.get("readers", [])) == 2,
          ", ".join(x["model"] for x in h.get("readers", [])))
    # the UI picks the optional reader out by this flag, not by its role name
    check("one reader is marked optional",
          sum(1 for x in h.get("readers", []) if x["required"] is False) == 1)

    r = client.get(f"/api/jobs/{doc}")
    check("job status", r.status_code == 200 and r.json()["status"] == "done")

    r = client.get(f"/api/jobs/{doc}/pages/{page}/preview.png")
    check("page preview", r.status_code == 200 and r.content[:4] == b"\x89PNG",
          f"page {page}, {len(r.content) // 1024} KB")

    r = client.get(f"/api/jobs/{doc}/fields/{fid}/crop.png")
    check("field crop", r.status_code == 200 and r.content[:4] == b"\x89PNG",
          f"{rec['label'][:28]}, {len(r.content) // 1024} KB")

    r = client.get(f"/api/jobs/{doc}/fields/{fid}/crop.png?scale=2")
    check("field crop scaled", r.status_code == 200)
    check("bad field id refused",
          client.get(f"/api/jobs/{doc}/fields/..%2Fetc/crop.png").status_code in (400, 404))

    # A hand-typed value is re-validated, not trusted: this one must come back invalid.
    r = client.patch(f"/api/jobs/{doc}/fields/{fid}", json={"value": "0399-1234567"})
    edited = r.json().get("field", {})
    check("edit accepted", r.status_code == 200 and edited.get("edited") is True)
    check("edit re-validated, not trusted", edited.get("value") != rec["value"],
          f"note: {edited.get('note', '')[:58]}")
    check("edit keeps the model's original", edited.get("edited_from") == rec["value"])

    r = client.delete(f"/api/jobs/{doc}/fields/{fid}/edit")
    check("edit reverted",
          r.status_code == 200 and r.json()["field"]["value"] == rec["value"])

    for fmt in ("json", "csv"):
        r = client.get(f"/api/jobs/{doc}/download.{fmt}")
        check(f"download {fmt}", r.status_code == 200 and len(r.content) > 0,
              f"{len(r.content) // 1024} KB")

    check("unknown job is a 404", client.get("/api/jobs/nope").status_code == 404)
    r = client.get("/")
    check("built UI served at /", r.status_code == 200 and "<div id=\"root\"" in r.text,
          "run `npm run build` in frontend/ if this fails")

    width = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [n for n, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "Cif-form"))
