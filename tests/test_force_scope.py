"""
Guard: --force must only discard the pages it is about to re-read.

This is a regression test for a bug that destroyed real work. `--force` emptied the whole
read cache and the run then saved that cache back over the file, so re-reading a single page
silently erased every other page's values - about an hour of model time, with no backup and
no warning. Re-reading one page must never cost another page its results.

Run: python tests/test_force_scope.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def scoped(cache: dict, index: list[dict], pages, force: bool) -> dict:
    """The cache `run()` starts from. Mirrors src/read_page.run - keep the two in step."""
    if not force:
        return dict(cache)
    page_of = {r["field"]: r["page"] for r in index}
    return {fid: v for fid, v in cache.items()
            if pages is not None and page_of.get(fid) not in pages}


def main() -> int:
    index = [{"field": "p1_a", "page": 1}, {"field": "p1_b", "page": 1},
             {"field": "p4_a", "page": 4}, {"field": "p9_a", "page": 9}]
    full = {r["field"]: {"value": "x", "source": "page"} for r in index}
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))

    kept = scoped(full, index, [4], force=True)
    check("re-reading page 4 keeps page 1", "p1_a" in kept and "p1_b" in kept)
    check("re-reading page 4 keeps page 9", "p9_a" in kept)
    check("re-reading page 4 drops page 4", "p4_a" not in kept)

    kept = scoped(full, index, None, force=True)
    check("--force over every page clears all", kept == {})

    kept = scoped(full, index, [4], force=False)
    check("without --force nothing is dropped", kept == full)

    kept = scoped(full, index, [1, 9], force=True)
    check("a multi-page force drops exactly those",
          set(kept) == {"p4_a"}, f"kept {sorted(kept)}")

    # The real module must agree with the mirror above, or this test guards nothing.
    import inspect

    import read_page
    src = inspect.getsource(read_page.run)
    check("read_page.run still scopes force by page",
          "page_of" in src and "pages is not None" in src,
          "if this fails, run() and this test have drifted apart")

    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    failed = [n for n, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
