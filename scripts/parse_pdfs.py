"""
Phase 1: Parse PDFs in data/pdfs/ to markdown via the Datalab Marker API.

We pivoted from local Marker because RTX 3050 Ti's 4 GB VRAM is below Marker's
5 GB minimum (severe thrashing) and CPU mode was equally slow. Datalab hosts
Marker on their GPUs.

Usage:
    python scripts/parse_pdfs.py --sample 1   # one paper
    python scripts/parse_pdfs.py --sample 3   # three papers
    python scripts/parse_pdfs.py              # all 100, skip already-done
    python scripts/parse_pdfs.py --force      # re-parse everything

Outputs:
    data/markdown/{paper_id}.md         -- markdown with section headers
    data/markdown/{paper_id}.meta.json  -- page count, parse time, cost
    data/parse_failures.json            -- list of failed papers w/ reason
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.core.budget import record_cost  # noqa: E402

ROOT = Path(__file__).parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
MD_DIR = ROOT / "data" / "markdown"
MANIFEST = ROOT / "data" / "manifest.csv"
FAILURES_PATH = ROOT / "data" / "parse_failures.json"

DATALAB_BASE = "https://www.datalab.to/api/v1"
DATALAB_CONVERT_URL = f"{DATALAB_BASE}/convert"

# Cost estimate — actual pricing TBD; placeholder used until invoice clarifies
EST_COST_PER_PAGE_USD = 0.0025

# Datalab limits: 400 RPM, 400 concurrent, 5000 in-flight pages.
# We use a small bounded concurrency for safety + easier debugging.
DEFAULT_CONCURRENCY = 8
POLL_INTERVAL_SECONDS = 3.0
POLL_TIMEOUT_SECONDS = 600.0
SUBMIT_RETRIES = 3


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="parse only first N papers (0=all)")
    p.add_argument("--force", action="store_true", help="re-parse even if .md exists")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="max in-flight requests")
    return p.parse_args()


# ── HTTP helpers ──────────────────────────────────────────────────────────────

# Multi-key rotation: when one key returns 403 (subscription exhausted) or
# repeated 429 (rate-limited / quota), it's marked dead and we move on.
class KeyPool:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.dead: set[str] = set()

    def alive(self) -> list[str]:
        return [k for k in self.keys if k not in self.dead]

    def mark_dead(self, key: str, reason: str) -> None:
        if key not in self.dead:
            self.dead.add(key)
            safe_print(f"  KEY EXHAUSTED ({reason}): ...{key[-6:]}")


async def submit_pdf(client: httpx.AsyncClient, key_pool: KeyPool, pdf_path: Path) -> dict:
    """Submit a PDF to Datalab. Rotates through keys on auth/quota errors. Returns the JSON response."""
    last_err: Exception | None = None
    for attempt in range(SUBMIT_RETRIES):
        for api_key in key_pool.alive():
            try:
                with open(pdf_path, "rb") as f:
                    files = {"file": (pdf_path.name, f, "application/pdf")}
                    data = {"output_format": "markdown"}
                    r = await client.post(
                        DATALAB_CONVERT_URL,
                        headers={"X-API-Key": api_key},
                        files=files,
                        data=data,
                        timeout=120.0,
                    )
                if r.status_code == 403:
                    # subscription exhausted on this key — try next
                    key_pool.mark_dead(api_key, "403 subscription required")
                    last_err = httpx.HTTPStatusError(f"403 on key ...{api_key[-6:]}", request=r.request, response=r)
                    continue
                if r.status_code == 429 and attempt == SUBMIT_RETRIES - 1:
                    # final attempt, persistent rate-limit — mark dead so we move on
                    key_pool.mark_dead(api_key, "429 persistent rate limit")
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                last_err = e
                # keep this key alive unless 403/429 already handled above
                continue
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        if not key_pool.alive():
            raise RuntimeError(f"All API keys exhausted. Last error: {last_err}")
        await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"Submit failed after {SUBMIT_RETRIES} attempts: {last_err}")


async def poll_result(client: httpx.AsyncClient, key_pool: KeyPool, check_url: str) -> dict:
    """Poll the check_url until status is 'complete' or 'error'. Tries any alive key."""
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    while time.time() < deadline:
        alive = key_pool.alive()
        if not alive:
            raise RuntimeError("All keys exhausted during polling")
        # use first alive key for polling — polling doesn't generally consume quota
        try:
            r = await client.get(check_url, headers={"X-API-Key": alive[0]}, timeout=30.0)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue
        status = data.get("status", "")
        if status == "complete":
            return data
        if status in ("error", "failed"):
            raise RuntimeError(f"Datalab reported failure: {data}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Polling exceeded {POLL_TIMEOUT_SECONDS}s for {check_url}")


# ── Per-paper worker ──────────────────────────────────────────────────────────

async def parse_one(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    key_pool: KeyPool,
    pid: str,
    title: str,
    pdf_path: Path,
    md_path: Path,
    progress: dict,
) -> tuple[bool, str | None]:
    """Submit + poll one PDF. Returns (success, error_message)."""
    async with sem:
        progress["in_flight"] += 1
        idx = progress["started"] = progress["started"] + 1
        total = progress["total"]
        safe_print(f"[{idx}/{total}] submitting {pid[:12]}... | {title[:55]}")
        t0 = time.time()
        try:
            sub = await submit_pdf(client, key_pool, pdf_path)
            check_url = sub.get("request_check_url")
            if not check_url:
                raise RuntimeError(f"No request_check_url in submit response: {sub}")
            result = await poll_result(client, key_pool, check_url)

            markdown = result.get("markdown") or ""
            if not markdown.strip():
                raise RuntimeError("Datalab returned empty markdown")

            page_count = int(result.get("page_count") or result.get("metadata", {}).get("page_count") or 0)
            md_path.write_text(markdown, encoding="utf-8")

            est_cost = round(EST_COST_PER_PAGE_USD * max(page_count, 1), 6)
            elapsed = round(time.time() - t0, 2)
            meta = {
                "char_count": len(markdown),
                "section_headers": markdown.count("\n#"),
                "page_count": page_count,
                "parse_seconds": elapsed,
                "est_cost_usd": est_cost,
            }
            md_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
            record_cost("datalab_parse", est_cost, paper_id=pid, pages=page_count)
            safe_print(
                f"  OK  {pid[:12]} | {elapsed}s | {meta['char_count']} chars | "
                f"{meta['section_headers']} sections | {page_count}p | est ${est_cost}"
            )
            progress["success"] += 1
            return True, None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            safe_print(f"  FAIL  {pid[:12]} | {err}")
            return False, err
        finally:
            progress["in_flight"] -= 1


# ── Main ──────────────────────────────────────────────────────────────────────

async def amain():
    args = parse_args()
    load_dotenv()
    # Collect all DATALAB_API_KEY* env vars in order: bare key first, then numbered
    keys: list[str] = []
    bare = os.getenv("DATALAB_API_KEY")
    if bare:
        keys.append(bare)
    for i in range(1, 10):
        v = os.getenv(f"DATALAB_API_KEY_{i}")
        if v and v not in keys:
            keys.append(v)
    if not keys:
        safe_print("No DATALAB_API_KEY* found in .env")
        sys.exit(1)
    safe_print(f"Loaded {len(keys)} Datalab API key(s): {[f'...{k[-6:]}' for k in keys]}")
    key_pool = KeyPool(keys)

    if not MANIFEST.exists():
        safe_print(f"Manifest not found at {MANIFEST}")
        sys.exit(1)

    MD_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST, dtype={"pdf_path": str})
    if args.sample:
        df = df.head(args.sample)

    todo = []
    for _, row in df.iterrows():
        pid = row["id"]
        pdf_path = PDF_DIR / f"{pid}.pdf"
        md_path = MD_DIR / f"{pid}.md"
        if not pdf_path.exists():
            continue
        if md_path.exists() and not args.force:
            continue
        todo.append((pid, row["title"], pdf_path, md_path))

    safe_print(f"Papers to parse: {len(todo)} (of {len(df)} in manifest)")
    if not todo:
        safe_print("Nothing to do. Use --force to re-parse.")
        return

    safe_print(f"Concurrency: {args.concurrency}")
    sem = asyncio.Semaphore(args.concurrency)
    progress = {"started": 0, "in_flight": 0, "success": 0, "total": len(todo)}
    failures: list[dict] = []
    t_start = time.time()

    async with httpx.AsyncClient() as client:
        tasks = [
            parse_one(sem, client, key_pool, pid, title, pdf_path, md_path, progress)
            for pid, title, pdf_path, md_path in todo
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for (pid, title, _pdf, _md), res in zip(todo, results):
        if isinstance(res, Exception):
            failures.append({"paper_id": pid, "title": title,
                             "error_type": type(res).__name__, "error": str(res)})
        elif res is None:
            continue
        else:
            ok, err = res
            if not ok:
                failures.append({"paper_id": pid, "title": title, "error": err or "unknown"})

    if failures:
        FAILURES_PATH.write_text(json.dumps(failures, indent=2))
        safe_print(f"\n{len(failures)} failures logged to {FAILURES_PATH}")

    total_t = time.time() - t_start
    safe_print(
        f"\nDone. {progress['success']} parsed, {len(failures)} failed in {total_t:.0f}s "
        f"({total_t/max(1,len(todo)):.1f}s/paper avg)"
    )


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        safe_print("\nInterrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
