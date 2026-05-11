"""
Phase 2: Extract structured data from markdown files via OpenAI GPT-4.1-mini.

Reads data/markdown/{paper_id}.md, extracts per-paper JSON using a frozen
system prompt (OpenAI auto-caches identical prefixes across all 100 papers),
writes to data/extractions/{paper_id}.json.

Usage:
    python scripts/extract_papers.py --sample 1   # single paper, verify output
    python scripts/extract_papers.py --sample 5   # spot-check 5
    python scripts/extract_papers.py              # all available markdowns
    python scripts/extract_papers.py --force      # re-extract even if .json exists

Strategy:
    - All 100 calls share the same system prompt. OpenAI automatically caches
      identical prompt prefixes (≥1024 tokens); cached tokens cost 0.1×.
    - Async concurrency with bounded semaphore (default 10) — safe and fast.
    - Fully idempotent: skips papers with an existing .json output.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.core.budget import record_cost, total_spent  # noqa: E402
from api.core.extraction_prompt import EXTRACTION_SYSTEM_PROMPT  # noqa: E402
from api.core.llm import (  # noqa: E402
    MODEL_GPT_MINI,
    get_openai_client,
    oai_cost_for_usage,
    oai_usage_summary,
)
from api.core.schemas import ExtractedPaper  # noqa: E402

ROOT = Path(__file__).parent.parent
MD_DIR = ROOT / "data" / "markdown"
OUT_DIR = ROOT / "data" / "extractions"
FAILURES_PATH = ROOT / "data" / "extract_failures.json"

DEFAULT_CONCURRENCY = 10

# Truncate very long papers to stay within context limits (150k chars ≈ ~37k tokens)
MAX_MD_CHARS = 150_000


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="process only first N papers (0=all)")
    p.add_argument("--force", action="store_true", help="re-extract even if .json exists")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--model", default=MODEL_GPT_MINI, help="OpenAI model ID to use")
    return p.parse_args()


def build_user_message(paper_id: str, markdown: str) -> str:
    md = markdown[:MAX_MD_CHARS]
    if len(markdown) > MAX_MD_CHARS:
        md += "\n\n[TRUNCATED — remainder omitted to fit context window]"
    return f"<paper_id>{paper_id}</paper_id>\n\n{md}"


async def extract_one(
    sem: asyncio.Semaphore,
    client,
    model: str,
    paper_id: str,
    md_path: Path,
    out_path: Path,
    progress: dict,
) -> tuple[bool, str | None]:
    """Extract one paper. Returns (success, error_message)."""
    async with sem:
        idx = progress["started"] = progress["started"] + 1
        total = progress["total"]
        t0 = time.time()
        safe_print(f"[{idx}/{total}] extracting {paper_id[:20]}...")
        try:
            markdown = md_path.read_text(encoding="utf-8", errors="replace")

            response = await client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(paper_id, markdown)},
                ],
                response_format=ExtractedPaper,
                temperature=0,
                max_completion_tokens=16384,
                # Prompt caching: same key across all 100 calls routes them to the
                # same cached prefix (the frozen ~4k-token system prompt).
                # Keep RPM per key < 15 — our default concurrency of 10 is safe.
                extra_body={
                    "prompt_cache_key": "research-extraction-v1",
                    "prompt_cache_retention": "in_memory",
                },
            )

            result: ExtractedPaper = response.choices[0].message.parsed
            if result is None:
                raise ValueError("Structured output returned None — possible refusal")

            # Ensure echoed paper_id is correct
            result.paper_id = paper_id

            out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            elapsed = round(time.time() - t0, 2)

            usage = response.usage
            cost = oai_cost_for_usage(model, usage)
            u = oai_usage_summary(usage)
            record_cost("extract_paper", cost, paper_id=paper_id, model=model, **u)

            cache_pct = int(100 * u["cached_tokens"] / max(u["input_tokens"], 1))
            safe_print(
                f"  OK  {paper_id[:20]} | {elapsed}s | "
                f"in:{u['input_tokens']} cached:{u['cached_tokens']}({cache_pct}%) "
                f"out:{u['output_tokens']} | ${cost:.4f}"
            )
            progress["success"] += 1
            return True, None

        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            safe_print(f"  FAIL {paper_id[:20]} | {err}")
            return False, err


async def amain():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(MD_DIR.glob("*.md"))
    if args.sample:
        md_files = md_files[: args.sample]

    todo = []
    skipped = 0
    for md_path in md_files:
        paper_id = md_path.stem
        out_path = OUT_DIR / f"{paper_id}.json"
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        todo.append((paper_id, md_path, out_path))

    safe_print(f"Papers to extract: {len(todo)}  (skipped {skipped} already done)")
    safe_print(f"Model: {args.model}  |  Concurrency: {args.concurrency}")
    safe_print(f"Total spend so far: ${total_spent():.4f}")

    if not todo:
        safe_print("Nothing to do. Use --force to re-extract.")
        return

    client = get_openai_client()
    sem = asyncio.Semaphore(args.concurrency)
    progress = {"started": 0, "success": 0, "total": len(todo)}
    budget_before = total_spent()

    tasks = [
        extract_one(sem, client, args.model, pid, mp, op, progress)
        for pid, mp, op in todo
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures: list[dict] = []
    for (pid, _mp, _op), res in zip(todo, results):
        if isinstance(res, Exception):
            failures.append({"paper_id": pid, "error": str(res)})
        elif res is not None:
            ok, err = res
            if not ok:
                failures.append({"paper_id": pid, "error": err or "unknown"})

    if failures:
        FAILURES_PATH.write_text(json.dumps(failures, indent=2))
        safe_print(f"\n{len(failures)} failures logged to {FAILURES_PATH}")

    spent = total_spent() - budget_before
    safe_print(
        f"\nDone. {progress['success']}/{len(todo)} extracted. "
        f"Run cost: ${spent:.4f} | Total: ${total_spent():.4f}"
    )


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        safe_print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
