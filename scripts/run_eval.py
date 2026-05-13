"""
Phase 9 — Quality-vs-budget evaluation runner.

Runs all 40 eval questions at $1, $5, and $20 budget levels.
Writes per-level JSON reports to eval/reports/ and a summary to eval/RESULTS.md.

Usage:
    python scripts/run_eval.py                  # all 3 budget levels
    python scripts/run_eval.py --budget '$5'    # single level
    python scripts/run_eval.py --limit 5        # first 5 questions only (dev)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.core.budget import record_cost, total_spent
from api.core.classifier import TierClassifier
from api.core.handlers import get_handler
from api.core.llm import MODEL_GPT_MINI, get_openai_client, oai_cost_for_usage
from api.core.retrieval import Retriever
from api.core.store import CorpusStore

ROOT = Path(__file__).parent.parent
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
REPORTS_DIR = ROOT / "eval" / "reports"
RESULTS_MD = ROOT / "eval" / "RESULTS.md"

ABORT_THRESHOLD = 25.0


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


# ── Match strategies ──────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


def _substring_match(answer: str, gold: str) -> tuple[bool, float]:
    passed = _normalize(gold) in _normalize(answer)
    return passed, 1.0 if passed else 0.0


def _structural_match(answer: str, gold_items: list[str]) -> tuple[bool, float]:
    if not gold_items:
        return True, 1.0
    answer_lo = answer.lower()
    matched = sum(1 for item in gold_items if item.lower() in answer_lo)
    overlap = matched / len(gold_items)
    return overlap >= 0.8, overlap


async def _llm_judge(question: str, answer: str, gold: str) -> tuple[bool, float]:
    client = get_openai_client()
    resp = await client.chat.completions.create(
        model=MODEL_GPT_MINI,
        messages=[
            {"role": "system", "content":
                "You are an answer-grader. Given QUESTION, GOLD ANSWER, and SYSTEM ANSWER, "
                "decide if the system answer is semantically equivalent to the gold. "
                "Reply ONLY 'PASS' or 'FAIL' on the first line."},
            {"role": "user", "content":
                f"QUESTION: {question}\n\nGOLD: {gold}\n\nSYSTEM: {answer}"},
        ],
        max_completion_tokens=64,
    )
    text = (resp.choices[0].message.content or "").strip().upper()
    cost = oai_cost_for_usage(MODEL_GPT_MINI, resp.usage)
    record_cost("eval_judge", cost)
    return text.startswith("PASS"), cost


# ── Single question runner ────────────────────────────────────────────────────

async def run_one(q: dict, store, retriever, classifier) -> dict:
    t0 = time.time()
    question = q["question"]
    gold_answer = q.get("gold_answer", "")
    expected_tier = q.get("tier")
    match_strategy = q.get("match_strategy", "substring")
    gold_items = q.get("gold_items", [])

    cost = 0.0
    judge_cost = 0.0
    error = None
    answer = ""
    actual_tier = 0
    actual_confidence = 0.0

    try:
        meta = await classifier.classify(question)
        cost += meta.get("cost_usd", 0.0)
        actual_tier = meta["tier"]
        actual_confidence = meta["confidence"]
        normalized_q = meta.get("normalized_question") or question

        handle = get_handler(actual_tier)
        result = await handle(normalized_q, store, retriever, classifier_meta=meta)
        cost += result.cost_usd
        answer = result.answer
    except Exception as e:
        import traceback
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    elapsed = round(time.time() - t0, 2)

    if error:
        passed, overlap = False, 0.0
    elif match_strategy == "substring":
        passed, overlap = _substring_match(answer, gold_answer)
    elif match_strategy == "structural":
        passed, overlap = _structural_match(answer, gold_items)
    elif match_strategy == "llm_judge":
        passed, judge_cost = await _llm_judge(question, answer, gold_answer)
        overlap = 1.0 if passed else 0.0
    else:
        passed, overlap = False, 0.0
        error = error or f"unknown match_strategy: {match_strategy}"

    return {
        "id": q.get("id", ""),
        "tier": expected_tier,
        "actual_tier": actual_tier,
        "tier_confidence": actual_confidence,
        "question": question,
        "gold_answer": gold_answer,
        "system_answer": answer,
        "match_strategy": match_strategy,
        "passed": passed,
        "overlap": overlap,
        "cost_usd": round(cost + judge_cost, 6),
        "elapsed_seconds": elapsed,
        "error": error,
    }


# ── Full budget-level run ─────────────────────────────────────────────────────

async def run_budget_level(budget: str, questions: list[dict], concurrency: int = 2) -> dict:
    prev = os.environ.get("BUDGET_LEVEL")
    os.environ["BUDGET_LEVEL"] = budget

    store = CorpusStore()
    retriever = Retriever()
    classifier = TierClassifier()

    budget_start = total_spent()
    sem = asyncio.Semaphore(concurrency)

    async def bounded(q):
        async with sem:
            return await run_one(q, store, retriever, classifier)

    safe_print(f"\n{'='*60}")
    safe_print(f"Running {len(questions)} questions at budget {budget}")
    safe_print(f"{'='*60}")
    safe_print(f"{'#':>3} {'ID':<8} {'T':<3} {'AT':<3} {'Pass':<5} {'$':>8} {'Time':>6}  Question")
    safe_print("-" * 80)

    results = []
    tasks = [bounded(q) for q in questions]
    # Run with controlled concurrency — gather returns in submission order
    for i, coro in enumerate(tasks, 1):
        r = await coro
        results.append(r)
        status = "OK " if r["passed"] else "FAIL"
        q_short = r["question"][:50]
        safe_print(
            f"{i:>3} {r['id']:<8} T{r['tier']:<2} T{r['actual_tier']:<2} "
            f"{status:<5} ${r['cost_usd']:>7.4f} {r['elapsed_seconds']:>5.1f}s  {q_short}"
        )
        if not r["passed"] and not r["error"]:
            gold_short = str(r["gold_answer"])[:60]
            ans_short = str(r["system_answer"])[:80]
            safe_print(f"     gold:   {gold_short}")
            safe_print(f"     answer: {ans_short}")
        if r["error"]:
            safe_print(f"     ERROR: {r['error']}")

        if total_spent() > ABORT_THRESHOLD:
            safe_print(f"\n!!! ABORT: total spend ${total_spent():.2f} exceeds ${ABORT_THRESHOLD} !!!")
            break

    # Per-tier aggregation
    per_tier: dict[int, list] = defaultdict(list)
    for r in results:
        per_tier[r["tier"]].append(r)

    tier_stats = {}
    for tier, rows in sorted(per_tier.items()):
        n = len(rows)
        passed = sum(1 for r in rows if r["passed"])
        tier_stats[tier] = {
            "n": n,
            "passed": passed,
            "pass_rate": round(passed / n, 3),
            "avg_cost": round(sum(r["cost_usd"] for r in rows) / n, 4),
            "avg_latency": round(sum(r["elapsed_seconds"] for r in rows) / n, 2),
            "avg_confidence": round(sum(r["tier_confidence"] for r in rows) / n, 3),
        }

    total_q = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    total_cost = round(sum(r["cost_usd"] for r in results), 4)
    total_time = round(sum(r["elapsed_seconds"] for r in results), 1)

    safe_print(f"\nSummary {budget}: {total_passed}/{total_q} passed ({100*total_passed/total_q:.0f}%)  "
               f"cost=${total_cost:.4f}  time={total_time:.0f}s  "
               f"(${total_cost/total_q:.4f}/q avg)")

    report = {
        "budget_level": budget,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_questions": total_q,
        "passed": total_passed,
        "pass_rate": round(total_passed / max(total_q, 1), 3),
        "total_cost_usd": total_cost,
        "total_latency_seconds": total_time,
        "budget_at_start_usd": round(budget_start, 4),
        "budget_at_end_usd": round(total_spent(), 4),
        "per_tier": tier_stats,
        "questions": results,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{report['timestamp']}_{budget.replace('$','')}usd.json"
    path = REPORTS_DIR / fname
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    safe_print(f"Report saved: {path}")

    if prev is None:
        os.environ.pop("BUDGET_LEVEL", None)
    else:
        os.environ["BUDGET_LEVEL"] = prev

    return report


# ── RESULTS.md writer ─────────────────────────────────────────────────────────

def write_results_md(reports: list[dict]) -> None:
    lines = [
        "# Eval Results — Quality vs Budget",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Overall Summary",
        "",
        "| Budget | Pass Rate | Passed/Total | Total Cost | Avg Cost/Q | Avg Latency/Q |",
        "|--------|-----------|--------------|------------|------------|---------------|",
    ]
    for r in reports:
        n = r["total_questions"]
        lines.append(
            f"| {r['budget_level']} "
            f"| {r['pass_rate']*100:.0f}% "
            f"| {r['passed']}/{n} "
            f"| ${r['total_cost_usd']:.3f} "
            f"| ${r['total_cost_usd']/n:.4f} "
            f"| {r['total_latency_seconds']/n:.1f}s |"
        )

    lines += ["", "## Per-Tier Accuracy", ""]

    # Gather all tiers
    all_tiers = sorted({int(t) for r in reports for t in r["per_tier"]})
    budgets = [r["budget_level"] for r in reports]

    # Header
    header = "| Tier |" + "".join(f" {b} pass | {b} cost |" for b in budgets)
    sep = "|------|" + "".join("|-----------|------------|" for _ in budgets)
    lines += [header, sep]

    for tier in all_tiers:
        row = f"| T{tier}   |"
        for r in reports:
            ts = r["per_tier"].get(str(tier)) or r["per_tier"].get(tier, {})
            if ts:
                pct = f"{ts['pass_rate']*100:.0f}% ({ts['passed']}/{ts['n']})"
                cost = f"${ts['avg_cost']:.4f}"
            else:
                pct, cost = "—", "—"
            row += f" {pct} | {cost} |"
        lines.append(row)

    lines += ["", "## Per-Question Detail", ""]

    for r in reports:
        lines += [f"### Budget: {r['budget_level']}", ""]
        lines += [
            "| ID | Tier | Pass | Match | Cost | Latency | Notes |",
            "|----|------|------|-------|------|---------|-------|",
        ]
        for q in r["questions"]:
            status = "✓" if q["passed"] else "✗"
            overlap = f"{q['overlap']*100:.0f}%"
            tier_match = "" if q["tier"] == q["actual_tier"] else f" (routed T{q['actual_tier']})"
            err = q.get("error") or ""
            note = f"{overlap}{tier_match}" + (f" ERR: {err[:40]}" if err else "")
            lines.append(
                f"| {q['id']} | T{q['tier']} | {status} | {q['match_strategy'][:10]} "
                f"| ${q['cost_usd']:.4f} | {q['elapsed_seconds']}s | {note} |"
            )
        lines.append("")

    lines += [
        "## Cost Notes",
        "",
        f"- Total spend including this eval: ${total_spent():.4f} / $30.00 cap",
        f"- Remaining budget: ${30.0 - total_spent():.4f}",
        "",
    ]

    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    safe_print(f"\nResults written to {RESULTS_MD}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", choices=["$1", "$5", "$20"],
                   help="Run a single budget level only (default: all three)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run first N questions (dev mode)")
    p.add_argument("--concurrency", type=int, default=2,
                   help="Parallel questions per budget level")
    args = p.parse_args()

    if not QUESTIONS_PATH.exists():
        safe_print(f"ERROR: {QUESTIONS_PATH} not found")
        sys.exit(1)

    questions = []
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if args.limit:
        questions = questions[:args.limit]

    budgets = [args.budget] if args.budget else ["$1", "$5", "$20"]

    all_reports = []
    for budget in budgets:
        report = await run_budget_level(budget, questions, args.concurrency)
        all_reports.append(report)

        if total_spent() > ABORT_THRESHOLD:
            safe_print("Budget threshold exceeded — stopping early.")
            break

    write_results_md(all_reports)

    safe_print(f"\nAll done. Total spend: ${total_spent():.4f} / $30")


if __name__ == "__main__":
    asyncio.run(amain())
