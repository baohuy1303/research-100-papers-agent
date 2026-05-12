"""
Battery test — runs ~20 diverse questions through the same code path as
ask_cli.py, captures pass/fail/error/answer-quality flags, prints a compact
summary so we can find bugs to fix.

Usage:
    python scripts/cli_battery.py
    python scripts/cli_battery.py --budget '$1'    # test $1 mode
"""
import argparse
import asyncio
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.core.classifier import TierClassifier  # noqa: E402
from api.core.handlers import get_handler  # noqa: E402
from api.core.handlers.base import is_adversarial  # noqa: E402
from api.core.retrieval import Retriever  # noqa: E402
from api.core.store import CorpusStore  # noqa: E402

# Question bank — designed to find bugs across tiers
# Each tuple: (expected_tier, question, sanity_check_substring or None)
BATTERY = [
    # ── Tier 1 ──
    (1, "What datasets did Swin Transformer use?", "ImageNet"),
    (1, "How many parameters does MAE have?", None),
    (1, "What activation function does ConvNeXt use?", None),
    (0, "What is the architecture of a paper that doesn't exist?", None),  # adversarial — expects refusal

    # ── Tier 2 ──
    (2, "How many papers in the corpus benchmark on COCO?", None),  # value varies (24 from results table vs 37 from mentions)
    (2, "List all papers from 2023.", None),
    (2, "What's the venue with the most papers?", None),

    # ── Tier 3 ──
    (3, "Do papers disagree on top-1 accuracy on CIFAR-100?", None),
    (3, "Are there contradictions about position embeddings across papers?", None),

    # ── Tier 4 ──
    (4, "How did the maximum top-1 accuracy on ImageNet change year over year?", None),
    (4, "What's the year-by-year average parameter count across papers?", None),

    # ── Tier 5 ──
    (5, "Which paper is the most cited within this corpus?", "Image is Worth"),
    (5, "What are the top 5 papers by pagerank?", None),
    (5, "What papers does Swin Transformer cite within this corpus?", None),

    # ── Tier 6 ──
    (6, "Among papers citing ViT, which has the largest model variant?", "Open X-Embodiment"),  # RT-2-X at 55B
    (6, "Among papers using both ImageNet AND COCO, which has the highest top-1 accuracy?", None),

    # ── Tier 7 ──
    (7, "Which standard CV datasets are NOT used by any paper in this corpus?", None),

    # ── Tier 8 ──
    (8, "What's the median parameter count across all model variants in the corpus?", None),
    (8, "What is the correlation between citation count and parameter count?", None),
]


def truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


async def run_one(q: str, expected_tier: int, sanity: str | None,
                  store, retriever, classifier) -> dict:
    t0 = time.time()
    out = {
        "question": q, "expected_tier": expected_tier,
        "actual_tier": None, "confidence": None,
        "answer": "", "cost_usd": 0.0, "elapsed_s": 0.0,
        "status": "", "notes": "",
    }
    try:
        # Adversarial pre-check (mirrors /ask + CLI). Maps to virtual tier 0.
        if is_adversarial(q):
            out["actual_tier"] = 0
            out["confidence"] = 1.0
            out["answer"] = "(adversarial pre-check refused)"
            ok_tier = (expected_tier == 0)
            out["status"] = "OK" if ok_tier else "TIER-MISMATCH"
            if not ok_tier:
                out["notes"] = f"expected T{expected_tier}, refused as adversarial (T0)"
            return out

        meta = await classifier.classify(q)
        out["actual_tier"] = meta["tier"]
        out["confidence"] = meta["confidence"]
        out["cost_usd"] += meta.get("cost_usd", 0.0)
        normalized = meta.get("normalized_question") or q

        handle = get_handler(meta["tier"])
        result = await handle(normalized, store, retriever, classifier_meta=meta)
        out["cost_usd"] += result.cost_usd
        out["answer"] = result.answer

        # Pass criteria:
        #   - no exception
        #   - tier matches expected (or close — fallback to T1 is fine)
        #   - answer is non-trivial
        #   - if sanity substring given, must be in answer (case-insensitive)
        ok_tier = meta["tier"] == expected_tier
        ok_answer = bool(result.answer) and len(result.answer) > 20
        ok_sanity = (sanity is None) or (sanity.lower() in result.answer.lower())

        if not ok_tier:
            out["status"] = "TIER-MISMATCH"
            out["notes"] = f"expected T{expected_tier} got T{meta['tier']}"
        elif not ok_answer:
            out["status"] = "EMPTY-ANSWER"
        elif not ok_sanity:
            out["status"] = "SANITY-FAIL"
            out["notes"] = f"missing '{sanity}' in answer"
        else:
            out["status"] = "OK"
    except Exception as e:
        out["status"] = "ERROR"
        out["notes"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
    finally:
        out["elapsed_s"] = round(time.time() - t0, 1)
    return out


async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", default="$5", choices=["$1", "$5", "$20"])
    args = p.parse_args()
    os.environ["BUDGET_LEVEL"] = args.budget

    store = CorpusStore()
    retriever = Retriever()
    classifier = TierClassifier()

    safe_print(f"Running {len(BATTERY)} battery tests at budget {args.budget}\n")
    safe_print(f"{'#':>3} {'Tier':>5} {'Conf':>5} {'Cost':>8} {'Time':>6}  {'Status':<14}  {'Question':<60}")
    safe_print("-" * 130)

    results = []
    for i, (expected, q, sanity) in enumerate(BATTERY, 1):
        out = await run_one(q, expected, sanity, store, retriever, classifier)
        results.append(out)
        tier_str = f"T{out['actual_tier']}" if out["actual_tier"] else "?"
        if out["actual_tier"] != expected and out["actual_tier"] is not None:
            tier_str += f"!=T{expected}"
        conf = out["confidence"] or 0.0
        safe_print(
            f"{i:>3} {tier_str:>5} {conf:>5.2f} ${out['cost_usd']:>6.4f} "
            f"{out['elapsed_s']:>5.1f}s  {out['status']:<14}  {truncate(q, 60)}"
        )
        if out["status"] not in ("OK",):
            safe_print(f"     >>> {out['notes']}")
            if out.get("traceback"):
                for line in out["traceback"].splitlines()[-3:]:
                    safe_print(f"         {line}")
            else:
                safe_print(f"     answer: {truncate(out['answer'], 100)}")

    # ── Summary ──
    n = len(results)
    by_status = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    total_cost = sum(r["cost_usd"] for r in results)
    total_time = sum(r["elapsed_s"] for r in results)

    safe_print("\n" + "=" * 60)
    safe_print(f"SUMMARY ({n} questions)")
    safe_print(f"  by status: {by_status}")
    safe_print(f"  total cost: ${total_cost:.4f}")
    safe_print(f"  total time: {total_time:.0f}s ({total_time/n:.1f}s avg)")


if __name__ == "__main__":
    asyncio.run(amain())
