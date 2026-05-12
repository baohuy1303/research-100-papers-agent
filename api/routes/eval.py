"""
POST /eval — run the eval set and report per-tier accuracy + cost.

Reads eval/questions.jsonl (built in Phase 8). Each entry runs through the
full /ask pipeline (classifier → handler → response). The result is judged
against the gold answer via per-tier match strategies:

  - "substring":  gold value appears as a substring of the system answer
  - "structural": set overlap (named years / missing items / paper_ids) >= 80%
  - "llm_judge":  gpt-5.4-mini scores 0/1 for semantic equivalence
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.core.budget import record_cost, total_spent
from api.core.classifier import TierClassifier
from api.core.handlers import get_handler
from api.core.llm import MODEL_GPT_MINI, get_openai_client, oai_cost_for_usage
from api.core.retrieval import Retriever
from api.core.store import CorpusStore

router = APIRouter(tags=["eval"])

ROOT = Path(__file__).parent.parent.parent
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
REPORTS_DIR = ROOT / "eval" / "reports"

ABORT_BUDGET_THRESHOLD_USD = 25.0  # leave $5 buffer for hidden test


# ── Schemas ──────────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    budget_level: str = Field(default="$5", pattern=r"^\$(1|5|20)$")
    limit: int | None = Field(default=None, ge=1, le=200,
                              description="Run only the first N questions (for dev).")
    concurrency: int = Field(default=1, ge=1, le=8,
                             description="How many questions to run in parallel.")


class TierStats(BaseModel):
    n: int = 0
    passed: int = 0
    avg_cost: float = 0.0
    avg_latency: float = 0.0
    avg_confidence: float = 0.0


class EvalResponse(BaseModel):
    budget_level: str
    total_questions: int
    passed: int
    pass_rate: float
    per_tier: dict[str, TierStats]
    total_cost_usd: float
    total_latency_seconds: float
    report_path: str
    aborted: bool = False
    abort_reason: str | None = None


# ── Per-tier match strategies ────────────────────────────────────────────────

def _normalize_for_substring(s: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation for substring matching."""
    import re
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


def _substring_match(answer: str, gold: str) -> bool:
    return _normalize_for_substring(gold) in _normalize_for_substring(answer)


def _structural_match(answer: str, gold_items: list[str]) -> tuple[bool, float]:
    """Check that ≥80% of gold items appear in answer."""
    if not gold_items:
        return True, 1.0
    answer_lo = answer.lower()
    matched = sum(1 for item in gold_items if item.lower() in answer_lo)
    overlap = matched / len(gold_items)
    return overlap >= 0.8, overlap


async def _llm_judge_match(question: str, answer: str, gold: str) -> tuple[bool, float]:
    """gpt-5.4-mini scores 0/1 for whether `answer` semantically matches `gold`."""
    client = get_openai_client()
    response = await client.chat.completions.create(
        model=MODEL_GPT_MINI,
        messages=[
            {"role": "system", "content":
                "You are an answer-grader. Given a QUESTION, a GOLD ANSWER, and a SYSTEM ANSWER, "
                "decide if the system answer is semantically equivalent to the gold answer. "
                "Reply ONLY 'PASS' or 'FAIL' on the first line."
            },
            {"role": "user", "content": f"QUESTION: {question}\n\nGOLD: {gold}\n\nSYSTEM: {answer}"},
        ],
        max_completion_tokens=64,
    )
    text = (response.choices[0].message.content or "").strip().upper()
    cost = oai_cost_for_usage(MODEL_GPT_MINI, response.usage)
    record_cost("eval_judge", cost)
    return text.startswith("PASS"), cost


# ── Eval runner ──────────────────────────────────────────────────────────────

def _load_questions(limit: int | None) -> list[dict]:
    if not QUESTIONS_PATH.exists():
        return []
    questions = []
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    if limit:
        questions = questions[:limit]
    return questions


async def _run_one_question(
    q: dict,
    store: CorpusStore,
    retriever: Retriever,
    classifier: TierClassifier,
) -> dict:
    """Run a single eval question end-to-end and judge it."""
    t0 = time.time()
    question = q["question"]
    gold_answer = q.get("gold_answer", "")
    expected_tier = q.get("tier")
    match_strategy = q.get("match_strategy", "substring")

    cost = 0.0
    error = None
    answer = ""
    actual_tier = 0
    actual_confidence = 0.0

    try:
        tier_meta = await classifier.classify(question)
        cost += tier_meta.get("cost_usd", 0.0)
        actual_tier = tier_meta["tier"]
        actual_confidence = tier_meta["confidence"]
        normalized_q = tier_meta.get("normalized_question") or question

        handle = get_handler(actual_tier)
        result = await handle(normalized_q, store, retriever, classifier_meta=tier_meta)
        cost += result.cost_usd
        answer = result.answer
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed = time.time() - t0

    # Judge
    judge_extra_cost = 0.0
    if error:
        passed, overlap = False, 0.0
    elif match_strategy == "substring":
        passed = _substring_match(answer, gold_answer)
        overlap = 1.0 if passed else 0.0
    elif match_strategy == "structural":
        gold_items = q.get("gold_items", [])
        passed, overlap = _structural_match(answer, gold_items)
    elif match_strategy == "llm_judge":
        passed, jc = await _llm_judge_match(question, answer, gold_answer)
        judge_extra_cost = jc
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
        "cost_usd": cost + judge_extra_cost,
        "elapsed_seconds": round(elapsed, 2),
        "error": error,
    }


# ── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/eval", response_model=EvalResponse)
async def evaluate(req: EvalRequest) -> EvalResponse:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"{timestamp}_{req.budget_level.replace('$','')}usd.json"

    # Override BUDGET_LEVEL for the duration of this run
    prev_budget = os.environ.get("BUDGET_LEVEL")
    os.environ["BUDGET_LEVEL"] = req.budget_level

    try:
        questions = _load_questions(req.limit)
        if not questions:
            return EvalResponse(
                budget_level=req.budget_level, total_questions=0, passed=0,
                pass_rate=0.0, per_tier={}, total_cost_usd=0.0,
                total_latency_seconds=0.0, report_path=str(report_path),
                aborted=True, abort_reason=f"no questions in {QUESTIONS_PATH}",
            )

        store = CorpusStore()
        retriever = Retriever()
        classifier = TierClassifier()

        budget_at_start = total_spent()
        per_question_results: list[dict] = []
        aborted = False
        abort_reason = None

        # Sequential by default; concurrency capped to avoid overwhelming the API
        sem = asyncio.Semaphore(req.concurrency)

        async def _bounded(q):
            async with sem:
                return await _run_one_question(q, store, retriever, classifier)

        results = await asyncio.gather(*[_bounded(q) for q in questions])

        # Check budget after each batch (currently runs all in parallel; this is
        # a simple post-hoc check)
        current_spent = total_spent()
        if current_spent > ABORT_BUDGET_THRESHOLD_USD:
            aborted = True
            abort_reason = f"total spend ${current_spent:.2f} exceeds ${ABORT_BUDGET_THRESHOLD_USD}"

        per_question_results.extend(results)

        # ── Aggregate per-tier ──
        per_tier_raw: dict[int, list[dict]] = defaultdict(list)
        for r in per_question_results:
            per_tier_raw[r["tier"]].append(r)

        per_tier_stats: dict[str, TierStats] = {}
        for tier, rows in sorted(per_tier_raw.items()):
            n = len(rows)
            passed = sum(1 for r in rows if r["passed"])
            avg_cost = sum(r["cost_usd"] for r in rows) / max(n, 1)
            avg_latency = sum(r["elapsed_seconds"] for r in rows) / max(n, 1)
            avg_conf = sum(r["tier_confidence"] for r in rows) / max(n, 1)
            per_tier_stats[str(tier)] = TierStats(
                n=n, passed=passed,
                avg_cost=round(avg_cost, 4),
                avg_latency=round(avg_latency, 2),
                avg_confidence=round(avg_conf, 3),
            )

        total_questions = len(per_question_results)
        passed = sum(1 for r in per_question_results if r["passed"])
        total_cost = sum(r["cost_usd"] for r in per_question_results)
        total_latency = sum(r["elapsed_seconds"] for r in per_question_results)

        # ── Write report ──
        report_payload = {
            "timestamp": timestamp,
            "budget_level": req.budget_level,
            "total_questions": total_questions,
            "passed": passed,
            "pass_rate": round(passed / max(total_questions, 1), 3),
            "total_cost_usd": round(total_cost, 4),
            "total_latency_seconds": round(total_latency, 1),
            "budget_at_start_usd": round(budget_at_start, 4),
            "budget_at_end_usd": round(total_spent(), 4),
            "aborted": aborted,
            "abort_reason": abort_reason,
            "per_tier": {k: v.model_dump() for k, v in per_tier_stats.items()},
            "questions": per_question_results,
        }
        report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return EvalResponse(
            budget_level=req.budget_level,
            total_questions=total_questions, passed=passed,
            pass_rate=round(passed / max(total_questions, 1), 3),
            per_tier=per_tier_stats,
            total_cost_usd=round(total_cost, 4),
            total_latency_seconds=round(total_latency, 1),
            report_path=str(report_path),
            aborted=aborted, abort_reason=abort_reason,
        )

    finally:
        if prev_budget is None:
            os.environ.pop("BUDGET_LEVEL", None)
        else:
            os.environ["BUDGET_LEVEL"] = prev_budget
