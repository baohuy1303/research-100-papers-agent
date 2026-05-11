"""
Budget control & cost tracking for the system.

Reads BUDGET_LEVEL from env: "$1", "$5", or "$20" (default: "$5").
Tracks per-call USD spend to data/cost_log.jsonl.
"""
import json
import os
import time
from pathlib import Path
from typing import Literal

BudgetLevel = Literal["$1", "$5", "$20"]
DEFAULT_BUDGET = "$5"

COST_LOG = Path(__file__).parent.parent.parent / "data" / "cost_log.jsonl"


def get_budget_level() -> BudgetLevel:
    """Current budget level from env."""
    val = os.getenv("BUDGET_LEVEL", DEFAULT_BUDGET)
    if val not in ("$1", "$5", "$20"):
        return DEFAULT_BUDGET
    return val  # type: ignore[return-value]


# Per-budget settings used by retrieval & handlers
BUDGET_PROFILES: dict[BudgetLevel, dict] = {
    "$1": {
        "default_model": "claude-haiku-4-5",
        "hard_tier_model": "claude-haiku-4-5",
        "retrieve_k": 3,
        "rerank_n": 0,           # 0 = skip reranker
        "decompose_max_steps": 1,
        "tier7_expansion": False,
    },
    "$5": {
        "default_model": "claude-haiku-4-5",
        "hard_tier_model": "claude-sonnet-4-6",
        "retrieve_k": 8,
        "rerank_n": 5,
        "decompose_max_steps": 2,
        "tier7_expansion": True,
    },
    "$20": {
        "default_model": "claude-sonnet-4-6",
        "hard_tier_model": "claude-sonnet-4-6",
        "retrieve_k": 15,
        "rerank_n": 8,
        "decompose_max_steps": 4,
        "tier7_expansion": True,
    },
}


def profile() -> dict:
    """Active profile dict for the current BUDGET_LEVEL."""
    return BUDGET_PROFILES[get_budget_level()]


def record_cost(label: str, usd: float, **meta) -> None:
    """Append a cost event to the log."""
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "label": label,
        "usd": float(usd),
        "budget_level": get_budget_level(),
        **meta,
    }
    with open(COST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def total_spent() -> float:
    """Sum of all recorded costs to date."""
    if not COST_LOG.exists():
        return 0.0
    total = 0.0
    with open(COST_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                total += json.loads(line).get("usd", 0.0)
            except json.JSONDecodeError:
                continue
    return total
