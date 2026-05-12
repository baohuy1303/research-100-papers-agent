"""
POST /ask — natural-language question over the corpus.

Flow:
  1. Optional override of BUDGET_LEVEL env var
  2. TierClassifier routes to one of 8 tiers
  3. Low-confidence (< 0.5) falls back to Tier 1 with cross-corpus retrieval
  4. Handler executes and returns HandlerResult
  5. Wrap with classifier metadata + tier into AskResponse
"""
from __future__ import annotations

import os
import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.core.classifier import TierClassifier
from api.core.handlers import Citation, get_handler
from api.core.handlers.base import ADVERSARIAL_REPLY, is_adversarial
from api.core.retrieval import Retriever
from api.core.store import CorpusStore

router = APIRouter(tags=["query"])


# ── Lazy module-global instances ────────────────────────────────────────────
# Constructed on first /ask call; survive subsequent requests.
_store: CorpusStore | None = None
_retriever: Retriever | None = None
_classifier: TierClassifier | None = None


def _get_store() -> CorpusStore:
    global _store
    if _store is None:
        _store = CorpusStore()
    return _store


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _get_classifier() -> TierClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TierClassifier()
    return _classifier


CONFIDENCE_FALLBACK_THRESHOLD = 0.5


# ── Request / response schemas ───────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    budget_level: str | None = Field(
        default=None,
        description="One of '$1', '$5', '$20'. If omitted, uses the BUDGET_LEVEL env var.",
    )
    target_paper_id: str | None = Field(
        default=None,
        description="If provided, biases Tier 1 toward this paper (skips paper resolution step).",
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    tier: int
    tier_confidence: float
    tier_reasoning: str
    tier_normalized_question: str
    citations: list[Citation]
    evidence: list[dict]
    cost_usd: float
    handler_reasoning: str | None = None
    handler_confidence: float
    elapsed_seconds: float
    fallback_used: bool = False
    error: str | None = None


# ── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    t_start = time.time()

    # 1. Optional budget override (set env var for the duration of this request)
    prev_budget = os.environ.get("BUDGET_LEVEL")
    if req.budget_level:
        if req.budget_level not in ("$1", "$5", "$20"):
            return AskResponse(
                question=req.question,
                answer=f"Invalid budget_level: {req.budget_level!r}. Use '$1', '$5', or '$20'.",
                tier=0, tier_confidence=0.0, tier_reasoning="invalid input",
                tier_normalized_question=req.question,
                citations=[], evidence=[], cost_usd=0.0,
                handler_confidence=0.0, elapsed_seconds=time.time() - t_start,
                error="invalid budget_level",
            )
        os.environ["BUDGET_LEVEL"] = req.budget_level

    try:
        store = _get_store()
        retriever = _get_retriever()
        classifier = _get_classifier()

        # 1.5. Adversarial pre-check (shared with CLI + battery via
        # api.core.handlers.base) — bail early on non-existent/fictional
        # references so we don't burn budget on the classifier + handler.
        if is_adversarial(req.question):
            return AskResponse(
                question=req.question,
                answer=ADVERSARIAL_REPLY,
                tier=0, tier_confidence=1.0,
                tier_reasoning="adversarial pre-check (marker matched)",
                tier_normalized_question=req.question,
                citations=[], evidence=[], cost_usd=0.0,
                handler_confidence=1.0,
                elapsed_seconds=time.time() - t_start,
            )

        # 2. Classify
        try:
            tier_meta = await classifier.classify(req.question)
        except Exception as e:
            return AskResponse(
                question=req.question,
                answer=f"Classifier failed: {type(e).__name__}: {e}",
                tier=0, tier_confidence=0.0, tier_reasoning="classifier error",
                tier_normalized_question=req.question,
                citations=[], evidence=[], cost_usd=0.0,
                handler_confidence=0.0, elapsed_seconds=time.time() - t_start,
                error=str(e),
            )

        tier = tier_meta["tier"]
        confidence = tier_meta["confidence"]
        normalized_q = tier_meta.get("normalized_question") or req.question
        classifier_cost = tier_meta.get("cost_usd", 0.0)

        # 3. Low confidence fallback → Tier 1 cross-corpus
        fallback = False
        if confidence < CONFIDENCE_FALLBACK_THRESHOLD:
            fallback = True
            tier = 1

        # 4. Execute handler (catch & report any errors)
        try:
            handle = get_handler(tier)
            result = await handle(
                normalized_q, store, retriever,
                classifier_meta=tier_meta,
            )
        except Exception as e:
            return AskResponse(
                question=req.question,
                answer=f"Handler T{tier} failed: {type(e).__name__}: {e}",
                tier=tier, tier_confidence=confidence,
                tier_reasoning=tier_meta.get("reasoning", ""),
                tier_normalized_question=normalized_q,
                citations=[], evidence=[], cost_usd=classifier_cost,
                handler_confidence=0.0,
                elapsed_seconds=time.time() - t_start,
                fallback_used=fallback, error=str(e),
            )

        # 5. Compose response
        return AskResponse(
            question=req.question,
            answer=result.answer,
            tier=result.tier,
            tier_confidence=confidence,
            tier_reasoning=tier_meta.get("reasoning", ""),
            tier_normalized_question=normalized_q,
            citations=result.citations,
            evidence=result.evidence,
            cost_usd=classifier_cost + result.cost_usd,
            handler_reasoning=result.reasoning,
            handler_confidence=result.confidence,
            elapsed_seconds=round(time.time() - t_start, 2),
            fallback_used=fallback,
        )

    finally:
        # Restore prior BUDGET_LEVEL env var
        if req.budget_level:
            if prev_budget is None:
                os.environ.pop("BUDGET_LEVEL", None)
            else:
                os.environ["BUDGET_LEVEL"] = prev_budget
