"""Shared question execution pipeline for API, CLI, battery, and eval."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from api.core.classifier import TierClassifier
from api.core.handlers import Citation, get_handler
from api.core.handlers.base import ADVERSARIAL_REPLY, is_adversarial
from api.core.retrieval import Retriever
from api.core.runtime import RuntimeConfig, use_runtime
from api.core.store import CorpusStore

CONFIDENCE_FALLBACK_THRESHOLD = 0.5

_store: CorpusStore | None = None
_retriever: Retriever | None = None
_classifier: TierClassifier | None = None


def get_store() -> CorpusStore:
    global _store
    if _store is None:
        _store = CorpusStore()
    return _store


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_classifier() -> TierClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TierClassifier()
    return _classifier


@dataclass
class PipelineResult:
    question: str
    answer: str
    tier: int
    tier_confidence: float
    tier_reasoning: str
    tier_normalized_question: str
    citations: list[Citation] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    handler_reasoning: str | None = None
    handler_confidence: float = 0.0
    elapsed_seconds: float = 0.0
    fallback_used: bool = False
    error: str | None = None


async def answer_question(
    question: str,
    *,
    runtime: RuntimeConfig | None = None,
    target_paper_id: str | None = None,
    store: CorpusStore | None = None,
    retriever: Retriever | None = None,
    classifier: TierClassifier | None = None,
) -> PipelineResult:
    """Run classification, tier handling, and response composition."""
    config = runtime or RuntimeConfig()
    with use_runtime(config):
        return await _answer_question(
            question,
            target_paper_id=target_paper_id,
            store=store or get_store(),
            retriever=retriever or get_retriever(),
            classifier=classifier or get_classifier(),
        )


async def _answer_question(
    question: str,
    *,
    target_paper_id: str | None,
    store: CorpusStore,
    retriever: Retriever,
    classifier: TierClassifier,
) -> PipelineResult:
    t_start = time.time()

    if is_adversarial(question):
        return PipelineResult(
            question=question,
            answer=ADVERSARIAL_REPLY,
            tier=0,
            tier_confidence=1.0,
            tier_reasoning="adversarial pre-check (marker matched)",
            tier_normalized_question=question,
            handler_confidence=1.0,
            elapsed_seconds=round(time.time() - t_start, 2),
        )

    try:
        tier_meta = await classifier.classify(question)
    except Exception as e:
        return PipelineResult(
            question=question,
            answer=f"Classifier failed: {type(e).__name__}: {e}",
            tier=0,
            tier_confidence=0.0,
            tier_reasoning="classifier error",
            tier_normalized_question=question,
            cost_usd=0.0,
            handler_confidence=0.0,
            elapsed_seconds=round(time.time() - t_start, 2),
            error=str(e),
        )

    tier = tier_meta["tier"]
    confidence = tier_meta["confidence"]
    normalized_q = tier_meta.get("normalized_question") or question
    classifier_cost = tier_meta.get("cost_usd", 0.0)
    fallback = False

    if confidence < CONFIDENCE_FALLBACK_THRESHOLD:
        fallback = True
        tier = 1

    handler_meta = dict(tier_meta)
    if target_paper_id:
        handler_meta["target_paper_id"] = target_paper_id

    try:
        handle = get_handler(tier)
        result = await handle(
            normalized_q,
            store,
            retriever,
            classifier_meta=handler_meta,
        )
    except Exception as e:
        return PipelineResult(
            question=question,
            answer=f"Handler T{tier} failed: {type(e).__name__}: {e}",
            tier=tier,
            tier_confidence=confidence,
            tier_reasoning=tier_meta.get("reasoning", ""),
            tier_normalized_question=normalized_q,
            cost_usd=classifier_cost,
            handler_confidence=0.0,
            elapsed_seconds=round(time.time() - t_start, 2),
            fallback_used=fallback,
            error=str(e),
        )

    return PipelineResult(
        question=question,
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
