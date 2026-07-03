"""POST /ask: natural-language questions over the paper corpus."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.core.handlers import Citation
from api.core.pipeline import answer_question
from api.core.runtime import RuntimeConfig, validate_budget_level

router = APIRouter(tags=["query"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    budget_level: str | None = Field(
        default=None,
        description="One of '$1', '$5', '$20'. If omitted, uses the default/runtime budget.",
    )
    target_paper_id: str | None = Field(
        default=None,
        description="Optional paper_id for Tier 1 questions to skip title resolution.",
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


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    try:
        budget_level = validate_budget_level(req.budget_level) if req.budget_level else None
    except ValueError as e:
        return AskResponse(
            question=req.question,
            answer=str(e),
            tier=0,
            tier_confidence=0.0,
            tier_reasoning="invalid input",
            tier_normalized_question=req.question,
            citations=[],
            evidence=[],
            cost_usd=0.0,
            handler_confidence=0.0,
            elapsed_seconds=0.0,
            error="invalid budget_level",
        )

    result = await answer_question(
        req.question,
        runtime=RuntimeConfig(budget_level=budget_level),
        target_paper_id=req.target_paper_id,
    )
    return AskResponse(**result.__dict__)
