"""
Tier 3 — Comparative / contradiction.

Two paths:
  - NUMERIC contradictions: Find rows in the `results` table where multiple
    papers report different numbers on the same (dataset, metric) pair.
    Compute STDDEV; flag pairs where the spread is large or where multiple
    papers each claim SOTA with different values.
  - TEXTUAL contradictions: Cross-paper claim retrieval — find chunks across
    papers that talk about the same topic, ask LLM to identify disagreements.

LLM picks which path based on the question wording.
"""
from __future__ import annotations

import statistics
from typing import Literal

from pydantic import BaseModel, Field

from api.core.budget import record_cost
from api.core.handlers.base import HandlerResult, build_citations, synthesize_answer
from api.core.llm import MODEL_GPT_MINI, get_openai_client, oai_cost_for_usage


class _ContradictPlan(BaseModel):
    mode: Literal["numeric", "textual"] = Field(
        description="numeric = compare benchmark numbers; textual = compare written claims."
    )
    dataset: str | None = Field(default=None, description="Canonical dataset (numeric mode).")
    metric: str | None = Field(default=None, description="Canonical metric (numeric mode).")
    topic_query: str | None = Field(
        default=None,
        description="For textual mode, a search query for relevant claims/chunks "
                    "(e.g. 'role of position embeddings').",
    )


async def handle(question: str, store, retriever, classifier_meta: dict | None = None) -> HandlerResult:
    client = get_openai_client()

    plan_resp = await client.beta.chat.completions.parse(
        model=MODEL_GPT_MINI,
        messages=[
            {"role": "system", "content":
                "Decide whether the contradiction question is about NUMERIC benchmark "
                "disagreement or TEXTUAL methodological disagreement. For numeric, fill "
                "dataset and metric (canonical names like 'ImageNet', 'top-1 accuracy', 'mIoU'). "
                "For textual, give a topic_query to retrieve relevant passages."
            },
            {"role": "user", "content": question},
        ],
        response_format=_ContradictPlan,
        temperature=0,
        max_completion_tokens=512,
    )
    plan = plan_resp.choices[0].message.parsed
    plan_cost = oai_cost_for_usage(MODEL_GPT_MINI, plan_resp.usage)
    record_cost("tier3_plan", plan_cost)

    extra_cost = 0.0

    if plan and plan.mode == "numeric" and plan.dataset and plan.metric:
        # Pull all results for this (dataset, metric) pair
        rows = store.results_for(dataset=plan.dataset, metric=plan.metric)
        if len(rows) < 2:
            return HandlerResult(
                tier=3,
                answer=f"Not enough data on {plan.dataset}/{plan.metric} to compare.",
                evidence=[{"row_count": len(rows)}],
                cost_usd=plan_cost, confidence=0.3,
            )

        values = [r["value_canonical"] for r in rows if r["value_canonical"] is not None]
        sota_rows = [r for r in rows if r["is_sota_claim"]]
        spread = max(values) - min(values) if values else 0
        stddev = statistics.pstdev(values) if len(values) > 1 else 0

        evidence = {
            "dataset": plan.dataset,
            "metric": plan.metric,
            "n_results": len(rows),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "spread": spread,
            "stddev": round(stddev, 2),
            "sota_claims": [
                {"paper_id": r["paper_id"], "paper_title": r["paper_title"],
                 "model": r["model"], "value": r["value_canonical"]}
                for r in sota_rows
            ],
            "top_5_results": [
                {"paper_id": r["paper_id"], "paper_title": r["paper_title"],
                 "model": r["model"], "value": r["value_canonical"]}
                for r in rows[:5]
            ],
        }
        instructions = (
            "Discuss the spread of numbers and any conflicting SOTA claims. "
            "If multiple papers claim SOTA with different values, that IS a contradiction."
        )

    else:
        # Textual mode: retrieve chunks across the corpus
        topic = plan.topic_query if plan and plan.topic_query else question
        retr = await retriever.search(topic, k=8)
        extra_cost += retr["cost_usd"]
        evidence = {
            "mode": "textual",
            "topic": topic,
            "passages": [
                {"paper_id": c["paper_id"], "section": c["section_title"],
                 "snippet": c["text"][:400]}
                for c in retr["chunks"]
            ],
        }
        instructions = (
            "Read the passages and identify any direct disagreements between papers. "
            "Quote the conflicting claims. If no real contradiction exists, say so."
        )

    answer, cited_ids, syn_cost = await synthesize_answer(
        question, evidence, instructions=instructions, max_completion_tokens=1500,
    )

    return HandlerResult(
        tier=3, answer=answer,
        citations=build_citations(cited_ids, store),
        evidence=[evidence],
        cost_usd=plan_cost + extra_cost + syn_cost,
        reasoning=f"mode={plan.mode if plan else 'textual'}",
    )
