"""
Tier 4 — Temporal / evolution.

Same NL→SQL approach as Tier 2 but the prompt is biased toward
GROUP BY year and YEAR-OVER-YEAR aggregations. We always include the year
column in the SELECT so the synthesis step can present a time series.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from api.core.budget import record_cost
from api.core.handlers.base import HandlerResult, build_citations, synthesize_answer
from api.core.handlers.tier2_aggregate import SCHEMA_DOC
from api.core.llm import MODEL_GPT_MINI, get_openai_client, oai_cost_for_usage


class _TemporalSQL(BaseModel):
    sql: str = Field(description="A SELECT with GROUP BY year (or year range) showing trend.")
    metric_name: str = Field(description="What is being tracked over time, in plain language.")


async def handle(question: str, store, retriever, classifier_meta: dict | None = None) -> HandlerResult:
    client = get_openai_client()
    plan_resp = await client.beta.chat.completions.parse(
        model=MODEL_GPT_MINI,
        messages=[
            {"role": "system", "content":
                "You translate temporal-evolution questions into a SQLite SELECT with "
                "GROUP BY year (or YEAR-bucketed). Always include the year column so the "
                "result reads as a time series.\n\n" + SCHEMA_DOC + "\n\n"
                "Examples:\n"
                "- 'How did model size grow over years?' -> "
                "  SELECT p.year, AVG(mv.param_count_millions), MAX(mv.param_count_millions) "
                "  FROM model_variants mv JOIN papers p USING(paper_id) "
                "  WHERE mv.param_count_millions IS NOT NULL "
                "  GROUP BY p.year ORDER BY p.year\n"
                "- 'Top-1 acc on ImageNet over years?' -> JOIN results+entities, "
                "  GROUP BY p.year, take MAX(value_canonical).\n"
            },
            {"role": "user", "content": question},
        ],
        response_format=_TemporalSQL,
        temperature=0,
        max_completion_tokens=512,
        extra_body={"prompt_cache_key": "tier4-temporal-v1"},
    )
    plan = plan_resp.choices[0].message.parsed
    plan_cost = oai_cost_for_usage(MODEL_GPT_MINI, plan_resp.usage)
    record_cost("tier4_plan", plan_cost)

    if plan is None:
        return HandlerResult(tier=4, answer="Failed to plan temporal query.",
                             cost_usd=plan_cost, confidence=0.2)

    try:
        rows = store.execute_sql(plan.sql)
    except Exception as e:
        return HandlerResult(
            tier=4, answer=f"SQL execution failed: {e}",
            evidence=[{"sql": plan.sql, "error": str(e)}],
            cost_usd=plan_cost, confidence=0.2,
        )

    evidence = {
        "metric_name": plan.metric_name,
        "sql": plan.sql,
        "time_series_rows": rows[:30],
    }

    answer, cited_ids, syn_cost = await synthesize_answer(
        question, evidence,
        instructions=(
            "Describe the trend across years using the rows. "
            "Mention the start, end, and direction (up/down/flat). "
            "Quote specific year-values where helpful."
        ),
    )

    return HandlerResult(
        tier=4, answer=answer,
        citations=build_citations(cited_ids, store),
        evidence=[evidence],
        cost_usd=plan_cost + syn_cost,
        reasoning=f"temporal: {plan.metric_name}",
    )
