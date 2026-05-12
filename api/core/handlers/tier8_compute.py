"""
Tier 8 — Quantitative computation.

LLM writes a Python snippet that operates on `pd.DataFrame` views of our
SQLite tables. We exec it in a sandboxed namespace (no file/network/import).
The snippet must assign its final result to `RESULT` (a dict, list, or scalar).

Used for: medians, correlations, regressions, custom math the SQL aggregator
can't express directly.
"""
from __future__ import annotations

import math
import sqlite3
import statistics
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from api.core.budget import record_cost
from api.core.handlers.base import HandlerResult, build_citations, synthesize_answer
from api.core.llm import MODEL_GPT_MINI, get_openai_client, oai_cost_for_usage


SCHEMA_BLURB = """Available pandas DataFrames (loaded from SQLite, you DON'T need to import):

  papers          (paper_id, title, year, venue, citation_count, ...)
  entities        (entity_id, canonical, type, paperswithcode_id, mention_count)
  mentions        (paper_id, entity_id, surface_form, purpose)
  results         (paper_id, model, dataset_id, metric_id,
                   value_canonical FLOAT, value_surface, is_sota_claim, table_caption)
  model_variants  (paper_id, name, param_count_millions FLOAT, param_count_surface)
  training        (paper_id, compute_surface, batch_size, epochs)
  claims          (paper_id, claim_text, evidence_section)
  paper_refs      (paper_id_src, paper_id_dst)

Available helpers (already imported): pd, math, statistics
You CANNOT import other modules or open files/sockets.

Convention: assign the final answer to a variable named RESULT.
RESULT may be a number, dict, or small DataFrame (use .to_dict('records') if so).
"""


class _CodePlan(BaseModel):
    code: str = Field(description="Python snippet that ends by assigning to RESULT.")
    explanation: str = Field(description="One sentence about what the code computes.")


def _load_dataframes(db_path) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    tables = ["papers", "entities", "mentions", "results", "model_variants",
              "training", "claims", "paper_refs"]
    dfs = {t: pd.read_sql_query(f"SELECT * FROM {t}", conn) for t in tables}
    conn.close()
    return dfs


def _run_sandboxed(code: str, dfs: dict[str, pd.DataFrame]) -> tuple[Any, str | None]:
    """Exec code in a restricted namespace. Returns (RESULT, error_message)."""
    # Namespace with only safe primitives
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "float": float, "int": int, "isinstance": isinstance,
        "len": len, "list": list, "max": max, "min": min, "range": range,
        "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
        "tuple": tuple, "zip": zip, "print": print, "True": True, "False": False,
        "None": None,
    }
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "pd": pd, "math": math, "statistics": statistics,
        **dfs,
        "RESULT": None,
    }
    try:
        exec(code, namespace)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    return namespace.get("RESULT"), None


async def handle(question: str, store, retriever, classifier_meta: dict | None = None) -> HandlerResult:
    client = get_openai_client()

    try:
        plan_resp = await client.beta.chat.completions.parse(
            model=MODEL_GPT_MINI,
            messages=[
                {"role": "system", "content":
                    "You write Python (pandas) snippets to compute quantitative answers from "
                    "the corpus. Code runs in a restricted sandbox.\n\n" + SCHEMA_BLURB
                },
                {"role": "user", "content": question},
            ],
            response_format=_CodePlan,
            temperature=0,
            max_completion_tokens=8192,  # gpt-5.4-mini is a reasoning model; needs headroom for reasoning + code
            extra_body={"prompt_cache_key": "tier8-codeplan-v1"},
        )
    except Exception as e:
        # Most common: JSON validation fails when max_completion_tokens runs out
        # mid-output and produces a truncated structured payload.
        return HandlerResult(tier=8, answer=f"Code planning failed: {e}",
                             cost_usd=0.0, confidence=0.0)

    plan = plan_resp.choices[0].message.parsed
    plan_cost = oai_cost_for_usage(MODEL_GPT_MINI, plan_resp.usage)
    record_cost("tier8_plan", plan_cost)

    if plan is None:
        return HandlerResult(tier=8, answer="Could not generate computation.",
                             cost_usd=plan_cost, confidence=0.2)

    dfs = _load_dataframes(store.db_path)
    result, err = _run_sandboxed(plan.code, dfs)

    if err:
        return HandlerResult(
            tier=8, answer=f"Computation failed: {err}",
            evidence=[{"code": plan.code, "error": err}],
            cost_usd=plan_cost, confidence=0.2,
        )

    # Format result for synthesis
    if isinstance(result, pd.DataFrame):
        result_repr = result.head(30).to_dict("records")
    elif isinstance(result, dict) and len(str(result)) > 4000:
        result_repr = {k: result[k] for k in list(result)[:20]}
    else:
        result_repr = result

    evidence = {
        "code": plan.code,
        "explanation": plan.explanation,
        "result": result_repr,
    }

    answer, cited_ids, syn_cost = await synthesize_answer(
        question, evidence,
        instructions="State the computed result with units. Cite contributing papers if applicable.",
    )

    return HandlerResult(
        tier=8, answer=answer,
        citations=build_citations(cited_ids, store),
        evidence=[evidence],
        cost_usd=plan_cost + syn_cost,
        reasoning=plan.explanation,
    )
