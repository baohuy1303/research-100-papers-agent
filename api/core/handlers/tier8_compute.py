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

CRITICAL PATTERN — filtering papers by a named entity (dataset / metric / method):
  # Step 1: find the entity_id(s) for the canonical name
  entity_ids = entities.loc[
      entities['canonical'].str.lower() == 'imagenet', 'entity_id'
  ]
  # Step 2: get paper_ids that mention those entities
  paper_ids_with_entity = mentions.loc[
      mentions['entity_id'].isin(entity_ids), 'paper_id'
  ].unique()
  # Step 3: filter the papers DataFrame
  imagenet_papers = papers.loc[papers['paper_id'].isin(paper_ids_with_entity)]
  other_papers    = papers.loc[~papers['paper_id'].isin(paper_ids_with_entity)]
  # Now compute statistics on each group:
  RESULT = {
      'imagenet_avg_citations': float(imagenet_papers['citation_count'].mean()),
      'other_avg_citations':    float(other_papers['citation_count'].mean()),
  }

Use this pattern ANY TIME the question asks about papers that use / benchmark on /
mention a specific dataset, metric, or method. Always use .isin() — never .merge()
on entity lookups, as merge can silently produce empty results.
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
    import json as _json
    import re as _re

    client = get_openai_client()

    try:
        plan_resp = await client.chat.completions.create(
            model=MODEL_GPT_MINI,
            messages=[
                {"role": "system", "content":
                    "You write Python (pandas) snippets to compute quantitative answers from "
                    "the corpus. Code runs in a restricted sandbox.\n\n" + SCHEMA_BLURB
                    + "\nRespond with ONLY a JSON object containing keys 'code' (Python string) "
                    "and 'explanation' (one sentence). No markdown fences."
                },
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=8192,
            extra_body={"prompt_cache_key": "tier8-codeplan-v2"},
        )
    except Exception as e:
        return HandlerResult(tier=8, answer=f"Code planning failed: {e}",
                             cost_usd=0.0, confidence=0.0)

    plan_cost = oai_cost_for_usage(MODEL_GPT_MINI, plan_resp.usage)
    raw = (plan_resp.choices[0].message.content or "").strip()

    # Extract JSON — use raw_decode to consume first valid object and ignore trailing content.
    # This handles cases where the model appends extra text or reasoning after the JSON.
    stripped = raw.lstrip()
    # Strip markdown fences if present
    stripped = _re.sub(r"^```(?:json)?\s*", "", stripped, flags=_re.DOTALL)
    stripped = _re.sub(r"\s*```$", "", stripped.rstrip(), flags=_re.DOTALL)
    # Find the first { to start parsing from
    start = stripped.find("{")
    if start == -1:
        return HandlerResult(tier=8, answer="Code planning failed: no JSON in response",
                             cost_usd=plan_cost, confidence=0.0)
    try:
        data, _ = _json.JSONDecoder().raw_decode(stripped, idx=start)
        plan = _CodePlan(**data)
    except Exception as e:
        return HandlerResult(tier=8, answer=f"Code planning failed: {e}",
                             cost_usd=plan_cost, confidence=0.0)
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
