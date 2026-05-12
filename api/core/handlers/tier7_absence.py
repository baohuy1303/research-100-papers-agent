"""
Tier 7 — Negation / absence.

Closed-world set difference. Strategy:
  1. Ask the LLM what type of "missing" we're looking for (datasets / metrics
     / methods / standard ViT benchmarks NOT covered).
  2. Ask the LLM for the EXPECTED set ("what would you typically see in a
     ViT paper for this question?").
  3. Compute observed_set from our entities table.
  4. Set difference = expected - observed.
  5. Synthesize answer naming what's missing.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from api.core.budget import record_cost
from api.core.handlers.base import HandlerResult, build_citations, synthesize_answer
from api.core.llm import MODEL_GPT_MINI, get_openai_client, oai_cost_for_usage


class _AbsencePlan(BaseModel):
    type: Literal["dataset", "metric", "method"] = Field(
        description="What kind of entity the question is about."
    )
    expected: list[str] = Field(
        description="The canonical entities you would EXPECT to see in a Vision Transformer "
                    "research corpus for this question. Use widely-recognized canonical names "
                    "(e.g. 'ImageNet', 'COCO', 'ADE20K' for datasets)."
    )
    scope_paper: str | None = Field(
        default=None,
        description="If the question asks about absence WITHIN a specific paper "
                    "(e.g. 'what datasets does ViT NOT report on'), give the paper search "
                    "string. Otherwise None for corpus-wide absence."
    )


async def handle(question: str, store, retriever, classifier_meta: dict | None = None) -> HandlerResult:
    client = get_openai_client()

    plan_resp = await client.beta.chat.completions.parse(
        model=MODEL_GPT_MINI,
        messages=[
            {"role": "system", "content":
                "You handle absence/negation questions over a 100-paper Vision Transformer "
                "corpus. First decide what entity type, then propose an EXPECTED set of "
                "canonical names that would normally appear in such a corpus. Be liberal "
                "with the expected set (15-30 items)."
            },
            {"role": "user", "content": question},
        ],
        response_format=_AbsencePlan,
        temperature=0,
        max_completion_tokens=2048,
    )
    plan = plan_resp.choices[0].message.parsed
    plan_cost = oai_cost_for_usage(MODEL_GPT_MINI, plan_resp.usage)
    record_cost("tier7_plan", plan_cost)

    if plan is None:
        return HandlerResult(tier=7, answer="Failed to plan absence query.",
                             cost_usd=plan_cost, confidence=0.2)

    # ── Build the OBSERVED set as union of canonical names + every alias ──
    # The LLM proposes "expected" entities using whatever names it knows
    # (e.g. "ImageNet-1K", "MS COCO"); we must match those against BOTH
    # the canonical names AND the aliases table to avoid false negatives.
    if plan.scope_paper:
        rows = store.execute_sql(
            "SELECT * FROM papers WHERE LOWER(title) LIKE LOWER(?) "
            "ORDER BY citation_count DESC LIMIT 1", (f"%{plan.scope_paper}%",))
        target = rows[0] if rows else None
        if target is None:
            return HandlerResult(tier=7, answer=f"Couldn't find paper matching '{plan.scope_paper}'.",
                                 cost_usd=plan_cost, confidence=0.2)
        # All entities mentioned by this paper, including their aliases
        observed_rows = store.execute_sql(
            """SELECT DISTINCT e.canonical, a.surface_form
               FROM mentions m
               JOIN entities e ON e.entity_id = m.entity_id
               LEFT JOIN aliases a ON a.entity_id = e.entity_id
               WHERE m.paper_id = ? AND e.type = ?""",
            (target["paper_id"], plan.type),
        )
        scope_label = f"in paper '{target['title']}'"
    else:
        observed_rows = store.execute_sql(
            """SELECT e.canonical, a.surface_form
               FROM entities e
               LEFT JOIN aliases a ON a.entity_id = e.entity_id
               WHERE e.type = ?""",
            (plan.type,),
        )
        scope_label = "across the corpus"

    # Build the lowercase observed lookup including BOTH canonical and aliases
    observed_set: set[str] = set()
    for r in observed_rows:
        if r["canonical"]:
            observed_set.add(r["canonical"].lower().strip())
        if r["surface_form"]:
            observed_set.add(r["surface_form"].lower().strip())

    # Also accept fuzzy substring matches — "MS COCO" should match "COCO"
    # because users often qualify common dataset names.
    def _is_present(needle: str) -> bool:
        n = needle.lower().strip()
        if n in observed_set:
            return True
        # Substring fallback: any observed name contains the needle, or vice versa.
        # Cap to ≥4 chars to avoid false positives like "AP" matching every entity.
        if len(n) >= 4:
            for obs in observed_set:
                if n == obs or n in obs or obs in n:
                    return True
        return False

    missing: list[str] = []
    present: list[str] = []
    for exp in plan.expected:
        if _is_present(exp):
            present.append(exp)
        else:
            missing.append(exp)

    evidence = {
        "type": plan.type,
        "scope": scope_label,
        "expected_set_size": len(plan.expected),
        "observed_set_size": len(observed_set),
        "missing": missing,
        "present_for_reference": present[:20],
    }

    answer, cited_ids, syn_cost = await synthesize_answer(
        question, evidence,
        instructions=(
            "List the missing items directly. Acknowledge that 'expected' is "
            "what a typical ViT corpus would include — not an authoritative list. "
            "If nothing is missing, say so."
        ),
    )

    return HandlerResult(
        tier=7, answer=answer,
        citations=build_citations(cited_ids, store),
        evidence=[evidence],
        cost_usd=plan_cost + syn_cost,
        reasoning=f"set_diff over {plan.type}, {len(missing)} missing",
        confidence=0.7 if len(plan.expected) >= 5 else 0.4,
    )
