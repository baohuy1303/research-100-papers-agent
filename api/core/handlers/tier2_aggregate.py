"""
Tier 2 — Corpus-level aggregation.

LLM writes a single read-only SELECT against our SQLite schema, we execute it
via store.execute_sql() (which blocks anything but SELECT/WITH), then we
synthesize the result rows into a NL answer.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from api.core.budget import record_cost
from api.core.handlers.base import HandlerResult, build_citations, synthesize_answer
from api.core.llm import MODEL_GPT_MINI, get_openai_client, oai_cost_for_usage


SCHEMA_DOC = """SQLite schema (read-only). Table: column descriptions:

papers              (paper_id PK, title, year, venue, citation_count,
                     architecture_summary, pdf_path)
entities            (entity_id PK, canonical, type ['dataset'|'metric'|'method'],
                     paperswithcode_id, hf_id, mention_count, source)
aliases             (entity_id FK, surface_form)             -- alt names per entity
mentions            (paper_id FK, entity_id FK, surface_form,
                     purpose ['pretrain'|'finetune'|'eval'|null])
results             (result_id PK, paper_id FK, model,
                     dataset_id FK, metric_id FK,
                     value_canonical REAL, value_surface, is_sota_claim INT, table_caption)
model_variants      (paper_id FK, name, param_count_millions REAL, param_count_surface)
training            (paper_id PK, compute_surface, batch_size, epochs)
claims              (claim_id PK, paper_id FK, claim_text, evidence_section)
paper_refs          (paper_id_src, paper_id_dst)              -- in-corpus citations

Important:
- Use canonical names from entities (e.g. 'ImageNet'), not surface forms.
- Datasets vs metrics are both in `entities` -- always filter by `type` column.
- value_canonical is a float (top-1 accuracy as %, mIoU as %, etc.).
- Use CASE-INSENSITIVE matching on text where the user might have minor variation.
"""


class _SQLPlan(BaseModel):
    sql: str = Field(description="A single read-only SELECT (or WITH-CTE) query.")
    explanation: str = Field(description="One sentence explaining what this query computes.")


async def _generate_sql(question: str) -> tuple[_SQLPlan, float]:
    client = get_openai_client()
    response = await client.beta.chat.completions.parse(
        model=MODEL_GPT_MINI,
        messages=[
            {"role": "system", "content":
                "You translate natural-language questions into a SINGLE read-only SQLite "
                "SELECT (or WITH ... SELECT) statement.\n\n" + SCHEMA_DOC + "\n\n"
                "Rules:\n"
                "- Output a single statement (no semicolons inside).\n"
                "- Use COUNT(DISTINCT ...) to avoid double-counting from joins.\n"
                "- Limit text columns where helpful with LIMIT 30.\n"
                "- Filter benchmark joins by entity type (e.g. e.type='dataset').\n"
                "\n"
                "CRITICAL — when the question names a metric (top-1 accuracy, mIoU, FID,\n"
                "AP, R@1, etc.), JOIN entities TWICE and filter BOTH dataset AND metric:\n"
                "  SELECT ...\n"
                "  FROM results r\n"
                "  JOIN entities ed ON ed.entity_id = r.dataset_id\n"
                "  JOIN entities em ON em.entity_id = r.metric_id\n"
                "  WHERE ed.canonical = 'ImageNet' AND em.canonical = 'top-1 accuracy'\n"
                "Without the metric filter you'll mix top-1 accuracy with FID, mIoU, etc.\n"
                "and return numerically meaningless rows.\n"
                "\n"
                "Use canonical names from the entities table:\n"
                "  metrics: 'top-1 accuracy', 'top-5 accuracy', 'mIoU', 'AP', 'mAP', 'R@1', 'PSNR', ...\n"
                "  datasets: 'ImageNet', 'ImageNet-21k', 'COCO', 'ADE20K', 'CIFAR-10', 'CIFAR-100', ...\n"
                "\n"
                "COUNTING ENTITIES — use the entities table DIRECTLY (not via mentions):\n"
                "  To count unique canonical datasets across all papers:\n"
                "    SELECT COUNT(DISTINCT canonical) FROM entities WHERE type='dataset'\n"
                "  To count entities that appear in at least one paper:\n"
                "    SELECT COUNT(DISTINCT e.canonical) FROM entities e JOIN mentions m ON m.entity_id=e.entity_id WHERE e.type='dataset'\n"
                "  Do NOT add extra WHERE clauses (like WHERE purpose IS NOT NULL) that would drastically reduce the count.\n"
            },
            {"role": "user", "content": question},
        ],
        response_format=_SQLPlan,
        temperature=0,
        max_completion_tokens=512,
        extra_body={"prompt_cache_key": "tier2-nl2sql-v1"},
    )
    plan = response.choices[0].message.parsed
    cost = oai_cost_for_usage(MODEL_GPT_MINI, response.usage)
    record_cost("tier2_nl2sql", cost)
    return plan, cost


async def handle(question: str, store, retriever, classifier_meta: dict | None = None) -> HandlerResult:
    plan, plan_cost = await _generate_sql(question)
    if plan is None:
        return HandlerResult(tier=2, answer="Failed to generate SQL.", cost_usd=plan_cost, confidence=0.2)

    rows = None
    for attempt in range(2):
        try:
            rows = store.execute_sql(plan.sql)
            break
        except Exception as e:
            if attempt == 1:
                return HandlerResult(
                    tier=2, answer=f"SQL execution failed: {e}",
                    evidence=[{"sql": plan.sql, "error": str(e)}],
                    cost_usd=plan_cost, confidence=0.2,
                )
            retry_plan, retry_cost = await _generate_sql(
                question + f"\n\nPrevious SQL failed: {e}\nFailed SQL was: {plan.sql}\nPlease write corrected SQL."
            )
            plan_cost += retry_cost
            if retry_plan:
                plan = retry_plan

    # Cap evidence size — if SQL returned thousands of rows, truncate
    truncated = False
    if len(rows) > 50:
        rows = rows[:50]
        truncated = True

    evidence = {
        "sql": plan.sql,
        "explanation": plan.explanation,
        "row_count": len(rows),
        "truncated": truncated,
        "rows": rows,
    }

    answer, cited_ids, syn_cost = await synthesize_answer(
        question, evidence,
        instructions="The result rows are authoritative. State the count or list directly.",
    )

    return HandlerResult(
        tier=2, answer=answer,
        citations=build_citations(cited_ids, store),
        evidence=[evidence],
        cost_usd=plan_cost + syn_cost,
        reasoning=plan.explanation,
    )
