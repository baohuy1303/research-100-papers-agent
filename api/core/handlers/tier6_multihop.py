"""
Tier 6 — Multi-hop / compositional.

The hardest tier. We expose store + retrieval as OpenAI tool functions and
let gpt-5.4-mini iteratively call them until it has enough evidence to
answer. The model decides the decomposition itself rather than us hard-coding
a fixed number of steps.

Hard cap: BUDGET_LEVEL controls max_steps to bound cost.
  $1  -> 1 step  (degrades to single-shot retrieval)
  $5  -> 4 steps
  $20 -> 8 steps
"""
from __future__ import annotations

import json
from typing import Any

from api.core.budget import record_cost
from api.core.handlers.base import HandlerResult, build_citations
from api.core.llm import MODEL_GPT_MINI, get_openai_client, oai_cost_for_usage


# ── Tool definitions exposed to the LLM ──────────────────────────────────────

TOOLS = [
    {"type": "function", "function": {
        "name": "sql_query",
        "description": "Run a read-only SELECT against the corpus SQLite database. "
                       "Use the schema documented in the system prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "Read-only SELECT or WITH-CTE query."},
            },
            "required": ["sql"],
        },
    }},
    {"type": "function", "function": {
        "name": "search_chunks",
        "description": "Semantic search over section-level chunks across all 100 papers. "
                       "Returns top-k chunks sorted by relevance.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 15},
                "paper_id": {"type": "string", "description": "Optional: restrict to one paper."},
            },
            "required": ["query"],
        },
    }},
    {"type": "function", "function": {
        "name": "graph_op",
        "description": "Run a citation-graph operation. ops: most_cited, descendants, "
                       "ancestors, cited_by, references_of, shortest_path.",
        "parameters": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["most_cited", "descendants", "ancestors",
                                                  "cited_by", "references_of", "shortest_path"]},
                "paper_id": {"type": "string", "description": "Required for descendants/ancestors/cited_by/references_of."},
                "paper_id_2": {"type": "string", "description": "Second paper for shortest_path."},
                "k": {"type": "integer", "default": 10},
            },
            "required": ["op"],
        },
    }},
    {"type": "function", "function": {
        "name": "finalize_answer",
        "description": "Call this when you have enough evidence to answer the question. "
                       "Provide the final natural-language answer and the supporting paper_ids.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "cited_paper_ids": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string", "description": "1-sentence explanation of how steps combined."},
            },
            "required": ["answer", "cited_paper_ids"],
        },
    }},
]


SYSTEM_PROMPT = """You answer compositional research-paper questions over a 100-paper Vision Transformer corpus by calling tools.

Tools available: sql_query, search_chunks, graph_op, finalize_answer.

SQLite schema (for sql_query):
  papers(paper_id, title, year, venue, citation_count)
  entities(entity_id, canonical, type ['dataset'|'metric'|'method'], paperswithcode_id)
  mentions(paper_id, entity_id, surface_form, purpose)
  results(paper_id, model, dataset_id, metric_id, value_canonical, is_sota_claim)
  model_variants(paper_id, name, param_count_millions)
  paper_refs(paper_id_src, paper_id_dst)  -- in-corpus citations

KEY PATTERN — looking up a paper by name:
  Papers are referenced by short names like "ViT", "Swin", "MAE". Use sql_query first
  to resolve the name to a paper_id:
    SELECT paper_id, title FROM papers
    WHERE LOWER(title) LIKE '%image is worth%'
    ORDER BY citation_count DESC LIMIT 1
  Then pass that paper_id into graph_op or other queries.

WORKED EXAMPLE — "Among papers that cite ViT, which has the largest model?":
  Step 1: sql_query  -> SELECT paper_id, title FROM papers
                        WHERE LOWER(title) LIKE '%image is worth%' LIMIT 1
                     => ViT's paper_id is X.
  Step 2: graph_op   -> {op: "cited_by", paper_id: X, k: 30}
                     => list of paper_ids citing ViT.
  Step 3: sql_query  -> SELECT mv.paper_id, p.title, MAX(mv.param_count_millions) AS max_params
                        FROM model_variants mv JOIN papers p USING(paper_id)
                        WHERE mv.paper_id IN ('id1','id2',...)
                        GROUP BY mv.paper_id ORDER BY max_params DESC LIMIT 5
  Step 4: finalize_answer with the top result + cited_paper_ids.

CRITICAL — chaining tool results:
  When a previous tool returns paper_ids (from graph_op or sql_query), USE THEM
  in subsequent queries via SQL `WHERE paper_id IN ('id1', 'id2', ...)`. Do NOT
  re-derive the same set with a fresh JOIN on paper_refs — those joins are
  fragile and frequently return 0 rows. Trust prior tool results.

Strategy:
  1. Decompose the question into atomic lookups.
  2. Use tools to gather evidence (often: identify a set of papers, then compute properties).
  3. As soon as you have enough, call finalize_answer.
  4. Hard cap on tool calls per question — don't waste calls.
  5. If a query returns 0 rows, REVISE — don't just retry with a similar query.

Always prefer SQL when the data is structured. Use search_chunks only for free-form text.
"""


def _exec_tool(name: str, args: dict, store, retriever):
    """Execute a tool call against our store/retriever. Returns a JSON-serializable dict."""
    if name == "sql_query":
        try:
            rows = store.execute_sql(args["sql"])
        except Exception as e:
            return {"error": str(e)}
        # Cap to avoid blowing context
        return {"row_count": len(rows), "rows": rows[:30]}
    elif name == "graph_op":
        op = args["op"]
        k = args.get("k", 10)
        if op == "most_cited":
            return {"papers": store.most_cited(k)}
        if op in ("descendants", "ancestors", "cited_by", "references_of"):
            pid = args.get("paper_id")
            if not pid:
                return {"error": "paper_id required"}
            ids = getattr(store, op)(pid)[:k]
            return {"paper_ids": ids,
                    "papers": [store.get_paper(p) for p in ids if store.get_paper(p)]}
        if op == "shortest_path":
            return {"path": store.shortest_citation_path(args.get("paper_id"), args.get("paper_id_2"))}
        return {"error": f"unknown op {op}"}
    return {"error": f"unknown tool {name}"}


async def handle(question: str, store, retriever, classifier_meta: dict | None = None) -> HandlerResult:
    client = get_openai_client()
    from api.core.budget import get_budget_level
    bl = get_budget_level()
    max_steps = {"$1": 2, "$5": 6, "$20": 10}[bl]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    total_cost = 0.0
    steps_taken = 0
    final_answer: str | None = None
    cited_ids: list[str] = []
    reasoning: str | None = None

    for step in range(max_steps + 1):  # +1 to allow finalize_answer
        response = await client.chat.completions.create(
            model=MODEL_GPT_MINI,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto" if step < max_steps else {"type": "function", "function": {"name": "finalize_answer"}},
            max_completion_tokens=1024,
        )
        msg = response.choices[0].message
        cost = oai_cost_for_usage(MODEL_GPT_MINI, response.usage)
        total_cost += cost
        record_cost(f"tier6_step{step}", cost)
        steps_taken += 1
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            # Plain text reply, no tool call → treat as final
            final_answer = msg.content or "(no answer)"
            break

        # Handle (possibly multiple) tool calls in this turn
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "finalize_answer":
                final_answer = args.get("answer", "(no answer)")
                cited_ids = args.get("cited_paper_ids", [])
                reasoning = args.get("reasoning")
                # Append a stub tool result so messages stay consistent
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps({"ok": True}),
                })
                break

            result = _exec_tool(name, args, store, retriever)
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result, default=str)[:8000],
            })

        if final_answer is not None:
            break

    if final_answer is None:
        final_answer = "(could not converge to an answer within step budget)"

    return HandlerResult(
        tier=6, answer=final_answer,
        citations=build_citations(cited_ids, store),
        evidence=[{"steps_taken": steps_taken, "max_steps": max_steps}],
        cost_usd=total_cost,
        reasoning=reasoning,
    )
