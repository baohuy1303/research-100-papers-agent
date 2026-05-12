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
  Step 2: graph_op   -> {op: "cited_by", paper_id: X, k: 100}
                        ALWAYS use k>=100 for cited_by/descendants/ancestors so
                        you don't silently miss high-param papers.
                     => paper_ids = ['id1','id2',...,'idN'].
  Step 3: sql_query  -> ALL of these clauses are REQUIRED for ranked aggregation:
                        SELECT mv.paper_id, p.title,
                               MAX(mv.param_count_millions) AS max_params
                        FROM model_variants mv
                        JOIN papers p ON p.paper_id = mv.paper_id
                        WHERE mv.paper_id IN ('id1','id2',...,'idN')
                          AND mv.param_count_millions IS NOT NULL
                        GROUP BY mv.paper_id, p.title          -- REQUIRED with MAX()
                        ORDER BY max_params DESC               -- REQUIRED for "largest"
                        LIMIT 5                                -- top-N
  Step 4: finalize_answer with the FIRST row's title + cited_paper_ids.

COMMON SQL MISTAKES TO AVOID:
  - MAX() / SUM() / AVG() without GROUP BY: produces undefined ordering, wrong row.
  - Missing ORDER BY when the question says "largest", "highest", "best", "most": you'll
    get an arbitrary row instead of the top one.
  - Missing IS NOT NULL guards on numeric columns: NULLs sort last in SQLite but if the
    set is mostly NULL you'll get nonsense.

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


MAX_TOOL_RESULT_CHARS = 8000


def _safe_serialize(result: dict, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Serialize a tool result to JSON, dropping array entries (not slicing strings)
    if it would otherwise exceed max_chars. Preserves valid JSON shape so the LLM
    never sees a half-cut object/string.

    Trim order matters: bulky verbose fields (e.g. `papers`, full row dicts) are
    redundant with their compact counterparts (e.g. `paper_ids`), so we trim the
    LARGEST list first and keep ID lists intact.
    """
    serialized = json.dumps(result, default=str)
    if len(serialized) <= max_chars:
        return serialized

    trimmed = dict(result)

    def _list_size(val) -> int:
        if not isinstance(val, list):
            return 0
        return len(json.dumps(val, default=str))

    # Iteratively trim the LARGEST list field
    while len(json.dumps(trimmed, default=str)) > max_chars:
        list_keys = [(k, _list_size(v)) for k, v in trimmed.items() if isinstance(v, list) and v]
        if not list_keys:
            break
        biggest_key = max(list_keys, key=lambda kv: kv[1])[0]
        trimmed[biggest_key].pop()  # drop one entry from the biggest list
        # If we just emptied it, replace with a count summary so the LLM
        # still knows the data existed (and to ask via more specific query).
        if not trimmed[biggest_key]:
            trimmed[biggest_key] = f"<emptied during truncation>"

    if len(json.dumps(trimmed, default=str)) > max_chars:
        # Last resort: replace remaining lists with counts
        for key, val in list(trimmed.items()):
            if isinstance(val, list):
                trimmed[key] = f"<{len(val)} items truncated>"

    trimmed["_truncated"] = True
    out = json.dumps(trimmed, default=str)
    return out[:max_chars] if len(out) > max_chars else out


def _exec_tool(name: str, args: dict, store, retriever):
    """Execute a tool call against our store/retriever. Returns a JSON-serializable dict."""
    if name == "sql_query":
        sql = args.get("sql")
        if not sql:
            return {"error": "missing 'sql' argument; pass {\"sql\": \"SELECT ...\"}"}
        try:
            rows = store.execute_sql(sql)
        except Exception as e:
            return {"error": str(e)}
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
            # gpt-5.4-mini is a reasoning model; needs headroom for both
            # internal reasoning AND constructing potentially long SQL
            # queries with IN-lists of paper_ids (~80 ids x 42 chars each).
            max_completion_tokens=4096,
        )
        msg = response.choices[0].message
        cost = oai_cost_for_usage(MODEL_GPT_MINI, response.usage)
        total_cost += cost
        record_cost(f"tier6_step{step}", cost)
        steps_taken += 1
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            content = (msg.content or "").strip()
            # Mid-loop narration like "Let me check...", "I'm gathering evidence..."
            # is NOT a final answer — push the model to commit. Only treat plain
            # text as final on the very last allowed step (where tool_choice is
            # already forced to finalize_answer).
            narration_markers = (
                "let me", "i'm checking", "i'm gathering", "i'll check",
                "i need to", "first,", "next,", "still gathering", "i will",
            )
            looks_like_narration = any(m in content.lower() for m in narration_markers)
            on_last_step = step >= max_steps
            if looks_like_narration and not on_last_step:
                # Nudge it to actually call a tool or finalize
                messages.append({
                    "role": "user",
                    "content": "Don't narrate. Call a tool now (sql_query, "
                               "search_chunks, graph_op) or call finalize_answer "
                               "with what you have.",
                })
                continue
            final_answer = content or "(no answer)"
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
                "content": _safe_serialize(result),
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
