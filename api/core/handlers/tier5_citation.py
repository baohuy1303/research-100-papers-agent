"""
Tier 5 — Citation-graph reasoning.

LLM picks one of a small set of graph operations (most_cited, descendants,
ancestors, shortest_path, pagerank), then we run it deterministically on the
NetworkX graph and synthesize the answer.

Examples:
  - "Which paper is most cited within this corpus?"      → most_cited
  - "What papers build on ViT?"                          → descendants
  - "Is there a citation chain from MAE to BEiT?"        → shortest_path
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from api.core.budget import record_cost
from api.core.handlers.base import HandlerResult, build_citations, synthesize_answer
from api.core.llm import MODEL_GPT_MINI, get_openai_client, oai_cost_for_usage


class _GraphPlan(BaseModel):
    op: Literal["most_cited", "pagerank", "descendants", "ancestors",
                "shortest_path", "cited_by", "references_of"]
    paper_query: str | None = Field(
        default=None,
        description="If op needs a paper, give a short search string for store.entity_search "
                    "(e.g. 'ViT', 'Swin Transformer', 'MAE'). Use None for most_cited/pagerank.",
    )
    paper_query_2: str | None = Field(
        default=None,
        description="Second paper for shortest_path queries.",
    )
    k: int = Field(default=10, ge=1, le=30)


_PAPER_NICKNAMES: dict[str, str] = {
    "vit": "image is worth 16x16",
    "deit": "data-efficient image transformers",
    "swin": "swin transformer",
    "mae": "masked autoencoders",
    "beit": "bert pre-training of image",
    "dino": "emerging properties in self-supervised",
    "clip": "learning transferable visual",
    "convnext": "convnet for the 2020s",
    "detr": "end-to-end object detection",
    "bert": "bert: pre-training of deep",
}


def _resolve_paper(query: str, store) -> dict | None:
    """Find a paper by title substring; prefer most-cited match.
    Falls back to well-known nickname expansions when acronym search finds nothing.
    """
    if not query:
        return None
    searches = [query]
    # If query looks like an acronym/nickname, add the expanded search term
    nickname_key = query.lower().strip()
    if nickname_key in _PAPER_NICKNAMES:
        searches.append(_PAPER_NICKNAMES[nickname_key])
    for term in searches:
        rows = [r for r in store.execute_sql(
            "SELECT * FROM papers WHERE LOWER(title) LIKE LOWER(?) "
            "ORDER BY citation_count DESC LIMIT 5",
            (f"%{term}%",),
        )]
        if rows:
            return rows[0]
    return None


async def handle(question: str, store, retriever, classifier_meta: dict | None = None) -> HandlerResult:
    client = get_openai_client()

    # ── Step 1: ask LLM which graph op to run ──
    plan_resp = await client.beta.chat.completions.parse(
        model=MODEL_GPT_MINI,
        messages=[
            {"role": "system", "content":
                "You convert citation-graph questions into a graph operation. "
                "Pick the most appropriate op. For 'most cited within corpus' use most_cited. "
                "For 'papers building on X' use descendants. For 'papers X is built on' use "
                "ancestors. For citation chain between A and B use shortest_path. "
                "For 'does X cite Y' use references_of on X and check if Y is in the result. "
                "IMPORTANT: If the question asks HOW MANY papers cite something, set k=100 so "
                "you retrieve all of them, not just the default top-10."
            },
            {"role": "user", "content": question},
        ],
        response_format=_GraphPlan,
        temperature=0,
        max_completion_tokens=1024,
    )
    plan = plan_resp.choices[0].message.parsed
    plan_cost = oai_cost_for_usage(MODEL_GPT_MINI, plan_resp.usage)
    record_cost("tier5_plan", plan_cost, model=MODEL_GPT_MINI)

    # ── Step 2: execute the op ──
    evidence: list[dict] = []
    notes: list[str] = []

    if plan.op == "most_cited":
        rows = store.most_cited(plan.k)
        evidence = [{"paper_id": r["paper_id"], "title": r["title"], "year": r["year"],
                     "in_corpus_citations": r["in_corpus_citations"]} for r in rows]

    elif plan.op == "pagerank":
        rows = store.pagerank_top(plan.k)
        evidence = [{"paper_id": r["paper_id"], "title": r["title"], "year": r["year"],
                     "pagerank": r["pagerank"]} for r in rows]

    elif plan.op in ("descendants", "ancestors", "cited_by", "references_of"):
        target = _resolve_paper(plan.paper_query or "", store)
        if target is None:
            return HandlerResult(tier=5, answer=f"No paper found matching '{plan.paper_query}'.",
                                 cost_usd=plan_cost, confidence=0.2)
        if plan.op == "descendants":
            ids = store.descendants(target["paper_id"])
        elif plan.op == "ancestors":
            ids = store.ancestors(target["paper_id"])
        elif plan.op == "cited_by":
            ids = store.cited_by(target["paper_id"])
        else:
            ids = store.references_of(target["paper_id"])
        total_count = len(ids)
        notes.append(f"target paper: {target['title']}")
        notes.append(f"total_count: {total_count}")
        # Cap to top by citation_count for evidence brevity, but keep total count
        rows = []
        for pid in ids:
            p = store.get_paper(pid)
            if p:
                rows.append(p)
        rows.sort(key=lambda p: -(p.get("citation_count") or 0))
        rows = rows[:plan.k]
        evidence = [{"paper_id": r["paper_id"], "title": r["title"], "year": r["year"]}
                    for r in rows]

    elif plan.op == "shortest_path":
        a = _resolve_paper(plan.paper_query or "", store)
        b = _resolve_paper(plan.paper_query_2 or "", store)
        if not a or not b:
            return HandlerResult(tier=5, answer="Could not resolve both papers for path query.",
                                 cost_usd=plan_cost, confidence=0.2)
        path = store.shortest_citation_path(a["paper_id"], b["paper_id"])
        if path is None:
            evidence = [{"path": None, "from": a["title"], "to": b["title"]}]
            notes.append("no citation path exists")
        else:
            evidence = [{"step": i, "paper_id": pid,
                         "title": store.get_paper(pid)["title"]}
                        for i, pid in enumerate(path)]

    instructions = (
        "Give a direct natural-language answer naming the top results. "
        "Prefer paper titles over IDs."
    )
    if notes:
        evidence_packet: dict = {"results": evidence, "notes": notes}
    else:
        evidence_packet = {"results": evidence}

    answer, cited_ids, syn_cost = await synthesize_answer(
        question, evidence_packet, instructions=instructions,
    )

    # If LLM didn't name papers, default to top evidence rows
    if not cited_ids:
        cited_ids = [e["paper_id"] for e in evidence if "paper_id" in e][:5]

    return HandlerResult(
        tier=5,
        answer=answer,
        citations=build_citations(cited_ids, store),
        evidence=evidence,
        cost_usd=plan_cost + syn_cost,
        reasoning=f"graph op = {plan.op}",
    )
