"""
Tier 1 — Single-document factual.

Pipeline:
  1. Identify which paper the question is about (LLM resolves "ViT", "Swin", etc.
     to a paper_id via title substring search).
  2. Try the structured store first — `architecture_summary`, model_variants,
     datasets_mentioned, training. If the question matches a structured field,
     answer from there (no retrieval needed).
  3. Fallback: Chroma retrieve top-k chunks restricted to that paper, synthesize.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from api.core.budget import record_cost
from api.core.handlers.base import HandlerResult, build_citations, synthesize_answer
from api.core.llm import MODEL_GPT_MINI, get_openai_client, oai_cost_for_usage


class _PaperResolution(BaseModel):
    paper_search: str = Field(
        description="Best title-substring search to find the target paper "
                    "(e.g. 'Image is Worth' for ViT, 'Swin Transformer' for Swin)."
    )


async def _resolve_paper(question: str, store) -> tuple[dict | None, float]:
    """Use LLM to extract the target paper name → SQL substring search."""
    client = get_openai_client()
    response = await client.beta.chat.completions.parse(
        model=MODEL_GPT_MINI,
        messages=[
            {"role": "system", "content":
                "Extract a unique paper title substring from the question. "
                "Examples: 'ViT' -> 'Image is Worth'; 'Swin' -> 'Swin Transformer'; "
                "'MAE' -> 'Masked Autoencoders'; 'DeiT' -> 'data-efficient image transformers'."
            },
            {"role": "user", "content": question},
        ],
        response_format=_PaperResolution,
        temperature=0,
        max_completion_tokens=256,
    )
    parsed = response.choices[0].message.parsed
    cost = oai_cost_for_usage(MODEL_GPT_MINI, response.usage)
    record_cost("tier1_resolve", cost)
    if parsed is None:
        return None, cost

    rows = store.execute_sql(
        "SELECT * FROM papers WHERE LOWER(title) LIKE LOWER(?) "
        "ORDER BY citation_count DESC LIMIT 1",
        (f"%{parsed.paper_search}%",),
    )
    return (rows[0] if rows else None), cost


async def handle(question: str, store, retriever, classifier_meta: dict | None = None) -> HandlerResult:
    paper, resolve_cost = await _resolve_paper(question, store)
    if paper is None:
        return HandlerResult(
            tier=1, answer="Could not identify which paper this question is about.",
            cost_usd=resolve_cost, confidence=0.2,
        )

    pid = paper["paper_id"]

    # Pull all structured data for this paper as evidence
    structured = {
        "paper_title": paper["title"],
        "year": paper["year"],
        "venue": paper["venue"],
        "architecture_summary": paper["architecture_summary"],
        "model_variants": store.model_variants_for(pid),
        "training": store.training_for(pid),
        "claims": store.claims_for(pid)[:5],
    }

    # Also retrieve top-k chunks from this paper
    retr = await retriever.search_in_paper(question, paper_id=pid, k=5)
    chunk_evidence = [
        {"section": c["section_title"], "snippet": c["text"][:500]}
        for c in retr["chunks"]
    ]

    evidence = {
        "structured": structured,
        "retrieved_chunks": chunk_evidence,
    }

    answer, cited_ids, syn_cost = await synthesize_answer(
        question, evidence,
        instructions=(
            "All evidence is from a single paper. Prefer structured fields when they answer "
            "the question directly. Fall back to retrieved_chunks for free-form details."
        ),
    )
    if not cited_ids:
        cited_ids = [pid]

    return HandlerResult(
        tier=1, answer=answer,
        citations=build_citations(cited_ids, store),
        evidence=[evidence],
        cost_usd=resolve_cost + retr["cost_usd"] + syn_cost,
        reasoning=f"target paper: {paper['title'][:60]}",
    )
