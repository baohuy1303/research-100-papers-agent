"""
Shared types and helpers for the 8 tier handlers.

All handlers expose:
    async def handle(question: str, store, retriever, classifier_meta=None) -> HandlerResult
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from api.core.budget import record_cost
from api.core.llm import (
    MODEL_GPT_MINI,
    get_openai_client,
    oai_cost_for_usage,
)


class Citation(BaseModel):
    paper_id: str
    paper_title: str | None = None
    section: str | None = None
    snippet: str | None = None


class HandlerResult(BaseModel):
    tier: int
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    cost_usd: float = 0.0
    confidence: float = 1.0
    reasoning: str | None = None


# ── Synthesis: ask gpt-5.4-mini to write a final NL answer from evidence ─────

class _Synthesis(BaseModel):
    answer: str = Field(description="Concise 1-3 sentence answer to the question, in plain English.")
    cited_paper_ids: list[str] = Field(
        default_factory=list,
        description="paper_ids from the evidence that support the answer.",
    )


async def synthesize_answer(
    question: str,
    evidence: list[dict] | dict,
    instructions: str = "",
    model: str = MODEL_GPT_MINI,
    max_completion_tokens: int = 1024,
) -> tuple[str, list[str], float]:
    """Ask the LLM to write a final NL answer from structured evidence.

    Returns: (answer_text, cited_paper_ids, cost_usd).
    """
    client = get_openai_client()
    sys_prompt = (
        "You are a research-paper QA assistant. You will be given a question "
        "and structured EVIDENCE drawn from a 100-paper Vision Transformer corpus. "
        "Your job is to write a faithful, concise answer using only the evidence. "
        "Do NOT invent facts beyond the evidence. If the evidence is insufficient, say so.\n"
        + (instructions + "\n" if instructions else "")
        + "\nReply as JSON: {answer: str, cited_paper_ids: [str]}."
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE (JSON):\n{json.dumps(evidence, indent=2, default=str, ensure_ascii=False)}"
    )
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        response_format=_Synthesis,
        temperature=0,
        max_completion_tokens=max_completion_tokens,
    )
    parsed = response.choices[0].message.parsed
    cost = oai_cost_for_usage(model, response.usage)
    record_cost("handler_synthesize", cost, model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens)
    if parsed is None:
        return "(no answer produced)", [], cost
    return parsed.answer, parsed.cited_paper_ids, cost


def build_citations(paper_ids: list[str], store, snippet_lookup: dict[str, str] | None = None) -> list[Citation]:
    """Resolve paper_ids → full Citation objects with title (and optional snippet)."""
    out = []
    for pid in paper_ids:
        paper = store.get_paper(pid)
        if paper is None:
            continue
        out.append(Citation(
            paper_id=pid,
            paper_title=paper.get("title"),
            snippet=(snippet_lookup or {}).get(pid),
        ))
    return out
