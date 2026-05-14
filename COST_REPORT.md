# Cost Report

## Pipeline cost (one-time prep)

| Phase | What | Tool | Cost | Time |
|-------|------|------|------|------|
| 1 — PDF parsing | 100 PDFs via Datalab Marker cloud API | Datalab (free tier) | ~$5.35 (free-tier keys used) | ~34 min |
| 2 — Structured extraction | GPT-5.4-mini, 100 papers, structured output, 95-99% prompt cache hit | OpenAI | **$1.41** | ~19 min |
| 3a — Number normalization | Regex-only, no LLM | — | **$0.00** | <1s |
| 3b — Entity normalization | Embeddings + HF API + GPT disambiguation (~6 iterations during dev) | OpenAI | **$0.20** | ~75s/run |
| 4 — Build indexes | text-embedding-3-small for 3768 Chroma chunks; S2 API for graph | OpenAI + S2 | **$0.05** | ~2.5 min |
| **Total prep** | | | **~$1.66 paid** ($5.35 free-tier) | **~57 min** |

> Datalab parsing used two free-tier API keys (~55 papers per key). No dollars were charged.  
> The ~$0.20 for entity normalization reflects iterative dev runs; a single production run costs ~$0.04.

---

## Eval run cost (per 40-question run)

| Budget level | Total cost | Avg cost/question | Pass rate |
|---|---|---|---|
| $1 | $0.066 | $0.0017 | **100% (40/40)** |
| $5 | $0.067 | $0.0017 | **100% (40/40)** |
| $20 | $0.069 | $0.0017 | **100% (40/40)** |

Includes classifier cost + handler cost + LLM judge cost (for llm_judge questions).

---

## Quality vs budget

### Accuracy by tier and budget level

| Tier | Description | $1 | $5 | $20 |
|------|-------------|----|----|-----|
| T1 | Single-paper factual | 100% | 100% | 100% |
| T2 | Corpus aggregation (SQL) | 100% | 100% | 100% |
| T3 | Contradiction / comparison | 100% | 100% | 100% |
| T4 | Temporal evolution | 100% | 100% | 100% |
| T5 | Citation graph | 100% | 100% | 100% |
| T6 | Multi-hop compositional | 100% | 100% | 100% |
| T7 | Negation / absence | 100% | 100% | 100% |
| T8 | Quantitative compute | 100% | 100% | 100% |
| **Overall** | | **100%** | **100%** | **100%** |

### Cost per question by tier (at $5 default)

| Tier | Avg cost | Avg latency | Driver |
|------|----------|-------------|--------|
| T1 | $0.0015 | 5–10s | 1 classify + 1 embed + 2 LLM calls |
| T2 | $0.0013 | 3–6s | 1 classify + 1 NL→SQL + 1 synthesize |
| T3 | $0.0018 | 5–15s | 1 classify + 1 plan + retrieval + 1 synthesize |
| T4 | $0.0014 | 3–6s | 1 classify + 1 NL→SQL + 1 synthesize |
| T5 | $0.0011 | 3–8s | 1 classify + 1 graph-op picker + 1 synthesize |
| T6 | $0.0036 | 15–40s | 1 classify + 3–6 tool-calling LLM steps |
| T7 | $0.0013 | 5–15s | 1 classify + 1 expected-set LLM + set diff + 1 synthesize |
| T8 | $0.0015 | 5–15s | 1 classify + 1 code-plan LLM + pandas exec + 1 synthesize |

### What budget level controls

- **Chroma retrieval k**: `$1`→3, `$5`→8, `$20`→15 chunks per search
- **T6 max tool steps**: `$1`→3, `$5`→6, `$20`→10 LLM calls per question
- **T7 expected-set size**: larger expected set at `$20` to catch more obscure absences

In practice, the quality curve is **flat** — all budget levels achieve 100% on our 40-question eval. The main differences are:
- T6 multi-hop questions at `$1` require the 3-step pattern to complete; increasing to $5/$20 provides more exploration headroom for novel question shapes
- Retrieval-heavy tiers (T1, T3) benefit marginally from larger k at `$20`

---

## Total project spend

| Category | Cost |
|----------|------|
| Phase 1 — PDF parsing (Datalab free tier) | $0.00 paid / ~$5.35 credit equivalent |
| Phase 2 — Extraction (OpenAI) | $1.41 |
| Phase 3b — Entity normalization (OpenAI, iterative) | $0.20 |
| Phase 4 — Chroma embeddings (OpenAI) | $0.05 |
| Eval runs + handler dev (OpenAI) | $4.37 |
| **Total paid to OpenAI** | **~$6.03** |
| **Total logged** | **$11.38** (includes all dev iterations) |
| **Hard cap** | $30.00 |
| **Remaining** | ~$18.62 |
