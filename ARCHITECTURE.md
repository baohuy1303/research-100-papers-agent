# Architecture — Research Comprehension System

## Context

Take-home assessment requires a system that answers natural-language questions across **8 difficulty tiers** over our 100-paper Vision Transformer corpus, with cited answers. Tasks 1 & 2 (corpus assembly) are done — 100 PDFs are at `data/pdfs/` with manifest at `data/manifest.csv`.

**Constraints**: $30 USD hard cap, no fine-tuning, results must be defensible/citeable, judged on robustness (45%), cost-efficiency (35%), live test-set performance (20%). The system must produce a quality-vs-budget curve at $1 / $5 / $20 levels.

**Core insight driving the design**: most "hard" tiers (2, 4, 7, 8) collapse into SQL queries *if* upfront extraction + normalization is thorough. So we invest one-time cost in a comprehensive extraction + normalization pass, then keep query-time cost low.

---

## Stack (decided)

| Layer | Choice | Why this over others |
|---|---|---|
| PDF parser | **Marker** | Markdown output with section headers, good tables, runs on CPU. Considered Nougat (slow without GPU), Docling (newer, less battle-tested), PyMuPDF (loses tables). |
| Extraction LLM | **Claude Haiku 4.5** | ~10× cheaper than Sonnet for structured extraction work. Sonnet reserved for hard query-time reasoning. Considered Gemini Flash (less reliable structured outputs), GPT-4o-mini (no benefit over Haiku). |
| Embeddings | **OpenAI text-embedding-3-small** | ~$0.05 total for our corpus, strong semantic match. Considered local BGE/MiniLM (free but lower quality), Voyage (overkill at our scale). |
| Vector store | **Chroma local** | Persistent local store, easy Python client, survives restarts. Considered sqlite-vec (single-file but less mature), LanceDB (better at larger scale). |
| Reranker | **BGE-reranker-v2 local** | Free, ~10× top-1 lift after dense retrieval. Considered Cohere rerank (paid, similar quality), LLM-rerank (Haiku, more flexible but adds latency/cost). |
| Structured store | **SQLite** | Universal, serverless, fits beside Chroma. DuckDB faster on analytics but unnecessary at this scale. |
| Citation graph | **Semantic Scholar `references` API** | Free, pre-resolved IDs to match against our 100. Considered Grobid (extract from PDF text, brittle), OpenAlex (equivalent, S2 already used for corpus). |
| Query routing | **LLM tier-classifier + per-tier handlers** | Predictable cost, debuggable, easy to instrument. Considered ReAct agent (unpredictable cost), DSPy (more setup), hybrid agent-per-tier (middle ground, deferred). |
| Query LLM | **Tiered: Haiku default, Sonnet for tiers 3/6/7** | Best $/quality. Hard reasoning tiers get Sonnet; lookups/aggregations stay on Haiku. |
| Caching | **Aggressive Anthropic prompt caching** | Cache extraction system prompt + schema (~3-5k tokens) repeated 100×; cuts extraction cost ~10×. |

---

## High-level pipeline

```
                  ┌────────────────────────────────────────────────┐
                  │ ONE-TIME PREP (~$5 of $30 budget)              │
                  │                                                │
PDFs ─► Marker ──►│ Pass 1: Verbatim Extraction (Haiku 4.5)        │
                  │   - per-paper structured JSON (verbatim)       │
                  │   - section-aware markdown chunks              │
                  │                                                │
                  │ Pass 2: Normalization                          │
                  │   - numeric: regex + Pint parser (no LLM)      │
                  │   - entity: PWC API → embed-cluster → Haiku    │
                  │     confirm canonical names                    │
                  │                                                │
                  │ Pass 3: Build indexes                          │
                  │   - SQLite: entities, mentions, results,       │
                  │     papers, claims                             │
                  │   - Chroma: section chunks + embeddings        │
                  │   - NetworkX: in-corpus citation graph         │
                  │     (S2 references API)                        │
                  └────────────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────────────┐
                  │ QUERY TIME                                     │
                  │                                                │
question ──►Tier  │  Tier classifier (Haiku) → routes to handler   │
classifier        │                                                │
                  │  Per-tier handlers (see table below)           │
                  │     - return: answer, citations,               │
                  │       evidence_spans, confidence               │
                  └────────────────────────────────────────────────┘
                                       │
                                       ▼
                                 FastAPI /ask
```

---

## Per-tier handlers

Each handler returns `{answer, citations: [(paper_id, page, span)], confidence, cost_usd}`.

### Tier 1 — Single-document factual
**Handler**: `pre_extracted_lookup → rag_fallback`
- First check the SQLite `papers` table — if the question is about a field we extracted (architecture, datasets, model size), return that directly.
- Else: dense retrieve top-15 chunks **from the target paper only** → BGE-rerank to top-5 → Haiku answers with citations.
- *Considered and skipped*: long-context whole-paper feed (too expensive per query); two-stage section-locate (extra LLM hop without quality gain at our chunk granularity).

### Tier 2 — Corpus-level aggregation
**Handler**: `sql_aggregate`
- Direct SQL over the normalized `entities` and `mentions` tables. Examples: `SELECT DISTINCT canonical_name FROM entities WHERE type='dataset'`.
- Haiku writes the SQL via NL→SQL prompt grounded in the schema.
- *Considered and skipped*: map-reduce over papers at query time (100 LLM calls per question, prohibitive); knowledge graph (overkill).

### Tier 3 — Comparative / contradiction
**Handler**: `numeric_variance + textual_claim_compare`
- For benchmark contradictions: `GROUP BY (canonical_dataset, canonical_metric)` over `results` table; flag rows where `STDDEV(value) > threshold` or where claimed-SOTA values disagree.
- For methodological/textual disagreements: retrieve relevant claim passages (cross-paper) → Sonnet identifies contradictions with span citations.
- *Considered and skipped*: NLI models (slower, less flexible than LLM at our scale); pure LLM over passages (more expensive than the structured-first split).

### Tier 4 — Temporal / evolution
**Handler**: `pandas_timeseries`
- pandas over the `results`/`papers` tables grouped by `year`. Sonnet writes the pandas snippet via the code-interpreter tool when the question is non-trivial; Haiku for simple year-bucketing.
- *Considered and skipped*: per-year LLM summaries (good for narrative answers but expensive; can layer on top later).

### Tier 5 — Citation-graph reasoning
**Handler**: `networkx_query`
- In-memory NetworkX graph built from S2 references intersected with our 100 paper IDs.
- Algorithms: `in_degree` for "most cited within corpus", `pagerank` for influence, `shortest_path` for citation chains, `descendants/ancestors` for "papers building on X".
- *Considered and skipped*: Neo4j (overkill at 100 nodes); Grobid PDF reference extraction (S2 covers it).

### Tier 6 — Multi-hop / compositional
**Handler**: `decompose → chained_queries`
- Sonnet decomposes the question into 2-4 explicit steps (each a SQL query, NetworkX call, or RAG fetch).
- Execute steps sequentially, threading results.
- Final synthesis call (Sonnet) composes the answer with all evidence cited.
- *Considered and skipped*: ReAct agent (unpredictable cost, harder to debug); GraphRAG (heavyweight upfront for marginal gain); HippoRAG (better for fuzzy hops, ours are predictable joins).

### Tier 7 — Negation / absence
**Handler**: `closed_world_set_diff` (hybrid)
- Build expected-set: pull canonical "standard ViT benchmarks/datasets" from Papers With Code → ask Sonnet to expand the list with reasoning ("what else would you expect for ViT?") → cache the expanded set.
- Compute set difference: `expected_set - observed_set` (from our `entities` table).
- Verify each "missing" item against the corpus once more (Haiku spot-check) to filter false negatives from extraction errors.
- *Considered and skipped*: pure frequency analysis (weaker definition of "absent"); contrastive prompting only (hallucinates non-existent benchmarks); pure hand-curated reference list (laborious, won't generalize to live test).

### Tier 8 — Quantitative computation
**Handler**: `code_interpreter`
- Expose the SQLite tables as a pandas DataFrame; Sonnet writes Python via tool use (`exec_python`); execute in a sandbox; return the result with intermediate values.
- Handles sums, medians, correlations, regressions, custom math.
- *Considered and skipped*: NL→SQL only (less expressive than pandas for stats); precomputed aggregations (rigid, won't handle hidden test questions).

---

## Extraction schema (Pass 1)

Per paper, Haiku 4.5 produces this JSON (verbatim surface forms — normalization happens in Pass 2):

```python
{
  "paper_id": str,
  "architecture_summary": str,           # 1-2 sentence
  "model_variants": [
    {"name": "ViT-L/16", "param_count_surface": "307M parameters", "page": 3}
  ],
  "datasets_mentioned": [
    {"surface": "ImageNet-1K", "purpose": "pretrain|finetune|eval", "page": 4}
  ],
  "benchmark_results": [
    {"model": "ViT-L/16", "dataset_surface": "ImageNet-1K",
     "metric_surface": "top-1 acc", "value_surface": "85.30",
     "is_sota_claim": bool, "page": 7, "table_caption": str}
  ],
  "training_details": {
    "compute_surface": "2.5k TPU-days",
    "batch_size": int_or_null,
    "epochs": int_or_null
  },
  "methods_used": ["self-attention", "patch embedding", "RandAugment", ...],
  "novel_contributions": [str],          # claims the paper makes
  "key_claims": [
    {"claim": str, "evidence_section": str, "page": int}
  ]
}
```

**Caching**: the system prompt + JSON schema + 2 few-shot examples (~4k tokens) is cached and reused for all 100 papers. Per-paper input is just the markdown.

---

## Normalization (Pass 2)

### Numbers (deterministic, no LLM)
Regex + Pint over `*_surface` fields. Targets:
- params → millions (float)
- FLOPs → GFLOPs
- accuracy → percent (0-100)
- compute → GPU/TPU-days

Sanity bounds per field; out-of-range flags for review.

### Entities (3-step)
1. Collect unique surface forms across corpus, per type (dataset, benchmark, metric, method).
2. **Papers With Code API lookup** for well-known entities (ImageNet, COCO, CIFAR, ADE20K, mIoU, etc.) — gets canonical_id for free.
3. For unresolved: encode with text-embedding-3-small → cluster at cosine ≥ 0.85 → batched Haiku call confirms canonical name per cluster.

Output written to `entities` and `aliases` tables in SQLite.

---

## Quality-vs-budget curve (assignment requirement)

Three modes toggled via `BUDGET_LEVEL` env var:

| Level | Setting |
|---|---|
| **$1** | Haiku everywhere, k=3 retrieve, no rerank, no decomposition (multi-hop falls back to single retrieve+answer). Skips Tier 7 expansion call. |
| **$5** | Haiku default, Sonnet for tiers 3/6/7 only. k=8 retrieve, BGE rerank to 5. 2-step decomposition cap. |
| **$20** | Sonnet default for answer-gen, Haiku for routing/extraction-lookup. k=15 retrieve, rerank to 8. Full decomposition. Sonnet expands Tier 7 expected-set. |

Each eval run reports `accuracy_per_tier × cost_usd_per_question` for the curve.

---

## File structure additions

```
research-100-papers-agent/
├── scripts/
│   ├── parse_pdfs.py          # NEW — Marker run over data/pdfs/ → data/markdown/
│   ├── extract_papers.py      # NEW — Haiku pass over markdown → data/extractions/*.json
│   ├── normalize_entities.py  # NEW — PWC lookup + cluster + Haiku confirm → SQLite
│   ├── build_indexes.py       # NEW — Chroma (chunks) + SQLite (structured) + NetworkX (graph)
│   └── (existing fetch/download scripts unchanged)
├── api/
│   ├── routes/
│   │   ├── papers.py          # existing
│   │   ├── ask.py             # NEW — POST /ask
│   │   └── eval.py            # NEW — POST /eval
│   └── core/
│       ├── classifier.py      # NEW — tier classifier
│       ├── handlers/          # NEW — one module per tier
│       │   ├── tier1_factual.py
│       │   ├── tier2_aggregate.py
│       │   ├── tier3_contradict.py
│       │   ├── tier4_temporal.py
│       │   ├── tier5_citation.py
│       │   ├── tier6_multihop.py
│       │   ├── tier7_absence.py
│       │   └── tier8_compute.py
│       ├── llm.py             # NEW — Anthropic client + caching helpers
│       ├── retrieval.py       # NEW — Chroma + reranker wrapper
│       ├── store.py           # NEW — SQLite + NetworkX wrappers
│       └── budget.py          # NEW — BUDGET_LEVEL config + cost tracking
├── data/
│   ├── markdown/              # NEW — Marker output
│   ├── extractions/           # NEW — per-paper JSON
│   ├── corpus.db              # NEW — SQLite store
│   ├── chroma/                # NEW — Chroma persistence
│   └── citation_graph.gpickle # NEW — NetworkX graph
└── eval/
    ├── questions.jsonl        # NEW — 40+ eval questions with gold
    └── reports/               # NEW — per-budget-level eval results
```

---

## SQLite schema (sketch)

```sql
papers(paper_id, title, year, venue, citation_count, ...)
entities(entity_id, canonical_name, type, external_id, aliases JSON)
mentions(paper_id, entity_id, surface_form, page, span, context)
results(paper_id, model, entity_id_dataset, entity_id_metric,
        value_canonical, value_surface, is_sota_claim, page)
claims(paper_id, claim_text, evidence_section, page)
references(paper_id_src, paper_id_dst)   -- in-corpus citations
```

---

## Budget allocation

| Bucket | Budget | What |
|---|---|---|
| One-time prep | ~$5 | $3 extraction (Haiku, 100 papers, cached prompt) + $1 normalization (Haiku confirm) + $1 dev queries |
| Eval × 3 budget levels | ~$20 | $1 + $5 + ~$14 across 40+ questions × 3 runs |
| Buffer | ~$5 | Re-runs, debugging, hidden test bandwidth |
| **Total** | **$30** | |

---

## FastAPI surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health |
| GET | `/papers` | Existing — corpus listing |
| GET | `/papers/{id}` | Existing |
| POST | `/papers/fetch`, `/papers/download` | Existing |
| **POST** | **`/ask`** | NEW — `{question, budget_level?}` → `{answer, tier, citations, evidence, cost_usd}` |
| **POST** | **`/eval`** | NEW — `{budget_level}` → runs `eval/questions.jsonl`, returns per-tier accuracy + cost |

---

## Eval set (40+ questions, ≥3 per tier)

- Sonnet drafts candidate questions per tier from our extracted DB ("generate 6 Tier 2 questions answerable from this corpus + your gold answer").
- Manual review/edit gold answers, especially for Tier 3/6/7 where Sonnet might over-confidently assert wrong gold.
- Stored in `eval/questions.jsonl` with `{tier, question, gold_answer, gold_citations, notes}`.

---

## Implementation phases

1. **Pipeline plumbing**: `parse_pdfs.py` (Marker), `extract_papers.py` (Haiku + caching), `normalize_entities.py` (PWC + cluster), `build_indexes.py` (Chroma + SQLite + NetworkX). Verify end-to-end on 5 papers, then full 100.
2. **Per-tier handlers**: implement and unit-test each handler with hand-written Q/A pairs. Wire through tier classifier.
3. **FastAPI `/ask` and `/eval`**: thin endpoints over the handlers.
4. **Eval set**: Sonnet-generate → manual review → stored.
5. **Quality-vs-budget runs**: at $1/$5/$20 levels, capture per-question cost + accuracy → report.

---

## Verification

- `python scripts/parse_pdfs.py` → `data/markdown/` has 100 .md files
- `python scripts/extract_papers.py` → `data/extractions/` has 100 .json files; spot-check 3 against PDFs
- `python scripts/normalize_entities.py` → `corpus.db` has populated `entities`; check that "ImageNet"/"ImageNet-1K"/"ILSVRC2012" all map to one canonical_id
- `python scripts/build_indexes.py` → Chroma + SQLite + citation graph all populated
- `uvicorn main:app --reload`, then `curl POST /ask` with a sample question per tier — verify citations and cost reported
- `POST /eval` at each budget level, compare per-tier accuracy
