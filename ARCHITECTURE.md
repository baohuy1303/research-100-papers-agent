# Architecture — Research Comprehension System

## Context

Take-home assessment requires a system that answers natural-language questions across **8 difficulty tiers** over our 100-paper Vision Transformer corpus, with cited answers. Tasks 1 & 2 (corpus assembly) are done — 100 PDFs are at `data/pdfs/` with manifest at `data/manifest.csv`.

**Constraints**: $30 USD hard cap, no fine-tuning, results must be defensible/citeable, judged on robustness (45%), cost-efficiency (35%), live test-set performance (20%). The system must produce a quality-vs-budget curve at $1 / $5 / $20 levels.

**Core insight driving the design**: most "hard" tiers (2, 4, 7, 8) collapse into SQL queries *if* upfront extraction + normalization is thorough. So we invest one-time cost in a comprehensive extraction + normalization pass, then keep query-time cost low.

---

## Stack (decided)

| Layer | Choice | Why this over others |
|---|---|---|
| PDF parser | **Marker via Datalab cloud API** | Same Marker model, Datalab-hosted GPUs. Local Marker attempted first but unworkable here: RTX 3050 Ti has only 4 GB VRAM vs Marker's documented 5 GB minimum — caused severe thrashing (5+ min/page). CPU mode equally slow (9 GB RAM, no progress in tens of minutes). Cloud API removes the hardware constraint. Considered Nougat, Docling, PyMuPDF — same quality concerns or worse. |
| Extraction LLM | **GPT-5.4-mini (OpenAI)** | Fast, cheap structured outputs; OpenAI auto-caches identical system-prompt prefixes (≥1024 tokens) at 0.1× input cost — no explicit cache markers needed. Anthropic Haiku 4.5 was the original choice but the account ran out of credits; switched to OpenAI. |
| Embeddings | **OpenAI text-embedding-3-small** | ~$0.05 total for our corpus, strong semantic match. Considered local BGE/MiniLM (free but lower quality), Voyage (overkill at our scale). |
| Vector store | **Chroma local** | Persistent local store, easy Python client, survives restarts. Considered sqlite-vec (single-file but less mature), LanceDB (better at larger scale). |
| Reranker | **BGE-reranker-v2 local** | Free, ~10× top-1 lift after dense retrieval. Considered Cohere rerank (paid, similar quality), LLM-rerank (Haiku, more flexible but adds latency/cost). |
| Structured store | **SQLite** | Universal, serverless, fits beside Chroma. DuckDB faster on analytics but unnecessary at this scale. |
| Citation graph | **Semantic Scholar `references` API** | Free, pre-resolved IDs to match against our 100. Considered Grobid (extract from PDF text, brittle), OpenAlex (equivalent, S2 already used for corpus). |
| Query routing | **LLM tier-classifier + per-tier handlers** | Predictable cost, debuggable, easy to instrument. Considered ReAct agent (unpredictable cost), DSPy (more setup), hybrid agent-per-tier (middle ground, deferred). |
| Query LLM | **Tiered: GPT-5.4-mini default, GPT-4.1 (or Sonnet) for tiers 3/6/7** | Best $/quality split. Hard reasoning tiers get the larger model; lookups/aggregations stay on mini. |
| Caching | **OpenAI automatic prefix caching** | Identical system-prompt prefixes cached automatically by OpenAI; no explicit `cache_control` markers required (unlike Anthropic). ~10× cost reduction on the frozen 4k-token extraction prompt. |

---

## High-level pipeline

```
                  ┌────────────────────────────────────────────────┐
                  │ ONE-TIME PREP (~$5 of $30 budget)              │
                  │                                                │
PDFs ─►Datalab──►│ Pass 1: Verbatim Extraction (GPT-5.4-mini)     │
       (Marker)  │
                  │   - per-paper structured JSON (verbatim)       │
                  │   - section-aware markdown chunks              │
                  │                                                │
                  │ Pass 2: Normalization                          │
                  │   - numeric: regex (no LLM)                    │
                  │   - entity: 6-stage hybrid                     │
                  │     curated → rule → fuzzy → embed → HF        │
                  │     → gpt-5-mini disambig                      │
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
question ──►Tier  │  Tier classifier (GPT-5.4-mini) → routes to handler │
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

**Caching**: the system prompt + JSON schema + 2 few-shot examples (~4k tokens) is identical across all 100 calls. OpenAI automatically caches this prefix; subsequent calls pay 0.1× input rate on the cached portion.

---

## Normalization (Pass 2)

### Numbers (`scripts/normalize_numbers.py` — deterministic, no LLM)

Regex parsing over `*_surface` fields. Targets:
- `param_count_surface` → `param_count_millions` (float)
  - Handles `"86M"`, `"86 M"`, `"86M parameters"`, `"1.2B"`, `"175 billion parameters"`, `"~ 45 million"`
- `value_surface` → `value_canonical` (float)
  - Strips `%`, takes primary of `"85.3 ± 0.2"` ranges, handles HTML `<b>...</b>` and LaTeX `_{...}` corruption

**Coverage on 100 papers**: 96% params (352/364), 98% metric values (3194/3239). Remaining nulls are genuinely qualitative ("BERT base", "state-of-the-art", "5 times fewer parameters"). Output: `data/normalized/{paper_id}.json`.

### Entities (`scripts/normalize_entities.py` — 6-stage hybrid pipeline)

PWC's API died in 2024 (302-redirects to HuggingFace). HF datasets API works and exposes a `paperswithcode_id` cross-reference field on dataset records, but only covers datasets, not metrics or methods. So we use a multi-stage pipeline that progressively escalates from cheap deterministic techniques to LLM disambiguation:

| Stage | Technique | Applies to | Catches |
|---|---|---|---|
| 1 | **Curated alias map** (~30 entries per type, hand-built) | datasets, metrics, methods | `"ImageNet"`/`"ILSVRC2012"`/`"ImageNet-1k"` → `ImageNet`; `"top-1"`/`"Top-1 (%)"`/`"ImageNet top-1 acc."` → `top-1 accuracy` |
| 2 | **Rule normalizer** (regex: case, suffix, "(%)") | datasets, metrics, methods | `"Top-1 Acc. (%)"` → `top-1 accuracy` |
| 3 | **Fuzzy match** (`rapidfuzz` WRatio ≥ 95) against existing canonicals | datasets, metrics, methods | typos, spacing variants — `"COCO"`/`"MS COCO"`/`"MSCOCO"` |
| 4 | **Embedding cluster** (`text-embedding-3-small`, cosine ≥ 0.92 auto-merge; 0.80–0.92 flagged) | datasets, metrics, methods | semantic neighbors |
| 5 | **HF Datasets lookup** on each cluster representative | **datasets only** | resolves `paperswithcode_id` and `hf_id` for citable IDs |
| 6 | **LLM disambiguation** (`gpt-5-mini`, reasoning model) | datasets, metrics, methods | gray-zone pairs in [0.80, 0.92) cosine that clustering can't decide |

**Embedding context**: surface-form only (no surrounding sentence). Embedding context would push two papers' "ImageNet" mentions apart even though they refer to the same dataset.

**Confidence flow**: ≥0.92 cosine → auto-merge; 0.80–0.92 → LLM check; <0.80 → keep separate.

**Coverage on 100 papers** (after Phase 3b):

| Type | Surface forms | → Entities | Reduction | PWC IDs |
|---|---|---|---|---|
| Datasets | 743 | 632 | 15% | 59 |
| Metrics | 450 | 365 | 19% | — |
| Methods | 1076 | 953 | 11% | — |

Output: single `data/entity_map.json` with `{canonical, type, aliases[], mention_count, source, paperswithcode_id, hf_id}` per entity. `source` ∈ `{curated, rule, fuzzy, clustered, hf-pwc, llm-confirmed}` for debugging which stage merged each entity. HF responses cached to `data/hf_cache/` for re-run resumability.

**Total Phase 3 cost**: $0.04 (embeddings ~$0.0001 + LLM disambig ~$0.04).

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
│   ├── fetch_papers.py        # DONE — Semantic Scholar bulk → data/manifest.csv
│   ├── download_pdfs.py       # DONE — manifest → data/pdfs/{paper_id}.pdf
│   ├── parse_pdfs.py          # DONE — Datalab API + KeyPool → data/markdown/{paper_id}.md
│   ├── extract_papers.py      # DONE — gpt-5.4-mini structured-output → data/extractions/*.json
│   ├── normalize_numbers.py   # DONE — regex → data/normalized/*.json
│   ├── normalize_entities.py  # DONE — 6-stage hybrid → data/entity_map.json
│   └── build_indexes.py       # NEXT — SQLite + Chroma + NetworkX
├── api/
│   ├── routes/
│   │   ├── papers.py          # DONE
│   │   ├── ask.py             # NEXT — POST /ask
│   │   └── eval.py            # NEXT — POST /eval
│   └── core/
│       ├── llm.py             # DONE — Anthropic + OpenAI clients, pricing, caching
│       ├── budget.py          # DONE — BUDGET_LEVEL config + cost tracking
│       ├── extraction_prompt.py  # DONE — frozen system prompt for Phase 2
│       ├── schemas.py         # DONE — Pydantic schema for ExtractedPaper
│       ├── classifier.py      # NEXT — tier classifier
│       ├── handlers/          # NEXT — one module per tier
│       │   ├── tier1_factual.py
│       │   ├── tier2_aggregate.py
│       │   ├── tier3_contradict.py
│       │   ├── tier4_temporal.py
│       │   ├── tier5_citation.py
│       │   ├── tier6_multihop.py
│       │   ├── tier7_absence.py
│       │   └── tier8_compute.py
│       ├── retrieval.py       # NEXT — Chroma + reranker wrapper
│       └── store.py           # NEXT — SQLite + NetworkX wrappers
├── data/
│   ├── manifest.csv           # DONE — 100 papers
│   ├── pdfs/                  # DONE — 100 PDFs
│   ├── markdown/              # DONE — 100 .md (Datalab Marker output)
│   ├── extractions/           # DONE — 100 .json (gpt-5.4-mini structured output)
│   ├── normalized/            # DONE — 100 .json (extractions + canonical numeric fields)
│   ├── entity_map.json        # DONE — canonical entities + aliases
│   ├── hf_cache/              # DONE — cached HF Datasets responses
│   ├── cost_log.jsonl         # DONE — running cost log
│   ├── corpus.db              # NEXT — SQLite store
│   ├── chroma/                # NEXT — Chroma persistence
│   └── citation_graph.gpickle # NEXT — NetworkX graph
└── eval/
    ├── questions.jsonl        # NEXT — 40+ eval questions with gold
    └── reports/               # NEXT — per-budget-level eval results
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

## Budget allocation (actuals so far)

| Bucket | Estimated | **Actual** | What |
|---|---|---|---|
| Phase 1: Datalab parsing | ~$3 | **~$3.22** | 100 papers via Datalab cloud Marker (multi-key rotation across 2 free-tier keys) |
| Phase 2: Extraction | ~$3 | **~$1.50** | 100 papers via gpt-5.4-mini structured output, OpenAI auto-cache on 4k system prompt |
| Phase 3a: Number normalization | $0 | **$0** | Pure regex, no LLM |
| Phase 3b: Entity normalization | ~$1 | **~$0.04** | Embeddings + gpt-5-mini disambig (~50 calls) |
| **One-time prep total** | ~$7 | **~$4.76** | Comes in well under estimate thanks to OpenAI prompt caching |
| Phases 4–7: Indexes, infra, handlers, API | ~$1 | TBD | Dev queries during implementation |
| Phase 8–9: Eval × 3 budget levels | ~$18 | TBD | $1 + $5 + ~$12 across 40+ questions × 3 runs |
| Buffer | ~$4 | TBD | Re-runs, debugging, hidden test bandwidth |
| **Total cap** | **$30** | **$6.80 spent** (23%) | $23 remaining |

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
