# Implementation Plan

High-level execution plan derived from [ARCHITECTURE.md](ARCHITECTURE.md). Phases are sequenced — each depends on the prior phase's outputs.

**Critical path**: Phases 1 → 2 → 3 → 4 (one-time prep) → 5 → 6 → 7 (query stack) → 8 → 9 (eval). Phase 6 unblocks 7, but 7 is a thin wrapper.

| Phase | Goal | Est. | **Actual** | Status |
|---|---|---|---|---|
| 0 | Project scaffolding & dependencies | $0 | $0 | ✅ done |
| 1 | Parse 100 PDFs (Datalab Marker API) | ~$3 | **~$3.22** | ✅ 100/100 |
| 2 | Extraction with gpt-5.4-mini | ~$3 | **~$1.50** | ✅ 100/100 |
| 3a | Normalize numbers (regex) | $0 | **$0** | ✅ 96% params / 98% values |
| 3b | Normalize entities (6-stage hybrid) | ~$1 | **~$0.04** | ✅ 632 datasets / 365 metrics / 953 methods |
| 4 | SQLite + Chroma + NetworkX indexes | ~$0.05 | **~$0.05** | ✅ 100 papers / 3768 chunks / 761 edges |
| 5 | Core query infrastructure (store, retrieval, classifier) | $0 | **~$0.005** | ✅ 28/28 sanity checks pass |
| 6 | 8 tier handlers | ~$1 | **~$0.02** | ✅ all 8 working end-to-end |
| 7 | FastAPI /ask and /eval | $0 | — | 🔜 next |
| 8 | Build 40+ eval question set | ~$0.50 | — | pending |
| 9 | Quality-vs-budget runs at $1/$5/$20 | ~$18 | — | pending |
| 10 | Cost report, README, submission polish | $0 | — | pending |
| **Total spent** | | | **~$6.90 / $30** (23%) | $23 remaining |

The `scripts/sanity_check.py` script exercises Phases 1–6 end-to-end (33 checks). Run before any major change.

---

## Phase 0 — Project scaffolding (DONE)

- [x] All Python deps in `requirements.txt`: `fastapi`, `uvicorn`, `httpx`, `pandas`, `python-dotenv`, `pydantic`, `anthropic`, `openai`, `chromadb`, `networkx`, `rapidfuzz`, `numpy`, `tqdm`
- [x] `.env` keys: `OPENAI_API_KEY`, `OPENALEX_API_KEY`, `ANTHROPIC_API_KEY`, `DATALAB_API_KEY_1`, `DATALAB_API_KEY_2`
- [x] Directory tree exists: `data/{markdown,extractions,normalized,chroma,hf_cache}`, `api/core/{handlers}`, `eval/{reports}`
- [x] `api/core/budget.py` with `BUDGET_LEVEL` env config + `record_cost()` + `total_spent()`
- [x] Local Marker deps stripped (we use Datalab cloud API instead)

---

## Phase 1 — PDF parsing (DONE)

Datalab cloud Marker API. Local Marker was unworkable: RTX 3050 Ti has 4 GB VRAM, Marker needs ≥5 GB; CPU mode equally slow.

- [x] `scripts/parse_pdfs.py` — async/httpx, bounded concurrency (8), `KeyPool` rotates between `DATALAB_API_KEY_*` env vars when one hits 403 (free-tier quota)
- [x] Each successful parse calls `record_cost("datalab_parse", ~$0.0025/page)`
- [x] All 100 papers parsed across 2 free-tier keys (KEY_1: ~55 papers before exhausted, KEY_2 finished the rest)
- [x] Zero permanent failures

**Cost: $5.35 estimated** (we used free-tier keys, so actual paid $0; the $0.0025/page is a placeholder since Datalab pricing wasn't accessible at planning time).
**Time: ~34 minutes wall-clock** spread across two sessions (KEY_1 exhausted partway, KEY_2 finished). Parse latency is bounded by Datalab's GPU queue (10-30s/paper for typical 15-30 page papers).

---

## Phase 2 — Verbatim extraction (DONE)

OpenAI gpt-5.4-mini structured output via Pydantic schema. Switched from Claude Haiku 4.5 (Anthropic credits ran out).

- [x] `api/core/schemas.py` — Pydantic `ExtractedPaper` schema
- [x] `api/core/extraction_prompt.py` — frozen ~4k-token system prompt with 2 few-shot examples (cached automatically by OpenAI when same prefix is reused)
- [x] `api/core/llm.py` — `get_openai_client()`, `oai_cost_for_usage()`, model constants
- [x] `scripts/extract_papers.py` — async with `prompt_cache_key` for routing all 100 calls to same cache; bumped `max_completion_tokens` 4096→16384 over 2 retries to handle papers with very large benchmark tables

**Cost: $1.41 actual.** 100/100 papers extracted with 95–99% cache hits on the system prompt.
**Time: ~19 minutes wall-clock** (concurrency=10). Per-paper latency 6–30s — most around 12–17s. Cache-hit calls were ~2× faster than cold calls because OpenAI's prefix cache skips re-tokenizing the 4k system prompt.

---

## Phase 3 — Normalization (DONE)

### 3a — Numbers (`scripts/normalize_numbers.py`)
Pure regex. Handles `param_count_surface` → `param_count_millions` (M/B suffix, "billion params", bare numbers), `value_surface` → `value_canonical` (% strip, ± ranges, HTML/LaTeX corruption).

**Coverage**: 96% params (352/364), 98% metric values (3194/3239). Remaining nulls are genuinely qualitative ("BERT base", "state-of-the-art").
**Time: <1 second** for all 100 papers. Pure regex, no I/O bottleneck.

### 3b — Entities (`scripts/normalize_entities.py`)
PWC API died (302→HuggingFace). HF Datasets API works and exposes `paperswithcode_id` cross-references. Implemented as 6-stage hybrid:

1. **Curated alias map** (~30 per type)
2. **Rule normalizer** (regex strips eval-split qualifiers like "val. set", "[8]", "256x256"; also reverse-lookup against existing canonicals to prevent fragmentation)
3. **Fuzzy match** (rapidfuzz WRatio ≥ 95)
4. **Embedding cluster** (text-embedding-3-small, cosine ≥ 0.92 auto-merge; 0.80–0.92 flagged for stage 6)
5. **HF Datasets lookup** on cluster reps → `paperswithcode_id` (datasets only)
6. **LLM disambiguation** on flagged pairs (gpt-5-mini reasoning model, `max_completion_tokens=512`)

**Coverage**: 743 → 632 datasets, 450 → 365 metrics, 1076 → 953 methods. ImageNet captures 22 aliases including ILSVRC2012; HF resolved 59 PWC IDs.

**Cost: $0.04 actual** per run (we re-ran ~6 times during iteration; cumulative ~$0.20).
**Time: ~60-90 seconds per run.** Breakdown:
- Curated map + rule normalize + fuzzy: <1s (CPU only)
- Embedding ~2,300 surface forms: ~2s (single batched OpenAI call)
- Greedy clustering: <1s (numpy on ~3k vectors)
- HF Datasets API lookups: ~30-45s (~600 cluster reps × ~50ms each, with caching to disk for re-runs)
- gpt-5-mini LLM disambig: ~30-45s for ~75-135 uncertain pairs (each call is a reasoning model, ~300-500ms)

---

## Phase 4 — Build indexes (DONE)

`scripts/build_indexes.py` builds three independent indexes (skip flags supported):

### SQLite (`data/corpus.db`)
9 tables: `papers`, `entities`, `aliases`, `mentions`, `results`, `model_variants`, `training`, `claims`, `paper_refs`. Cross-references built via `surface_form → entity_id` lookup at insertion.
- 100 papers, 529 model variants, 2980 mentions, 3204 benchmark results (37 skipped because `metric_surface` was empty in extraction), 728 claims

### Chroma (`data/chroma/`)
3768 section-level chunks (split on H1/H2/H3 markdown markers, sub-split if >4000 chars). Embedded with text-embedding-3-small (2.6M tokens, $0.053).

### NetworkX (`data/citation_graph.gpickle`)
S2 `/paper/{id}/references` API → intersect with our 100 corpus IDs → in-memory `DiGraph`. **761 in-corpus edges** (very dense — ViT alone is cited by 83 of the 99 other papers).

**Cost: $0.05 actual** (Chroma embeddings only; SQLite + graph are free).
**Time per index** (one full `build_indexes.py` run):
- SQLite: **~3 seconds** (DROP + CREATE schema + INSERT 100 papers + 529 model variants + 2980 mentions + 3204 results + 728 claims)
- Chroma: **~45 seconds** (3768 chunks across 38 batches of 100; bottleneck is OpenAI embedding API, not chunking or Chroma writes)
- NetworkX graph: **~95 seconds** (100 papers × S2 API at 3 RPS bounded concurrency, including pagination)
- **Total: ~2.5 minutes**

---

## Phase 5 — Core query infrastructure (DONE)

Three modules in [api/core/](api/core/):

- **`store.py`** — typed wrapper around SQLite + NetworkX. Methods: `get_paper`, `papers_using`, `results_for`, `best_on`, `entity_by_alias`, `most_cited`, `descendants`/`ancestors`, `shortest_citation_path`, `pagerank_top`, `execute_sql` (read-only guard rejects DROP/UPDATE/INSERT)
- **`retrieval.py`** — Chroma query wrapper with budget-aware default `k`. `search()` and `search_in_paper()` return chunks with cosine similarity scores
- **`classifier.py`** — gpt-5.4-mini tier router with frozen prompt + cache key. **8/8 test questions classified correctly with ≥0.96 confidence**

Reusing Phase 2's `llm.py` for OpenAI client + cost tracking. Sanity checks: 28/28 green.

**Cost: $0.004 across one full sanity-check run.**
**Per-call latency** (single in-process call, warm cache):
- `store.*` SQL methods: **<10ms** (SQLite indexed lookups)
- `store.most_cited(k=10)`, `store.descendants()`: **<50ms** (NetworkX in-memory)
- `retriever.search(query, k=8)`: **~300-500ms** (one OpenAI embedding call + Chroma lookup)
- `classifier.classify(question)`: **~600-1500ms** (one gpt-5.4-mini call with cached system prompt)

---

## Phase 6 — Tier handlers (DONE)

8 handlers in [api/core/handlers/](api/core/handlers/), all using gpt-5.4-mini (no model switching across tiers per user direction). Each returns `HandlerResult{tier, answer, citations, evidence, cost_usd, confidence, reasoning}`:

| Tier | Mechanism | Verified output |
|---|---|---|
| 1 | Resolve target paper → structured store + Chroma RAG → synthesize | "ViT uses a pure Transformer that splits images into fixed-size patches…" |
| 2 | NL→SQL via gpt-5.4-mini → `execute_sql` (read-only) → synthesize | "51 papers in the corpus benchmark on ImageNet." |
| 3 | LLM picks `numeric` (variance over results table) or `textual` (cross-paper RAG) → synthesize | "Yes — CoCa claims 91.0, PaLI-17B claims 90.9, both labeled SOTA." |
| 4 | NL→SQL with GROUP BY year → time-series synthesis | "Top-1 increased from 85.2 in 2020 to 90.0 in 2022." |
| 5 | LLM picks graph op (`most_cited`/`pagerank`/`descendants`/`ancestors`/`shortest_path`/`cited_by`/`references_of`) → execute → synthesize | "ViT is the most cited within this corpus (83 in-corpus citations)." |
| 6 | Tool-calling agent: 4 tools (`sql_query`, `search_chunks`, `graph_op`, `finalize_answer`); `max_steps` budget-gated (1/6/10) | "Among ViT-citers, MLP-Mixer has the largest variant at 431M params." |
| 7 | LLM proposes EXPECTED set → set diff against entities table → synthesize what's missing | "Missing: <list of expected datasets not present>" |
| 8 | LLM writes Python over pandas DataFrames of SQLite tables → exec in sandboxed namespace → synthesize | "Median: 61.69 million parameters." |

**Cost: ~$0.02 across all 8 handler smoke tests.** All wired into `scripts/sanity_check.py` (33/33 green).
**Per-tier latency** (one question, $5 budget level):

| Tier | Typical latency | Why |
|---|---|---|
| 1 | 5–10s | resolve paper (1 LLM) + Chroma RAG (1 embed) + synthesis (1 LLM) |
| 2 | 3–6s | NL→SQL (1 LLM) + SQLite (<10ms) + synthesis (1 LLM) |
| 3 | 5–15s | plan (1 LLM) + numeric SQL OR cross-corpus retrieval + synthesis (1 LLM) |
| 4 | 3–6s | NL→SQL (1 LLM) + SQLite (<10ms) + synthesis (1 LLM) |
| 5 | 3–8s | graph op picker (1 LLM) + NetworkX (<50ms) + synthesis (1 LLM) |
| 6 | 15–40s | tool-calling agent: 3–6 LLM calls with tool execution between each |
| 7 | 5–15s | LLM expected-set proposal (1 LLM, larger output) + set diff (instant) + synthesis (1 LLM) |
| 8 | 5–15s | code planner (1 LLM) + sandboxed pandas exec (50ms) + synthesis (1 LLM) |

---

## Phase 7 — FastAPI endpoints (NEXT)

**Goal**: HTTP surface that ties classifier → handler → response, plus a batch eval runner.

### `api/routes/ask.py` — `POST /ask`

```json
Request:  {"question": "...", "budget_level": "$5", "target_paper_id": null}
Response: {
  "answer": "...",
  "tier": 2,
  "tier_confidence": 0.99,
  "tier_reasoning": "...",
  "citations": [{"paper_id": "...", "paper_title": "...", "section": null, "snippet": null}],
  "evidence": [...],
  "cost_usd": 0.0012,
  "handler_reasoning": "..."
}
```

Pipeline:
1. Override `BUDGET_LEVEL` env if `budget_level` supplied
2. `TierClassifier().classify(question)` → `{tier, confidence, reasoning, normalized_question}`
3. `get_handler(tier)(normalized_question, store, retriever, classifier_meta)` → `HandlerResult`
4. Add `tier_*` fields and return

Edge cases:
- Classifier confidence < 0.5 → log warning, default to Tier 1 with cross-corpus retrieval (fallback)
- Handler raises → catch, return `{answer: "Internal error: <msg>", cost_usd: <classifier_cost>, tier}`
- `target_paper_id` provided → bypass paper resolution in Tier 1 handler

### `api/routes/eval.py` — `POST /eval`

```json
Request:  {"budget_level": "$5", "limit": null}
Response: {
  "total_questions": 40,
  "passed": 32,
  "per_tier": {"1": {"n": 5, "pass": 4, "avg_cost": 0.001}, ...},
  "total_cost_usd": 0.45,
  "report_path": "eval/reports/20260512_5usd.json"
}
```

Reads `eval/questions.jsonl`, runs each question through `/ask` internals, compares to gold. Pass criterion is per-tier:
- Tier 1, 2, 5, 8: substring match of gold value in answer (e.g. "70 papers" in answer)
- Tier 4, 7: structural match (set overlap of named years / missing items ≥ 80%)
- Tier 3, 6: LLM judges semantic equivalence to gold (gpt-5.4-mini scored 0/1)

Each report saved to `eval/reports/{timestamp}_{budget}.json` with full per-question details.

### Wire-up

- [x] `main.py` already has FastAPI app and `papers.py` router → just import and mount the new routers
- Smoke test via `curl` + a representative question per tier
- Add the `/ask` endpoint to `scripts/sanity_check.py` as final check

**Cost estimate**: $0 (just dev queries). Should take ~150 lines total.

---

## Phase 8 — Eval set construction

**Goal**: `eval/questions.jsonl` with **≥40 questions** (≥5 per tier), each with gold answer + citations.

Refined approach (vs the original "Sonnet drafts" plan): generate questions **from the actual extracted DB**, not from imagination — this guarantees they're answerable.

### `scripts/generate_eval_set.py`

For each tier 1–8, fetch real data from `store.py`, pass it to gpt-5.4-mini with a tier-specific prompt:
- **Tier 1**: pick 5 random papers, ask LLM to write a factual question + extract gold answer from `architecture_summary` / `model_variants`
- **Tier 2**: hand-write 5 prompt templates (e.g. "How many papers benchmark on {dataset}?") and fill from entity table; gold = SQL result
- **Tier 3**: scan for (dataset, metric) pairs with stddev > 5 → "Do papers disagree on X benchmark?"; gold = the spread
- **Tier 4**: hand-write 5 templates ("How did {metric} on {dataset} change 2020–2024?"); gold = year-bucketed mean
- **Tier 5**: hand-write 5 templates from graph data ("Most cited paper", "Papers building on {top-3 papers}", "Path from A to B"); gold = computed
- **Tier 6**: pick 5 compositional templates ("Among papers using {dataset}, which has highest {metric}"); gold = SQL chain result
- **Tier 7**: 3 absence questions per scope (corpus-wide / per-paper); gold = our set diff
- **Tier 8**: pick 5 templates ("Median X", "Correlation between Y and Z"); gold = pandas computation

### Manual review pass

Tier 3, 6, 7 outputs are the highest-risk for wrong gold (LLM may over-confidently assert false equivalences). Spot-check each.

**Output**: 40+ entries in `eval/questions.jsonl`:
```json
{"id": "T1-001", "tier": 1, "question": "What architecture does ViT use?",
 "gold_answer": "Pure Transformer with patch embeddings",
 "gold_citation_paper_ids": ["268d347e8a55b5eb82fb5e7d2f800e33c75ab18a"],
 "match_strategy": "substring", "notes": "..."}
```

**Cost: ~$0.30** (40 questions × ~$0.005 to generate + a bit for spot-check synthesis).

---

## Phase 9 — Quality-vs-budget runs

**Goal**: produce the assessment-required quality-vs-budget curve.

### Three runs

| Level | What changes | Expected behavior |
|---|---|---|
| **$1** | `BUDGET_LEVEL=$1`: Chroma `retrieve_k=3`, no rerank, Tier 6 `max_steps=2`, skip Tier 7 LLM expansion | Lower accuracy on T1/T6/T7; T2/T4/T5/T8 unaffected (deterministic) |
| **$5** | Default. `retrieve_k=8`, Tier 6 `max_steps=6`, full Tier 7 expansion | Baseline accuracy |
| **$20** | `retrieve_k=15`, Tier 6 `max_steps=10`, larger Tier 7 expected set, optional rerank | Marginal lift on hard tiers |

### Per-budget reports

- `eval/reports/{timestamp}_{level}.json` — per-question detail
- `eval/reports/quality_vs_budget.png` — matplotlib scatter: x = avg cost / question, y = % accuracy, one point per (tier × budget); color by tier; annotated with totals
- `eval/RESULTS.md` — human-readable table

### Hard guardrails

- `budget.total_spent()` checked between runs; abort if > $25 (leave $5 buffer for hidden test)
- Each `/eval` run logs its starting + ending total to console

**Cost: ~$15** estimated ($1 + $5 + $9 for the $20 run, scaled by 40 questions).

---

## Phase 10 — Submission polish

- [ ] `README.md` rewrite: setup, env vars, full pipeline reproduction (`python scripts/parse_pdfs.py && extract_papers.py && normalize_numbers.py && normalize_entities.py && build_indexes.py`), how to run sanity check, how to query API, link to ARCHITECTURE.md
- [ ] `COST_REPORT.md`: $-spent table per phase, the quality-vs-budget plot
- [ ] `eval/RESULTS.md`: gold vs system answer per question per budget level
- [ ] Verify clean clone reproduces (or document local-only deps clearly)
- [ ] Final commit + tag + push

---

## Lessons learned (informing future steps)

| Lesson | Where it bit us | What we'll do |
|---|---|---|
| Reasoning models eat tokens before emitting content | Stage 6 LLM disambig with `max_completion_tokens=8` returned empty | Use ≥512 for any gpt-5-mini call |
| Surface forms come from multiple JSON paths | Initial entity normalization missed `benchmark_results.dataset_surface` → 568 results dropped | When building indexes, gather surface forms from EVERY field that references entities |
| Rule-bucket Stage 2 must check existing canonicals | "ImageNet val. set" became its own entity instead of merging into "ImageNet" | Stage 2 reverse-looks-up rule-key against canonicals from Stage 1 |
| Multi-hop agents need ≥6 steps | Tier 6 with max_steps=4 ran out before converging | $5 budget gives 6 steps; $20 gives 10 |
| Chroma metadata `where=` filters need exact paper_id strings | First Tier 1 attempt used a substring search that returned no results | Resolve to paper_id first, then filter |
| Windows console is cp1252 | Unicode arrows / box characters crash print() | Helper `safe_print()` ASCII-encodes everything |
| IDE diagnostics check system Python, not venv | Constant false positives for chromadb / rapidfuzz | Ignore — actual scripts run from venv fine |

---

## Performance summary (for README)

One-shot cost / time / output trade-offs across the full prep pipeline. Useful for the README's "what does this cost / how long does it take?" section.

| Phase | Wall time | Cost | Output | Bottleneck |
|---|---|---|---|---|
| 1. Parse PDFs | ~34 min | $5.35 (free-tier) | 100 markdown files (avg 30k chars each) | Datalab GPU queue |
| 2. Extract structured | ~19 min | $1.41 | 100 verbatim JSONs | OpenAI inference at concurrency 10 |
| 3a. Normalize numbers | <1s | $0 | 100 normalized JSONs | None (regex) |
| 3b. Normalize entities | ~75s | $0.04 | 1 entity_map.json (~1,950 canonical entities) | HF API + LLM disambig |
| 4. Build indexes | ~2.5 min | $0.05 | corpus.db + chroma/ + citation_graph.gpickle | OpenAI embeddings (45s) + S2 API (95s) |
| 5. Smoke check (Phase 1–5) | ~20s | $0.005 | sanity log | Classifier LLM calls (5×) |
| **Full prep, fresh** | **~57 min** | **~$6.85** | All indexes ready for query-time | Datalab + extraction (LLM-bound) |

### Query-time trade-offs (per question, $5 budget level)

| Tier | Typical latency | Typical cost | Interaction shape |
|---|---|---|---|
| 1 | 5–10s | $0.001–0.002 | factual lookup |
| 2 | 3–6s | $0.0005–0.001 | NL→SQL aggregation |
| 3 | 5–15s | $0.001–0.003 | numeric or textual contradiction |
| 4 | 3–6s | $0.0005–0.001 | year-bucketed time series |
| 5 | 3–8s | $0.0003–0.001 | citation-graph op |
| 6 | 15–40s | $0.003–0.008 | multi-step agent (3–6 tool calls) |
| 7 | 5–15s | $0.001–0.003 | absence / set diff |
| 8 | 5–15s | $0.0005–0.002 | sandboxed pandas computation |

**Caching effect**: OpenAI's automatic prefix cache applies to every call that reuses the system prompt (extraction, classifier, NL→SQL planner, etc.). When the same prompt is reused within ~5 minutes, cached input tokens cost 0.1× and reduce latency ~2×. Our extraction run got 95–99% cache hits because all 100 calls used the identical 4k-token system prompt.

**Concurrency knobs**:
- `parse_pdfs.py --concurrency N` (default 8) — Datalab is the bottleneck; going above 16 doesn't help
- `extract_papers.py --concurrency N` (default 10) — OpenAI tier 4 supports much higher; 10 keeps us polite
- `normalize_entities.py` — fixed concurrency (1 batch embed + sequential HF lookups + ~10 parallel LLM disambigs would be a future improvement)
- `build_indexes.py` — sequential by design (each index depends on the prior); embedding batches are 100 chunks each
- Query-time (Phase 6 handlers) — fully sequential within one question; FastAPI serves requests concurrently across questions

---

## Risks & contingencies

| Risk | Mitigation |
|---|---|
| Eval gold answers wrong (esp. Tier 3/6/7) | Manual review + label LLM-generated gold as "high-confidence" / "needs-review" |
| Cost overrun during eval Phase 9 | Pre-run dry estimate ($cost_per_question × 40); abort if > $20 |
| Hidden test set has unanticipated question shapes | Tier classifier returns a tier even when uncertain; if confidence < 0.5 we log + fall back to Tier 1 cross-corpus RAG (most general) |
| OpenAI API outage during eval | Retry with exponential backoff (already in openai SDK); if persistent, write partial results and resume |
| Tier 8 sandbox escape attempt by LLM | `_run_sandboxed` blocks `import` and exposes only `pd`, `math`, `statistics`; no `__builtins__.open`/`__import__`/etc. |
| FastAPI returns serialization error on `evidence` field | `HandlerResult.evidence: list[dict]` — ensure all values are JSON-serializable (no datetime, no Path, no DataFrame) at handler-write time |
