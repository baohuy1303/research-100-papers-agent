# Implementation Plan

High-level execution plan derived from [ARCHITECTURE.md](ARCHITECTURE.md). Phases are sequenced — each depends on the prior phase's outputs.

**Critical path**: Phase 1 → 2 → 3 → 4 (one-time prep, ~$5 spend) must complete before query handlers can be tested. Phases 5–6 can develop in parallel once the indexes exist.

| Phase | Goal | Est. cost | Critical? |
|---|---|---|---|
| 0 | Project scaffolding & dependencies | $0 | yes |
| 1 | Parse 100 PDFs to markdown (**Datalab Marker API**) | ~$3 | yes |
| 2 | Verbatim extraction with GPT-5.4-mini | ~$1.50 | yes |
| 3 | Normalize numbers + entities | ~$1 | yes |
| 4 | Build SQLite + Chroma + NetworkX indexes | ~$0.05 | yes |
| 5 | Core query infrastructure (LLM, retrieval, store, budget) | $0 | yes |
| 6 | 8 tier handlers + tier classifier | dev queries ~$1 | yes |
| 7 | FastAPI `/ask` and `/eval` endpoints | $0 | yes |
| 8 | Build 40+ eval question set | ~$0.50 | yes |
| 9 | Quality-vs-budget runs at $1 / $5 / $20 | ~$18 | yes |
| 10 | Cost report, README, submission polish | $0 | yes |
| **Total** | | **~$26 + $4 buffer** | |

---

## Phase 0 — Project scaffolding

**Goal**: dependencies installed, env keys present, directory tree ready. (Mostly done; updated for Datalab pivot.)

- [x] Add `pydantic`, `httpx`, `pandas`, `python-dotenv`, `fastapi`, `uvicorn` to `requirements.txt`
- [x] Add `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` to `.env`
- [ ] Add `DATALAB_API_KEY` to `.env` (user provides)
- [x] Create directory tree: `data/markdown/`, `data/extractions/`, `data/chroma/`, `eval/reports/`, `api/core/handlers/`
- [x] `api/core/budget.py` already stubbed with `BUDGET_LEVEL` + `record_cost()` + `total_spent()`
- [ ] **Strip local-Marker deps**: remove `marker-pdf`, `torch`, `torchvision`, `torchaudio` (no longer needed; cloud API does parsing)
- [ ] Re-enable `anthropic` (needed for Phase 2). Other later-phase deps (`chromadb`, `sentence-transformers`, `FlagEmbedding`, `networkx`, `pint`, `tiktoken`, `tqdm`) stay commented until their phase

---

## Phase 1 — PDF parsing (Datalab Marker API)

**Goal**: convert all 100 PDFs to clean markdown via the Datalab cloud API (Marker hosted on their GPUs).

**Why we pivoted from local Marker**: RTX 3050 Ti's 4 GB VRAM is below Marker's 5 GB minimum; first attempt thrashed at 5+ min/page on layout alone. CPU mode equally slow (9 GB RAM, no progress in tens of minutes). Cloud API removes the constraint.

**API contract**:
- `POST https://www.datalab.to/api/v1/convert` — multipart upload (`file=@pdf`, `output_format=markdown`), header `X-API-Key`. Returns `{request_id, request_check_url}`.
- `GET {request_check_url}` — poll every ~2s until `status=="complete"`; result has `markdown` field.
- Limits: 400 RPM, 400 concurrent, 5000 in-flight pages, 200 MB max file.

**Tasks**:
- [ ] Rewrite `scripts/parse_pdfs.py` — async/httpx, **bounded concurrency (8 in flight)**, submit + poll, write `data/markdown/{paper_id}.md`
- [ ] Each successful parse: call `budget.record_cost("datalab_parse", est_per_page * pages, paper_id=...)`
- [ ] Test on `--sample 1` (one paper); verify markdown has section headers and tables
- [ ] Test on `--sample 3` (ViT, Swin, DeiT); spot-check tables
- [ ] Run on all 100; track `data/parse_failures.json`
- [ ] Inspect any failures; per-paper retry once before logging final failure
- [ ] Decide per-failed-paper: fall back to PyMuPDF or skip

**Verification**: `data/markdown/` has ≥95 `.md` files; ViT, Swin, DeiT spot-checked manually; cost log shows ≤$5 spent on Datalab.

---

## Phase 2 — Verbatim extraction

**Goal**: per-paper structured JSON populated by GPT-5.4-mini (OpenAI), with automatic prefix caching.

**Why switched from Haiku 4.5**: Anthropic account ran out of credits; switched to OpenAI GPT-5.4-mini which has the same structured-output capability. OpenAI auto-caches identical system-prompt prefixes (≥1024 tokens) without explicit `cache_control` markers.

**Note**: gpt-5.4-mini requires `max_completion_tokens` instead of `max_tokens` in the API call.

- [x] Define Pydantic schema (`api/core/schemas.py`)
- [x] Write extraction system prompt + 2 few-shot examples (`api/core/extraction_prompt.py`, ~4k tokens)
- [x] Build `api/core/llm.py` with both Anthropic + OpenAI clients, cost tracking
- [x] Write `scripts/extract_papers.py` — async OpenAI structured output, bounded concurrency, idempotent
- [x] Tested on 1 paper — working ($0.0146/paper, 23k tokens in / 3.2k out)
- [ ] Test on 5 papers; review extracted JSON against PDFs (benchmark tables, model variants)
- [ ] Run on all 81 available markdowns; verify total cost stays under $3

**Verification**: ≥81 .json files in `data/extractions/`; 3 spot-checked vs PDFs; cost log under budget.

---

## Phase 3 — Normalization

**Goal**: canonical entities + canonical numeric values, written to SQLite.

- [ ] Write `scripts/normalize_numbers.py` — pure-Python parser (regex + Pint) over all `*_surface` numeric fields (params, FLOPs, accuracy, compute)
- [ ] Write `scripts/normalize_entities.py`:
  - Collect unique surface forms per type (datasets, benchmarks, metrics, methods)
  - Try Papers With Code API for canonical resolution
  - For unresolved: embed with `text-embedding-3-small` → cluster at cosine ≥ 0.85
  - Batched Haiku call confirms canonical name per cluster
- [ ] Add a `data/manual_aliases.json` for hand-fixing obvious clustering errors after first run
- [ ] Verify: "ImageNet" / "ImageNet-1K" / "ILSVRC2012" all map to one canonical_id

**Verification**: SQLite `entities` table populated; alias coverage spot-checked on 5 well-known datasets.

---

## Phase 4 — Build indexes

**Goal**: SQLite, Chroma, and NetworkX populated and queryable.

- [ ] Write `scripts/build_indexes.py`:
  - Create SQLite schema (`papers`, `entities`, `mentions`, `results`, `claims`, `references`)
  - Insert from `data/extractions/*.json` joined with normalization output
  - Embed each section chunk (Marker H2/H3 boundaries) with `text-embedding-3-small` → Chroma persistent collection
  - Build NetworkX citation graph: query S2 `/paper/{id}/references` for each corpus paper, intersect with corpus IDs → in-memory `DiGraph` → pickle to `data/citation_graph.gpickle`
- [ ] Sanity queries: `SELECT COUNT(*) FROM mentions GROUP BY type;` `nx.in_degree(graph, 'ViT_id')`

**Verification**: 100 papers in SQLite, ~5k chunks in Chroma, citation graph has ≥50 in-corpus edges.

---

## Phase 5 — Core query infrastructure

**Goal**: shared modules that all tier handlers reuse.

- [x] `api/core/llm.py` — Anthropic + OpenAI clients, model selector by `BUDGET_LEVEL`, OpenAI auto-cache + Anthropic cache_control helpers, cost tracking
- [ ] `api/core/retrieval.py` — Chroma query wrapper + BGE-reranker-v2; respects `BUDGET_LEVEL` for `k` and rerank-N
- [ ] `api/core/store.py` — typed wrappers for SQLite tables (papers, entities, mentions, results, citations) + NetworkX graph access
- [ ] `api/core/classifier.py` — Haiku tier classifier with few-shot examples per tier, returns `{tier, confidence, normalized_question}`
- [ ] Unit tests for each (smoke tests against the built indexes)

**Verification**: each module callable independently; classifier returns correct tier on 5 hand-written questions per tier.

---

## Phase 6 — Tier handlers

**Goal**: 8 handler modules, each returning `{answer, citations, confidence, cost_usd}`.

Implement in this order (simpler tiers first to validate the framework):

- [ ] **Tier 5** (citation-graph) — pure NetworkX, no LLM in core path
- [ ] **Tier 4** (temporal) — pandas over `results`/`papers`
- [ ] **Tier 2** (aggregation) — Haiku NL→SQL over `entities`/`mentions`
- [ ] **Tier 8** (quantitative) — Sonnet code-interpreter via tool use, executes pandas in sandbox
- [ ] **Tier 1** (single-doc factual) — pre-extracted lookup → RAG fallback (Chroma + rerank + Haiku)
- [ ] **Tier 3** (contradiction) — variance scan over `results` + Sonnet textual claim compare
- [ ] **Tier 6** (multi-hop) — Sonnet decompose → chain of store/retrieval calls → Sonnet synthesize
- [ ] **Tier 7** (negation) — PWC + Sonnet expansion → set-diff against `entities` → Haiku verify-each
- [ ] Hand-test each handler with 2–3 questions before moving on

**Verification**: each handler returns valid output structure with citations on its hand-written questions.

---

## Phase 7 — FastAPI endpoints

**Goal**: HTTP surface for query and evaluation.

- [ ] `api/routes/ask.py` — `POST /ask` accepts `{question, budget_level?, target_paper_id?}`, calls classifier → handler, returns full response with citations and cost
- [ ] `api/routes/eval.py` — `POST /eval` accepts `{budget_level}`, runs all questions in `eval/questions.jsonl`, returns per-tier accuracy + cost summary, writes to `eval/reports/{budget_level}_{timestamp}.json`
- [ ] Wire routers into `main.py`
- [ ] Smoke-test via curl with one question per tier

**Verification**: `POST /ask` returns valid JSON for sample questions across all 8 tiers.

---

## Phase 8 — Eval set construction

**Goal**: `eval/questions.jsonl` with ≥40 questions (≥3 per tier), gold answers + citations.

- [ ] Write `scripts/generate_eval_set.py` — Sonnet drafts ~6 candidate questions per tier from the extracted DB, includes proposed gold answer + citations
- [ ] Manual review pass: validate every gold answer against the actual PDFs/DB; tighten ambiguous questions
- [ ] Mark difficulty (easy/medium/hard) and expected handler tier per question
- [ ] Save final set to `eval/questions.jsonl`

**Verification**: 40+ entries, ≥3 per tier, every entry has `{tier, question, gold_answer, gold_citations, notes}`.

---

## Phase 9 — Quality-vs-budget eval runs

**Goal**: produce the assessment-required quality-vs-budget curve.

- [ ] Run `POST /eval {budget_level: "$1"}` → save report
- [ ] Run `POST /eval {budget_level: "$5"}` → save report
- [ ] Run `POST /eval {budget_level: "$20"}` → save report
- [ ] Generate plot: per-tier accuracy vs cost-per-question (matplotlib → PNG in `eval/reports/`)
- [ ] Compute: total cost, mean/median/max cost per question
- [ ] Track running budget — abort early if approaching $30 cap

**Verification**: 3 budget reports + 1 quality-vs-budget plot exist.

---

## Phase 10 — Submission polish

**Goal**: repo is reproducible end-to-end, docs are accurate.

- [ ] Update `README.md`: setup, env vars, run instructions for full pipeline + how to query, link to ARCHITECTURE.md
- [ ] Cost report (`COST_REPORT.md`): final $ spent per phase, per-question stats, the quality-vs-budget plot
- [ ] Eval results (`eval/RESULTS.md`): each question + gold + system answer per budget level
- [ ] Verify clean clone + setup works on a fresh checkout (or document any local-only dependencies)
- [ ] Final commit, tag, push

**Verification**: a fresh user with this repo + the API keys + `pip install -r requirements.txt` can reproduce all reported numbers.

---

## Risks & contingencies

| Risk | Mitigation |
|---|---|
| Marker fails on some PDFs | Fall back to PyMuPDF for those papers; note in manifest |
| Extraction quality poor for benchmark tables | Add table-specific prompt section; consider one-off Sonnet pass on the worst cases |
| Entity normalization mis-merges (e.g., CIFAR-10 ↔ CIFAR-100) | Manual override file `data/manual_aliases.json` |
| Cost overrun during eval | `budget.py` aborts a question if running total exceeds 90% of phase budget |
| S2 rate limits on references API | Throttle 1 req/s; cache responses to disk |
| Hidden test set has unanticipated question shapes | Tier classifier returns `unknown` → fall back to Tier 6 (multi-hop) which is most general |
