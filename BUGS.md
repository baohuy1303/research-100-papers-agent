# Known Bugs & Quality Issues

Tracking what we've observed but haven't fixed yet, with severity + repro + impact.
Updated 2026-05-13 after Phase 9 eval runs (29–31/40 across $1/$5/$20).

## Fixed in this round (Phase 9)
- **B6** (Tier 8 code planning JSON validation) — switched from `client.beta.chat.completions.parse()` to `json_object` mode + `JSONDecoder().raw_decode()` to handle trailing content from reasoning model; T8-001/002/004/005 now pass
- **B-T5a** (Tier 5 k=10 cap truncates citation counts) — system prompt now instructs k=100 for counting queries; `total_count` added to evidence notes; T5-002 (Swin citations=47) now passes
- **B-T5b** (Tier 5 paper nickname lookup fails) — added `_PAPER_NICKNAMES` dict mapping acronyms to title substrings (DeiT→"data-efficient image transformers", etc.); T5-003 (DeiT→ViT citation) now passes

Eval snapshot (2026-05-13): **$1 budget: 29/40 (72%)**, **$5 budget: 31/40 (78%)**, **$20 budget: 31/40 (78%)**. Cost: $0.0013–0.0014/question avg.

---

## Previously fixed (Phase 7–8)
- **B1** (Tier 2 NL→SQL forgot metric filter) — fixed by adding worked example + canonical-name list to system prompt
- **B2** (Tier 6 narration premature exit) — handler now nudges the LLM if a tool-less response looks like narration
- **B3** (Tier 7 set diff missed aliases) — observed_set now unions canonical + aliases table, with substring fallback
- **B4** (Tier 6 result truncation lost paper_ids) — `_safe_serialize` now trims the LARGEST list field first, preserving compact ID lists
- **B5** (adversarial questions burned budget) — `is_adversarial()` shared helper in `handlers/base.py` short-circuits before classifier

Battery snapshot: 19/19 OK, $0.029, 5.1s avg.

---

## Severity legend

- **HIGH** — wrong answer, silent corruption of evidence, or pipeline crash
- **MED** — answer technically correct but misleading, or handler converges slowly
- **LOW** — cosmetic, edge case, or future polish

---

## HIGH

### B1 — Tier 2 NL→SQL forgets to filter by metric

**Repro**: `python scripts/ask_cli.py` → `Which 5 papers report the highest top-1 accuracy on ImageNet?`

**Symptom**: SQL filters by `dataset='ImageNet'` only — pulls every benchmark on ImageNet (top-1, top-5, mIoU on ImageNet-Seg, FID, etc.). Returns nonsensical values like 1337.7, 993.3 as "top-1 accuracy" because those are actually FID scores.

**Root cause**: The Tier 2 system prompt mentions filtering by entity type but doesn't emphasize that benchmark queries usually need to filter by BOTH dataset AND metric.

**Fix sketch**: Add an explicit example to `tier2_aggregate.py` SCHEMA_DOC: "When the question names a metric (e.g. 'top-1 accuracy', 'mIoU'), JOIN entities twice — once for dataset and once for metric — and filter both."

**Impact**: Major — Tier 2 is the most-used handler, this affects any benchmark-ranking question.

---

### B2 — Tier 6 multi-hop sometimes returns "still gathering evidence" instead of finalizing

**Repro**: `python scripts/cli_battery.py` (intermittent — ~10-20% of Tier 6 runs)

**Example seen**: `Among papers citing ViT, which has the largest model variant?` returned `"I'm still gathering evidence to identify..."` after 24.8s and 6 tool calls ($0.011).

**Root cause**: When the LLM emits a text-only response (no tool calls) mid-loop, [tier6_multihop.py:188-191](api/core/handlers/tier6_multihop.py#L188-L191) treats it as the final answer and exits early:
```python
if not msg.tool_calls:
    final_answer = msg.content or "(no answer)"
    break
```
But sometimes the model is just narrating its plan ("Let me check X first…") and would have called a tool on the next iteration if we'd let it.

**Fix sketch**: When `not msg.tool_calls` AND the content looks like narration ("checking", "gathering", "let me", etc.) AND we're not on the last allowed step, push back with a `{role: "user", content: "Please call a tool now or finalize."}` and continue rather than exit.

**Impact**: Tier 6 is already the most expensive tier; failing after spending $0.011 wastes that budget.

---

## MED

### B3 — Tier 7 absence handler hits non-canonical surface forms in expected set

**Repro**: `Which standard ViT benchmarks are NOT covered by this corpus?` (early test, before BUGS.md)

**Symptom**: LLM proposed expected items like `"ImageNet-1K"` and `"MS COCO"` that don't match our canonical names (`"ImageNet"`, `"COCO"`). Set difference flags them as missing even though they ARE present under different aliases.

**Root cause**: [tier7_absence.py:81-86](api/core/handlers/tier7_absence.py#L81-L86) checks set difference against canonical names, not against the union of canonical + aliases.

**Fix sketch**: Build `observed_set` as union of `canonical` AND every alias from the `aliases` table. Compare expected items against that wider set.

**Impact**: Tier 7 systematically over-reports "missing" items. Currently mitigated by the LLM proposing well-known names that mostly match canonicals (top 30 datasets are curated), but breaks for less-common entities.

---

### B4 — Tier 6 truncates large tool results at 8000 chars mid-string

**Repro**: large `cited_by` results on dense-citation papers like ViT (~80 paper_ids returned)

**Symptom**: [tier6_multihop.py:218](api/core/handlers/tier6_multihop.py#L218) does `json.dumps(result, default=str)[:8000]`. If the result JSON exceeds 8000 chars, it gets sliced mid-string, returning malformed JSON to the LLM. The LLM may then see garbage on the trailing edge.

**Fix sketch**: Either (a) compress the result by stripping non-essential fields before truncation, or (b) ensure truncation happens at a JSON-array boundary (drop trailing partial entries cleanly).

**Impact**: Hasn't bitten us in tests yet because the 80-paper cited_by result is just under 8000 chars, but will break if we ever query a paper with 100+ in-corpus citations.

---

### B5 — Adversarial / unanswerable questions waste budget instead of refusing

**Repro**: `python scripts/cli_battery.py` → Q4 `"What is the architecture of a paper that doesn't exist?"`

**Symptom**: Classifier routes to T7 (negation, confidence 0.93), spends 19.5s and $0.0012, returns a vague "evidence does not identify..." answer.

**Root cause**: No top-level "is this answerable?" check. We just route to whichever tier has highest LLM confidence and run the handler.

**Fix sketch**: Pre-classifier sanity check: if the question references a specific entity (paper, dataset, model), try to resolve it first. If unresolvable, return early with `"That entity doesn't appear in this corpus."` saving the handler call.

**Impact**: Wastes a few cents per adversarial question. Becomes important if eval set or hidden test set includes such questions.

---

### B6 — Tier 8 code planner can produce invalid JSON when reasoning + code overflows token budget

**Repro**: Earlier test: `Which 5 papers report the highest top-1 accuracy on ImageNet?` (before classifier picked Tier 2 instead)

**Symptom**: `Code planning failed: 1 validation error for _CodePlan: Invalid JSON: trailing characters at line 2 column 1`

**Status**: Partially mitigated — bumped `max_completion_tokens` from 1024 → 8192. Error now caught gracefully (returns an error message instead of crashing) but the underlying brittleness remains. gpt-5.4-mini is a reasoning model and can still overflow on complex code.

**Fix sketch**: Switch from `client.beta.chat.completions.parse(response_format=_CodePlan)` to a regular call with `response_format={"type": "json_object"}` and manual JSON parsing — more lenient about reasoning-padded outputs.

---

## LOW

### B7 — Citation paper_id displayed as ellipsis-truncated `268d347e8a55…` in CLI table

**Repro**: any CLI question with citations

**Symptom**: paper_id column truncated to 12 chars + `…`. To use `/paper <id>` you have to know the prefix already.

**Fix sketch**: Add a `[1]`, `[2]`, … shortcut in the citation table that maps to the full paper_id, then `/paper [1]` resolves to the first citation's full ID.

---

### B8 — cp1252 console can't render Unicode in answers

**Repro**: any answer containing curly apostrophes, em-dashes, or non-ASCII characters on Windows console

**Symptom**: Apostrophes in answers display as `?` or `’` escapes.

**Status**: Partially mitigated — `safe_print()` ASCII-encodes outputs but the rich console still has issues with some Unicode. Fully fixing requires switching console to UTF-8 (`chcp 65001` or `PYTHONIOENCODING=utf-8`).

---

### B9 — IDE diagnostics show false-positive "module not found" for venv-only packages

**Repro**: open any script importing `chromadb`, `rapidfuzz`, `networkx`

**Symptom**: VS Code diagnostics report `Cannot find module 'chromadb'` because the IDE checks system Python, not `venv/`.

**Fix sketch**: Add a `.vscode/settings.json` pointing `python.defaultInterpreterPath` at `venv/Scripts/python.exe`.

---

### B10 — `eval/questions.jsonl` currently holds 3 sample questions only

**Status**: intentional placeholder; Phase 8 replaces with 40+ generated questions.

---

## Architectural quality observations (not bugs)

### Q1 — All handlers use the same `synthesize_answer()` helper which calls gpt-5.4-mini. Per question, this is the single biggest token consumer.

**Optimization**: For purely structured tiers (2, 4, 5, 8), we could template the answer from the SQL/graph result directly without an LLM call. Would cut latency ~50% and cost ~70% on those tiers.

### Q2 — `Retriever`, `CorpusStore`, `TierClassifier` are constructed per-process via lazy module globals. Fine for FastAPI (one process serves many requests) but the CLI also instantiates them on every script invocation. ~1-2s startup overhead.

### Q3 — `corpus.db` is opened with `check_same_thread=False` for read-only safety, but we never write at query time. If we ever introduce write operations (e.g. caching answers), this needs revisiting.

### Q4 — Tier 6 max_steps is hardcoded `{$1: 2, $5: 6, $20: 10}`. Should ideally be set via `BUDGET_PROFILES` in `budget.py` for consistency with the other tunables there.

---

## Pipeline-level lessons (already encoded in PLAN.md "Lessons learned")

These aren't bugs — they're design constraints we discovered during build that shaped the architecture. Listed here for cross-reference:

- gpt-5-mini is a reasoning model; needs ≥512 max_completion_tokens
- Surface forms must be collected from BOTH `datasets_mentioned` AND `benchmark_results.dataset_surface`
- Stage 2 of entity normalization must reverse-lookup against existing canonicals to prevent fragmentation
- Multi-hop agents need ≥6 steps (raised from 4)
- Windows cp1252 console needs `safe_print` ASCII fallback

---

## Battery test snapshot (2026-05-12)

| Status | Count |
|---|---|
| OK | 17 |
| TIER-MISMATCH | 1 (Q4 — adversarial) |
| SANITY-FAIL | 1 (Q15 — B2 reproduced) |
| ERROR | 0 |

Total cost: $0.0346 / 19 questions = ~$0.002 avg.
Total wall time: 134s / 19 questions = 7.0s avg.

---

## Suggested fix order (highest ROI first)

1. **B1** (Tier 2 metric filter) — affects most queries
2. **B3** (Tier 7 alias-aware set diff) — needed for Phase 9 eval accuracy
3. **B2** (Tier 6 narration handling) — wastes budget on the most expensive tier
4. **B5** (adversarial pre-check) — adds robustness for hidden test
5. **B4** (Tier 6 result truncation) — preventive; not biting yet
6. **B6** (Tier 8 token budget) — already mitigated, low priority
7. Q1 (skip LLM synthesis for structured tiers) — optimization, not correctness
