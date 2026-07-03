# Architecture

## Goal

Answer natural-language questions over 100 Vision Transformer papers with cited, defensible answers under a $30 spend cap and without fine-tuning.

The central design choice is to spend once on extraction and normalization, then keep query-time answers cheap by routing to structured tools whenever possible.

## Data Flow

```text
PDFs
  -> Datalab Marker parse
  -> markdown
  -> OpenAI structured extraction
  -> numeric and entity normalization
  -> SQLite + Chroma + NetworkX indexes
  -> API / CLI / eval pipeline
```

## Query-Time Flow

`api/core/pipeline.py` is the single orchestration path for `/ask`, the CLI, the battery script, and eval:

```text
question
  -> adversarial pre-check
  -> TierClassifier
  -> confidence fallback if needed
  -> tier handler
  -> PipelineResult
```

Budget selection is request-scoped via `api/core/runtime.py`; request handlers no longer mutate process-wide environment variables.

## Core Modules

| Module | Responsibility |
|---|---|
| `api/core/runtime.py` | request-scoped `RuntimeConfig` and budget context |
| `api/core/budget.py` | budget profiles and append-only cost logging |
| `api/core/pipeline.py` | shared classify/handle/compose flow |
| `api/core/sql_safety.py` | single-statement read-only SQL validation |
| `api/core/store.py` | SQLite and citation graph access |
| `api/core/retrieval.py` | Chroma search and query embeddings |
| `api/core/classifier.py` | 8-tier LLM router |
| `api/core/handlers/` | tier-specific reasoning logic |

## Tier Handlers

| Tier | Handler | Notes |
|---|---|---|
| T1 | `tier1_factual.py` | uses `target_paper_id` when supplied, otherwise resolves the paper by title |
| T2 | `tier2_aggregate.py` | LLM writes read-only SQL, store enforces safety |
| T3 | `tier3_contradict.py` | numeric benchmark spread or textual claim comparison |
| T4 | `tier4_temporal.py` | time-series SQL with year grouping |
| T5 | `tier5_citation.py` | graph operations over in-corpus citations |
| T6 | `tier6_multihop.py` | tool-calling with SQL, graph, retrieval, and final answer tools |
| T7 | `tier7_absence.py` | expected set minus observed corpus set |
| T8 | `tier8_compute.py` | AST-validated sandboxed pandas snippets |

## Safety Boundaries

- LLM-generated SQL is validated by `validate_readonly_sql()` before SQLite execution.
- Tier 8 generated code is AST-checked before `exec()` and receives copies of DataFrames.
- Runtime budget is context-local, so concurrent API requests cannot leak budget profiles.
- The API returns structured errors in the normal response envelope instead of throwing raw tracebacks.

## Artifacts

Key artifacts are intentionally checked in so a reviewer can run the project without spending an hour rebuilding the corpus:

- `data/corpus.db`
- `data/chroma/`
- `data/citation_graph.gpickle`
- extracted and normalized JSON
- eval questions and reports

Transient cost logs, failure logs, caches, and local environments are ignored.
