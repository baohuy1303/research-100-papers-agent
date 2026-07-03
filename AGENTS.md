# AGENTS.md

Guidance for agents working in this repository.

## Project Purpose

Build and polish a Research Comprehension System over the 100 most-cited Vision Transformer papers. The system answers questions across eight tiers: single-doc factual, corpus aggregation, contradiction, temporal, citation graph, multi-hop, negation, and quantitative.

Hard constraints:

- $30 USD total spend cap.
- No fine-tuning.
- Answers must be cited and defensible.

## Current State

The corpus, extraction pipeline, indexes, API, CLI, and eval set are implemented.

Key artifacts:

- `data/manifest.csv` lists all 100 papers.
- `data/pdfs/` contains the source PDFs.
- `data/markdown/`, `data/extractions/`, and `data/normalized/` contain processed paper data.
- `data/corpus.db`, `data/chroma/`, and `data/citation_graph.gpickle` power query-time reasoning.
- `eval/questions.jsonl` and `eval/RESULTS.md` define and summarize the eval suite.

## Common Commands

```bash
source venv/Scripts/activate
pip install -r requirements.txt

uvicorn main:app --reload
python scripts/ask_cli.py
python scripts/sanity_check.py --skip-llm
python scripts/run_eval.py --budget '$5' --limit 5
```

## Architecture Notes

- HTTP routes in `api/routes/` should stay thin.
- Shared question execution belongs in `api/core/pipeline.py`.
- Request-scoped budget belongs in `api/core/runtime.py`; do not mutate `os.environ` for per-request budget changes.
- Tier-specific logic belongs in `api/core/handlers/`.
- LLM-generated SQL must pass through `api/core/sql_safety.py`.
- Tier 8 code execution must remain AST-validated and sandboxed.

## Data Gotchas

- Pandas should read CSV columns with `dtype=str` and `fillna("")` when handling manifest data.
- Windows console output should use ASCII-safe printing for paper titles and user-data strings.
- Some publisher PDF URLs fail; arXiv fallback URLs are often available through Semantic Scholar metadata.
- Keep demo-critical data artifacts in git, but keep secrets, bytecode, transient cost logs, and failure logs ignored.
