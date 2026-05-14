# Research Comprehension System — 100 Vision Transformer Papers

A retrieval-augmented system that answers natural-language questions over a corpus of the 100 most-cited Vision Transformer papers, with cited answers across 8 difficulty tiers.

**Total pipeline cost**: ~$6.90 (one-time prep) + ~$0.07 per 40-question eval run  
**Quality**: 100% pass rate at $1 / $5 / $20 budget levels on the 40-question eval set

---

## Setup

```bash
# Clone and create venv (Python 3.11+)
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
# or: source venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
```

Create `.env` in the project root:

```env
OPENAI_API_KEY=sk-...
DATALAB_API_KEY_1=...          # free-tier key for Phase 1 (PDF parsing)
DATALAB_API_KEY_2=...          # second free-tier key (optional, for failover)
```

---

## Reproducing the full pipeline

The one-time prep pipeline takes ~57 minutes and costs ~$6.90 (dominated by Datalab PDF parsing). Each step is idempotent — re-running skips already-completed work.

```bash
# 1. Assemble corpus manifest (queries Semantic Scholar Bulk API, no cost)
python scripts/fetch_papers.py
# Output: data/manifest.csv (100 papers, sorted by citation count)

# 2. Download PDFs
python scripts/download_pdfs.py
# Output: data/pdfs/{paper_id}.pdf  (~2–3 minutes, skips existing)

# 3. Parse PDFs to markdown (Datalab cloud Marker API, ~$5.35 if paid)
python scripts/parse_pdfs.py
# Output: data/markdown/{paper_id}.md  (~30 minutes, bounded concurrency=8)

# 4. Extract structured data (GPT — ~$1.41, ~19 min, concurrency=10)
python scripts/extract_papers.py
# Output: data/extractions/{paper_id}.json

# 5. Normalize numbers (regex, free, <1s)
python scripts/normalize_numbers.py
# Output: data/normalized/{paper_id}.json

# 6. Normalize entities — collapse 2,269 surface forms → 1,950 canonical entities
#    (embeddings + HF API + GPT disambiguation, ~$0.04, ~75s)
python scripts/normalize_entities.py
# Output: data/entity_map.json

# 7. Build indexes (SQLite + Chroma + NetworkX, ~$0.05, ~2.5 min)
python scripts/build_indexes.py
# Output: data/corpus.db, data/chroma/, data/citation_graph.gpickle

# 8. Verify everything is wired up (33 sanity checks, ~20s, ~$0.005)
python scripts/sanity_check.py
```

> **Note on PDF parsing**: Datalab's free tier allows ~55 papers per API key. Two free-tier keys (`DATALAB_API_KEY_1` + `DATALAB_API_KEY_2`) are sufficient. The script rotates keys automatically on 403 rate-limit errors.

---

## Running the API

```bash
uvicorn main:app --reload
# → http://localhost:8000
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which paper has the highest ImageNet top-1 accuracy?", "budget_level": "$5"}'
```

Response:
```json
{
  "answer": "CoCa achieves the highest reported ImageNet top-1 accuracy at 91.0%...",
  "tier": 3,
  "tier_confidence": 0.97,
  "citations": [{"paper_id": "...", "paper_title": "CoCa: Contrastive Captioners..."}],
  "cost_usd": 0.0018,
  "handler_reasoning": "mode=numeric"
}
```

The `budget_level` parameter controls retrieval depth and multi-hop step count:

| Level | Chroma k | T6 max steps | Use case |
|-------|----------|--------------|----------|
| `$1`  | 3        | 3            | Fast / cheap (still 100% on eval) |
| `$5`  | 8        | 6            | Default balanced setting |
| `$20` | 15       | 10           | Thorough / complex questions |

### Run the eval suite

```bash
# All 3 budget levels, 40 questions each
python scripts/run_eval.py

# Single budget level
python scripts/run_eval.py --budget '$5'

# Quick smoke test (first 5 questions)
python scripts/run_eval.py --budget '$1' --limit 5
```

Results are written to `eval/RESULTS.md` and `eval/reports/{timestamp}_{budget}.json`.

---

## Question tiers

The system routes each question to one of 8 handlers based on a GPT classifier:

| Tier | Type | Example |
|------|------|---------|
| T1 | Single-paper factual | "What architecture does ViT use?" |
| T2 | Corpus aggregation (SQL) | "How many papers benchmark on ImageNet?" |
| T3 | Contradiction / comparison | "Do papers agree on ADE20K SOTA?" |
| T4 | Temporal evolution | "How did top-1 accuracy change year over year?" |
| T5 | Citation graph | "Which paper is most cited within this corpus?" |
| T6 | Multi-hop compositional | "Among ViT-citing papers, which has the largest model?" |
| T7 | Negation / absence | "Which segmentation datasets are NOT used in this corpus?" |
| T8 | Quantitative compute | "What is the median parameter count across all models?" |

---

## Repo layout

```
data/
  manifest.csv          # 100-paper corpus metadata
  pdfs/                 # downloaded PDFs
  markdown/             # Marker-parsed markdown
  extractions/          # GPT structured extraction JSONs
  normalized/           # number-normalized JSONs
  entity_map.json       # canonical entity → aliases map
  corpus.db             # SQLite: papers, entities, results, claims, ...
  chroma/               # Chroma vector store (3768 chunks)
  citation_graph.gpickle # NetworkX DiGraph (761 in-corpus edges)
  cost_log.jsonl        # append-only spend log

api/
  core/
    store.py            # typed SQLite + NetworkX wrapper
    retrieval.py        # Chroma search
    classifier.py       # tier router
    handlers/           # tier1_*.py … tier8_*.py
    budget.py           # spend tracking + BUDGET_LEVEL config
    llm.py              # OpenAI client + cost helpers
  routes/
    ask.py              # POST /ask
    eval.py             # POST /eval

scripts/
  fetch_papers.py       # Semantic Scholar corpus assembly
  download_pdfs.py      # PDF downloader
  parse_pdfs.py         # Datalab Marker cloud API
  extract_papers.py     # GPT structured extraction
  normalize_numbers.py  # regex number normalization
  normalize_entities.py # 6-stage entity canonicalization
  build_indexes.py      # SQLite + Chroma + NetworkX builder
  sanity_check.py       # 33 end-to-end checks
  run_eval.py           # quality-vs-budget evaluation runner
  ask_cli.py            # quick interactive CLI query tool

eval/
  questions.jsonl       # 40 eval questions with gold answers
  RESULTS.md            # latest eval results
  reports/              # per-run JSON reports
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions: why Marker over Nougat, why SQLite over DuckDB, why a tiered handler over a single ReAct agent, and the full quality-vs-budget tradeoff analysis.

See [COST_REPORT.md](COST_REPORT.md) for the phase-by-phase cost breakdown and quality-vs-budget curve.
