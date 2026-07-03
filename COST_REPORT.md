# Cost Report

## One-Time Prep

| Phase | Tooling | Paid cost | Time |
|---|---|---:|---:|
| PDF parsing | Datalab Marker API, free-tier keys | $0 paid / about $5.35 credit value | about 34 min |
| Structured extraction | OpenAI GPT-5.4-mini structured output | $1.41 | about 19 min |
| Number normalization | regex only | $0.00 | <1s |
| Entity normalization | embeddings + HF lookup + GPT disambiguation | about $0.20 during dev | about 75s/run |
| Index build | SQLite + Chroma embeddings + Semantic Scholar graph | about $0.05 | about 2.5 min |

Single clean prep run is about $1.50 paid plus free-tier Datalab usage. Logged project spend is higher because it includes development iterations and evals.

## Eval Cost

Latest recorded 40-question eval:

| Budget | Cost | Avg/question | Pass rate |
|---|---:|---:|---:|
| `$1` | about $0.066 | about $0.0017 | 100% |
| `$5` | about $0.067 | about $0.0017 | 100% |
| `$20` | about $0.069 | about $0.0017 | 100% |

## Budget Controls

| Setting | `$1` | `$5` | `$20` |
|---|---:|---:|---:|
| Retrieval `k` | 3 | 8 | 15 |
| Tier 6 max tool steps | 3 | 6 | 10 |
| Tier 7 expansion | compact | default | broader |

## Spend Guardrails

- `api/core/budget.py` records every LLM/embedding cost event to `data/cost_log.jsonl`.
- Eval runners abort once total recorded spend exceeds the configured safety threshold.
- The hidden-test buffer target is at least $5 under the $30 cap.
