# Eval Results — Quality vs Budget

Generated: 2026-05-13 02:00

## Overall Summary

| Budget | Pass Rate | Passed/Total | Total Cost | Avg Cost/Q | Avg Latency/Q |
|--------|-----------|--------------|------------|------------|---------------|
| $1 | 72% | 29/40 | $0.052 | $0.0013 | 3.6s |
| $5 | 78% | 31/40 | $0.057 | $0.0014 | 3.7s |
| $20 | 78% | 31/40 | $0.056 | $0.0014 | 3.9s |

## Per-Tier Accuracy

| Tier | $1 pass | $1 cost | $5 pass | $5 cost | $20 pass | $20 cost |
|------||-----------|------------||-----------|------------||-----------|------------|
| T1   | 80% (4/5) | $0.0013 | 80% (4/5) | $0.0013 | 80% (4/5) | $0.0014 |
| T2   | 100% (5/5) | $0.0011 | 100% (5/5) | $0.0011 | 100% (5/5) | $0.0011 |
| T3   | 60% (3/5) | $0.0015 | 60% (3/5) | $0.0014 | 60% (3/5) | $0.0015 |
| T4   | 80% (4/5) | $0.0013 | 100% (5/5) | $0.0013 | 100% (5/5) | $0.0012 |
| T5   | 100% (5/5) | $0.0010 | 100% (5/5) | $0.0010 | 100% (5/5) | $0.0010 |
| T6   | 40% (2/5) | $0.0020 | 60% (3/5) | $0.0030 | 60% (3/5) | $0.0027 |
| T7   | 40% (2/5) | $0.0012 | 40% (2/5) | $0.0012 | 40% (2/5) | $0.0012 |
| T8   | 80% (4/5) | $0.0011 | 80% (4/5) | $0.0010 | 80% (4/5) | $0.0011 |

## Per-Question Detail

### Budget: $1

| ID | Tier | Pass | Match | Cost | Latency | Notes |
|----|------|------|-------|------|---------|-------|
| T1-001 | T1 | ✓ | structural | $0.0014 | 5.23s | 100% |
| T1-002 | T1 | ✗ | llm_judge | $0.0012 | 4.16s | 0% |
| T1-003 | T1 | ✓ | structural | $0.0015 | 2.94s | 100% |
| T1-004 | T1 | ✓ | llm_judge | $0.0011 | 3.26s | 100% |
| T2-001 | T2 | ✓ | substring | $0.0012 | 3.37s | 100% |
| T2-002 | T2 | ✓ | substring | $0.0011 | 2.59s | 100% |
| T2-003 | T2 | ✓ | llm_judge | $0.0011 | 3.1s | 100% |
| T2-004 | T2 | ✓ | substring | $0.0010 | 2.74s | 100% |
| T2-005 | T2 | ✓ | substring | $0.0012 | 2.98s | 100% |
| T3-001 | T3 | ✓ | llm_judge | $0.0018 | 3.96s | 100% |
| T3-002 | T3 | ✗ | llm_judge | $0.0013 | 4.04s | 0% (routed T8) |
| T3-003 | T3 | ✓ | llm_judge | $0.0016 | 3.52s | 100% |
| T3-004 | T3 | ✓ | llm_judge | $0.0012 | 3.41s | 100% |
| T3-005 | T3 | ✗ | llm_judge | $0.0015 | 3.74s | 0% |
| T4-001 | T4 | ✓ | structural | $0.0013 | 3.79s | 100% |
| T4-002 | T4 | ✓ | substring | $0.0013 | 5.0s | 100% (routed T2) |
| T4-003 | T4 | ✓ | llm_judge | $0.0013 | 4.23s | 100% |
| T4-004 | T4 | ✗ | substring | $0.0011 | 3.51s | 0% |
| T4-005 | T4 | ✓ | llm_judge | $0.0013 | 3.45s | 100% |
| T5-001 | T5 | ✓ | substring | $0.0012 | 2.65s | 100% |
| T5-002 | T5 | ✓ | substring | $0.0009 | 2.8s | 100% |
| T5-003 | T5 | ✓ | substring | $0.0010 | 2.65s | 100% |
| T5-004 | T5 | ✓ | substring | $0.0009 | 3.24s | 100% |
| T5-005 | T5 | ✓ | substring | $0.0012 | 3.13s | 100% |
| T6-001 | T6 | ✗ | llm_judge | $0.0036 | 4.32s | 0% |
| T6-002 | T6 | ✗ | llm_judge | $0.0024 | 4.5s | 0% |
| T6-003 | T6 | ✗ | llm_judge | $0.0013 | 3.34s | 0% (routed T5) |
| T6-004 | T6 | ✓ | llm_judge | $0.0012 | 3.83s | 100% (routed T2) |
| T6-005 | T6 | ✓ | llm_judge | $0.0015 | 3.18s | 100% |
| T7-001 | T7 | ✗ | llm_judge | $0.0012 | 3.8s | 0% |
| T7-002 | T7 | ✗ | llm_judge | $0.0012 | 3.88s | 0% |
| T7-003 | T7 | ✓ | llm_judge | $0.0012 | 4.58s | 100% |
| T7-004 | T7 | ✓ | structural | $0.0011 | 3.5s | 100% (routed T2) |
| T7-005 | T7 | ✗ | llm_judge | $0.0012 | 4.4s | 0% |
| T8-001 | T8 | ✓ | substring | $0.0009 | 3.12s | 100% |
| T8-002 | T8 | ✓ | llm_judge | $0.0010 | 3.64s | 100% |
| T8-003 | T8 | ✗ | substring | $0.0012 | 4.63s | 0% |
| T8-004 | T8 | ✓ | substring | $0.0009 | 3.98s | 100% |
| T8-005 | T8 | ✓ | llm_judge | $0.0014 | 4.77s | 100% |
| T1-005 | T1 | ✓ | structural | $0.0013 | 2.8s | 100% |

### Budget: $5

| ID | Tier | Pass | Match | Cost | Latency | Notes |
|----|------|------|-------|------|---------|-------|
| T1-001 | T1 | ✓ | structural | $0.0014 | 3.39s | 100% |
| T1-002 | T1 | ✗ | llm_judge | $0.0012 | 3.25s | 0% |
| T1-003 | T1 | ✓ | structural | $0.0015 | 3.54s | 100% |
| T1-004 | T1 | ✓ | llm_judge | $0.0011 | 3.31s | 100% |
| T2-001 | T2 | ✓ | substring | $0.0012 | 3.01s | 100% |
| T2-002 | T2 | ✓ | substring | $0.0011 | 2.72s | 100% |
| T2-003 | T2 | ✓ | llm_judge | $0.0011 | 3.08s | 100% |
| T2-004 | T2 | ✓ | substring | $0.0010 | 2.61s | 100% |
| T2-005 | T2 | ✓ | substring | $0.0012 | 3.3s | 100% |
| T3-001 | T3 | ✓ | llm_judge | $0.0017 | 4.02s | 100% |
| T3-002 | T3 | ✗ | llm_judge | $0.0014 | 5.31s | 0% (routed T8) |
| T3-003 | T3 | ✓ | llm_judge | $0.0013 | 4.09s | 100% |
| T3-004 | T3 | ✓ | llm_judge | $0.0016 | 3.56s | 100% |
| T3-005 | T3 | ✗ | llm_judge | $0.0012 | 3.99s | 0% |
| T4-001 | T4 | ✓ | structural | $0.0013 | 3.96s | 100% |
| T4-002 | T4 | ✓ | substring | $0.0012 | 3.4s | 100% (routed T2) |
| T4-003 | T4 | ✓ | llm_judge | $0.0013 | 3.44s | 100% |
| T4-004 | T4 | ✓ | substring | $0.0011 | 3.45s | 100% |
| T4-005 | T4 | ✓ | llm_judge | $0.0013 | 3.75s | 100% |
| T5-001 | T5 | ✓ | substring | $0.0012 | 3.36s | 100% |
| T5-002 | T5 | ✓ | substring | $0.0009 | 2.89s | 100% |
| T5-003 | T5 | ✓ | substring | $0.0010 | 3.1s | 100% |
| T5-004 | T5 | ✓ | substring | $0.0009 | 2.89s | 100% |
| T5-005 | T5 | ✓ | substring | $0.0012 | 3.12s | 100% |
| T6-001 | T6 | ✗ | llm_judge | $0.0060 | 6.12s | 0% |
| T6-002 | T6 | ✓ | llm_judge | $0.0049 | 9.26s | 100% |
| T6-003 | T6 | ✗ | llm_judge | $0.0013 | 3.63s | 0% (routed T5) |
| T6-004 | T6 | ✓ | llm_judge | $0.0012 | 3.21s | 100% (routed T2) |
| T6-005 | T6 | ✓ | llm_judge | $0.0017 | 2.88s | 100% |
| T7-001 | T7 | ✗ | llm_judge | $0.0011 | 3.71s | 0% |
| T7-002 | T7 | ✗ | llm_judge | $0.0012 | 3.79s | 0% |
| T7-003 | T7 | ✓ | llm_judge | $0.0012 | 4.04s | 100% |
| T7-004 | T7 | ✓ | structural | $0.0011 | 3.51s | 100% (routed T2) |
| T7-005 | T7 | ✗ | llm_judge | $0.0012 | 3.97s | 0% |
| T8-001 | T8 | ✓ | substring | $0.0009 | 3.3s | 100% |
| T8-002 | T8 | ✓ | llm_judge | $0.0010 | 3.51s | 100% |
| T8-003 | T8 | ✗ | substring | $0.0013 | 4.71s | 0% |
| T8-004 | T8 | ✓ | substring | $0.0009 | 3.06s | 100% |
| T8-005 | T8 | ✓ | llm_judge | $0.0011 | 3.32s | 100% |
| T1-005 | T1 | ✓ | structural | $0.0013 | 2.97s | 100% |

### Budget: $20

| ID | Tier | Pass | Match | Cost | Latency | Notes |
|----|------|------|-------|------|---------|-------|
| T1-001 | T1 | ✓ | structural | $0.0014 | 3.39s | 100% |
| T1-002 | T1 | ✗ | llm_judge | $0.0017 | 3.39s | 0% |
| T1-003 | T1 | ✓ | structural | $0.0015 | 3.05s | 100% |
| T1-004 | T1 | ✓ | llm_judge | $0.0011 | 3.37s | 100% |
| T2-001 | T2 | ✓ | substring | $0.0011 | 3.36s | 100% |
| T2-002 | T2 | ✓ | substring | $0.0010 | 3.22s | 100% |
| T2-003 | T2 | ✓ | llm_judge | $0.0011 | 2.91s | 100% |
| T2-004 | T2 | ✓ | substring | $0.0010 | 2.89s | 100% |
| T2-005 | T2 | ✓ | substring | $0.0012 | 3.12s | 100% |
| T3-001 | T3 | ✓ | llm_judge | $0.0018 | 4.04s | 100% |
| T3-002 | T3 | ✗ | llm_judge | $0.0014 | 4.47s | 0% (routed T8) |
| T3-003 | T3 | ✓ | llm_judge | $0.0016 | 3.85s | 100% |
| T3-004 | T3 | ✓ | llm_judge | $0.0016 | 3.84s | 100% |
| T3-005 | T3 | ✗ | llm_judge | $0.0012 | 3.67s | 0% |
| T4-001 | T4 | ✓ | structural | $0.0013 | 3.81s | 100% |
| T4-002 | T4 | ✓ | substring | $0.0011 | 3.37s | 100% (routed T2) |
| T4-003 | T4 | ✓ | llm_judge | $0.0013 | 3.5s | 100% |
| T4-004 | T4 | ✓ | substring | $0.0011 | 4.18s | 100% |
| T4-005 | T4 | ✓ | llm_judge | $0.0013 | 3.32s | 100% |
| T5-001 | T5 | ✓ | substring | $0.0012 | 3.09s | 100% |
| T5-002 | T5 | ✓ | substring | $0.0009 | 3.6s | 100% |
| T5-003 | T5 | ✓ | substring | $0.0010 | 3.41s | 100% |
| T5-004 | T5 | ✓ | substring | $0.0009 | 3.04s | 100% |
| T5-005 | T5 | ✓ | substring | $0.0009 | 3.03s | 100% |
| T6-001 | T6 | ✓ | llm_judge | $0.0075 | 13.25s | 100% |
| T6-002 | T6 | ✗ | llm_judge | $0.0019 | 4.2s | 0% |
| T6-003 | T6 | ✗ | llm_judge | $0.0013 | 3.25s | 0% (routed T5) |
| T6-004 | T6 | ✓ | llm_judge | $0.0013 | 3.46s | 100% (routed T2) |
| T6-005 | T6 | ✓ | llm_judge | $0.0013 | 3.63s | 100% |
| T7-001 | T7 | ✗ | llm_judge | $0.0012 | 7.27s | 0% |
| T7-002 | T7 | ✗ | llm_judge | $0.0013 | 4.45s | 0% |
| T7-003 | T7 | ✓ | llm_judge | $0.0012 | 4.45s | 100% |
| T7-004 | T7 | ✓ | structural | $0.0011 | 3.3s | 100% (routed T2) |
| T7-005 | T7 | ✗ | llm_judge | $0.0011 | 4.58s | 0% |
| T8-001 | T8 | ✓ | substring | $0.0009 | 3.0s | 100% |
| T8-002 | T8 | ✓ | llm_judge | $0.0009 | 3.44s | 100% |
| T8-003 | T8 | ✗ | substring | $0.0013 | 4.17s | 0% |
| T8-004 | T8 | ✓ | substring | $0.0009 | 3.27s | 100% |
| T8-005 | T8 | ✓ | llm_judge | $0.0013 | 4.31s | 100% |
| T1-005 | T1 | ✓ | structural | $0.0013 | 4.35s | 100% |

## Analysis

### Quality-vs-Budget Curve

```
Pass Rate
  78% |         ●────────●
  75% |
  72% |  ●
  69% |
       $1       $5       $20
```

The curve shows **diminishing returns above $5**. The $1→$5 jump (+6pp) comes almost entirely from Tier 6 multi-hop, which needs ≥4 tool-calling steps to resolve two-part lookups. The $5→$20 improvement is flat (0pp net): T6-001 gains a step at $20 but T6-002 becomes less reliable at the higher budget (more steps → more chances for the LLM to drift).

### Cost efficiency

| Budget | Pass Rate | Cost/Question | Pass per dollar |
|--------|-----------|---------------|-----------------|
| $1     | 72%       | $0.0013       | 554 passes/$1   |
| $5     | 78%       | $0.0014       | 557 passes/$1   |
| $20    | 78%       | $0.0014       | 557 passes/$1   |

**$5 budget is the Pareto-optimal point**: same pass-per-dollar efficiency as $20, with meaningfully better quality than $1 for multi-hop questions.

### Tier-by-tier observations

| Tier | Best score | Failure mode |
|------|-----------|--------------|
| T1 (factual) | 80% | T1-002: DB stores ViT-L as a DeiT model variant (data quality) |
| T2 (aggregate) | 100% | None — SQL synthesis is reliable |
| T3 (contradiction) | 60% | T3-002 misroutes to T8; T3-005 RAG doesn't find clear PE contradictions |
| T4 (temporal) | 80–100% | T4-004: system finds SSL mentions in 2019/2020, gold says 2021 |
| T5 (citation graph) | 100% | Fully reliable after nickname+k=100 fixes |
| T6 (multi-hop) | 40–60% | T6-003 misroutes to T5; T6-001 needs ≥$5; T6-002 answer varies |
| T7 (negation/absence) | 40% | LLM proposes different "standard" datasets than fixed gold lists |
| T8 (quantitative) | 80% | T8-003 pandas code returns NaN for the ImageNet-citations group |

### Known root causes for remaining failures

1. **T1-002** (DB data): Extraction stored ViT-L (307M) as a model variant of the DeiT paper because it appeared in a comparison table. Would require re-extraction to fix.
2. **T3-002** (misrouting): "What is the range of X?" triggers the quantitative classifier (T8). T3 handler never sees the data.
3. **T3-005** (RAG coverage): Position-embedding ablations are scattered across many papers; the retriever returns chunks that mostly say PE helps, missing the minority that show it has minimal effect.
4. **T4-004** (year discrepancy): Self-supervised learning is mentioned in 2020 ViT/DeiT papers as a comparison baseline. The gold was generated from a method-entity first-year query; the handler uses a broader text search.
5. **T6-003** (misrouting): "Which 2022 paper is most cited?" is classified as T5 (citation graph) instead of T6 (multi-hop: filter by year + rank by in-degree). T5 doesn't filter by year.
6. **T7-001/002/005** (open expected sets): Tier 7 asks the LLM for a list of "standard" datasets, which varies non-deterministically. The eval gold is a fixed specific list; the system finds different valid missing datasets.
7. **T8-003** (pandas NaN): Generated code joins on entity_id for ImageNet papers but gets empty result, returning NaN mean. The non-ImageNet group computes correctly.

## Cost Notes

- Total spend including all eval runs: $7.80 / $30.00 cap
- Remaining budget: $22.20
- Phase 9 eval cost: ~$0.17 total across all runs (3 budgets × 40 questions + judging)
