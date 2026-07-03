# Eval Results - Quality vs Budget

Generated: 2026-07-03 10:48

## Overall Summary

| Budget | Pass Rate | Passed/Total | Total Cost | Avg Cost/Q | Avg Latency/Q |
|--------|-----------|--------------|------------|------------|---------------|
| $1 | 98% | 39/40 | $0.070 | $0.0018 | 5.8s |
| $5 | 100% | 40/40 | $0.068 | $0.0017 | 4.5s |
| $20 | 98% | 39/40 | $0.065 | $0.0016 | 4.6s |

## Per-Tier Accuracy

| Tier | $1 pass | $1 cost | $5 pass | $5 cost | $20 pass | $20 cost |
|------|-----------|------------|-----------|------------|-----------|------------|
| T1   | 80% (4/5) | $0.0019 | 100% (5/5) | $0.0014 | 100% (5/5) | $0.0013 |
| T2   | 100% (5/5) | $0.0013 | 100% (5/5) | $0.0013 | 100% (5/5) | $0.0013 |
| T3   | 100% (5/5) | $0.0020 | 100% (5/5) | $0.0018 | 100% (5/5) | $0.0017 |
| T4   | 100% (5/5) | $0.0014 | 100% (5/5) | $0.0014 | 100% (5/5) | $0.0014 |
| T5   | 100% (5/5) | $0.0012 | 100% (5/5) | $0.0012 | 100% (5/5) | $0.0012 |
| T6   | 100% (5/5) | $0.0035 | 100% (5/5) | $0.0038 | 80% (4/5) | $0.0033 |
| T7   | 100% (5/5) | $0.0013 | 100% (5/5) | $0.0013 | 100% (5/5) | $0.0013 |
| T8   | 100% (5/5) | $0.0015 | 100% (5/5) | $0.0014 | 100% (5/5) | $0.0015 |

## Per-Question Detail

### Budget: $1

| ID | Tier | Pass | Match | Cost | Latency | Notes |
|----|------|------|-------|------|---------|-------|
| T1-001 | T1 | OK | structural | $0.0021 | 7.18s | 100% |
| T1-002 | T1 | OK | llm_judge | $0.0018 | 5.85s | 100% |
| T1-003 | T1 | FAIL | structural | $0.0017 | 4.74s | 50% |
| T1-004 | T1 | OK | llm_judge | $0.0018 | 4.33s | 100% |
| T2-001 | T2 | OK | substring | $0.0012 | 3.34s | 100% |
| T2-002 | T2 | OK | substring | $0.0012 | 7.27s | 100% |
| T2-003 | T2 | OK | llm_judge | $0.0013 | 4.63s | 100% |
| T2-004 | T2 | OK | substring | $0.0012 | 3.11s | 100% |
| T2-005 | T2 | OK | substring | $0.0014 | 4.03s | 100% |
| T3-001 | T3 | OK | llm_judge | $0.0020 | 45.96s | 100% |
| T3-002 | T3 | OK | llm_judge | $0.0015 | 4.63s | 100% |
| T3-003 | T3 | OK | llm_judge | $0.0018 | 4.45s | 100% |
| T3-004 | T3 | OK | llm_judge | $0.0018 | 3.88s | 100% |
| T3-005 | T3 | OK | llm_judge | $0.0030 | 5.14s | 100% |
| T4-001 | T4 | OK | structural | $0.0014 | 4.21s | 100% |
| T4-002 | T4 | OK | substring | $0.0015 | 4.11s | 100% (routed T2) |
| T4-003 | T4 | OK | llm_judge | $0.0014 | 3.66s | 100% |
| T4-004 | T4 | OK | substring | $0.0013 | 4.08s | 100% |
| T4-005 | T4 | OK | llm_judge | $0.0015 | 4.11s | 100% |
| T5-001 | T5 | OK | substring | $0.0013 | 3.87s | 100% |
| T5-002 | T5 | OK | substring | $0.0011 | 3.96s | 100% |
| T5-003 | T5 | OK | substring | $0.0011 | 3.58s | 100% |
| T5-004 | T5 | OK | substring | $0.0011 | 3.58s | 100% |
| T5-005 | T5 | OK | substring | $0.0013 | 5.4s | 100% |
| T6-001 | T6 | OK | llm_judge | $0.0079 | 14.71s | 100% |
| T6-002 | T6 | OK | llm_judge | $0.0041 | 8.51s | 100% |
| T6-003 | T6 | OK | llm_judge | $0.0024 | 3.32s | 100% |
| T6-004 | T6 | OK | llm_judge | $0.0014 | 3.63s | 100% (routed T2) |
| T6-005 | T6 | OK | llm_judge | $0.0014 | 3.47s | 100% |
| T7-001 | T7 | OK | llm_judge | $0.0013 | 3.81s | 100% |
| T7-002 | T7 | OK | llm_judge | $0.0013 | 4.43s | 100% |
| T7-003 | T7 | OK | llm_judge | $0.0013 | 4.2s | 100% |
| T7-004 | T7 | OK | structural | $0.0013 | 3.75s | 100% (routed T2) |
| T7-005 | T7 | OK | llm_judge | $0.0013 | 3.71s | 100% |
| T8-001 | T8 | OK | substring | $0.0011 | 3.91s | 100% |
| T8-002 | T8 | OK | llm_judge | $0.0013 | 4.27s | 100% |
| T8-003 | T8 | OK | substring | $0.0019 | 6.98s | 100% |
| T8-004 | T8 | OK | substring | $0.0016 | 5.41s | 100% |
| T8-005 | T8 | OK | llm_judge | $0.0016 | 4.65s | 100% |
| T1-005 | T1 | OK | structural | $0.0020 | 3.7s | 100% |

### Budget: $5

| ID | Tier | Pass | Match | Cost | Latency | Notes |
|----|------|------|-------|------|---------|-------|
| T1-001 | T1 | OK | structural | $0.0014 | 4.28s | 100% |
| T1-002 | T1 | OK | llm_judge | $0.0013 | 4.45s | 100% |
| T1-003 | T1 | OK | structural | $0.0013 | 4.0s | 100% |
| T1-004 | T1 | OK | llm_judge | $0.0013 | 4.49s | 100% |
| T2-001 | T2 | OK | substring | $0.0013 | 3.68s | 100% |
| T2-002 | T2 | OK | substring | $0.0012 | 4.34s | 100% |
| T2-003 | T2 | OK | llm_judge | $0.0013 | 3.31s | 100% |
| T2-004 | T2 | OK | substring | $0.0012 | 3.89s | 100% |
| T2-005 | T2 | OK | substring | $0.0014 | 3.59s | 100% |
| T3-001 | T3 | OK | llm_judge | $0.0020 | 4.45s | 100% |
| T3-002 | T3 | OK | llm_judge | $0.0014 | 4.02s | 100% |
| T3-003 | T3 | OK | llm_judge | $0.0018 | 4.23s | 100% |
| T3-004 | T3 | OK | llm_judge | $0.0018 | 4.28s | 100% |
| T3-005 | T3 | OK | llm_judge | $0.0018 | 4.66s | 100% |
| T4-001 | T4 | OK | structural | $0.0014 | 4.29s | 100% |
| T4-002 | T4 | OK | substring | $0.0015 | 3.58s | 100% (routed T2) |
| T4-003 | T4 | OK | llm_judge | $0.0015 | 4.42s | 100% |
| T4-004 | T4 | OK | substring | $0.0013 | 3.51s | 100% |
| T4-005 | T4 | OK | llm_judge | $0.0014 | 3.9s | 100% |
| T5-001 | T5 | OK | substring | $0.0013 | 3.42s | 100% |
| T5-002 | T5 | OK | substring | $0.0011 | 3.83s | 100% |
| T5-003 | T5 | OK | substring | $0.0011 | 3.34s | 100% |
| T5-004 | T5 | OK | substring | $0.0011 | 3.26s | 100% |
| T5-005 | T5 | OK | substring | $0.0013 | 3.56s | 100% |
| T6-001 | T6 | OK | llm_judge | $0.0108 | 21.7s | 100% |
| T6-002 | T6 | OK | llm_judge | $0.0037 | 8.2s | 100% |
| T6-003 | T6 | OK | llm_judge | $0.0017 | 3.15s | 100% |
| T6-004 | T6 | OK | llm_judge | $0.0014 | 4.09s | 100% (routed T2) |
| T6-005 | T6 | OK | llm_judge | $0.0014 | 3.39s | 100% |
| T7-001 | T7 | OK | llm_judge | $0.0013 | 3.33s | 100% |
| T7-002 | T7 | OK | llm_judge | $0.0013 | 3.83s | 100% |
| T7-003 | T7 | OK | llm_judge | $0.0013 | 4.1s | 100% |
| T7-004 | T7 | OK | structural | $0.0013 | 3.45s | 100% (routed T2) |
| T7-005 | T7 | OK | llm_judge | $0.0013 | 3.62s | 100% |
| T8-001 | T8 | OK | substring | $0.0012 | 3.39s | 100% |
| T8-002 | T8 | OK | llm_judge | $0.0012 | 3.66s | 100% |
| T8-003 | T8 | OK | substring | $0.0019 | 5.91s | 100% |
| T8-004 | T8 | OK | substring | $0.0011 | 3.97s | 100% |
| T8-005 | T8 | OK | llm_judge | $0.0016 | 4.99s | 100% |
| T1-005 | T1 | OK | structural | $0.0020 | 4.07s | 100% |

### Budget: $20

| ID | Tier | Pass | Match | Cost | Latency | Notes |
|----|------|------|-------|------|---------|-------|
| T1-001 | T1 | OK | structural | $0.0014 | 3.85s | 100% |
| T1-002 | T1 | OK | llm_judge | $0.0013 | 3.92s | 100% |
| T1-003 | T1 | OK | structural | $0.0013 | 4.66s | 100% |
| T1-004 | T1 | OK | llm_judge | $0.0013 | 3.7s | 100% |
| T2-001 | T2 | OK | substring | $0.0013 | 3.47s | 100% |
| T2-002 | T2 | OK | substring | $0.0012 | 3.52s | 100% |
| T2-003 | T2 | OK | llm_judge | $0.0013 | 3.9s | 100% |
| T2-004 | T2 | OK | substring | $0.0012 | 2.9s | 100% |
| T2-005 | T2 | OK | substring | $0.0014 | 3.35s | 100% |
| T3-001 | T3 | OK | llm_judge | $0.0020 | 4.09s | 100% |
| T3-002 | T3 | OK | llm_judge | $0.0015 | 4.46s | 100% |
| T3-003 | T3 | OK | llm_judge | $0.0018 | 3.91s | 100% |
| T3-004 | T3 | OK | llm_judge | $0.0014 | 3.86s | 100% |
| T3-005 | T3 | OK | llm_judge | $0.0018 | 4.98s | 100% |
| T4-001 | T4 | OK | structural | $0.0015 | 3.97s | 100% |
| T4-002 | T4 | OK | substring | $0.0015 | 3.74s | 100% (routed T2) |
| T4-003 | T4 | OK | llm_judge | $0.0014 | 3.67s | 100% |
| T4-004 | T4 | OK | substring | $0.0012 | 3.19s | 100% |
| T4-005 | T4 | OK | llm_judge | $0.0015 | 4.01s | 100% |
| T5-001 | T5 | OK | substring | $0.0013 | 3.17s | 100% |
| T5-002 | T5 | OK | substring | $0.0011 | 3.6s | 100% |
| T5-003 | T5 | OK | substring | $0.0011 | 3.36s | 100% |
| T5-004 | T5 | OK | substring | $0.0011 | 3.76s | 100% |
| T5-005 | T5 | OK | substring | $0.0013 | 3.52s | 100% |
| T6-001 | T6 | OK | llm_judge | $0.0075 | 13.38s | 100% |
| T6-002 | T6 | OK | llm_judge | $0.0036 | 8.33s | 100% |
| T6-003 | T6 | OK | llm_judge | $0.0025 | 3.64s | 100% |
| T6-004 | T6 | OK | llm_judge | $0.0014 | 3.71s | 100% (routed T2) |
| T6-005 | T6 | FAIL | llm_judge | $0.0015 | 3.15s | 0% |
| T7-001 | T7 | OK | llm_judge | $0.0013 | 3.57s | 100% |
| T7-002 | T7 | OK | llm_judge | $0.0014 | 3.7s | 100% |
| T7-003 | T7 | OK | llm_judge | $0.0013 | 18.65s | 100% |
| T7-004 | T7 | OK | structural | $0.0013 | 3.29s | 100% (routed T2) |
| T7-005 | T7 | OK | llm_judge | $0.0013 | 3.86s | 100% |
| T8-001 | T8 | OK | substring | $0.0012 | 3.62s | 100% |
| T8-002 | T8 | OK | llm_judge | $0.0012 | 3.5s | 100% |
| T8-003 | T8 | OK | substring | $0.0019 | 5.74s | 100% |
| T8-004 | T8 | OK | substring | $0.0016 | 4.61s | 100% |
| T8-005 | T8 | OK | llm_judge | $0.0016 | 4.51s | 100% |
| T1-005 | T1 | OK | structural | $0.0013 | 4.62s | 100% |

## Cost Notes

- Total spend including this eval: $11.6019 / $30.00 cap
- Remaining budget: $18.3981
