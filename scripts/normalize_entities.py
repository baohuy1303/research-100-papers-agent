"""
Phase 3b: Normalize entity surface forms (datasets, metrics, methods).

Pipeline per type:
  1. Collect every unique surface form from data/normalized/*.json
  2. Apply deterministic rules (lowercase, strip suffixes, hand-curated aliases)
     — collapses ~60% of duplicates instantly (e.g. all "Top-1*" variants)
  3. For the remaining unresolved forms, embed with text-embedding-3-small
  4. Greedy cluster at cosine ≥ 0.92 — pure numpy, no sklearn
  5. Within each cluster the canonical = most-frequent original surface form

Outputs:
  data/entity_map.json — {datasets|metrics|methods: [{canonical, aliases:[...], type}]}

Idempotent. Run with --report to see clustering stats.

Estimated cost: ~$0.005 (embedding ~2300 short strings).
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.core.budget import record_cost  # noqa: E402
from api.core.llm import get_openai_client  # noqa: E402

ROOT = Path(__file__).parent.parent
NORM_DIR = ROOT / "data" / "normalized"
OUT_PATH = ROOT / "data" / "entity_map.json"

EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE_PER_TOKEN = 0.02 / 1_000_000  # $0.02 per 1M tokens

CLUSTER_THRESHOLD = 0.92  # cosine similarity to merge


# ── Rule-based pre-normalization ──────────────────────────────────────────────

def rule_normalize_dataset(s: str) -> str:
    """Aggressive lowercase / suffix strip for datasets."""
    s = s.strip()
    # Common case-variant collapses
    s = re.sub(r"-1[Kk]\b", "-1k", s)
    s = re.sub(r"-21[Kk]\b", "-21k", s)
    s = re.sub(r"-22[Kk]\b", "-22k", s)
    return s.lower()


def rule_normalize_metric(s: str) -> str:
    """Strip percentage / parenthetical suffixes from metric names."""
    s = s.strip()
    # Strip "(%)" "(percent)" trailing
    s = re.sub(r"\s*\([%a-z]+\)\s*$", "", s, flags=re.IGNORECASE)
    # Normalize 'top-1 acc.' / 'top1-acc' / 'top 1 accuracy' families
    s = re.sub(r"\btop[\s\-]?1[\s\-]?acc(uracy|\.?)?\b", "top-1 accuracy", s, flags=re.IGNORECASE)
    s = re.sub(r"\btop[\s\-]?5[\s\-]?acc(uracy|\.?)?\b", "top-5 accuracy", s, flags=re.IGNORECASE)
    # Strip "imagenet" qualifier — same metric, different dataset
    s = re.sub(r"^(imagenet|imnet)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(imagenet|imnet)\s+top-1.*", "top-1 accuracy", s, flags=re.IGNORECASE)
    return s.strip().lower()


def rule_normalize_method(s: str) -> str:
    return s.strip().lower()


# ── Embedding & clustering ────────────────────────────────────────────────────

async def embed_batch(client, texts: list[str]) -> tuple[np.ndarray, int]:
    """Embed a list of strings; returns (matrix, total_tokens)."""
    response = await client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in response.data], dtype=np.float32)
    # text-embedding-3-small returns L2-normalized vectors → dot = cosine
    return vecs, response.usage.total_tokens


def greedy_cluster(vecs: np.ndarray, threshold: float) -> list[list[int]]:
    """Greedy O(n²) clustering: assign each item to first cluster within threshold.

    Cluster representative = first item assigned. Good enough at our scale (n<3k).
    """
    n = len(vecs)
    cluster_reps: list[int] = []   # indices of representative vectors
    assignments: list[int] = []    # cluster id per item

    for i in range(n):
        if not cluster_reps:
            cluster_reps.append(i)
            assignments.append(0)
            continue
        rep_vecs = vecs[cluster_reps]
        sims = rep_vecs @ vecs[i]
        best = int(sims.argmax())
        if sims[best] >= threshold:
            assignments.append(best)
        else:
            assignments.append(len(cluster_reps))
            cluster_reps.append(i)

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(assignments):
        clusters[cid].append(idx)
    return list(clusters.values())


# ── Per-type pipeline ─────────────────────────────────────────────────────────

async def normalize_type(
    client,
    type_name: str,
    counter: Counter,
    rule_fn,
) -> tuple[list[dict], int]:
    """Cluster surface forms for one type. Returns (entities, tokens_used)."""

    # Step 1: rule-based bucket — group surface forms by normalized key
    by_norm: dict[str, list[str]] = defaultdict(list)
    for surface, _count in counter.most_common():
        by_norm[rule_fn(surface)].append(surface)

    print(f"  {type_name}: {len(counter)} surface forms -> "
          f"{len(by_norm)} after rule normalization")

    norm_keys = sorted(by_norm.keys())
    if not norm_keys:
        return [], 0

    # Step 2: embed normalized keys
    print(f"  embedding {len(norm_keys)} {type_name} keys...")
    vecs, tokens = await embed_batch(client, norm_keys)

    # Step 3: greedy cluster
    cluster_indices = greedy_cluster(vecs, CLUSTER_THRESHOLD)
    print(f"  -> {len(cluster_indices)} clusters")

    # Step 4: build canonical entries
    entities: list[dict] = []
    for cluster in cluster_indices:
        # All surface forms whose normalized key is in this cluster
        all_surfaces: list[str] = []
        for idx in cluster:
            all_surfaces.extend(by_norm[norm_keys[idx]])
        # Canonical = most-frequent original surface form (by paper-mention count)
        canonical = max(all_surfaces, key=lambda s: counter[s])
        entities.append({
            "canonical": canonical,
            "type": type_name,
            "aliases": sorted(set(all_surfaces)),
            "mention_count": sum(counter[s] for s in all_surfaces),
        })

    entities.sort(key=lambda e: -e["mention_count"])
    return entities, tokens


# ── Main ─────────────────────────────────────────────────────────────────────

async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true", help="print top clusters per type")
    args = p.parse_args()

    # Collect surface forms
    datasets, metrics, methods = Counter(), Counter(), Counter()
    for f in sorted(NORM_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for ds in d.get("datasets_mentioned", []):
            if ds.get("surface"):
                datasets[ds["surface"].strip()] += 1
        for br in d.get("benchmark_results", []):
            if br.get("metric_surface"):
                metrics[br["metric_surface"].strip()] += 1
        for m in d.get("methods_used", []):
            if m and m.strip():
                methods[m.strip()] += 1

    print(f"Found {len(datasets)} datasets, {len(metrics)} metrics, "
          f"{len(methods)} methods across {len(list(NORM_DIR.glob('*.json')))} papers")

    client = get_openai_client()
    total_tokens = 0

    print("\n[datasets]")
    ds_entities, t1 = await normalize_type(client, "dataset", datasets, rule_normalize_dataset)
    total_tokens += t1

    print("\n[metrics]")
    m_entities, t2 = await normalize_type(client, "metric", metrics, rule_normalize_metric)
    total_tokens += t2

    print("\n[methods]")
    me_entities, t3 = await normalize_type(client, "method", methods, rule_normalize_method)
    total_tokens += t3

    # Write output
    out = {
        "datasets": ds_entities,
        "metrics": m_entities,
        "methods": me_entities,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    cost = total_tokens * EMBED_PRICE_PER_TOKEN
    record_cost("normalize_entities", cost,
                model=EMBED_MODEL, total_tokens=total_tokens)

    print(f"\nWrote {OUT_PATH}")
    print(f"Tokens: {total_tokens:,}  Cost: ${cost:.4f}")

    if args.report:
        for tname, ents in [("datasets", ds_entities), ("metrics", m_entities), ("methods", me_entities)]:
            print(f"\n=== Top 10 {tname} clusters ===")
            for e in ents[:10]:
                aliases = ", ".join(e["aliases"][:5])
                more = f" (+{len(e['aliases'])-5} more)" if len(e["aliases"]) > 5 else ""
                print(f"  {e['mention_count']:4d}x  {e['canonical']!r}  <- {aliases}{more}")


def main():
    import asyncio
    asyncio.run(amain())


if __name__ == "__main__":
    main()
