"""
Phase 3b: Normalize entity surface forms (datasets, metrics, methods).

Six-stage hybrid pipeline. Each stage only processes what earlier stages
couldn't resolve.

  1. Curated alias map         - hand-curated top ~30 per type
  2. Rule-based normalizer     - lowercase, strip suffixes, regex collapses
  3. Fuzzy match (rapidfuzz)   - score >= 90 against existing canonicals
  4. Embedding cluster         - text-embedding-3-small, cosine >= 0.92
  5. HF Datasets lookup        - cluster reps -> paperswithcode_id (datasets only)
  6. LLM disambiguation        - gpt-5-mini decides medium-confidence pairs (0.80-0.92)

Outputs:
  data/entity_map.json         - canonical entities + aliases per type
  data/hf_cache/<key>.json     - HF response cache (resumable / idempotent)

Run:
    python scripts/normalize_entities.py
    python scripts/normalize_entities.py --report   # print top clusters
"""
import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.core.budget import record_cost  # noqa: E402
from api.core.llm import (  # noqa: E402
    MODEL_GPT_5_MINI,
    get_openai_client,
    oai_cost_for_usage,
)

ROOT = Path(__file__).parent.parent
NORM_DIR = ROOT / "data" / "normalized"
OUT_PATH = ROOT / "data" / "entity_map.json"
HF_CACHE_DIR = ROOT / "data" / "hf_cache"

EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE_PER_TOKEN = 0.02 / 1_000_000

# Tunable thresholds
FUZZY_THRESHOLD = 95        # rapidfuzz WRatio — 95 avoids "Kinetics-400"/"Kinetics-700" false merges
CLUSTER_HIGH = 0.92         # cosine >= : auto-merge
CLUSTER_LOW = 0.80          # cosine in [LOW, HIGH): LLM decides
HF_BASE = "https://huggingface.co/api/datasets"


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1: hand-curated alias maps
# ─────────────────────────────────────────────────────────────────────────────

# Surface form (lowercased) -> canonical name. Built from inspecting the top
# 30 most-mentioned forms per type in our 100-paper corpus.

DATASET_ALIASES: dict[str, str] = {
    # ImageNet family
    "imagenet": "ImageNet",
    "imagenet-1k": "ImageNet",
    "imagenet-1K".lower(): "ImageNet",
    "imagenet1k": "ImageNet",
    "ilsvrc2012": "ImageNet",
    "ilsvrc-2012": "ImageNet",
    "imagenet val": "ImageNet",
    "imagenet validation": "ImageNet",
    "imagenet-21k": "ImageNet-21k",
    "imagenet21k": "ImageNet-21k",
    "imagenet-22k": "ImageNet-21k",   # 21k and 22k often used interchangeably
    "imagenet22k": "ImageNet-21k",
    "imagenet-v2": "ImageNet-V2",
    "imagenet v2": "ImageNet-V2",
    "imagenet-r": "ImageNet-R",
    "imagenet-a": "ImageNet-A",
    "imagenet-c": "ImageNet-C",
    # COCO family
    "coco": "COCO",
    "ms-coco": "COCO",
    "mscoco": "COCO",
    "ms coco": "COCO",
    "coco 2017": "COCO",
    "coco2017": "COCO",
    "coco captions": "COCO Captions",
    # CIFAR
    "cifar-10": "CIFAR-10",
    "cifar10": "CIFAR-10",
    "cifar-100": "CIFAR-100",
    "cifar100": "CIFAR-100",
    # Segmentation
    "ade20k": "ADE20K",
    "ade-20k": "ADE20K",
    "cityscapes": "Cityscapes",
    "pascal voc": "PASCAL VOC",
    "voc": "PASCAL VOC",
    # Large-scale pretraining
    "jft-300m": "JFT-300M",
    "jft300m": "JFT-300M",
    "jft-3b": "JFT-3B",
    "laion-400m": "LAION-400M",
    "laion400m": "LAION-400M",
    "laion-5b": "LAION-5B",
    # VQA / captioning
    "vqav2": "VQAv2",
    "vqa v2": "VQAv2",
    "vqa-v2": "VQAv2",
    "gqa": "GQA",
    "nlvr2": "NLVR2",
    "flickr30k": "Flickr30K",
    "flickr-30k": "Flickr30K",
    "nocaps": "NoCaps",
    "visual genome": "Visual Genome",
    "vg": "Visual Genome",
    "sbu captions": "SBU Captions",
    "conceptual captions": "Conceptual Captions",
    "cc3m": "Conceptual Captions",
    "cc12m": "Conceptual Captions 12M",
    # Video
    "kinetics-400": "Kinetics-400",
    "kinetics400": "Kinetics-400",
    "k400": "Kinetics-400",
    "kinetics 400": "Kinetics-400",
    "kinetics-600": "Kinetics-600",
    "kinetics600": "Kinetics-600",
    "k600": "Kinetics-600",
    "kinetics 600": "Kinetics-600",
    "kinetics-700": "Kinetics-700",
    "kinetics700": "Kinetics-700",
    "k700": "Kinetics-700",
    "kinetics 700": "Kinetics-700",
    # Misc
    "vtab": "VTAB",
    "vtab-1k": "VTAB-1k",
    "objectnet": "ObjectNet",
}

METRIC_ALIASES: dict[str, str] = {
    # top-1 accuracy family
    "top-1": "top-1 accuracy",
    "top1": "top-1 accuracy",
    "top 1": "top-1 accuracy",
    "top-1 acc": "top-1 accuracy",
    "top-1 acc.": "top-1 accuracy",
    "top1-acc": "top-1 accuracy",
    "top1-acc (%)": "top-1 accuracy",
    "top-1 (%)": "top-1 accuracy",
    "top-1 acc (%)": "top-1 accuracy",
    "top-1 acc. (%)": "top-1 accuracy",
    "top-1 accuracy": "top-1 accuracy",
    "top-1 accuracy (%)": "top-1 accuracy",
    "imagenet": "top-1 accuracy",
    "imagenet top-1": "top-1 accuracy",
    "imagenet top-1 acc": "top-1 accuracy",
    "imagenet top-1 acc.": "top-1 accuracy",
    "imagenet top-1 accuracy": "top-1 accuracy",
    "imagenet top-1 accuracy (%)": "top-1 accuracy",
    "imagenet top-1 (%)": "top-1 accuracy",
    "imagenet classification accuracy": "top-1 accuracy",
    "imagenet top1": "top-1 accuracy",
    "imagenet top-1 classification accuracy": "top-1 accuracy",
    "imnet top-1": "top-1 accuracy",
    "imnet top-1 %": "top-1 accuracy",
    "accu": "top-1 accuracy",
    "top-1 classification accuracy": "top-1 accuracy",
    "top-1 err": "top-1 error",
    "top-1 err (%)": "top-1 error",
    "top-1 error": "top-1 error",
    # top-5
    "top-5": "top-5 accuracy",
    "top5": "top-5 accuracy",
    "top-5 acc": "top-5 accuracy",
    "top-5 acc.": "top-5 accuracy",
    "top-5 (%)": "top-5 accuracy",
    "top-5 accuracy": "top-5 accuracy",
    # mIoU
    "miou": "mIoU",
    "miou (%)": "mIoU",
    "mean iou": "mIoU",
    "iou": "IoU",
    # Detection
    "ap": "AP",
    "map": "mAP",
    "ap box": "AP_box",
    "ap_box": "AP_box",
    "ap mask": "AP_mask",
    "ap_mask": "AP_mask",
    "ap^m": "AP_mask",
    # Retrieval
    "r@1": "R@1",
    "r@5": "R@5",
    "r@10": "R@10",
    # General
    "accuracy": "accuracy",
    "acc": "accuracy",
    "acc.": "accuracy",
    "f1": "F1",
    "f1-score": "F1",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "wer": "WER",
    "bleu": "BLEU",
    "linear acc": "linear probe accuracy",
    "linear acc.": "linear probe accuracy",
}

METHOD_ALIASES: dict[str, str] = {
    "self-attention": "self-attention",
    "self attention": "self-attention",
    "multi-head attention": "multi-head attention",
    "multi-head self-attention": "multi-head attention",
    "mha": "multi-head attention",
    "mhsa": "multi-head attention",
    "patch embedding": "patch embedding",
    "patch embeddings": "patch embedding",
    "position embedding": "position embedding",
    "positional embedding": "position embedding",
    "positional encoding": "position embedding",
    "layer normalization": "layer normalization",
    "layernorm": "layer normalization",
    "batch normalization": "batch normalization",
    "batchnorm": "batch normalization",
    "knowledge distillation": "knowledge distillation",
    "distillation": "knowledge distillation",
    "data augmentation": "data augmentation",
    "randaugment": "RandAugment",
    "rand augment": "RandAugment",
    "mixup": "MixUp",
    "cutmix": "CutMix",
    "cutout": "Cutout",
    "dropout": "dropout",
    "stochastic depth": "stochastic depth",
    "drop path": "stochastic depth",
    "drop-path": "stochastic depth",
    "adamw": "AdamW",
    "adam": "Adam",
    "sgd": "SGD",
    "contrastive learning": "contrastive learning",
    "contrastive pretraining": "contrastive learning",
    "masked image modeling": "masked image modeling",
    "mim": "masked image modeling",
    "masked autoencoding": "masked image modeling",
    "mae": "masked image modeling",
    "self-supervised learning": "self-supervised learning",
    "self supervised learning": "self-supervised learning",
    "ssl": "self-supervised learning",
    "fine-tuning": "fine-tuning",
    "finetuning": "fine-tuning",
    "linear probing": "linear probing",
    "transfer learning": "transfer learning",
}

ALIAS_MAPS = {
    "dataset": DATASET_ALIASES,
    "metric":  METRIC_ALIASES,
    "method":  METHOD_ALIASES,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2: rule-based normalizer
# ─────────────────────────────────────────────────────────────────────────────

def rule_normalize_dataset(s: str) -> str:
    s = s.strip()
    s = re.sub(r"-1[Kk]\b", "-1k", s)
    s = re.sub(r"-21[Kk]\b", "-21k", s)
    s = re.sub(r"-22[Kk]\b", "-22k", s)
    return s.lower()


def rule_normalize_metric(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s*\([%a-z]+\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(imagenet|imnet)\s+", "", s, flags=re.IGNORECASE)
    return s.strip().lower()


def rule_normalize_method(s: str) -> str:
    return s.strip().lower()


RULE_FNS: dict[str, Callable[[str], str]] = {
    "dataset": rule_normalize_dataset,
    "metric":  rule_normalize_metric,
    "method":  rule_normalize_method,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3: fuzzy matching
# ─────────────────────────────────────────────────────────────────────────────

def fuzzy_assign(unresolved: list[str], canonicals: list[str], threshold: int) -> dict[str, str]:
    """Assign each unresolved surface to nearest canonical above threshold."""
    assignments: dict[str, str] = {}
    if not canonicals:
        return assignments
    for surface in unresolved:
        match = process.extractOne(
            surface, canonicals, scorer=fuzz.WRatio, score_cutoff=threshold
        )
        if match:
            assignments[surface] = match[0]
    return assignments


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 4: embedding & clustering
# ─────────────────────────────────────────────────────────────────────────────

async def embed_batch(client, texts: list[str]) -> tuple[np.ndarray, int]:
    """Embed a list of strings; returns (matrix, total_tokens)."""
    response = await client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in response.data], dtype=np.float32)
    return vecs, response.usage.total_tokens


def greedy_cluster_with_uncertain(
    vecs: np.ndarray,
    high: float,
    low: float,
) -> tuple[list[list[int]], list[tuple[int, int, float]]]:
    """Greedy O(n^2) clustering. Returns (clusters, uncertain_pairs).

    uncertain_pairs[i] = (item_idx, cluster_rep_idx, cosine) for points where
    similarity falls in [low, high) — flagged for Stage 6 LLM check.
    """
    n = len(vecs)
    cluster_reps: list[int] = []
    assignments: list[int] = []
    uncertain: list[tuple[int, int, float]] = []

    for i in range(n):
        if not cluster_reps:
            cluster_reps.append(i)
            assignments.append(0)
            continue
        sims = vecs[cluster_reps] @ vecs[i]
        best = int(sims.argmax())
        best_sim = float(sims[best])
        if best_sim >= high:
            assignments.append(best)
        elif best_sim >= low:
            uncertain.append((i, cluster_reps[best], best_sim))
            assignments.append(len(cluster_reps))
            cluster_reps.append(i)
        else:
            assignments.append(len(cluster_reps))
            cluster_reps.append(i)

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(assignments):
        clusters[cid].append(idx)
    return list(clusters.values()), uncertain


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 5: HF Datasets lookup
# ─────────────────────────────────────────────────────────────────────────────

async def hf_lookup_dataset(http: httpx.AsyncClient, query: str) -> dict | None:
    """Search HF Datasets; return best result with paperswithcode_id if any.

    Cached to data/hf_cache/<sanitized_key>.json. Best = top result that
    has paperswithcode_id, else top result by downloads.
    """
    cache_key = re.sub(r"[^a-zA-Z0-9]+", "_", query.lower()).strip("_")[:80]
    if not cache_key:
        return None
    cache_path = HF_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        r = await http.get(HF_BASE, params={"search": query, "limit": 20}, timeout=10.0)
        r.raise_for_status()
        results = r.json()
    except Exception:
        cache_path.write_text("null", encoding="utf-8")
        return None

    # Prefer entry with paperswithcode_id
    pwc_hits = [r for r in results if r.get("paperswithcode_id")]
    chosen = pwc_hits[0] if pwc_hits else (results[0] if results else None)

    out = None
    if chosen:
        out = {
            "hf_id": chosen.get("id"),
            "paperswithcode_id": chosen.get("paperswithcode_id"),
            "downloads": chosen.get("downloads", 0),
        }
    cache_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 6: LLM disambiguation
# ─────────────────────────────────────────────────────────────────────────────

DISAMBIG_PROMPT = """You are normalizing entity names in research papers.
Given two surface forms, decide if they refer to the SAME entity.

Surface A: {a}
Surface B: {b}
Type: {type}

Reply with ONLY 'yes' or 'no'."""


async def llm_disambiguate(client, model: str, a: str, b: str, typ: str) -> tuple[bool, float]:
    """Ask gpt-5-mini if two surface forms are the same entity.

    gpt-5-mini is a reasoning model: it spends most tokens on internal reasoning
    before emitting visible content, so we need a generous max_completion_tokens.

    Returns (is_same, cost_usd).
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": DISAMBIG_PROMPT.format(a=a, b=b, type=typ)}],
        max_completion_tokens=512,
    )
    content = (response.choices[0].message.content or "").strip().lower()
    is_same = content.startswith("y")
    cost = oai_cost_for_usage(model, response.usage)
    return is_same, cost


# ─────────────────────────────────────────────────────────────────────────────
#  Per-type pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def normalize_type(
    type_name: str,
    counter: Counter,
    oai_client,
    http_client: httpx.AsyncClient,
) -> tuple[list[dict], dict]:
    """Run the 6-stage pipeline for one entity type. Returns (entities, stats)."""
    print(f"\n[{type_name}] starting with {len(counter)} unique surface forms")

    alias_map = ALIAS_MAPS[type_name]
    rule_fn = RULE_FNS[type_name]

    # canonical_name -> {aliases: set[str], mention_count: int, source: str,
    #                    paperswithcode_id: str|None, hf_id: str|None}
    entities: dict[str, dict] = {}

    def _add(canonical: str, surface: str, source: str, **extra):
        e = entities.setdefault(canonical, {
            "canonical": canonical, "type": type_name,
            "aliases": set(), "mention_count": 0, "source": source,
            "paperswithcode_id": None, "hf_id": None,
        })
        e["aliases"].add(surface)
        e["mention_count"] += counter[surface]
        for k, v in extra.items():
            if v is not None:
                e[k] = v

    # ── Stage 1: curated alias map ──
    stage1_resolved = 0
    unresolved: list[str] = []
    for surface in counter:
        canonical = alias_map.get(surface.lower())
        if canonical:
            _add(canonical, surface, "curated")
            stage1_resolved += 1
        else:
            unresolved.append(surface)
    print(f"  stage 1 (curated):  {stage1_resolved}/{len(counter)} resolved, "
          f"{len(unresolved)} remaining")

    # ── Stage 2: rule normalization ──
    by_norm: dict[str, list[str]] = defaultdict(list)
    for surface in unresolved:
        by_norm[rule_fn(surface)].append(surface)
    # Within each rule-bucket, all surfaces share a canonical (first by mention count)
    rule_assigned = 0
    still_unresolved: list[str] = []
    for norm_key, surfaces in by_norm.items():
        if len(surfaces) > 1:
            canonical = max(surfaces, key=lambda s: counter[s])
            for s in surfaces:
                _add(canonical, s, "rule")
                rule_assigned += 1
        else:
            still_unresolved.append(surfaces[0])
    unresolved = still_unresolved
    print(f"  stage 2 (rule):     {rule_assigned} merged into existing buckets, "
          f"{len(unresolved)} singletons remaining")

    # ── Stage 3: fuzzy match against existing canonicals ──
    canonicals_so_far = list(entities.keys())
    fuzzy_assigns = fuzzy_assign(unresolved, canonicals_so_far, FUZZY_THRESHOLD)
    for surface, canonical in fuzzy_assigns.items():
        _add(canonical, surface, "fuzzy")
    unresolved = [s for s in unresolved if s not in fuzzy_assigns]
    print(f"  stage 3 (fuzzy):    {len(fuzzy_assigns)} merged, "
          f"{len(unresolved)} remaining")

    # ── Stage 4: embedding cluster (only the long tail) ──
    uncertain_pairs: list[tuple[str, str, float]] = []
    embed_tokens = 0
    if unresolved:
        print(f"  stage 4 (embed):    embedding {len(unresolved)} surfaces...")
        # Embed in chunks of <=2048 (OpenAI per-request limit)
        all_vecs = []
        for i in range(0, len(unresolved), 2048):
            chunk = unresolved[i:i + 2048]
            vecs, tk = await embed_batch(oai_client, chunk)
            all_vecs.append(vecs)
            embed_tokens += tk
        vecs = np.concatenate(all_vecs, axis=0)
        clusters, uncertain_idx = greedy_cluster_with_uncertain(
            vecs, high=CLUSTER_HIGH, low=CLUSTER_LOW
        )
        for cluster in clusters:
            cluster_surfaces = [unresolved[i] for i in cluster]
            canonical = max(cluster_surfaces, key=lambda s: counter[s])
            for s in cluster_surfaces:
                _add(canonical, s, "clustered")
        for i, rep_i, sim in uncertain_idx:
            uncertain_pairs.append((unresolved[i], unresolved[rep_i], sim))
        print(f"                     -> {len(clusters)} clusters, "
              f"{len(uncertain_pairs)} uncertain pairs flagged")

    # ── Stage 5: HF Datasets lookup (datasets only) ──
    hf_resolved = 0
    if type_name == "dataset":
        print(f"  stage 5 (HF):       looking up {len(entities)} cluster reps...")
        for canonical in list(entities.keys()):
            ent = entities[canonical]
            hf = await hf_lookup_dataset(http_client, canonical)
            if hf and hf.get("paperswithcode_id"):
                ent["paperswithcode_id"] = hf["paperswithcode_id"]
                ent["hf_id"] = hf["hf_id"]
                if ent["source"] in ("clustered", "fuzzy", "rule"):
                    ent["source"] = "hf-pwc"
                hf_resolved += 1
        print(f"                     -> {hf_resolved} entities got PWC IDs")

    # ── Stage 6: LLM disambiguation on uncertain pairs ──
    llm_cost = 0.0
    llm_merged = 0
    if uncertain_pairs:
        print(f"  stage 6 (LLM):      checking {len(uncertain_pairs)} uncertain pairs...")
        # Cap to avoid runaway cost
        for surface_a, surface_b, sim in uncertain_pairs[:50]:
            try:
                same, cost = await llm_disambiguate(
                    oai_client, MODEL_GPT_5_MINI, surface_a, surface_b, type_name
                )
                llm_cost += cost
                if not same:
                    continue
                # Merge surface_a's entity into surface_b's
                ent_a = next((e for e in entities.values() if surface_a in e["aliases"]), None)
                ent_b = next((e for e in entities.values() if surface_b in e["aliases"]), None)
                if ent_a is not None and ent_b is not None and ent_a is not ent_b:
                    target = ent_a if ent_a["mention_count"] >= ent_b["mention_count"] else ent_b
                    other = ent_b if target is ent_a else ent_a
                    target["aliases"].update(other["aliases"])
                    target["mention_count"] += other["mention_count"]
                    target["source"] = "llm-confirmed"
                    if other["paperswithcode_id"] and not target["paperswithcode_id"]:
                        target["paperswithcode_id"] = other["paperswithcode_id"]
                        target["hf_id"] = other["hf_id"]
                    del entities[other["canonical"]]
                    llm_merged += 1
            except Exception as e:
                print(f"    LLM disambig error: {e}")
        print(f"                     -> {llm_merged} pairs merged, ${llm_cost:.4f} spent")

    # Convert sets to sorted lists for JSON
    out_list = []
    for e in entities.values():
        e["aliases"] = sorted(e["aliases"])
        out_list.append(e)
    out_list.sort(key=lambda e: -e["mention_count"])

    stats = {
        "input_surfaces": len(counter),
        "output_entities": len(out_list),
        "stage1_curated": stage1_resolved,
        "stage5_hf_resolved": hf_resolved,
        "stage6_llm_merged": llm_merged,
        "embed_tokens": embed_tokens,
        "llm_cost_usd": llm_cost,
    }
    return out_list, stats


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def safe_print(msg: str) -> None:
    print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true")
    args = p.parse_args()

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Collect surface forms from normalized JSONs.
    # Datasets are pulled from BOTH datasets_mentioned AND benchmark_results.dataset_surface
    # — papers often use slightly different surface forms in benchmark tables
    # (e.g. "ImageNet val", "COCO val2017") that aren't in the higher-level mention list.
    datasets, metrics, methods = Counter(), Counter(), Counter()
    files = sorted(NORM_DIR.glob("*.json"))
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for ds in d.get("datasets_mentioned", []):
            if ds.get("surface"):
                datasets[ds["surface"].strip()] += 1
        for br in d.get("benchmark_results", []):
            if br.get("dataset_surface"):
                datasets[br["dataset_surface"].strip()] += 1
            if br.get("metric_surface"):
                metrics[br["metric_surface"].strip()] += 1
        for m in d.get("methods_used", []):
            if m and m.strip():
                methods[m.strip()] += 1

    safe_print(f"Loaded {len(datasets)} dataset / {len(metrics)} metric / "
               f"{len(methods)} method surface forms across {len(files)} papers")

    oai_client = get_openai_client()
    total_embed_tokens = 0
    total_llm_cost = 0.0
    output: dict[str, list[dict]] = {}

    async with httpx.AsyncClient() as http_client:
        for type_name, counter in [
            ("dataset", datasets), ("metric", metrics), ("method", methods),
        ]:
            entities, stats = await normalize_type(
                type_name, counter, oai_client, http_client
            )
            output[type_name + "s"] = entities
            total_embed_tokens += stats["embed_tokens"]
            total_llm_cost += stats["llm_cost_usd"]

    # Write entity_map.json
    OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    embed_cost = total_embed_tokens * EMBED_PRICE_PER_TOKEN
    total_cost = embed_cost + total_llm_cost
    record_cost("normalize_entities", total_cost,
                embed_tokens=total_embed_tokens,
                llm_cost=total_llm_cost)

    safe_print(f"\nWrote {OUT_PATH}")
    safe_print(f"Embed tokens: {total_embed_tokens:,}  Embed cost: ${embed_cost:.4f}")
    safe_print(f"LLM cost: ${total_llm_cost:.4f}")
    safe_print(f"Total Phase 3b cost: ${total_cost:.4f}")

    if args.report:
        for type_key in ("datasets", "metrics", "methods"):
            safe_print(f"\n=== Top 10 {type_key} ===")
            for e in output[type_key][:10]:
                aliases = ", ".join(e["aliases"][:5])
                more = f" (+{len(e['aliases'])-5} more)" if len(e["aliases"]) > 5 else ""
                pwc = f" [pwc:{e['paperswithcode_id']}]" if e["paperswithcode_id"] else ""
                safe_print(f"  {e['mention_count']:4d}x  {e['canonical']!r}  "
                           f"({e['source']}){pwc}  <- {aliases}{more}")


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        safe_print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
