"""
Frozen prompt for Phase 2 paper extraction. Deterministic — content is built
once at import time. The system prompt is cached via Anthropic prompt caching;
this module must not produce per-call-varying output.
"""
import json

from api.core.schemas import ExtractedPaper


_INSTRUCTIONS = """You are a research-paper structured-extraction assistant.

You will receive the markdown text of a single computer-vision research paper.
Extract a faithful, verbatim record of the paper's claims, models, datasets, and
benchmark results into the JSON schema described below.

Hard rules:
1. Preserve surface forms exactly as written. Do NOT normalize numbers, units,
   dataset names, or metric names. "ImageNet-1K" and "ImageNet" and "ILSVRC2012"
   are all valid surface forms — return whichever the paper actually uses.
2. Do not invent fields. If the paper does not mention training compute, set
   training_details.compute_surface to null. Do not guess.
3. For benchmark_results: only include rows where you can identify all four of
   (model, dataset, metric, value). Skip ambiguous rows. Include the table
   caption when available — it disambiguates many benchmarks.
4. is_sota_claim should be true only when the paper EXPLICITLY claims SOTA on
   that specific result. Don't infer from context.
5. methods_used should list techniques the paper uses (not just defines). For
   example: "self-attention", "patch embedding", "data augmentation", "knowledge
   distillation", "masked image modeling", "contrastive pretraining".
6. novel_contributions are the paper's stated novelty claims. Pull these from
   the abstract, introduction, or "contributions" bullets — usually 1-5 items.
7. key_claims should be 3-8 headline empirical or methodological claims the
   reader would extract. Each must include a section pointer.
8. paper_id: echo back the exact value enclosed in <paper_id>...</paper_id>
   tags in the user message.

The user message will contain the paper's markdown after the paper_id tags.
"""


_SCHEMA_JSON = json.dumps(ExtractedPaper.model_json_schema(), sort_keys=True, indent=2)


_FEWSHOT_1 = """Few-shot example 1 — simple architecture paper.

User message (truncated for brevity):
<paper_id>example_001</paper_id>

# Tiny-ViT: Efficient Vision Transformers for Edge Devices
## Abstract
We introduce Tiny-ViT, a family of vision transformers optimized for mobile
inference. Our models reach 78.5% top-1 accuracy on ImageNet-1K with only 5.2M
parameters, outperforming MobileNetV3 (75.2%) under matched FLOPs. We train on
ImageNet-1K from scratch using RandAugment and CutMix.
## 1. Method
Tiny-ViT replaces the MLP head with depthwise convolutions and uses a shifted
window attention pattern. We provide three variants: Tiny-ViT-S (5.2M params),
Tiny-ViT-B (12M), Tiny-ViT-L (28M).
## 4. Experiments
Table 1: ImageNet-1K top-1 accuracy. Tiny-ViT-S: 78.5%, Tiny-ViT-B: 81.2%,
Tiny-ViT-L: 83.4%. MobileNetV3-L (baseline): 75.2%.

Expected JSON output:
{
  "paper_id": "example_001",
  "architecture_summary": "Tiny-ViT replaces the MLP head in vision transformers with depthwise convolutions and uses shifted-window attention, optimized for mobile inference.",
  "model_variants": [
    {"name": "Tiny-ViT-S", "param_count_surface": "5.2M parameters", "page": null},
    {"name": "Tiny-ViT-B", "param_count_surface": "12M", "page": null},
    {"name": "Tiny-ViT-L", "param_count_surface": "28M", "page": null}
  ],
  "datasets_mentioned": [
    {"surface": "ImageNet-1K", "purpose": "pretrain", "page": null},
    {"surface": "ImageNet-1K", "purpose": "eval", "page": null}
  ],
  "benchmark_results": [
    {"model": "Tiny-ViT-S", "dataset_surface": "ImageNet-1K", "metric_surface": "top-1 accuracy", "value_surface": "78.5%", "is_sota_claim": false, "page": null, "table_caption": "ImageNet-1K top-1 accuracy"},
    {"model": "Tiny-ViT-B", "dataset_surface": "ImageNet-1K", "metric_surface": "top-1 accuracy", "value_surface": "81.2%", "is_sota_claim": false, "page": null, "table_caption": "ImageNet-1K top-1 accuracy"},
    {"model": "Tiny-ViT-L", "dataset_surface": "ImageNet-1K", "metric_surface": "top-1 accuracy", "value_surface": "83.4%", "is_sota_claim": false, "page": null, "table_caption": "ImageNet-1K top-1 accuracy"},
    {"model": "MobileNetV3-L", "dataset_surface": "ImageNet-1K", "metric_surface": "top-1 accuracy", "value_surface": "75.2%", "is_sota_claim": false, "page": null, "table_caption": "ImageNet-1K top-1 accuracy"}
  ],
  "training_details": {"compute_surface": null, "batch_size": null, "epochs": null},
  "methods_used": ["shifted window attention", "depthwise convolutions", "RandAugment", "CutMix"],
  "novel_contributions": [
    "Tiny-ViT family of vision transformers optimized for mobile inference",
    "Replaces MLP head with depthwise convolutions",
    "Outperforms MobileNetV3 at matched FLOPs"
  ],
  "key_claims": [
    {"claim": "Tiny-ViT-S reaches 78.5% top-1 on ImageNet-1K with only 5.2M parameters", "evidence_section": "Abstract", "page": null},
    {"claim": "Tiny-ViT outperforms MobileNetV3 under matched FLOPs", "evidence_section": "Abstract", "page": null},
    {"claim": "Training uses ImageNet-1K from scratch with RandAugment and CutMix", "evidence_section": "Abstract", "page": null}
  ]
}
"""


_FEWSHOT_2 = """Few-shot example 2 — complex paper with SOTA claims, pretrain/finetune split, and compute.

User message (truncated for brevity):
<paper_id>example_002</paper_id>

# MegaViT: Pretraining Scaling Laws for Vision Transformers
## Abstract
We pretrain MegaViT on the proprietary WebImage-2B dataset (2 billion image-text
pairs), then finetune on ImageNet-21k and evaluate on ImageNet-1K and ADE20K.
MegaViT-L (304M parameters) achieves 89.2% top-1 on ImageNet-1K — state of the
art at submission. On ADE20K semantic segmentation we report 58.4 mIoU,
surpassing the previous best (Swin-L, 57.3 mIoU).
## 3. Method
Standard ViT architecture with patch size 14, depth 24, width 1024. Two
variants: MegaViT-B (87M params) and MegaViT-L (304M params).
## 4. Training
Pretrained on 256 TPU v4 cores for 32 days using AdamW with batch size 4096.
Finetuned with batch size 512 for 100 epochs on ImageNet-21k.
## 5. Results
Table 2: ImageNet-1K top-1 (SOTA in bold). MegaViT-L: 89.2 (bold), MegaViT-B:
86.4, Swin-L: 87.3, ConvNeXt-L: 87.8.
Table 3: ADE20K mIoU. MegaViT-L: 58.4 (SOTA), Swin-L: 57.3, ConvNeXt-L: 57.7.

Expected JSON output:
{
  "paper_id": "example_002",
  "architecture_summary": "MegaViT uses a standard ViT architecture (patch 14, depth 24, width 1024) pretrained on a 2B image-text web dataset and finetuned on ImageNet-21k.",
  "model_variants": [
    {"name": "MegaViT-B", "param_count_surface": "87M params", "page": null},
    {"name": "MegaViT-L", "param_count_surface": "304M parameters", "page": null}
  ],
  "datasets_mentioned": [
    {"surface": "WebImage-2B", "purpose": "pretrain", "page": null},
    {"surface": "ImageNet-21k", "purpose": "finetune", "page": null},
    {"surface": "ImageNet-1K", "purpose": "eval", "page": null},
    {"surface": "ADE20K", "purpose": "eval", "page": null}
  ],
  "benchmark_results": [
    {"model": "MegaViT-L", "dataset_surface": "ImageNet-1K", "metric_surface": "top-1", "value_surface": "89.2", "is_sota_claim": true, "page": null, "table_caption": "ImageNet-1K top-1 (SOTA in bold)"},
    {"model": "MegaViT-B", "dataset_surface": "ImageNet-1K", "metric_surface": "top-1", "value_surface": "86.4", "is_sota_claim": false, "page": null, "table_caption": "ImageNet-1K top-1 (SOTA in bold)"},
    {"model": "Swin-L", "dataset_surface": "ImageNet-1K", "metric_surface": "top-1", "value_surface": "87.3", "is_sota_claim": false, "page": null, "table_caption": "ImageNet-1K top-1 (SOTA in bold)"},
    {"model": "ConvNeXt-L", "dataset_surface": "ImageNet-1K", "metric_surface": "top-1", "value_surface": "87.8", "is_sota_claim": false, "page": null, "table_caption": "ImageNet-1K top-1 (SOTA in bold)"},
    {"model": "MegaViT-L", "dataset_surface": "ADE20K", "metric_surface": "mIoU", "value_surface": "58.4", "is_sota_claim": true, "page": null, "table_caption": "ADE20K mIoU"},
    {"model": "Swin-L", "dataset_surface": "ADE20K", "metric_surface": "mIoU", "value_surface": "57.3", "is_sota_claim": false, "page": null, "table_caption": "ADE20K mIoU"},
    {"model": "ConvNeXt-L", "dataset_surface": "ADE20K", "metric_surface": "mIoU", "value_surface": "57.7", "is_sota_claim": false, "page": null, "table_caption": "ADE20K mIoU"}
  ],
  "training_details": {"compute_surface": "256 TPU v4 cores for 32 days", "batch_size": 4096, "epochs": 100},
  "methods_used": ["patch embedding", "self-attention", "AdamW", "large-scale pretraining"],
  "novel_contributions": [
    "Pretraining on 2B image-text pairs (WebImage-2B)",
    "Scaling laws analysis for vision transformer pretraining",
    "SOTA on ImageNet-1K and ADE20K at submission"
  ],
  "key_claims": [
    {"claim": "MegaViT-L achieves 89.2% top-1 on ImageNet-1K (SOTA at submission)", "evidence_section": "Abstract", "page": null},
    {"claim": "MegaViT-L reaches 58.4 mIoU on ADE20K, surpassing Swin-L (57.3)", "evidence_section": "Abstract", "page": null},
    {"claim": "Pretrained on 256 TPU v4 cores for 32 days", "evidence_section": "4. Training", "page": null}
  ]
}
"""


# Final frozen system prompt. Built once at import time. Token-stable across runs.
EXTRACTION_SYSTEM_PROMPT = "\n\n".join([
    _INSTRUCTIONS,
    "JSON schema (sorted keys for cache stability):",
    _SCHEMA_JSON,
    _FEWSHOT_1,
    _FEWSHOT_2,
    "End of examples. The next user message will contain the actual paper.",
])
