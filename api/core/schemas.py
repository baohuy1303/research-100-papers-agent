"""
Pydantic schemas for verbatim per-paper extraction (Phase 2).

These describe the "raw" extraction output — surface forms preserved as-is.
Normalization (Phase 3) maps surface forms to canonical entities + numeric values.
"""
from pydantic import BaseModel, Field


class ModelVariant(BaseModel):
    name: str = Field(description="Model name as written, e.g. 'ViT-L/16', 'Swin-B'")
    param_count_surface: str | None = Field(
        default=None,
        description="Parameter count exactly as written, e.g. '307M parameters', '1.2B', '86M'",
    )
    page: int | None = Field(default=None, description="PDF page number where mentioned")


class DatasetMention(BaseModel):
    surface: str = Field(description="Dataset name as written, e.g. 'ImageNet-1K', 'JFT-300M'")
    purpose: str | None = Field(
        default=None,
        description="One of: pretrain | finetune | eval | other",
    )
    page: int | None = None


class BenchmarkResult(BaseModel):
    model: str = Field(description="Model name (matches a model_variant.name)")
    dataset_surface: str = Field(description="Dataset name as written")
    metric_surface: str = Field(description="Metric as written, e.g. 'top-1 acc', 'mIoU', 'AP'")
    value_surface: str = Field(description="Value as written, e.g. '85.30', '85.3%', '85.30 ± 0.2'")
    is_sota_claim: bool = Field(default=False, description="True if paper claims this is SOTA")
    page: int | None = None
    table_caption: str | None = Field(
        default=None, description="Caption of the table this came from, if any"
    )


class TrainingDetails(BaseModel):
    compute_surface: str | None = Field(
        default=None, description="Training compute as written, e.g. '2.5k TPU-days', '8 V100 GPUs for 7 days'"
    )
    batch_size: int | None = None
    epochs: int | None = None


class KeyClaim(BaseModel):
    claim: str = Field(description="The claim, paraphrased in 1-2 sentences")
    evidence_section: str | None = Field(
        default=None, description="Section header where supporting evidence appears"
    )
    page: int | None = None


class ExtractedPaper(BaseModel):
    paper_id: str = Field(description="The paper_id passed in (echo back)")
    architecture_summary: str = Field(
        description="1-2 sentence description of the paper's architecture/method"
    )
    model_variants: list[ModelVariant] = Field(default_factory=list)
    datasets_mentioned: list[DatasetMention] = Field(default_factory=list)
    benchmark_results: list[BenchmarkResult] = Field(default_factory=list)
    training_details: TrainingDetails = Field(default_factory=TrainingDetails)
    methods_used: list[str] = Field(
        default_factory=list,
        description="Techniques/methods, e.g. 'self-attention', 'patch embedding', 'RandAugment', 'distillation'",
    )
    novel_contributions: list[str] = Field(
        default_factory=list, description="Claims the paper makes about what's new"
    )
    key_claims: list[KeyClaim] = Field(
        default_factory=list,
        description="Headline empirical or methodological claims, with evidence pointers",
    )
