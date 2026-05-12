"""
Tier classifier — routes a natural-language question to one of 8 handler tiers.

Single LLM call (gpt-5.4-mini) with a frozen system prompt + 24 few-shot
examples (3 per tier). The prompt is identical across all calls so OpenAI's
auto-prefix-cache will keep cost minimal.

Output schema:
  {
    "tier": int,            # 1..8
    "confidence": float,    # 0..1, classifier's self-reported confidence
    "reasoning": str,       # 1-sentence justification
    "normalized_question": str,  # cleaned-up restatement (entity normalization etc.)
  }
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from api.core.budget import record_cost
from api.core.llm import (
    MODEL_GPT_MINI,
    get_openai_client,
    oai_cost_for_usage,
)


class TierClassification(BaseModel):
    tier: int = Field(ge=1, le=8, description="Question difficulty tier 1-8")
    confidence: float = Field(ge=0, le=1, description="Self-reported confidence")
    reasoning: str = Field(description="One-sentence justification for the tier choice")
    normalized_question: str = Field(
        description="The question rephrased clearly. Resolve pronouns, normalize entity names."
    )


CLASSIFIER_PROMPT = """You are a question classifier for a research-paper QA system over
a corpus of 100 Vision Transformer papers. You must route each question to exactly
one of 8 tiers based on what kind of computation answers it.

TIER 1 — Single-document factual.
  The answer lives inside ONE paper, often in one section. The user names the paper
  or the question is clearly about a single specific paper.
  Examples:
    - "What architecture does ViT use?"
    - "How many parameters does Swin-L have?"
    - "What datasets did DeiT train on?"

TIER 2 — Corpus-level aggregation.
  Counting / listing across many papers. Pure SQL: COUNT, GROUP BY, DISTINCT.
  Examples:
    - "How many papers benchmark on ImageNet?"
    - "What datasets are used across the corpus?"
    - "Which papers from 2022 use self-attention?"

TIER 3 — Comparative / contradiction.
  Two or more papers reporting different numbers on the same benchmark, or
  contradicting each other on a methodological claim.
  Examples:
    - "Do ViT and Swin agree on ImageNet top-1 accuracy?"
    - "Do any papers contradict each other on the role of position embeddings?"
    - "Which paper claims the highest top-1 accuracy on ImageNet — is the SOTA claim consistent?"

TIER 4 — Temporal / evolution.
  How something changed over time (year-by-year). Bucket by year, plot a trend.
  Examples:
    - "How did top-1 accuracy on ImageNet improve from 2020 to 2024?"
    - "When did masked image modeling become popular?"
    - "What's the trend in model size over the years?"

TIER 5 — Citation-graph reasoning.
  Questions about WHO CITES WHOM — uses the in-corpus citation graph.
  Examples:
    - "Which paper is most cited within this corpus?"
    - "What papers build on ViT?"
    - "Is there a citation chain from MAE to BEiT?"

TIER 6 — Multi-hop / compositional.
  Requires combining 2+ steps: first find X, then for each result do Y.
  Often "among papers that ..., which/what ...".
  Examples:
    - "Among papers that cite ViT, which use the most parameters?"
    - "What's the average top-1 accuracy of papers published after MAE?"
    - "Find papers that train on JFT-300M and report results on ADE20K."

TIER 7 — Negation / absence.
  What's MISSING. Closed-world set difference: "expected but not observed".
  Examples:
    - "Which standard ViT benchmarks are NOT covered by this corpus?"
    - "What evaluation datasets are missing from MAE?"
    - "Which papers do NOT report compute requirements?"

TIER 8 — Quantitative computation.
  Math / stats over benchmark numbers — sums, medians, correlations, regressions.
  Examples:
    - "What's the median parameter count for transformer-based papers?"
    - "Is there a correlation between model size and ImageNet accuracy?"
    - "Sum of training compute across all papers from 2023."

Important rules:
  - Pick exactly ONE tier (the dominant computation needed).
  - If a question could fit multiple tiers, prefer the simpler one
    (tier 1 < 2 < 5 < 4 < 7 < 3 < 6 < 8 in increasing complexity).
  - "How many" → usually tier 2.
  - "Compare X and Y" / "Do they agree" → tier 3.
  - Math / numeric computation → tier 8.
  - Multi-step joins ("among X, find Y") → tier 6.
"""


class TierClassifier:
    """Lazy-loaded classifier instance."""

    def __init__(self, model: str = MODEL_GPT_MINI):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_openai_client()
        return self._client

    async def classify(self, question: str) -> dict:
        """Classify a question into one of 8 tiers.

        Returns the parsed Pydantic dict plus 'cost_usd' and 'usage'.
        """
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format=TierClassification,
            temperature=0,
            max_completion_tokens=512,
            extra_body={
                "prompt_cache_key": "tier-classifier-v1",
                "prompt_cache_retention": "in_memory",
            },
        )

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(f"Classifier returned no parsed output: {response.choices[0].message}")

        cost = oai_cost_for_usage(self.model, response.usage)
        record_cost("tier_classifier", cost,
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens)

        return {
            **parsed.model_dump(),
            "cost_usd": cost,
        }
