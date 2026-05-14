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
  The answer lives inside ONE paper. The user names a specific paper explicitly.
  Examples:
    - "What architecture does ViT use?"
    - "How many parameters does Swin-L have?"
    - "What datasets did DeiT train on?"
    - "What training hyperparameters did MAE use?"

TIER 2 — Corpus-level aggregation.
  Counting / listing across many papers. Pure SQL: COUNT, GROUP BY, DISTINCT.
  Examples:
    - "How many papers benchmark on ImageNet?"
    - "What datasets are used across the corpus?"
    - "Which papers from 2022 use self-attention?"
    - "How many papers were published in 2023?"

TIER 3 — Comparative / contradiction / range across papers.
  Comparing numbers or claims ACROSS multiple papers — disagreements, spread,
  range of reported values, conflicting SOTA claims.
  Use T3 whenever the question asks about variation, disagreement, or range of a
  metric across many papers — even if it doesn't use the word "contradiction".
  Examples:
    - "Do ViT and Swin agree on ImageNet top-1 accuracy?"
    - "Do any papers contradict each other on the role of position embeddings?"
    - "What is the range of top-1 accuracy values reported on CIFAR-100?" ← T3, not T8
    - "How much do reported mIoU values on ADE20K vary across papers?" ← T3
    - "Which paper claims the highest SOTA on ImageNet — are the claims consistent?"

TIER 4 — Temporal / evolution.
  How something changed OVER TIME (year-by-year trends).
  Examples:
    - "How did top-1 accuracy on ImageNet improve from 2020 to 2024?"
    - "When did masked image modeling become popular?"
    - "What's the trend in model size over the years?"
    - "In which year did papers first mention self-supervised learning?"

TIER 5 — Citation-graph reasoning (simple).
  Direct citation-graph lookups with NO additional filter: WHO CITES WHOM,
  most-cited, PageRank, citation paths. No year filter, no dataset filter.
  Examples:
    - "Which paper is most cited within this corpus?" ← pure citation rank
    - "What papers build on ViT?" ← who cites ViT
    - "Is there a citation chain from MAE to BEiT?"
    - "How many papers cite Swin?" ← direct count

TIER 6 — Multi-hop / compositional.
  Requires 2+ steps: first find/filter a SET, then rank/analyze that set.
  Key signal: "among [filtered set], which/what ..." or ranking within a
  constrained group (year, dataset usage, method).
  Examples:
    - "Among papers that cite ViT, which has the largest model?" ← cite-then-rank
    - "Which paper published in 2022 is the most cited?" ← year-filter THEN citation-rank → T6
    - "Among papers using ADE20K, which has the highest ImageNet accuracy?" ← T6
    - "What is the most cited paper among those published after 2021?" ← T6
    - "Which 2023 paper uses the most parameters?" ← year-filter + property-rank → T6

TIER 7 — Negation / absence.
  What's MISSING. Closed-world set difference: "expected but not observed".
  Examples:
    - "Which standard ViT benchmarks are NOT covered by this corpus?"
    - "What evaluation datasets are missing from MAE?"
    - "Which papers do NOT report compute requirements?"

TIER 8 — Quantitative computation (novel math).
  Statistical calculations that require writing code: medians, correlations,
  regressions, percentages, aggregations OVER the whole corpus structure.
  NOT for min/max/range of a benchmark (that's T3). Only for computations
  that can't be expressed as a simple GROUP BY query.
  Examples:
    - "What's the median parameter count across all model variants?" ← needs pandas median
    - "Is there a Pearson correlation between citation count and model size?" ← regression
    - "What percentage of benchmark results include a SOTA claim?" ← ratio computation
    - "Do ImageNet papers have more citations on average than non-ImageNet papers?" ← comparison of group means

DECISION RULES (apply in order):
  1. If the user names ONE specific paper → T1.
  2. If asking for a count/list across the whole corpus, no filters → T2.
  3. If asking for range / spread / variation of a benchmark metric across papers → T3.
  4. If asking about change over time / trends / first year → T4.
  5. If a citation-graph question WITH a year or dataset filter ("most cited 2022 paper") → T6.
  6. If a pure citation-graph question (no other filter) → T5.
  7. If asking what's ABSENT / NOT present → T7.
  8. If multi-step: find a set, then rank/analyze that set → T6.
  9. If requires computing a statistic not expressible as SQL GROUP BY → T8.
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
            max_completion_tokens=1024,
            extra_body={
                "prompt_cache_key": "tier-classifier-v3",
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
