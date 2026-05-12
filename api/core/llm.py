"""
LLM client wrappers with structured outputs and cost tracking.

Supports both Anthropic (prompt caching) and OpenAI (auto-caching) clients.
The extraction pipeline uses OpenAI; query-time reasoning may use either.

Usage (OpenAI):
    from api.core.llm import get_openai_client, oai_cost_for_usage, MODEL_GPT_MINI

    client = get_openai_client()
    response = await client.beta.chat.completions.parse(
        model=MODEL_GPT_MINI,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        response_format=MyPydanticModel,
        temperature=0,
    )
    result = response.choices[0].message.parsed
    cost = oai_cost_for_usage(MODEL_GPT_MINI, response.usage)

Usage (Anthropic):
    from api.core.llm import get_client, cached_system, cost_for_usage, MODEL_HAIKU

    client = get_client()
    # cached_system() adds cache_control for Haiku/Sonnet (≥4096 tokens to cache)
"""
import os

import anthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ── Anthropic model IDs ───────────────────────────────────────────────────────
MODEL_HAIKU = "claude-haiku-4-5"
MODEL_SONNET = "claude-sonnet-4-6"

# ── OpenAI model IDs ──────────────────────────────────────────────────────────
# gpt-5.4-mini — extraction default. gpt-5-mini — cheap fallback / disambiguation.
MODEL_GPT_MINI = "gpt-5.4-mini"
MODEL_GPT_5_MINI = "gpt-5-mini"

# ── Pricing tables (USD per token) ───────────────────────────────────────────

# Anthropic: cache write = 1.25× input (5-min TTL), read = 0.1× input
_ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    MODEL_HAIKU: {
        "input": 1.00 / 1_000_000,
        "cache_write_5m": 1.25 / 1_000_000,
        "cache_read": 0.10 / 1_000_000,
        "output": 5.00 / 1_000_000,
    },
    MODEL_SONNET: {
        "input": 3.00 / 1_000_000,
        "cache_write_5m": 3.75 / 1_000_000,
        "cache_read": 0.30 / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
}

# OpenAI: cached_tokens billed at cache_read rate (automatic prefix caching).
# gpt-5.4-mini pricing — update if OpenAI's pricing page differs.
_OAI_PRICING: dict[str, dict[str, float]] = {
    MODEL_GPT_MINI: {
        "input": 0.40 / 1_000_000,
        "cache_read": 0.10 / 1_000_000,   # auto-cached prefix tokens
        "output": 1.60 / 1_000_000,
    },
    MODEL_GPT_5_MINI: {
        "input": 0.25 / 1_000_000,
        "cache_read": 0.025 / 1_000_000,
        "output": 2.00 / 1_000_000,
    },
}


# ── Anthropic helpers ─────────────────────────────────────────────────────────

def get_client(max_retries: int = 5, timeout: float = 120.0) -> anthropic.Anthropic:
    """Anthropic client with sensible defaults."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing from environment")
    return anthropic.Anthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)


def cached_system(prefix_text: str) -> list[dict]:
    """System-prompt list with a cache_control marker (Anthropic explicit caching).

    Haiku 4.5 requires ≥4096 tokens to actually cache. Shorter prefixes silently skip.
    """
    return [
        {
            "type": "text",
            "text": prefix_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def cost_for_usage(model: str, usage) -> float:
    """USD cost from an Anthropic response.usage object."""
    if model not in _ANTHROPIC_PRICING:
        raise ValueError(f"No Anthropic pricing table for model {model!r}")
    p = _ANTHROPIC_PRICING[model]
    return (
        (usage.input_tokens or 0) * p["input"]
        + (usage.cache_creation_input_tokens or 0) * p["cache_write_5m"]
        + (usage.cache_read_input_tokens or 0) * p["cache_read"]
        + (usage.output_tokens or 0) * p["output"]
    )


def usage_summary(usage) -> dict:
    """Flatten Anthropic usage to a plain dict."""
    return {
        "input_tokens": usage.input_tokens or 0,
        "cache_write": usage.cache_creation_input_tokens or 0,
        "cache_read": usage.cache_read_input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
    }


# ── OpenAI helpers ────────────────────────────────────────────────────────────

def get_openai_client(timeout: float = 120.0) -> AsyncOpenAI:
    """Async OpenAI client (structured outputs use beta.chat.completions.parse)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing from environment")
    return AsyncOpenAI(api_key=api_key, timeout=timeout)


def oai_cost_for_usage(model: str, usage) -> float:
    """USD cost from an OpenAI completion.usage object.

    OpenAI auto-caches the prefix; cached tokens appear in
    usage.prompt_tokens_details.cached_tokens and are billed at cache_read rate.
    """
    if model not in _OAI_PRICING:
        # Unknown model — fall back to raw input pricing only
        return 0.0
    p = _OAI_PRICING[model]
    total_input = getattr(usage, "prompt_tokens", 0) or 0
    output = getattr(usage, "completion_tokens", 0) or 0
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        cached = getattr(details, "cached_tokens", 0) or 0
    uncached_input = total_input - cached
    return uncached_input * p["input"] + cached * p["cache_read"] + output * p["output"]


def oai_usage_summary(usage) -> dict:
    """Flatten OpenAI usage to a plain dict."""
    total_input = getattr(usage, "prompt_tokens", 0) or 0
    output = getattr(usage, "completion_tokens", 0) or 0
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        cached = getattr(details, "cached_tokens", 0) or 0
    return {
        "input_tokens": total_input,
        "cached_tokens": cached,
        "output_tokens": output,
    }
