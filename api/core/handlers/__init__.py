"""Per-tier question handlers. Tier 1-8."""
from api.core.handlers.base import Citation, HandlerResult

__all__ = ["Citation", "HandlerResult", "get_handler"]


def get_handler(tier: int):
    """Lazy-import a handler by tier number to avoid loading all on every call."""
    if tier == 1:
        from api.core.handlers.tier1_factual import handle
    elif tier == 2:
        from api.core.handlers.tier2_aggregate import handle
    elif tier == 3:
        from api.core.handlers.tier3_contradict import handle
    elif tier == 4:
        from api.core.handlers.tier4_temporal import handle
    elif tier == 5:
        from api.core.handlers.tier5_citation import handle
    elif tier == 6:
        from api.core.handlers.tier6_multihop import handle
    elif tier == 7:
        from api.core.handlers.tier7_absence import handle
    elif tier == 8:
        from api.core.handlers.tier8_compute import handle
    else:
        raise ValueError(f"Unknown tier: {tier}")
    return handle
