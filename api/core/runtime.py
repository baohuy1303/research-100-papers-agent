"""Request-scoped runtime configuration.

The API, CLI, and eval runner may execute multiple questions concurrently.
Keep budget selection in a context variable instead of mutating process-wide
environment variables during a request.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Literal

BudgetLevel = Literal["$1", "$5", "$20"]
DEFAULT_BUDGET: BudgetLevel = "$5"
VALID_BUDGETS: tuple[BudgetLevel, ...] = ("$1", "$5", "$20")


@dataclass(frozen=True)
class RuntimeConfig:
    budget_level: BudgetLevel | None = None


_runtime: ContextVar[RuntimeConfig] = ContextVar(
    "runtime_config",
    default=RuntimeConfig(),
)


def validate_budget_level(value: str | None) -> BudgetLevel:
    """Return a valid budget level or raise ValueError."""
    if value is None:
        return DEFAULT_BUDGET
    if value not in VALID_BUDGETS:
        raise ValueError(f"Invalid budget level {value!r}; use '$1', '$5', or '$20'.")
    return value  # type: ignore[return-value]


def current_runtime() -> RuntimeConfig:
    """Runtime config active for the current request/task."""
    return _runtime.get()


@contextmanager
def use_runtime(config: RuntimeConfig) -> Iterator[None]:
    """Temporarily bind runtime config in the current context."""
    token = _runtime.set(config)
    try:
        yield
    finally:
        _runtime.reset(token)
