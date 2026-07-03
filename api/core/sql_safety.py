"""Read-only SQL validation for LLM-generated SQLite queries."""
from __future__ import annotations

import re

READONLY_START_TOKENS = {"SELECT", "WITH"}
BLOCKED_TOKENS = {
    "ALTER",
    "ATTACH",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "REPLACE",
    "UPDATE",
    "VACUUM",
}


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql.strip()


def _has_multiple_statements(sql: str) -> bool:
    """True when semicolons contain anything except an optional final terminator."""
    stripped = sql.strip()
    if ";" not in stripped:
        return False
    return bool(stripped.rstrip(";").replace(";", "").strip() != stripped.rstrip(";").strip())


def validate_readonly_sql(sql: str) -> str:
    """Validate and normalize one read-only SELECT/WITH SQLite statement."""
    cleaned = _strip_comments(sql)
    if not cleaned:
        raise ValueError("SQL query is empty.")
    if _has_multiple_statements(cleaned):
        raise ValueError("Only one SQL statement is allowed.")

    cleaned = cleaned.rstrip(";").strip()
    first = cleaned.split(None, 1)[0].upper()
    if first not in READONLY_START_TOKENS:
        raise ValueError(f"Only SELECT/WITH statements are allowed, got {first!r}.")

    tokens = {t.upper() for t in re.findall(r"\b[A-Za-z_]+\b", cleaned)}
    blocked = sorted(tokens & BLOCKED_TOKENS)
    if blocked:
        raise ValueError(f"Blocked SQL token(s): {', '.join(blocked)}.")
    return cleaned
