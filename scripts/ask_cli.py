"""
Interactive terminal UI for testing the Research Comprehension System.

Usage:
    python scripts/ask_cli.py
    python scripts/ask_cli.py --budget '$5'

Commands inside the REPL:
    /q, /quit, /exit       — leave
    /budget $1|$5|$20      — switch budget level for subsequent questions
    /last                  — print the FULL evidence dump from the last answer
    /paper <id_prefix>     — show details for a paper from a citation
    /help                  — list commands

Everything else is treated as a question.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.core.budget import total_spent  # noqa: E402
from api.core.classifier import TierClassifier  # noqa: E402
from api.core.handlers import get_handler  # noqa: E402
from api.core.handlers.base import ADVERSARIAL_REPLY, is_adversarial  # noqa: E402
from api.core.retrieval import Retriever  # noqa: E402
from api.core.store import CorpusStore  # noqa: E402

console = Console()

VALID_BUDGETS = ("$1", "$5", "$20")
DEFAULT_BUDGET = "$5"


def banner():
    console.print(Panel.fit(
        "[bold cyan]Research Comprehension System -- CLI[/bold cyan]\n"
        "[dim]100 Vision Transformer papers | 8 question tiers[/dim]\n\n"
        "Type a question, or [yellow]/help[/yellow] for commands.",
        border_style="cyan",
    ))


def show_help():
    table = Table(title="Commands", show_header=True, header_style="bold")
    table.add_column("Command")
    table.add_column("Description")
    table.add_row("/q, /quit, /exit",  "Leave the REPL")
    table.add_row("/budget $1|$5|$20", "Switch budget level for next questions")
    table.add_row("/last",             "Show FULL evidence dump for last answer")
    table.add_row("/paper <id_prefix>", "Show details for a paper")
    table.add_row("/help",             "This help text")
    table.add_row("(anything else)",   "Treated as a question to /ask")
    console.print(table)


def render_classification(meta: dict, fallback: bool):
    conf = meta["confidence"]
    color = "green" if conf > 0.8 else "yellow" if conf > 0.5 else "red"
    line = (
        f"[bold]Tier {meta['tier']}[/bold]   "
        f"confidence: [{color}]{conf:.2f}[/{color}]   "
        f"[dim]{meta['reasoning']}[/dim]"
    )
    if fallback:
        line += "  [yellow](low-conf fallback to T1)[/yellow]"
    console.print(line)


def render_answer(answer: str, cost: float, elapsed: float):
    body = Markdown(answer)
    console.print(Panel(body, title="Answer",
                        title_align="left", border_style="green"))
    console.print(
        f"[dim]cost: [bold]${cost:.4f}[/bold]   "
        f"latency: [bold]{elapsed:.1f}s[/bold][/dim]"
    )


def render_citations(citations: list[dict]):
    if not citations:
        console.print("[dim]No citations.[/dim]")
        return
    table = Table(title=f"Citations ({len(citations)})", header_style="bold cyan",
                  show_lines=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Paper Title", style="bold")
    table.add_column("paper_id", style="dim")
    table.add_column("Section")
    for i, c in enumerate(citations, 1):
        title = (c.get("paper_title") or "")[:65]
        pid = (c.get("paper_id") or "")[:12] + "…"
        section = (c.get("section") or "—")[:30]
        table.add_row(str(i), title, pid, section)
    console.print(table)


def render_evidence_summary(evidence: list[dict]):
    """Compact evidence display tailored to the typical shape of each tier's payload."""
    if not evidence:
        return
    e = evidence[0] if isinstance(evidence, list) else evidence

    # Tier 2 / 4 / 8: SQL or pandas — show the query
    if "sql" in e:
        console.print(
            Panel(Syntax(e["sql"], "sql", theme="monokai", word_wrap=True),
                  title="SQL Executed", title_align="left", border_style="blue"),
        )
        if "row_count" in e:
            console.print(f"[dim]rows returned: {e['row_count']}"
                          + ("  [yellow](truncated)[/yellow]" if e.get("truncated") else "")
                          + "[/dim]")
    if "code" in e:
        console.print(
            Panel(Syntax(e["code"], "python", theme="monokai", word_wrap=True),
                  title="Python Executed", title_align="left", border_style="blue"),
        )
        result = e.get("result")
        if result is not None:
            shown = json.dumps(result, indent=2, default=str)[:600]
            console.print(f"[bold]Result:[/bold] [white]{shown}[/white]")

    # Tier 1: structured + retrieved chunks
    if "retrieved_chunks" in e:
        chunks = e["retrieved_chunks"]
        if chunks:
            t = Table(title=f"Retrieved Chunks ({len(chunks)})",
                      header_style="bold magenta", show_lines=False)
            t.add_column("#", justify="right", style="dim", width=3)
            t.add_column("Section")
            t.add_column("Snippet")
            for i, c in enumerate(chunks, 1):
                t.add_row(str(i),
                          (c.get("section") or "—")[:35],
                          (c.get("snippet") or "")[:90].replace("\n", " "))
            console.print(t)

    # Tier 3 numeric: spread + sota
    if "sota_claims" in e:
        claims = e.get("sota_claims", [])
        if claims:
            t = Table(title="SOTA Claims", header_style="bold yellow")
            t.add_column("Paper")
            t.add_column("Model")
            t.add_column("Value", justify="right")
            for c in claims:
                t.add_row((c.get("paper_title") or "")[:50],
                          (c.get("model") or "")[:25],
                          str(c.get("value")))
            console.print(t)

    # Tier 5: graph results
    if "results" in e and isinstance(e["results"], list) and e["results"]:
        first = e["results"][0]
        if "in_corpus_citations" in first or "pagerank" in first:
            t = Table(title="Graph Results", header_style="bold cyan")
            t.add_column("#", justify="right", style="dim")
            t.add_column("Paper")
            t.add_column("Year", justify="right")
            score_col = "Citations" if "in_corpus_citations" in first else "PageRank"
            t.add_column(score_col, justify="right")
            for i, r in enumerate(e["results"][:10], 1):
                score = r.get("in_corpus_citations") or r.get("pagerank")
                t.add_row(str(i), (r.get("title") or "")[:55],
                          str(r.get("year") or ""), f"{score}")
            console.print(t)

    # Tier 7: missing items
    if "missing" in e:
        missing = e.get("missing", [])
        present = e.get("present_for_reference", [])
        if missing:
            console.print(f"[bold red]Missing ({len(missing)}):[/bold red] "
                          + ", ".join(missing[:15])
                          + (" …" if len(missing) > 15 else ""))
        if present:
            console.print(f"[dim]Present ({len(present)}):[/dim] "
                          + ", ".join(present[:8])
                          + (" …" if len(present) > 8 else ""))

    # Tier 6 only carries step count
    if "steps_taken" in e:
        console.print(f"[dim]Tool-calling steps: {e['steps_taken']} / {e.get('max_steps', '?')}[/dim]")


# ── Main loop ────────────────────────────────────────────────────────────────

async def repl(initial_budget: str):
    store = CorpusStore()
    retriever = Retriever()
    classifier = TierClassifier()
    budget = initial_budget
    last_result: dict | None = None

    banner()
    console.print(f"[dim]Budget: [bold]{budget}[/bold]   "
                  f"Total spent so far: [bold]${total_spent():.4f}[/bold] / $30[/dim]\n")

    while True:
        try:
            q = Prompt.ask(f"[bold cyan]({budget})[/bold cyan] >", console=console).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            return

        if not q:
            continue

        if q in ("/q", "/quit", "/exit"):
            console.print("[dim]bye.[/dim]")
            return
        if q == "/help":
            show_help()
            continue
        if q.startswith("/budget"):
            parts = q.split()
            if len(parts) == 2 and parts[1] in VALID_BUDGETS:
                budget = parts[1]
                console.print(f"[green]budget set to {budget}[/green]")
            else:
                console.print(f"[red]usage:[/red] /budget {'|'.join(VALID_BUDGETS)}")
            continue
        if q == "/last":
            if last_result is None:
                console.print("[dim]no previous answer[/dim]")
            else:
                console.print(Panel(
                    Syntax(json.dumps(last_result, indent=2, default=str)[:8000],
                           "json", theme="monokai", word_wrap=True),
                    title="Last full result",
                    border_style="dim",
                ))
            continue
        if q.startswith("/paper "):
            prefix = q.split(maxsplit=1)[1].strip()
            rows = store.execute_sql(
                "SELECT * FROM papers WHERE paper_id LIKE ? LIMIT 1",
                (f"{prefix}%",),
            )
            if rows:
                p = rows[0]
                console.print(Panel(
                    f"[bold]{p['title']}[/bold]\n"
                    f"id: [dim]{p['paper_id']}[/dim]\n"
                    f"year: {p['year']}   venue: {p['venue']}   citations: {p['citation_count']}\n\n"
                    f"{p['architecture_summary']}",
                    title="Paper", border_style="magenta",
                ))
            else:
                console.print(f"[red]no paper with id starting {prefix!r}[/red]")
            continue

        # Otherwise: treat as a question.
        if is_adversarial(q):
            console.print(Panel(ADVERSARIAL_REPLY, title="Adversarial pre-check",
                                border_style="yellow"))
            console.print()
            continue

        prev_budget = os.environ.get("BUDGET_LEVEL")
        os.environ["BUDGET_LEVEL"] = budget
        t0 = time.time()
        try:
            with console.status("[dim]classifying…[/dim]", spinner="dots"):
                tier_meta = await classifier.classify(q)
            tier = tier_meta["tier"]
            conf = tier_meta["confidence"]
            normalized_q = tier_meta.get("normalized_question") or q
            classifier_cost = tier_meta.get("cost_usd", 0.0)

            fallback = conf < 0.5
            if fallback:
                tier = 1

            render_classification(tier_meta, fallback)

            with console.status(f"[dim]running Tier {tier} handler…[/dim]", spinner="dots"):
                handle = get_handler(tier)
                result = await handle(normalized_q, store, retriever, classifier_meta=tier_meta)

            elapsed = time.time() - t0
            total_cost = classifier_cost + result.cost_usd

            render_answer(result.answer, total_cost, elapsed)
            render_citations([c.model_dump() for c in result.citations])
            render_evidence_summary(result.evidence)

            last_result = {
                "question": q, "tier": tier,
                "answer": result.answer,
                "citations": [c.model_dump() for c in result.citations],
                "evidence": result.evidence,
                "cost_usd": total_cost, "elapsed_seconds": elapsed,
            }

        except Exception as e:
            console.print(f"[bold red]ERROR:[/bold red] {type(e).__name__}: {e}")
        finally:
            if prev_budget is None:
                os.environ.pop("BUDGET_LEVEL", None)
            else:
                os.environ["BUDGET_LEVEL"] = prev_budget
            console.print()  # blank line between turns


def main():
    p = argparse.ArgumentParser(description="Interactive REPL for the QA system.")
    p.add_argument("--budget", default=DEFAULT_BUDGET, choices=VALID_BUDGETS)
    args = p.parse_args()
    try:
        asyncio.run(repl(args.budget))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
