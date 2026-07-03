"""Interactive terminal UI for testing the Research Comprehension System."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.core.budget import total_spent  # noqa: E402
from api.core.pipeline import answer_question  # noqa: E402
from api.core.runtime import RuntimeConfig  # noqa: E402
from api.core.store import CorpusStore  # noqa: E402

console = Console()

VALID_BUDGETS = ("$1", "$5", "$20")
DEFAULT_BUDGET = "$5"


def banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]Research Comprehension System - CLI[/bold cyan]\n"
        "[dim]100 Vision Transformer papers | 8 question tiers[/dim]\n\n"
        "Type a question, or [yellow]/help[/yellow] for commands.",
        border_style="cyan",
    ))


def show_help() -> None:
    table = Table(title="Commands", show_header=True, header_style="bold")
    table.add_column("Command")
    table.add_column("Description")
    table.add_row("/q, /quit, /exit", "Leave the REPL")
    table.add_row("/budget $1|$5|$20", "Switch budget level for next questions")
    table.add_row("/last", "Show full evidence dump for last answer")
    table.add_row("/paper <id_prefix>", "Show details for a paper")
    table.add_row("/help", "This help text")
    table.add_row("(anything else)", "Ask the corpus")
    console.print(table)


def render_classification(tier: int, confidence: float, reasoning: str, fallback: bool) -> None:
    color = "green" if confidence > 0.8 else "yellow" if confidence > 0.5 else "red"
    line = (
        f"[bold]Tier {tier}[/bold]   "
        f"confidence: [{color}]{confidence:.2f}[/{color}]   "
        f"[dim]{reasoning}[/dim]"
    )
    if fallback:
        line += "  [yellow](low-confidence fallback to T1)[/yellow]"
    console.print(line)


def render_answer(answer: str, cost: float, elapsed: float) -> None:
    console.print(Panel(Markdown(answer), title="Answer", title_align="left", border_style="green"))
    console.print(f"[dim]cost: [bold]${cost:.4f}[/bold]   latency: [bold]{elapsed:.1f}s[/bold][/dim]")


def render_citations(citations: list[dict]) -> None:
    if not citations:
        console.print("[dim]No citations.[/dim]")
        return
    table = Table(title=f"Citations ({len(citations)})", header_style="bold cyan", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Paper Title", style="bold")
    table.add_column("paper_id", style="dim")
    table.add_column("Section")
    for i, citation in enumerate(citations, 1):
        title = (citation.get("paper_title") or "")[:65]
        paper_id = (citation.get("paper_id") or "")[:12] + "..."
        section = (citation.get("section") or "-")[:30]
        table.add_row(str(i), title, paper_id, section)
    console.print(table)


def render_evidence_summary(evidence: list[dict]) -> None:
    if not evidence:
        return
    item = evidence[0] if isinstance(evidence, list) else evidence

    if "sql" in item:
        console.print(Panel(
            Syntax(item["sql"], "sql", theme="monokai", word_wrap=True),
            title="SQL Executed",
            title_align="left",
            border_style="blue",
        ))
        if "row_count" in item:
            suffix = "  [yellow](truncated)[/yellow]" if item.get("truncated") else ""
            console.print(f"[dim]rows returned: {item['row_count']}{suffix}[/dim]")

    if "code" in item:
        console.print(Panel(
            Syntax(item["code"], "python", theme="monokai", word_wrap=True),
            title="Python Executed",
            title_align="left",
            border_style="blue",
        ))
        if item.get("result") is not None:
            shown = json.dumps(item["result"], indent=2, default=str)[:600]
            console.print(f"[bold]Result:[/bold] [white]{shown}[/white]")

    if "retrieved_chunks" in item and item["retrieved_chunks"]:
        table = Table(title=f"Retrieved Chunks ({len(item['retrieved_chunks'])})", header_style="bold magenta")
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Section")
        table.add_column("Snippet")
        for i, chunk in enumerate(item["retrieved_chunks"], 1):
            table.add_row(
                str(i),
                (chunk.get("section") or "-")[:35],
                (chunk.get("snippet") or "")[:90].replace("\n", " "),
            )
        console.print(table)

    if "missing" in item:
        missing = item.get("missing", [])
        present = item.get("present_for_reference", [])
        if missing:
            console.print(f"[bold red]Missing ({len(missing)}):[/bold red] " + ", ".join(missing[:15]))
        if present:
            console.print(f"[dim]Present ({len(present)}):[/dim] " + ", ".join(present[:8]))

    if "steps_taken" in item:
        console.print(f"[dim]Tool-calling steps: {item['steps_taken']} / {item.get('max_steps', '?')}[/dim]")


async def repl(initial_budget: str) -> None:
    store = CorpusStore()
    budget = initial_budget
    last_result: dict | None = None

    banner()
    console.print(f"[dim]Budget: [bold]{budget}[/bold]   Total spent: [bold]${total_spent():.4f}[/bold] / $30[/dim]\n")

    while True:
        try:
            question = Prompt.ask(f"[bold cyan]({budget})[/bold cyan] >", console=console).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            return

        if not question:
            continue
        if question in ("/q", "/quit", "/exit"):
            console.print("[dim]bye.[/dim]")
            return
        if question == "/help":
            show_help()
            continue
        if question.startswith("/budget"):
            parts = question.split()
            if len(parts) == 2 and parts[1] in VALID_BUDGETS:
                budget = parts[1]
                console.print(f"[green]budget set to {budget}[/green]")
            else:
                console.print(f"[red]usage:[/red] /budget {'|'.join(VALID_BUDGETS)}")
            continue
        if question == "/last":
            if last_result is None:
                console.print("[dim]no previous answer[/dim]")
            else:
                console.print(Panel(
                    Syntax(json.dumps(last_result, indent=2, default=str)[:8000], "json", theme="monokai", word_wrap=True),
                    title="Last full result",
                    border_style="dim",
                ))
            continue
        if question.startswith("/paper "):
            prefix = question.split(maxsplit=1)[1].strip()
            rows = store.execute_sql("SELECT * FROM papers WHERE paper_id LIKE ? LIMIT 1", (f"{prefix}%",))
            if not rows:
                console.print(f"[red]no paper with id starting {prefix!r}[/red]")
                continue
            paper = rows[0]
            console.print(Panel(
                f"[bold]{paper['title']}[/bold]\n"
                f"id: [dim]{paper['paper_id']}[/dim]\n"
                f"year: {paper['year']}   venue: {paper['venue']}   citations: {paper['citation_count']}\n\n"
                f"{paper['architecture_summary']}",
                title="Paper",
                border_style="magenta",
            ))
            continue

        try:
            with console.status("[dim]answering...[/dim]", spinner="dots"):
                result = await answer_question(question, runtime=RuntimeConfig(budget_level=budget))
            render_classification(
                result.tier,
                result.tier_confidence,
                result.tier_reasoning,
                result.fallback_used,
            )
            render_answer(result.answer, result.cost_usd, result.elapsed_seconds)
            citations = [citation.model_dump() for citation in result.citations]
            render_citations(citations)
            render_evidence_summary(result.evidence)
            last_result = {
                "question": question,
                "tier": result.tier,
                "answer": result.answer,
                "citations": citations,
                "evidence": result.evidence,
                "cost_usd": result.cost_usd,
                "elapsed_seconds": result.elapsed_seconds,
                "error": result.error,
            }
        except Exception as e:
            console.print(f"[bold red]ERROR:[/bold red] {type(e).__name__}: {e}")
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive REPL for the QA system.")
    parser.add_argument("--budget", default=DEFAULT_BUDGET, choices=VALID_BUDGETS)
    args = parser.parse_args()
    try:
        asyncio.run(repl(args.budget))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
