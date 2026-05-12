"""
Typed wrappers around the corpus.db SQLite store and the citation_graph
NetworkX pickle.

A single CorpusStore() instance lazily opens the DB and graph and exposes
high-level methods that the per-tier handlers (and OpenAI tool wrappers in
api/core/tools.py) call. Designed so the same methods can be invoked either:
  - Directly from Python (deterministic handlers, Tiers 1/2/4/5)
  - As tool-call results from an LLM (agent-style handlers, Tiers 3/6/7/8)

All methods return plain dicts/lists — no ORM, easy to JSON-serialize.
"""
from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Any

import networkx as nx

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "corpus.db"
GRAPH_PATH = ROOT / "data" / "citation_graph.gpickle"


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


class CorpusStore:
    """Lazy-loaded handle to SQLite + NetworkX graph.

    Construct once per process; methods are thread-safe for SQLite reads
    via check_same_thread=False (we never write at query time).
    """

    def __init__(self, db_path: Path | str = DB_PATH, graph_path: Path | str = GRAPH_PATH):
        self.db_path = Path(db_path)
        self.graph_path = Path(graph_path)
        self._conn: sqlite3.Connection | None = None
        self._graph: nx.DiGraph | None = None

    # ── lazy resources ────────────────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.db_path.exists():
                raise FileNotFoundError(f"SQLite store missing: {self.db_path} — run scripts/build_indexes.py")
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @property
    def graph(self) -> nx.DiGraph:
        if self._graph is None:
            if not self.graph_path.exists():
                raise FileNotFoundError(f"Citation graph missing: {self.graph_path}")
            with open(self.graph_path, "rb") as f:
                self._graph = pickle.load(f)
        return self._graph

    # ── papers ────────────────────────────────────────────────────────────

    def get_paper(self, paper_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None

    def papers_in_year(self, year: int) -> list[dict]:
        return [_row_to_dict(r) for r in self.conn.execute(
            "SELECT * FROM papers WHERE year = ? ORDER BY citation_count DESC",
            (year,),
        )]

    def papers_in_year_range(self, year_min: int, year_max: int) -> list[dict]:
        return [_row_to_dict(r) for r in self.conn.execute(
            "SELECT * FROM papers WHERE year BETWEEN ? AND ? "
            "ORDER BY year ASC, citation_count DESC",
            (year_min, year_max),
        )]

    def all_papers(self) -> list[dict]:
        return [_row_to_dict(r) for r in self.conn.execute(
            "SELECT * FROM papers ORDER BY citation_count DESC"
        )]

    # ── entities ──────────────────────────────────────────────────────────

    def entity_by_alias(self, surface: str, type: str | None = None) -> dict | None:
        """Resolve a surface form to its canonical entity via the aliases table."""
        if type:
            row = self.conn.execute(
                """SELECT e.* FROM entities e
                   JOIN aliases a ON a.entity_id = e.entity_id
                   WHERE a.surface_form = ? AND e.type = ?
                   LIMIT 1""",
                (surface, type),
            ).fetchone()
        else:
            row = self.conn.execute(
                """SELECT e.* FROM entities e
                   JOIN aliases a ON a.entity_id = e.entity_id
                   WHERE a.surface_form = ?
                   LIMIT 1""",
                (surface,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def entity_by_canonical(self, canonical: str, type: str | None = None) -> dict | None:
        if type:
            row = self.conn.execute(
                "SELECT * FROM entities WHERE canonical = ? AND type = ?",
                (canonical, type),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM entities WHERE canonical = ?",
                (canonical,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def entity_search(self, query: str, type: str | None = None, limit: int = 10) -> list[dict]:
        """Substring search on canonical names (case-insensitive)."""
        sql = "SELECT * FROM entities WHERE canonical LIKE ?"
        params: list[Any] = [f"%{query}%"]
        if type:
            sql += " AND type = ?"
            params.append(type)
        sql += " ORDER BY mention_count DESC LIMIT ?"
        params.append(limit)
        return [_row_to_dict(r) for r in self.conn.execute(sql, params)]

    def all_entities_by_type(self, type: str) -> list[dict]:
        return [_row_to_dict(r) for r in self.conn.execute(
            "SELECT * FROM entities WHERE type = ? ORDER BY mention_count DESC",
            (type,),
        )]

    # ── papers_using / mentions ───────────────────────────────────────────

    def papers_using(
        self,
        canonical: str,
        type: str,
        purpose: str | None = None,
    ) -> list[dict]:
        """Papers that mention an entity. Optionally filter by purpose
        (pretrain | finetune | eval) — only meaningful for datasets."""
        sql = """SELECT DISTINCT p.* FROM papers p
                 JOIN mentions m ON m.paper_id = p.paper_id
                 JOIN entities e ON e.entity_id = m.entity_id
                 WHERE e.canonical = ? AND e.type = ?"""
        params: list[Any] = [canonical, type]
        if purpose:
            sql += " AND m.purpose = ?"
            params.append(purpose)
        sql += " ORDER BY p.citation_count DESC"
        return [_row_to_dict(r) for r in self.conn.execute(sql, params)]

    # ── benchmark results ─────────────────────────────────────────────────

    def results_for(
        self,
        dataset: str | None = None,
        metric: str | None = None,
        paper_id: str | None = None,
        sota_only: bool = False,
        limit: int | None = None,
    ) -> list[dict]:
        """Return benchmark rows joining papers + dataset/metric entities.

        Each row carries: paper_id, paper_title, year, model,
            dataset_canonical, metric_canonical,
            value_canonical, value_surface, is_sota_claim, table_caption.
        """
        sql = """SELECT
                    p.paper_id, p.title AS paper_title, p.year,
                    r.model,
                    ed.canonical AS dataset_canonical,
                    em.canonical AS metric_canonical,
                    r.value_canonical, r.value_surface,
                    r.is_sota_claim, r.table_caption
                 FROM results r
                 JOIN papers   p  ON p.paper_id   = r.paper_id
                 JOIN entities ed ON ed.entity_id = r.dataset_id
                 JOIN entities em ON em.entity_id = r.metric_id
                 WHERE 1=1"""
        params: list[Any] = []
        if dataset:
            sql += " AND ed.canonical = ?"
            params.append(dataset)
        if metric:
            sql += " AND em.canonical = ?"
            params.append(metric)
        if paper_id:
            sql += " AND p.paper_id = ?"
            params.append(paper_id)
        if sota_only:
            sql += " AND r.is_sota_claim = 1"
        sql += " ORDER BY r.value_canonical DESC NULLS LAST"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [_row_to_dict(r) for r in self.conn.execute(sql, params)]

    def best_on(self, dataset: str, metric: str, k: int = 10) -> list[dict]:
        """Top-k results ranked by value_canonical (descending)."""
        return self.results_for(dataset=dataset, metric=metric, limit=k)

    # ── model variants ────────────────────────────────────────────────────

    def model_variants_for(self, paper_id: str) -> list[dict]:
        return [_row_to_dict(r) for r in self.conn.execute(
            "SELECT * FROM model_variants WHERE paper_id = ?", (paper_id,)
        )]

    def all_model_variants(self) -> list[dict]:
        """All variants across the corpus, joined with paper title/year for context."""
        return [_row_to_dict(r) for r in self.conn.execute(
            """SELECT mv.*, p.title AS paper_title, p.year
               FROM model_variants mv JOIN papers p ON p.paper_id = mv.paper_id
               ORDER BY mv.param_count_millions DESC NULLS LAST"""
        )]

    # ── claims ────────────────────────────────────────────────────────────

    def claims_for(self, paper_id: str) -> list[dict]:
        return [_row_to_dict(r) for r in self.conn.execute(
            "SELECT * FROM claims WHERE paper_id = ?", (paper_id,)
        )]

    # ── training ──────────────────────────────────────────────────────────

    def training_for(self, paper_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM training WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None

    # ── raw SQL escape hatch ──────────────────────────────────────────────

    def execute_sql(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run a read-only SELECT. Used by Tier 2 / Tier 8 NL→SQL handlers.

        Refuses any statement starting with INSERT/UPDATE/DELETE/DROP/etc.
        """
        first = sql.strip().split(None, 1)[0].upper()
        if first not in {"SELECT", "WITH"}:
            raise ValueError(f"Only SELECT/WITH allowed, got {first!r}")
        return [_row_to_dict(r) for r in self.conn.execute(sql, params)]

    # ── citation graph ────────────────────────────────────────────────────

    def cites(self, src: str, dst: str) -> bool:
        return self.graph.has_edge(src, dst)

    def references_of(self, paper_id: str) -> list[str]:
        """Papers that paper_id cites (within corpus)."""
        if paper_id not in self.graph:
            return []
        return list(self.graph.successors(paper_id))

    def cited_by(self, paper_id: str) -> list[str]:
        """Papers that cite paper_id (within corpus)."""
        if paper_id not in self.graph:
            return []
        return list(self.graph.predecessors(paper_id))

    def descendants(self, paper_id: str) -> list[str]:
        """Transitive closure of papers building on paper_id."""
        if paper_id not in self.graph:
            return []
        return list(nx.descendants(self.graph.reverse(copy=False), paper_id))

    def ancestors(self, paper_id: str) -> list[str]:
        """Transitive closure of papers paper_id is built on."""
        if paper_id not in self.graph:
            return []
        return list(nx.ancestors(self.graph.reverse(copy=False), paper_id))

    def shortest_citation_path(self, src: str, dst: str) -> list[str] | None:
        try:
            return nx.shortest_path(self.graph, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def most_cited(self, k: int = 10) -> list[dict]:
        """Top-k papers by in-degree (in-corpus citations received)."""
        ranked = sorted(self.graph.in_degree(), key=lambda x: -x[1])[:k]
        out = []
        for pid, deg in ranked:
            paper = self.get_paper(pid)
            if paper:
                paper["in_corpus_citations"] = deg
                out.append(paper)
        return out

    def pagerank_top(self, k: int = 10) -> list[dict]:
        scores = nx.pagerank(self.graph)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        out = []
        for pid, score in ranked:
            paper = self.get_paper(pid)
            if paper:
                paper["pagerank"] = round(score, 5)
                out.append(paper)
        return out

    # ── housekeeping ──────────────────────────────────────────────────────

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
