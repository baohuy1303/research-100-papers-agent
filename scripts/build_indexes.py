"""
Phase 4: Build the three indexes that power query-time tier handlers.

  1. SQLite (data/corpus.db)         — relational store for SQL aggregations
                                        (Tiers 2, 3, 4, 7, 8)
  2. Chroma (data/chroma/)           — section-chunked embeddings for semantic
                                        retrieval (Tier 1, RAG fallback)
  3. NetworkX (data/citation_graph.gpickle)
                                      — in-corpus citation graph from S2
                                        (Tier 5)

Each index is independent — pass --skip-{sqlite,chroma,graph} to skip.

Usage:
    python scripts/build_indexes.py                  # all three
    python scripts/build_indexes.py --skip-chroma    # SQLite + graph only
    python scripts/build_indexes.py --skip-sqlite --skip-graph
"""
import argparse
import asyncio
import json
import pickle
import re
import sqlite3
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.core.budget import record_cost  # noqa: E402
from api.core.llm import get_openai_client  # noqa: E402

ROOT = Path(__file__).parent.parent
NORM_DIR = ROOT / "data" / "normalized"
MD_DIR = ROOT / "data" / "markdown"
ENTITY_MAP = ROOT / "data" / "entity_map.json"
MANIFEST = ROOT / "data" / "manifest.csv"

DB_PATH = ROOT / "data" / "corpus.db"
CHROMA_DIR = ROOT / "data" / "chroma"
GRAPH_PATH = ROOT / "data" / "citation_graph.gpickle"

EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE_PER_TOKEN = 0.02 / 1_000_000

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  1. SQLite
# ─────────────────────────────────────────────────────────────────────────────

SQL_SCHEMA = """
DROP TABLE IF EXISTS papers;
DROP TABLE IF EXISTS entities;
DROP TABLE IF EXISTS aliases;
DROP TABLE IF EXISTS mentions;
DROP TABLE IF EXISTS results;
DROP TABLE IF EXISTS model_variants;
DROP TABLE IF EXISTS training;
DROP TABLE IF EXISTS claims;
DROP TABLE IF EXISTS paper_refs;

CREATE TABLE papers (
    paper_id              TEXT PRIMARY KEY,
    title                 TEXT,
    year                  INTEGER,
    venue                 TEXT,
    citation_count        INTEGER,
    architecture_summary  TEXT,
    pdf_path              TEXT
);

CREATE TABLE entities (
    entity_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical           TEXT NOT NULL,
    type                TEXT NOT NULL,        -- dataset | metric | method
    paperswithcode_id   TEXT,
    hf_id               TEXT,
    mention_count       INTEGER,
    source              TEXT,
    UNIQUE(canonical, type)
);

CREATE TABLE aliases (
    entity_id     INTEGER,
    surface_form  TEXT,
    PRIMARY KEY(entity_id, surface_form),
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
);
CREATE INDEX idx_aliases_surface ON aliases(surface_form);

CREATE TABLE mentions (
    paper_id      TEXT,
    entity_id     INTEGER,
    surface_form  TEXT,
    purpose       TEXT,                       -- pretrain | finetune | eval (datasets only)
    FOREIGN KEY(paper_id)  REFERENCES papers(paper_id),
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
);
CREATE INDEX idx_mentions_paper  ON mentions(paper_id);
CREATE INDEX idx_mentions_entity ON mentions(entity_id);

CREATE TABLE results (
    result_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id         TEXT,
    model            TEXT,
    dataset_id       INTEGER,
    metric_id        INTEGER,
    value_canonical  REAL,
    value_surface    TEXT,
    is_sota_claim    INTEGER,                 -- 0/1
    table_caption    TEXT,
    FOREIGN KEY(paper_id)   REFERENCES papers(paper_id),
    FOREIGN KEY(dataset_id) REFERENCES entities(entity_id),
    FOREIGN KEY(metric_id)  REFERENCES entities(entity_id)
);
CREATE INDEX idx_results_dataset ON results(dataset_id);
CREATE INDEX idx_results_metric  ON results(metric_id);
CREATE INDEX idx_results_paper   ON results(paper_id);

CREATE TABLE model_variants (
    paper_id              TEXT,
    name                  TEXT,
    param_count_millions  REAL,
    param_count_surface   TEXT,
    PRIMARY KEY(paper_id, name),
    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
);

CREATE TABLE training (
    paper_id          TEXT PRIMARY KEY,
    compute_surface   TEXT,
    batch_size        INTEGER,
    epochs            INTEGER,
    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
);

CREATE TABLE claims (
    claim_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id          TEXT,
    claim_text        TEXT,
    evidence_section  TEXT,
    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
);
CREATE INDEX idx_claims_paper ON claims(paper_id);

CREATE TABLE paper_refs (
    paper_id_src  TEXT,
    paper_id_dst  TEXT,
    PRIMARY KEY(paper_id_src, paper_id_dst)
);
CREATE INDEX idx_refs_dst ON paper_refs(paper_id_dst);
"""


def build_sqlite() -> dict:
    """Build SQLite from data/normalized/*.json + data/entity_map.json + manifest.csv."""
    safe_print("\n[1/3] Building SQLite at data/corpus.db ...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SQL_SCHEMA)

    # ── Insert entities + aliases, build surface→entity_id lookup ──
    em = json.loads(ENTITY_MAP.read_text(encoding="utf-8"))
    surface_to_id: dict[tuple[str, str], int] = {}     # (type, lowercased surface) -> entity_id
    for type_key, type_name in [("datasets", "dataset"), ("metrics", "metric"), ("methods", "method")]:
        for ent in em.get(type_key, []):
            cur = conn.execute(
                """INSERT INTO entities(canonical, type, paperswithcode_id, hf_id,
                                        mention_count, source)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ent["canonical"], type_name,
                 ent.get("paperswithcode_id"), ent.get("hf_id"),
                 ent.get("mention_count", 0), ent.get("source", "")),
            )
            entity_id = cur.lastrowid
            for alias in ent["aliases"]:
                conn.execute(
                    "INSERT OR IGNORE INTO aliases(entity_id, surface_form) VALUES (?, ?)",
                    (entity_id, alias),
                )
                surface_to_id[(type_name, alias.strip().lower())] = entity_id
            # Also index the canonical itself so lookups by canonical succeed
            surface_to_id.setdefault((type_name, ent["canonical"].strip().lower()), entity_id)

    # ── Insert papers from manifest ──
    df = pd.read_csv(MANIFEST, dtype=str).fillna("")

    # Build paper_id → metadata lookup; also collect normalized JSON per paper
    norm_jsons = {}
    for f in NORM_DIR.glob("*.json"):
        norm_jsons[f.stem] = json.loads(f.read_text(encoding="utf-8"))

    inserted = {"papers": 0, "model_variants": 0, "training": 0, "mentions": 0,
                "results": 0, "claims": 0}
    skipped_results = 0

    for _, row in df.iterrows():
        pid = row["id"]
        norm = norm_jsons.get(pid)
        if norm is None:
            continue  # paper exists in manifest but no extraction yet

        try:
            year = int(row["year"]) if row["year"] else None
        except ValueError:
            year = None
        try:
            citations = int(row["citation_count"]) if row["citation_count"] else None
        except ValueError:
            citations = None

        conn.execute(
            """INSERT INTO papers(paper_id, title, year, venue, citation_count,
                                  architecture_summary, pdf_path)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (pid, row["title"], year, row["venue"], citations,
             norm.get("architecture_summary", ""), row["pdf_path"]),
        )
        inserted["papers"] += 1

        # model_variants
        for mv in norm.get("model_variants", []):
            if not mv.get("name"):
                continue
            conn.execute(
                """INSERT OR IGNORE INTO model_variants(paper_id, name,
                                                        param_count_millions,
                                                        param_count_surface)
                   VALUES (?, ?, ?, ?)""",
                (pid, mv["name"], mv.get("param_count_millions"),
                 mv.get("param_count_surface")),
            )
            inserted["model_variants"] += 1

        # training
        td = norm.get("training_details", {}) or {}
        conn.execute(
            """INSERT INTO training(paper_id, compute_surface, batch_size, epochs)
               VALUES (?, ?, ?, ?)""",
            (pid, td.get("compute_surface"), td.get("batch_size"), td.get("epochs")),
        )
        inserted["training"] += 1

        # mentions (datasets + methods)
        for ds in norm.get("datasets_mentioned", []):
            sf = (ds.get("surface") or "").strip()
            if not sf:
                continue
            eid = surface_to_id.get(("dataset", sf.lower()))
            if eid is None:
                continue   # surface form didn't make it into entity_map (rare)
            conn.execute(
                """INSERT INTO mentions(paper_id, entity_id, surface_form, purpose)
                   VALUES (?, ?, ?, ?)""",
                (pid, eid, sf, ds.get("purpose")),
            )
            inserted["mentions"] += 1
        for m in norm.get("methods_used", []):
            sf = (m or "").strip()
            if not sf:
                continue
            eid = surface_to_id.get(("method", sf.lower()))
            if eid is None:
                continue
            conn.execute(
                """INSERT INTO mentions(paper_id, entity_id, surface_form, purpose)
                   VALUES (?, ?, ?, ?)""",
                (pid, eid, sf, None),
            )
            inserted["mentions"] += 1

        # results — need both dataset_id AND metric_id resolved
        for br in norm.get("benchmark_results", []):
            ds_sf = (br.get("dataset_surface") or "").strip()
            mt_sf = (br.get("metric_surface") or "").strip()
            ds_id = surface_to_id.get(("dataset", ds_sf.lower())) if ds_sf else None
            mt_id = surface_to_id.get(("metric", mt_sf.lower())) if mt_sf else None
            if ds_id is None or mt_id is None:
                skipped_results += 1
                continue
            conn.execute(
                """INSERT INTO results(paper_id, model, dataset_id, metric_id,
                                       value_canonical, value_surface,
                                       is_sota_claim, table_caption)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, br.get("model"), ds_id, mt_id,
                 br.get("value_canonical"), br.get("value_surface"),
                 1 if br.get("is_sota_claim") else 0, br.get("table_caption")),
            )
            inserted["results"] += 1

        # claims
        for c in norm.get("key_claims", []):
            txt = c.get("claim")
            if not txt:
                continue
            conn.execute(
                """INSERT INTO claims(paper_id, claim_text, evidence_section)
                   VALUES (?, ?, ?)""",
                (pid, txt, c.get("evidence_section")),
            )
            inserted["claims"] += 1

    conn.commit()
    conn.close()

    safe_print(f"  papers:         {inserted['papers']}")
    safe_print(f"  model_variants: {inserted['model_variants']}")
    safe_print(f"  training:       {inserted['training']}")
    safe_print(f"  mentions:       {inserted['mentions']}")
    safe_print(f"  results:        {inserted['results']}  (skipped: {skipped_results} unresolved)")
    safe_print(f"  claims:         {inserted['claims']}")
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
#  2. Chroma — section-chunked embeddings
# ─────────────────────────────────────────────────────────────────────────────

# Split markdown on H1/H2/H3 boundaries
SECTION_PATTERN = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
CHUNK_MAX_CHARS = 4000      # ~1000 tokens
CHUNK_MIN_CHARS = 100       # skip tiny stub sections


def chunk_markdown(md: str, paper_id: str) -> list[dict]:
    """Yield {paper_id, section_title, char_offset, text} chunks."""
    matches = list(SECTION_PATTERN.finditer(md))
    if not matches:
        return [{"paper_id": paper_id, "section_title": "(full)",
                 "char_offset": 0, "text": md.strip()}]

    chunks = []
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[body_start:body_end].strip()
        if len(body) < CHUNK_MIN_CHARS:
            continue
        # Sub-split very long sections
        if len(body) <= CHUNK_MAX_CHARS:
            chunks.append({
                "paper_id": paper_id,
                "section_title": title,
                "char_offset": body_start,
                "text": f"# {title}\n\n{body}",
            })
        else:
            for sub_i in range(0, len(body), CHUNK_MAX_CHARS):
                sub_text = body[sub_i:sub_i + CHUNK_MAX_CHARS]
                chunks.append({
                    "paper_id": paper_id,
                    "section_title": f"{title} (part {sub_i // CHUNK_MAX_CHARS + 1})",
                    "char_offset": body_start + sub_i,
                    "text": f"# {title}\n\n{sub_text}",
                })
    return chunks


async def build_chroma() -> dict:
    """Embed section chunks and store in Chroma."""
    safe_print("\n[2/3] Building Chroma at data/chroma/ ...")
    import chromadb

    if CHROMA_DIR.exists():
        # Reset by removing the old persistent dir
        import shutil
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.create_collection(name="paper_chunks")

    # ── Build all chunks ──
    all_chunks = []
    for md_path in sorted(MD_DIR.glob("*.md")):
        md = md_path.read_text(encoding="utf-8", errors="replace")
        all_chunks.extend(chunk_markdown(md, md_path.stem))

    safe_print(f"  built {len(all_chunks)} chunks across {len(list(MD_DIR.glob('*.md')))} papers")

    # ── Embed in batches ──
    oai = get_openai_client()
    BATCH = 100
    total_tokens = 0
    t0 = time.time()

    for batch_start in range(0, len(all_chunks), BATCH):
        batch = all_chunks[batch_start:batch_start + BATCH]
        texts = [c["text"] for c in batch]
        response = await oai.embeddings.create(model=EMBED_MODEL, input=texts)
        embeddings = [d.embedding for d in response.data]
        total_tokens += response.usage.total_tokens

        ids = [f"{c['paper_id']}__{c['char_offset']}" for c in batch]
        metadatas = [
            {"paper_id": c["paper_id"], "section_title": c["section_title"],
             "char_offset": c["char_offset"]}
            for c in batch
        ]
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
        safe_print(f"  embedded batch {batch_start // BATCH + 1}/"
                   f"{(len(all_chunks) + BATCH - 1) // BATCH} "
                   f"({len(batch)} chunks)")

    elapsed = time.time() - t0
    cost = total_tokens * EMBED_PRICE_PER_TOKEN
    record_cost("chroma_embed", cost, model=EMBED_MODEL,
                total_tokens=total_tokens, n_chunks=len(all_chunks))
    safe_print(f"  Chroma done in {elapsed:.0f}s | tokens: {total_tokens:,} | ${cost:.4f}")

    return {"chunks": len(all_chunks), "tokens": total_tokens, "cost_usd": cost}


# ─────────────────────────────────────────────────────────────────────────────
#  3. NetworkX — in-corpus citation graph
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_references(http: httpx.AsyncClient, paper_id: str) -> list[str]:
    """Fetch list of cited paper_ids from S2 references API."""
    url = f"{S2_API_BASE}/paper/{paper_id}/references"
    refs: list[str] = []
    offset = 0
    LIMIT = 100
    while True:
        try:
            r = await http.get(url, params={"offset": offset, "limit": LIMIT,
                                            "fields": "paperId"}, timeout=30.0)
            if r.status_code == 429:
                await asyncio.sleep(5)
                continue
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            safe_print(f"    refs fetch failed for {paper_id[:12]}: {e}")
            return refs
        for item in data.get("data", []):
            cited = item.get("citedPaper", {}).get("paperId")
            if cited:
                refs.append(cited)
        if "next" not in data or not data.get("data"):
            break
        offset = data["next"]
    return refs


async def build_graph() -> dict:
    """Build NetworkX DiGraph of in-corpus citations."""
    import networkx as nx
    safe_print("\n[3/3] Building citation graph at data/citation_graph.gpickle ...")

    df = pd.read_csv(MANIFEST, dtype=str).fillna("")
    corpus_ids = set(df["id"].dropna())
    safe_print(f"  corpus has {len(corpus_ids)} papers")

    g = nx.DiGraph()
    for _, row in df.iterrows():
        g.add_node(row["id"], title=row["title"], year=row["year"])

    # Polite to S2: ~3 RPS
    sem = asyncio.Semaphore(3)

    async def one(http, pid):
        async with sem:
            await asyncio.sleep(0.1)
            refs = await fetch_references(http, pid)
            in_corpus = [r for r in refs if r in corpus_ids and r != pid]
            for r in in_corpus:
                g.add_edge(pid, r)
            return pid, len(refs), len(in_corpus)

    edges_total = 0
    async with httpx.AsyncClient() as http:
        results = []
        tasks = [one(http, pid) for pid in df["id"] if pid]
        for i in range(0, len(tasks), 10):
            batch_results = await asyncio.gather(*tasks[i:i + 10], return_exceptions=True)
            for res in batch_results:
                if isinstance(res, Exception):
                    continue
                pid, n_refs, n_in = res
                edges_total += n_in
                results.append(res)
            safe_print(f"  fetched refs for {min(i + 10, len(tasks))}/{len(tasks)} papers, "
                       f"{edges_total} in-corpus edges so far")

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(g, f)

    safe_print(f"  graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    safe_print(f"  saved to {GRAPH_PATH}")
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()}


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-sqlite", action="store_true")
    p.add_argument("--skip-chroma", action="store_true")
    p.add_argument("--skip-graph", action="store_true")
    args = p.parse_args()

    summary = {}
    if not args.skip_sqlite:
        summary["sqlite"] = build_sqlite()
    if not args.skip_chroma:
        summary["chroma"] = await build_chroma()
    if not args.skip_graph:
        summary["graph"] = await build_graph()

    safe_print("\n" + "=" * 60)
    safe_print("Phase 4 complete.")
    for k, v in summary.items():
        safe_print(f"  {k}: {v}")


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        safe_print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
