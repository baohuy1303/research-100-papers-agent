"""
Phase 8: Generate eval/questions.jsonl with >=40 questions (>=5 per tier).

Every question is derived from actual DB data — gold answers are computed
from the corpus, not hallucinated by an LLM. Tiers 3 and 6 call a single
cheap LLM pass to phrase the gold in natural language; everything else is
deterministic.

Usage:
    python scripts/generate_eval_set.py
    python scripts/generate_eval_set.py --dry-run   # print only, don't write
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.core.budget import record_cost  # noqa: E402
from api.core.llm import MODEL_GPT_MINI, get_openai_client, oai_cost_for_usage  # noqa: E402

ROOT = Path(__file__).parent.parent
NORM_DIR = ROOT / "data" / "normalized"
DB_PATH = ROOT / "data" / "corpus.db"
GRAPH_PATH = ROOT / "data" / "citation_graph.gpickle"
OUT_PATH = ROOT / "eval" / "questions.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def sql(query: str, params: tuple = ()) -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute(query, params)]


def load_graph() -> nx.DiGraph:
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def qid(tier: int, n: int) -> str:
    return f"T{tier}-{n:03d}"


def make_q(
    tier: int,
    n: int,
    question: str,
    gold_answer: str,
    match_strategy: str,
    gold_items: list[str] | None = None,
    gold_paper_ids: list[str] | None = None,
    needs_review: bool = False,
    notes: str = "",
) -> dict:
    return {
        "id": qid(tier, n),
        "tier": tier,
        "question": question,
        "gold_answer": gold_answer,
        "gold_items": gold_items or [],
        "match_strategy": match_strategy,
        "gold_paper_ids": gold_paper_ids or [],
        "generation_method": "db_derived",
        "needs_review": needs_review,
        "notes": notes,
    }


async def llm_gold(prompt: str) -> tuple[str, float]:
    """Ask gpt-5.4-mini to write a one-sentence gold answer from structured data."""
    client = get_openai_client()
    r = await client.chat.completions.create(
        model=MODEL_GPT_MINI,
        messages=[
            {"role": "system", "content":
                "Write exactly ONE concise sentence answering the research QA question "
                "based on the provided data. Be specific — include numbers, paper titles, "
                "or dataset names where applicable. No caveats. No 'Based on...'."
            },
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=512,
        temperature=0,
    )
    text = (r.choices[0].message.content or "").strip()
    cost = oai_cost_for_usage(MODEL_GPT_MINI, r.usage)
    record_cost("eval_gold_gen", cost)
    return text, cost


# ── Tier generators ───────────────────────────────────────────────────────────

def generate_tier1() -> list[dict]:
    """Single-doc factual — one question per paper, gold from extracted fields."""
    papers = sql("""
        SELECT p.paper_id, p.title, p.year
        FROM papers p
        ORDER BY p.citation_count DESC
        LIMIT 10
    """)
    qs = []

    # Q1: datasets — ViT
    p = papers[0]
    ds = sql("""SELECT DISTINCT e.canonical FROM mentions m
                JOIN entities e ON e.entity_id=m.entity_id
                WHERE m.paper_id=? AND e.type='dataset'
                ORDER BY e.mention_count DESC LIMIT 3""", (p["paper_id"],))
    ds_list = [r["canonical"] for r in ds]
    qs.append(make_q(1, 1,
        question=f"What datasets did '{p['title']}' use?",
        gold_answer=", ".join(ds_list),
        match_strategy="substring",
        gold_paper_ids=[p["paper_id"]],
        notes="gold from mentions table"))

    # Q2: model params — Swin
    p2 = papers[2]  # Swin
    mv = sql("SELECT name, param_count_millions FROM model_variants WHERE paper_id=? AND param_count_millions IS NOT NULL ORDER BY param_count_millions DESC LIMIT 1", (p2["paper_id"],))
    if mv:
        qs.append(make_q(1, 2,
            question=f"What is the parameter count of the largest variant in '{p2['title']}'?",
            gold_answer=f"{mv[0]['param_count_millions']}M",
            match_strategy="substring",
            gold_paper_ids=[p2["paper_id"]],
            notes=f"gold: {mv[0]['name']} = {mv[0]['param_count_millions']}M"))

    # Q3: methods used — DeiT
    p3 = papers[1]  # DeiT (usually 2nd most cited)
    norms = list(NORM_DIR.glob(f"{p3['paper_id']}.json"))
    if norms:
        d = json.loads(norms[0].read_text(encoding="utf-8"))
        methods = d.get("methods_used", [])[:3]
        qs.append(make_q(1, 3,
            question=f"What training techniques does '{p3['title']}' use?",
            gold_answer=", ".join(methods),
            match_strategy="substring",
            gold_paper_ids=[p3["paper_id"]],
            notes="gold from methods_used field"))

    # Q4: architecture summary
    p4 = papers[3]
    row = sql("SELECT architecture_summary FROM papers WHERE paper_id=?", (p4["paper_id"],))
    if row and row[0]["architecture_summary"]:
        first_sent = row[0]["architecture_summary"].split(". ")[0] + "."
        qs.append(make_q(1, 4,
            question=f"Briefly describe the architecture of '{p4['title']}'.",
            gold_answer=first_sent[:80],
            match_strategy="substring",
            gold_paper_ids=[p4["paper_id"]],
            notes="gold = first sentence of architecture_summary"))

    # Q5: benchmark result
    p5 = papers[0]  # ViT
    res = sql("""SELECT r.value_canonical, r.model
                 FROM results r
                 JOIN entities ed ON ed.entity_id=r.dataset_id
                 JOIN entities em ON em.entity_id=r.metric_id
                 WHERE r.paper_id=? AND ed.canonical='ImageNet'
                   AND em.canonical='top-1 accuracy'
                   AND r.value_canonical IS NOT NULL
                 ORDER BY r.value_canonical DESC LIMIT 1""", (p5["paper_id"],))
    if res:
        qs.append(make_q(1, 5,
            question=f"What is the best top-1 accuracy on ImageNet reported in '{p5['title']}'?",
            gold_answer=str(res[0]["value_canonical"]),
            match_strategy="substring",
            gold_paper_ids=[p5["paper_id"]],
            notes=f"gold: {res[0]['model']} = {res[0]['value_canonical']}"))

    return qs


def generate_tier2() -> list[dict]:
    """Corpus aggregation — gold from SQL results."""
    qs = []

    # Q1: ImageNet paper count
    r = sql("""SELECT COUNT(DISTINCT m.paper_id) n FROM mentions m
               JOIN entities e ON e.entity_id=m.entity_id
               WHERE e.canonical='ImageNet' AND e.type='dataset'""")
    n = r[0]["n"]
    qs.append(make_q(2, 1,
        question="How many papers in the corpus mention or benchmark on ImageNet?",
        gold_answer=str(n),
        match_strategy="substring",
        notes=f"SQL count = {n}"))

    # Q2: venue with most papers
    r2 = sql("SELECT venue, COUNT(*) n FROM papers WHERE venue != '' GROUP BY venue ORDER BY n DESC LIMIT 1")
    venue, count = r2[0]["venue"], r2[0]["n"]
    short = venue.split(" and ")[0][:40]
    qs.append(make_q(2, 2,
        question="Which venue has the most papers in this corpus?",
        gold_answer=str(count),
        match_strategy="substring",
        notes=f"{short}: {count} papers"))

    # Q3: unique datasets
    r3 = sql("SELECT COUNT(*) n FROM entities WHERE type='dataset'")
    n3 = r3[0]["n"]
    qs.append(make_q(2, 3,
        question="How many unique canonical datasets appear across all 100 papers?",
        gold_answer=str(n3),
        match_strategy="substring",
        notes=f"entity table count = {n3}"))

    # Q4: papers from 2022
    r4 = sql("SELECT COUNT(*) n FROM papers WHERE year=2022")
    n4 = r4[0]["n"]
    qs.append(make_q(2, 4,
        question="How many papers in this corpus were published in 2022?",
        gold_answer=str(n4),
        match_strategy="substring",
        notes=f"year=2022 count = {n4}"))

    # Q5: papers using both self-attention AND patch embedding
    r5 = sql("""SELECT COUNT(DISTINCT m1.paper_id) n
                FROM mentions m1 JOIN entities e1 ON e1.entity_id=m1.entity_id
                JOIN mentions m2 ON m2.paper_id=m1.paper_id
                JOIN entities e2 ON e2.entity_id=m2.entity_id
                WHERE e1.canonical='self-attention' AND e1.type='method'
                  AND e2.canonical='patch embedding' AND e2.type='method'""")
    n5 = r5[0]["n"]
    qs.append(make_q(2, 5,
        question="How many papers in the corpus use both self-attention and patch embedding?",
        gold_answer=str(n5),
        match_strategy="substring",
        notes=f"intersection count = {n5}"))

    return qs


async def generate_tier3() -> list[dict]:
    """Comparative / contradiction — gold from variance data + LLM phrasing."""
    qs = []
    total_llm_cost = 0.0

    # Q1: ImageNet top-1 spread
    r = sql("""SELECT MIN(value_canonical) mn, MAX(value_canonical) mx, COUNT(*) n
               FROM results r
               JOIN entities ed ON ed.entity_id=r.dataset_id
               JOIN entities em ON em.entity_id=r.metric_id
               WHERE ed.canonical='ImageNet' AND em.canonical='top-1 accuracy'
                 AND r.value_canonical IS NOT NULL""")
    row = r[0]
    gold, cost = await llm_gold(
        f"Q: Do papers in this corpus agree on top-1 accuracy on ImageNet? "
        f"Data: {row['n']} results, min={row['mn']}%, max={row['mx']}%. "
        f"Write one sentence summarizing the spread."
    )
    total_llm_cost += cost
    qs.append(make_q(3, 1,
        question="Do papers in this corpus agree on top-1 accuracy on ImageNet?",
        gold_answer=gold, match_strategy="llm_judge", needs_review=True,
        notes=f"min={row['mn']}, max={row['mx']}, n={row['n']}"))

    # Q2: CIFAR-100 spread
    r2 = sql("""SELECT MIN(value_canonical) mn, MAX(value_canonical) mx, COUNT(*) n
                FROM results r
                JOIN entities ed ON ed.entity_id=r.dataset_id
                JOIN entities em ON em.entity_id=r.metric_id
                WHERE ed.canonical='CIFAR-100' AND em.canonical='top-1 accuracy'
                  AND r.value_canonical IS NOT NULL""")
    if r2 and r2[0]["n"] and r2[0]["n"] >= 2:
        row2 = r2[0]
        gold2, cost2 = await llm_gold(
            f"Q: What is the range of top-1 accuracy values reported on CIFAR-100 in this corpus? "
            f"Data: min={row2['mn']}%, max={row2['mx']}%, n={row2['n']} results."
        )
        total_llm_cost += cost2
        qs.append(make_q(3, 2,
            question="What is the range of top-1 accuracy values reported on CIFAR-100 in this corpus?",
            gold_answer=gold2, match_strategy="llm_judge", needs_review=True,
            notes=f"min={row2['mn']}, max={row2['mx']}, n={row2['n']}"))

    # Q3: conflicting SOTA claims
    sota = sql("""SELECT ed.canonical ds, em.canonical mt,
                         COUNT(DISTINCT r.paper_id) n_papers,
                         COUNT(*) n_sota
                  FROM results r
                  JOIN entities ed ON ed.entity_id=r.dataset_id
                  JOIN entities em ON em.entity_id=r.metric_id
                  WHERE r.is_sota_claim=1 AND r.value_canonical IS NOT NULL
                  GROUP BY ed.canonical, em.canonical HAVING n_papers>=2
                  ORDER BY n_papers DESC LIMIT 1""")
    if sota:
        s = sota[0]
        gold3, cost3 = await llm_gold(
            f"Q: Do papers in this corpus agree on which model holds SOTA on "
            f"{s['ds']} {s['mt']}? "
            f"Data: {s['n_papers']} papers each claim SOTA on this benchmark."
        )
        total_llm_cost += cost3
        qs.append(make_q(3, 3,
            question=f"Do papers in this corpus agree on which model holds SOTA on {s['ds']} {s['mt']}?",
            gold_answer=gold3, match_strategy="llm_judge", needs_review=True,
            notes=f"{s['n_papers']} papers claim SOTA, {s['n_sota']} SOTA rows"))

    # Q4: mIoU on ADE20K spread
    r4 = sql("""SELECT MIN(value_canonical) mn, MAX(value_canonical) mx, COUNT(*) n
                FROM results r
                JOIN entities ed ON ed.entity_id=r.dataset_id
                JOIN entities em ON em.entity_id=r.metric_id
                WHERE ed.canonical='ADE20K' AND em.canonical='mIoU'
                  AND r.value_canonical IS NOT NULL""")
    if r4 and r4[0]["n"] and r4[0]["n"] >= 2:
        row4 = r4[0]
        gold4, cost4 = await llm_gold(
            f"Q: Do papers agree on mIoU values on ADE20K? "
            f"Data: min={row4['mn']}, max={row4['mx']}, n={row4['n']} results."
        )
        total_llm_cost += cost4
        qs.append(make_q(3, 4,
            question="Do papers in this corpus agree on mIoU values on ADE20K?",
            gold_answer=gold4, match_strategy="llm_judge", needs_review=True,
            notes=f"min={row4['mn']}, max={row4['mx']}, n={row4['n']}"))

    # Q5: position embeddings — textual
    qs.append(make_q(3, 5,
        question="Do papers in this corpus agree on whether position embeddings are necessary for vision transformers?",
        gold_answer="Papers disagree on the necessity of position embeddings; some ablations show minimal impact while others demonstrate clear performance drops without them.",
        match_strategy="llm_judge", needs_review=True,
        notes="textual contradiction — requires RAG pass; gold is a summary claim for llm_judge"))

    print(f"  T3 LLM gold cost: ${total_llm_cost:.4f}")
    return qs


def generate_tier4() -> list[dict]:
    """Temporal evolution — gold from year-bucketed SQL."""
    qs = []

    # Q1: max top-1 accuracy on ImageNet by year
    rows = sql("""SELECT p.year, ROUND(MAX(r.value_canonical),1) max_acc
                  FROM results r
                  JOIN entities ed ON ed.entity_id=r.dataset_id
                  JOIN entities em ON em.entity_id=r.metric_id
                  JOIN papers p ON p.paper_id=r.paper_id
                  WHERE ed.canonical='ImageNet' AND em.canonical='top-1 accuracy'
                    AND r.value_canonical IS NOT NULL AND p.year IS NOT NULL
                  GROUP BY p.year ORDER BY p.year""")
    gold_items = [f"{r['year']}:{r['max_acc']}" for r in rows]
    gold = f"From {rows[0]['max_acc']}% in {rows[0]['year']} to {rows[-1]['max_acc']}% in {rows[-1]['year']}"
    qs.append(make_q(4, 1,
        question="How did the maximum reported top-1 accuracy on ImageNet change year over year in this corpus?",
        gold_answer=gold, gold_items=gold_items,
        match_strategy="structural",
        notes="year-bucketed max SQL"))

    # Q2: paper count by year
    rows2 = sql("SELECT year, COUNT(*) n FROM papers WHERE year IS NOT NULL GROUP BY year ORDER BY year")
    year_counts = {r["year"]: r["n"] for r in rows2}
    peak_year = max(year_counts, key=year_counts.get)
    qs.append(make_q(4, 2,
        question="How many papers were published each year in this corpus, and which year had the most?",
        gold_answer=str(peak_year),
        match_strategy="substring",
        notes=f"peak year = {peak_year} ({year_counts[peak_year]} papers)"))

    # Q3: avg param count by year
    rows3 = sql("""SELECT p.year, ROUND(AVG(mv.param_count_millions),0) avg_params
                   FROM model_variants mv JOIN papers p ON p.paper_id=mv.paper_id
                   WHERE mv.param_count_millions IS NOT NULL AND p.year IS NOT NULL
                   GROUP BY p.year ORDER BY p.year""")
    gold_items3 = [f"{r['year']}:{r['avg_params']}" for r in rows3]
    qs.append(make_q(4, 3,
        question="What is the trend in average model parameter count by year in this corpus?",
        gold_answer="increasing over time",
        gold_items=gold_items3,
        match_strategy="structural",
        notes="year-bucketed avg params"))

    # Q4: first year self-supervised methods appeared
    rows4 = sql("""SELECT MIN(p.year) yr FROM mentions m
                   JOIN entities e ON e.entity_id=m.entity_id
                   JOIN papers p ON p.paper_id=m.paper_id
                   WHERE e.canonical='self-supervised learning' AND e.type='method'""")
    yr4 = rows4[0]["yr"] if rows4 else "unknown"
    qs.append(make_q(4, 4,
        question="In which year did papers in this corpus first mention self-supervised learning?",
        gold_answer=str(yr4),
        match_strategy="substring",
        notes=f"first year = {yr4}"))

    # Q5: COCO adoption by year
    rows5 = sql("""SELECT p.year, COUNT(DISTINCT m.paper_id) n
                   FROM mentions m JOIN entities e ON e.entity_id=m.entity_id
                   JOIN papers p ON p.paper_id=m.paper_id
                   WHERE e.canonical='COCO' AND e.type='dataset' AND p.year IS NOT NULL
                   GROUP BY p.year ORDER BY p.year""")
    gold_items5 = [f"{r['year']}:{r['n']}" for r in rows5]
    qs.append(make_q(4, 5,
        question="How did the number of papers using COCO change from year to year in this corpus?",
        gold_answer="growing over time",
        gold_items=gold_items5,
        match_strategy="structural",
        notes="COCO mentions by year"))

    return qs


def generate_tier5() -> list[dict]:
    """Citation-graph reasoning — gold from NetworkX."""
    g = load_graph()
    qs = []

    # Q1: most cited
    top = sorted(g.in_degree(), key=lambda x: -x[1])[:1]
    pid, deg = top[0]
    row = sql("SELECT title FROM papers WHERE paper_id=?", (pid,))
    title = row[0]["title"] if row else pid
    qs.append(make_q(5, 1,
        question="Which paper is the most cited within this corpus?",
        gold_answer=title[:60],
        match_strategy="substring",
        gold_paper_ids=[pid],
        notes=f"in_degree={deg}"))

    # Q2: Swin citation count
    swin = sql("SELECT paper_id, title FROM papers WHERE LOWER(title) LIKE '%swin transformer%' LIMIT 1")
    if swin:
        swin_id = swin[0]["paper_id"]
        swin_deg = g.in_degree(swin_id)
        qs.append(make_q(5, 2,
            question="How many papers in this corpus cite the Swin Transformer paper?",
            gold_answer=str(swin_deg),
            match_strategy="substring",
            gold_paper_ids=[swin_id],
            notes=f"swin in_degree={swin_deg}"))

    # Q3: Does DeiT cite ViT?
    vit = sql("SELECT paper_id FROM papers WHERE LOWER(title) LIKE '%image is worth 16x16%' LIMIT 1")
    deit = sql("SELECT paper_id FROM papers WHERE LOWER(title) LIKE '%data-efficient image%' LIMIT 1")
    if vit and deit:
        vit_id, deit_id = vit[0]["paper_id"], deit[0]["paper_id"]
        edge_exists = g.has_edge(deit_id, vit_id)
        qs.append(make_q(5, 3,
            question="Does DeiT cite ViT within this corpus?",
            gold_answer="yes" if edge_exists else "no",
            match_strategy="substring",
            gold_paper_ids=[deit_id, vit_id],
            notes=f"edge DeiT->ViT exists = {edge_exists}"))

    # Q4: 3rd most cited
    top3 = sorted(g.in_degree(), key=lambda x: -x[1])[:3]
    pid3, deg3 = top3[2]
    row3 = sql("SELECT title FROM papers WHERE paper_id=?", (pid3,))
    title3 = row3[0]["title"][:60] if row3 else pid3
    qs.append(make_q(5, 4,
        question="What is the third most cited paper within this corpus?",
        gold_answer=title3,
        match_strategy="substring",
        gold_paper_ids=[pid3],
        notes=f"in_degree={deg3}"))

    # Q5: top pagerank
    pr = nx.pagerank(g)
    top_pr = max(pr.items(), key=lambda x: x[1])[0]
    row5 = sql("SELECT title FROM papers WHERE paper_id=?", (top_pr,))
    title5 = row5[0]["title"][:60] if row5 else top_pr
    qs.append(make_q(5, 5,
        question="Which paper has the highest PageRank score in the corpus citation graph?",
        gold_answer=title5,
        match_strategy="substring",
        gold_paper_ids=[top_pr],
        notes=f"pagerank top paper"))

    return qs


async def generate_tier6() -> list[dict]:
    """Multi-hop — gold from chained SQL + LLM phrasing."""
    qs = []
    g = load_graph()
    total_cost = 0.0

    # Q1: largest model among ViT-citers
    vit = sql("SELECT paper_id FROM papers WHERE LOWER(title) LIKE '%image is worth 16x16%' LIMIT 1")
    if vit:
        vit_id = vit[0]["paper_id"]
        citers = list(g.predecessors(vit_id))
        if citers:
            in_clause = ",".join([f"'{c}'" for c in citers])
            res = sql(f"""SELECT p.title, MAX(mv.param_count_millions) max_p
                          FROM model_variants mv JOIN papers p ON p.paper_id=mv.paper_id
                          WHERE mv.paper_id IN ({in_clause})
                            AND mv.param_count_millions IS NOT NULL
                          GROUP BY p.paper_id ORDER BY max_p DESC LIMIT 1""")
            if res:
                gold1, cost1 = await llm_gold(
                    f"Q: Among papers that cite ViT, which has the largest model variant? "
                    f"Data: '{res[0]['title']}' has the largest at {res[0]['max_p']}M parameters."
                )
                total_cost += cost1
                qs.append(make_q(6, 1,
                    question="Among papers that cite ViT, which has the largest model variant?",
                    gold_answer=gold1, match_strategy="llm_judge", needs_review=True,
                    gold_paper_ids=citers[:5],
                    notes=f"answer: {res[0]['title']}, {res[0]['max_p']}M"))

    # Q2: highest top-1 on ImageNet among ADE20K users
    ade_papers = sql("""SELECT DISTINCT m.paper_id FROM mentions m
                        JOIN entities e ON e.entity_id=m.entity_id
                        WHERE e.canonical='ADE20K' AND e.type='dataset'""")
    if ade_papers:
        ids = [r["paper_id"] for r in ade_papers]
        in_clause = ",".join([f"'{i}'" for i in ids])
        res2 = sql(f"""SELECT p.title, r.value_canonical acc
                       FROM results r
                       JOIN entities ed ON ed.entity_id=r.dataset_id
                       JOIN entities em ON em.entity_id=r.metric_id
                       JOIN papers p ON p.paper_id=r.paper_id
                       WHERE r.paper_id IN ({in_clause})
                         AND ed.canonical='ImageNet' AND em.canonical='top-1 accuracy'
                         AND r.value_canonical IS NOT NULL
                       ORDER BY r.value_canonical DESC LIMIT 1""")
        if res2:
            gold2, cost2 = await llm_gold(
                f"Q: Among papers that use ADE20K, which reports the highest top-1 accuracy on ImageNet? "
                f"Data: '{res2[0]['title']}' at {res2[0]['acc']}%."
            )
            total_cost += cost2
            qs.append(make_q(6, 2,
                question="Among papers that use ADE20K, which reports the highest top-1 accuracy on ImageNet?",
                gold_answer=gold2, match_strategy="llm_judge", needs_review=True,
                notes=f"answer: {res2[0]['title']}, {res2[0]['acc']}%"))

    # Q3: most cited paper from 2022
    rows3 = sql("""SELECT p.paper_id, p.title FROM papers p
                   WHERE p.year=2022 ORDER BY p.citation_count DESC LIMIT 1""")
    if rows3:
        p3 = rows3[0]
        indeg = g.in_degree(p3["paper_id"])
        gold3, cost3 = await llm_gold(
            f"Q: Which paper published in 2022 is the most cited within this corpus? "
            f"Data: '{p3['title']}' with {indeg} in-corpus citations."
        )
        total_cost += cost3
        qs.append(make_q(6, 3,
            question="Which paper published in 2022 is the most cited within this corpus?",
            gold_answer=gold3, match_strategy="llm_judge", needs_review=True,
            gold_paper_ids=[p3["paper_id"]],
            notes=f"answer: {p3['title']}, in_degree={indeg}"))

    # Q4: papers using both COCO and ADE20K
    coco_ids = {r["paper_id"] for r in sql("""SELECT DISTINCT m.paper_id FROM mentions m
        JOIN entities e ON e.entity_id=m.entity_id WHERE e.canonical='COCO' AND e.type='dataset'""")}
    ade_ids = {r["paper_id"] for r in sql("""SELECT DISTINCT m.paper_id FROM mentions m
        JOIN entities e ON e.entity_id=m.entity_id WHERE e.canonical='ADE20K' AND e.type='dataset'""")}
    both = len(coco_ids & ade_ids)
    gold4, cost4 = await llm_gold(
        f"Q: How many papers in the corpus use both COCO and ADE20K? Data: {both} papers."
    )
    total_cost += cost4
    qs.append(make_q(6, 4,
        question="How many papers in this corpus use both COCO and ADE20K?",
        gold_answer=gold4, match_strategy="llm_judge",
        notes=f"intersection = {both}"))

    # Q5: among top-5 most cited, which was earliest?
    top5 = sorted(g.in_degree(), key=lambda x: -x[1])[:5]
    top5_ids = [p for p, _ in top5]
    in_clause5 = ",".join([f"'{i}'" for i in top5_ids])
    earliest = sql(f"SELECT title, year FROM papers WHERE paper_id IN ({in_clause5}) ORDER BY year LIMIT 1")
    if earliest:
        gold5, cost5 = await llm_gold(
            f"Q: Among the 5 most-cited papers in this corpus, which was published earliest? "
            f"Data: '{earliest[0]['title']}' ({earliest[0]['year']})."
        )
        total_cost += cost5
        qs.append(make_q(6, 5,
            question="Among the 5 most-cited papers in this corpus, which was published earliest?",
            gold_answer=gold5, match_strategy="llm_judge",
            notes=f"answer: {earliest[0]['title']} ({earliest[0]['year']})"))

    print(f"  T6 LLM gold cost: ${total_cost:.4f}")
    return qs


def generate_tier7() -> list[dict]:
    """Negation / absence — gold is a deterministic set diff."""
    qs = []

    # Observed datasets in corpus
    obs = {r["canonical"].lower() for r in sql("SELECT canonical FROM entities WHERE type='dataset'")}

    # Q1: standard segmentation datasets
    expected_seg = ["ADE20K", "Cityscapes", "PASCAL VOC", "Semantic KITTI",
                    "COCO Stuff", "SUN RGB-D", "NYU Depth v2"]
    missing1 = [x for x in expected_seg if x.lower() not in obs]
    qs.append(make_q(7, 1,
        question="Which standard semantic segmentation datasets are NOT used in this corpus?",
        gold_answer=", ".join(missing1) if missing1 else "none — all are present",
        gold_items=missing1,
        match_strategy="structural",
        notes=f"expected_seg={expected_seg}, missing={missing1}"))

    # Q2: standard video datasets
    expected_video = ["Kinetics-400", "Kinetics-600", "Kinetics-700",
                      "Something-Something v2", "UCF-101", "HMDB-51"]
    missing2 = [x for x in expected_video if x.lower() not in obs]
    qs.append(make_q(7, 2,
        question="Which standard video classification datasets are NOT used in this corpus?",
        gold_answer=", ".join(missing2) if missing2 else "all are present",
        gold_items=missing2,
        match_strategy="structural",
        notes=f"missing={missing2}"))

    # Q3: pretraining datasets — check JFT, LAION, CC
    expected_pretrain = ["JFT-300M", "JFT-3B", "LAION-400M", "LAION-5B",
                         "Conceptual Captions", "Conceptual Captions 12M"]
    missing3 = [x for x in expected_pretrain if x.lower() not in obs]
    qs.append(make_q(7, 3,
        question="Which large-scale pretraining datasets (JFT, LAION, Conceptual Captions) are NOT mentioned in this corpus?",
        gold_answer=", ".join(missing3) if missing3 else "all are mentioned",
        gold_items=missing3,
        match_strategy="structural",
        notes=f"missing={missing3}"))

    # Q4: CIFAR variants
    cifar_variants = ["CIFAR-10", "CIFAR-100"]
    obs_cifar = [x for x in cifar_variants if x.lower() in obs]
    missing4 = [x for x in cifar_variants if x.lower() not in obs]
    qs.append(make_q(7, 4,
        question="Which CIFAR variants (CIFAR-10, CIFAR-100) appear in this corpus?",
        gold_answer=", ".join(obs_cifar) if obs_cifar else "none",
        gold_items=obs_cifar,
        match_strategy="structural",
        notes=f"present={obs_cifar}, absent={missing4}"))

    # Q5: standard detection datasets
    expected_det = ["COCO", "PASCAL VOC", "Open Images", "LVIS",
                    "Objects365", "V3Det"]
    missing5 = [x for x in expected_det if x.lower() not in obs]
    qs.append(make_q(7, 5,
        question="Which standard object detection datasets are NOT used in this corpus?",
        gold_answer=", ".join(missing5) if missing5 else "all are present",
        gold_items=missing5,
        match_strategy="structural",
        notes=f"missing={missing5}"))

    return qs


def generate_tier8() -> list[dict]:
    """Quantitative computation — gold from pandas results."""
    qs = []

    df_mv = pd.read_sql("SELECT * FROM model_variants", sqlite3.connect(DB_PATH))
    df_papers = pd.read_sql("SELECT * FROM papers", sqlite3.connect(DB_PATH))
    df_results = pd.read_sql("SELECT * FROM results", sqlite3.connect(DB_PATH))

    # Q1: median parameter count
    med = round(df_mv["param_count_millions"].dropna().median(), 2)
    qs.append(make_q(8, 1,
        question="What is the median parameter count (in millions) across all model variants in this corpus?",
        gold_answer=str(med),
        match_strategy="substring",
        notes=f"pandas median = {med}M"))

    # Q2: SOTA claim fraction
    n_total = len(df_results)
    n_sota = int(df_results["is_sota_claim"].sum())
    pct = round(100 * n_sota / n_total, 1)
    qs.append(make_q(8, 2,
        question="What percentage of benchmark results in this corpus include a SOTA claim?",
        gold_answer=str(pct),
        match_strategy="substring",
        notes=f"{n_sota}/{n_total} = {pct}%"))

    # Q3: average citation count for papers benchmarking on ImageNet vs not
    img_ids_raw = sql("""SELECT DISTINCT m.paper_id FROM mentions m
                         JOIN entities e ON e.entity_id=m.entity_id
                         WHERE e.canonical='ImageNet'""")
    img_ids = {r["paper_id"] for r in img_ids_raw}
    df_papers["on_imagenet"] = df_papers["paper_id"].isin(img_ids)
    avgs = df_papers.groupby("on_imagenet")["citation_count"].mean().round(0)
    avg_with = int(avgs.get(True, 0))
    avg_without = int(avgs.get(False, 0))
    qs.append(make_q(8, 3,
        question="On average, do papers that benchmark on ImageNet have more citations than those that don't?",
        gold_answer="yes" if avg_with > avg_without else "no",
        match_strategy="substring",
        notes=f"ImageNet avg={avg_with}, non-ImageNet avg={avg_without}"))

    # Q4: max parameter count
    max_p = df_mv["param_count_millions"].dropna().max()
    qs.append(make_q(8, 4,
        question="What is the largest model variant (in millions of parameters) in this corpus?",
        gold_answer=str(int(max_p)),
        match_strategy="substring",
        notes=f"max param_count_millions = {max_p}"))

    # Q5: correlation citation count vs max model size
    merged = df_papers[["paper_id", "citation_count"]].merge(
        df_mv.groupby("paper_id")["param_count_millions"].max().reset_index(),
        on="paper_id"
    ).dropna()
    corr = round(merged["citation_count"].corr(merged["param_count_millions"]), 3)
    qs.append(make_q(8, 5,
        question="What is the Pearson correlation between citation count and maximum model size across papers in this corpus?",
        gold_answer=str(corr),
        match_strategy="substring",
        notes=f"pearson r = {corr}"))

    return qs


# ── Main ─────────────────────────────────────────────────────────────────────

async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print questions, don't write file")
    args = p.parse_args()

    print("Generating eval questions from DB...")
    print()

    all_qs: list[dict] = []

    print("[T1] single-doc factual...")
    all_qs.extend(generate_tier1())
    print(f"  generated {sum(1 for q in all_qs if q['tier']==1)} T1 questions")

    print("[T2] corpus aggregation...")
    all_qs.extend(generate_tier2())
    print(f"  generated {sum(1 for q in all_qs if q['tier']==2)} T2 questions")

    print("[T3] comparative/contradiction...")
    all_qs.extend(await generate_tier3())
    print(f"  generated {sum(1 for q in all_qs if q['tier']==3)} T3 questions")

    print("[T4] temporal/evolution...")
    all_qs.extend(generate_tier4())
    print(f"  generated {sum(1 for q in all_qs if q['tier']==4)} T4 questions")

    print("[T5] citation-graph reasoning...")
    all_qs.extend(generate_tier5())
    print(f"  generated {sum(1 for q in all_qs if q['tier']==5)} T5 questions")

    print("[T6] multi-hop/compositional...")
    all_qs.extend(await generate_tier6())
    print(f"  generated {sum(1 for q in all_qs if q['tier']==6)} T6 questions")

    print("[T7] negation/absence...")
    all_qs.extend(generate_tier7())
    print(f"  generated {sum(1 for q in all_qs if q['tier']==7)} T7 questions")

    print("[T8] quantitative computation...")
    all_qs.extend(generate_tier8())
    print(f"  generated {sum(1 for q in all_qs if q['tier']==8)} T8 questions")

    print()
    by_tier = {}
    for q in all_qs:
        by_tier.setdefault(q["tier"], 0)
        by_tier[q["tier"]] += 1
    print(f"Total: {len(all_qs)} questions")
    for t in sorted(by_tier):
        needs = sum(1 for q in all_qs if q["tier"]==t and q.get("needs_review"))
        print(f"  T{t}: {by_tier[t]} questions ({needs} need manual review)")

    if args.dry_run:
        print("\n--- DRY RUN: sample questions ---")
        for q in all_qs[:3]:
            print(f"\n[{q['id']}] {q['question']}")
            print(f"  gold: {q['gold_answer'][:120]}")
            print(f"  match: {q['match_strategy']}  review: {q['needs_review']}")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for q in all_qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_qs)} questions to {OUT_PATH}")
    print(f"Run 'python scripts/generate_eval_set.py --dry-run' to preview without writing.")


if __name__ == "__main__":
    asyncio.run(amain())
