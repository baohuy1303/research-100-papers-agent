"""
End-to-end sanity check across Phases 1-5.

Runs ~15 checks covering: parsed markdown, extractions, normalized JSONs,
entity map, SQLite, Chroma, NetworkX graph, store wrappers, retrieval,
and the tier classifier. Prints PASS/FAIL per check and a final summary.

Usage:
    python scripts/sanity_check.py
    python scripts/sanity_check.py --skip-llm   # skip classifier (no API call)
"""
import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
PASS = "[ OK ]"
FAIL = "[FAIL]"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    marker = PASS if ok else FAIL
    line = f"{marker}  {name}"
    if detail:
        line += f"  -- {detail}"
    print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)


def header(text: str) -> None:
    print(f"\n-- {text} " + "-" * (60 - len(text)), flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1-3 — file artifacts
# ─────────────────────────────────────────────────────────────────────────────

def check_phase_1_2_3():
    header("Phase 1-3 artifacts")

    md = list((ROOT / "data" / "markdown").glob("*.md"))
    check("Phase 1: 100 markdown files", len(md) == 100, f"{len(md)} files")

    ext = list((ROOT / "data" / "extractions").glob("*.json"))
    check("Phase 2: 100 extraction JSONs", len(ext) == 100, f"{len(ext)} files")

    norm = list((ROOT / "data" / "normalized").glob("*.json"))
    check("Phase 3a: 100 normalized JSONs", len(norm) == 100, f"{len(norm)} files")

    # Spot-check a normalized file has canonical fields
    if norm:
        d = json.loads(norm[0].read_text(encoding="utf-8"))
        has_param_canon = any(mv.get("param_count_millions") is not None
                              for mv in d.get("model_variants", []))
        has_value_canon = any(br.get("value_canonical") is not None
                              for br in d.get("benchmark_results", []))
        check("Phase 3a: canonical numeric fields present",
              has_param_canon and has_value_canon,
              f"params={has_param_canon}, values={has_value_canon}")

    em_path = ROOT / "data" / "entity_map.json"
    if em_path.exists():
        em = json.loads(em_path.read_text(encoding="utf-8"))
        n_ds, n_mt, n_me = len(em["datasets"]), len(em["metrics"]), len(em["methods"])
        check("Phase 3b: entity_map exists", True,
              f"{n_ds} datasets, {n_mt} metrics, {n_me} methods")
        # Verify ImageNet captures multiple aliases
        img = next((e for e in em["datasets"] if e["canonical"] == "ImageNet"), None)
        check("Phase 3b: ImageNet aliases", img is not None and len(img["aliases"]) >= 10,
              f"{len(img['aliases']) if img else 0} aliases")
        check("Phase 3b: ImageNet has PWC ID",
              img is not None and img.get("paperswithcode_id") == "imagenet-1k-1",
              f"pwc_id={img.get('paperswithcode_id') if img else None}")
    else:
        check("Phase 3b: entity_map.json exists", False, "missing")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 4 — indexes
# ─────────────────────────────────────────────────────────────────────────────

def check_phase_4():
    header("Phase 4 indexes")

    # SQLite
    db = ROOT / "data" / "corpus.db"
    if db.exists():
        conn = sqlite3.connect(db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        expected = {"papers", "entities", "aliases", "mentions", "results",
                    "model_variants", "training", "claims", "paper_refs"}
        check("SQLite: all 9 tables exist",
              expected.issubset(tables),
              f"missing: {expected - tables}" if not expected.issubset(tables) else "9/9")

        n_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        check("SQLite: 100 papers", n_papers == 100, f"{n_papers}")

        n_results = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        check("SQLite: results > 3000", n_results > 3000, f"{n_results}")

        n_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        check("SQLite: entities > 1500", n_entities > 1500, f"{n_entities}")
        conn.close()
    else:
        check("SQLite: corpus.db exists", False, "missing")

    # Chroma
    chroma_dir = ROOT / "data" / "chroma"
    if chroma_dir.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_dir))
            col = client.get_collection("paper_chunks")
            n_chunks = col.count()
            check("Chroma: collection exists with chunks", n_chunks > 3000, f"{n_chunks} chunks")
        except Exception as e:
            check("Chroma: openable", False, f"{e}")
    else:
        check("Chroma: data/chroma exists", False, "missing")

    # Graph
    graph_path = ROOT / "data" / "citation_graph.gpickle"
    if graph_path.exists():
        import pickle
        g = pickle.load(open(graph_path, "rb"))
        check("Graph: 100 nodes", g.number_of_nodes() == 100, f"{g.number_of_nodes()}")
        check("Graph: edges > 500", g.number_of_edges() > 500, f"{g.number_of_edges()}")
    else:
        check("Graph: citation_graph.gpickle exists", False, "missing")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 5 — query infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def check_store():
    header("Phase 5: store.py")
    from api.core.store import CorpusStore
    s = CorpusStore()

    # Most cited within corpus → expect ViT
    mc = s.most_cited(1)
    is_vit = mc and "Image is Worth" in (mc[0].get("title") or "")
    check("store.most_cited(1) returns ViT", is_vit,
          f"top: {mc[0]['title'][:50] if mc else 'none'}")

    # papers_using ImageNet → expect ~70 papers
    n = len(s.papers_using("ImageNet", "dataset"))
    check("store.papers_using(ImageNet) returns 60-90 papers", 60 <= n <= 90, f"{n}")

    # results_for → returns sane benchmark rows
    rs = s.best_on("ImageNet", "top-1 accuracy", k=3)
    check("store.best_on(ImageNet, top-1 accuracy) returns 3 rows",
          len(rs) == 3,
          f"{len(rs)} rows, top val={rs[0]['value_canonical'] if rs else 'NA'}")

    # entity_by_alias resolves ILSVRC2012 → ImageNet
    e = s.entity_by_alias("ILSVRC2012", "dataset")
    check("store.entity_by_alias(ILSVRC2012) -> ImageNet",
          e and e.get("canonical") == "ImageNet",
          f"resolved to: {e['canonical'] if e else None}")

    # SQL safety: only SELECT/WITH allowed
    blocked = False
    try:
        s.execute_sql("DROP TABLE papers")
    except ValueError:
        blocked = True
    check("store.execute_sql blocks DROP", blocked)

    # Graph descendants of ViT
    if mc:
        desc = s.descendants(mc[0]["paper_id"])
        check("store.descendants(ViT) > 50", len(desc) > 50, f"{len(desc)} descendants")


async def check_retrieval():
    header("Phase 5: retrieval.py")
    from api.core.retrieval import Retriever
    from api.core.store import CorpusStore
    r = Retriever()
    s = CorpusStore()

    res = await r.search("shifted window self-attention", k=3)
    # Swin chunks have section "3.2. Shifted Window based Self-Attention" — check
    # the paper title via store (chunk body starts with "# section_title", not the
    # paper title, so we need a join).
    top_paper_id = res["chunks"][0]["paper_id"] if res["chunks"] else None
    top_title = s.get_paper(top_paper_id)["title"] if top_paper_id else ""
    top_ok = "Swin" in top_title
    check("retrieval.search(swin) -> top hit is Swin paper", bool(top_ok),
          f"top: {top_title[:50]} (score={res['chunks'][0]['score']:.3f})")
    check("retrieval cost is tiny (<$0.001)", res["cost_usd"] < 0.001,
          f"${res['cost_usd']:.6f}")


async def check_handlers():
    header("Phase 6: tier handlers")
    from api.core.handlers import get_handler
    from api.core.retrieval import Retriever
    from api.core.store import CorpusStore
    store = CorpusStore()
    retriever = Retriever()

    cases = [
        (1, "What architecture does ViT use?"),
        (2, "How many papers in the corpus benchmark on ImageNet?"),
        (4, "How did the maximum top-1 accuracy on ImageNet change over years?"),
        (5, "Which paper is the most cited within this corpus?"),
        (7, "Which standard ViT benchmarks are NOT used by any paper in this corpus?"),
    ]
    total_cost = 0.0
    for tier, q in cases:
        try:
            handle = get_handler(tier)
            result = await handle(q, store, retriever)
            ok = bool(result.answer) and len(result.answer) > 10
            total_cost += result.cost_usd
            check(f"T{tier}: {q[:50]}", ok,
                  f"answer: {result.answer[:80]} | cost=${result.cost_usd:.4f}")
        except Exception as e:
            check(f"T{tier}: {q[:50]}", False, f"{type(e).__name__}: {e}")
    print(f"\n  total handler cost: ${total_cost:.4f}")


async def check_classifier():
    header("Phase 5: classifier.py")
    from api.core.classifier import TierClassifier
    c = TierClassifier()

    # Pick one easy question per tier
    cases = [
        ("What architecture does ViT use?", 1),
        ("How many papers benchmark on ImageNet?", 2),
        ("Do ViT and Swin agree on top-1 accuracy?", 3),
        ("Which paper is most cited within this corpus?", 5),
        ("What's the median parameter count across the corpus?", 8),
    ]
    correct = 0
    for q, expected in cases:
        result = await c.classify(q)
        ok = result["tier"] == expected
        if ok:
            correct += 1
        check(f"classify T{expected}: {q[:55]}",
              ok, f"got T{result['tier']} (conf={result['confidence']:.2f})")
    check("classifier: ≥4/5 correct", correct >= 4, f"{correct}/5")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-llm", action="store_true",
                   help="Skip classifier checks (no API call needed)")
    args = p.parse_args()

    check_phase_1_2_3()
    check_phase_4()
    check_store()
    await check_retrieval()
    if not args.skip_llm:
        await check_classifier()
        await check_handlers()

    # Summary
    n_total = len(results)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_pass}/{n_total} checks passed")
    if n_pass < n_total:
        print("\nFailures:")
        for name, ok, detail in results:
            if not ok:
                msg = f"  - {name}  ({detail})"
                print(msg.encode("ascii", errors="replace").decode("ascii"))
        sys.exit(1)
    print("ALL GREEN")


if __name__ == "__main__":
    asyncio.run(amain())
