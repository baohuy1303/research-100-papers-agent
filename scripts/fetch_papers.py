"""
Fetches the top 100 most-cited Vision Transformer papers using the
Semantic Scholar Bulk Search API.

Approach:
1. Bulk-search "vision transformer" sorted by citationCount descending.
2. Paginate until we have >= CANDIDATE_TARGET papers with open-access PDFs.
3. Sort by citation count, take top 100 with PDF URLs.
4. Write data/manifest.csv.

Auth: optional S2_API_KEY in .env gives 1 req/s; without it we sleep longer.
"""
import os
import time
import httpx
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
S2_API_KEY = os.getenv("S2_API_KEY", "")

BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "paperId,title,authors,year,venue,citationCount,openAccessPdf,externalIds,publicationTypes"
TARGET = 100
CANDIDATE_TARGET = 300  # collect until we have this many with OA PDF
PAGE_LIMIT = 500        # max results per bulk request

HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
SLEEP = 1.1 if not S2_API_KEY else 0.5


def fetch_bulk_page(token: str | None = None) -> dict:
    params: dict = {
        "query": "vision transformer",
        "sort": "citationCount:desc",
        "fields": FIELDS,
        "limit": PAGE_LIMIT,
    }
    if token:
        params["token"] = token
    resp = httpx.get(BULK_URL, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()


def parse_authors(authors: list) -> str:
    return " | ".join(a.get("name", "") for a in authors if a.get("name"))


def to_row(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    doi = ext.get("DOI", "")
    arxiv_id = ext.get("ArXiv", "")
    oa = p.get("openAccessPdf") or {}
    oa_url = oa.get("url", "")

    # If S2 doesn't have a PDF URL but we have an arXiv ID, construct it
    if not oa_url and arxiv_id:
        oa_url = f"https://arxiv.org/pdf/{arxiv_id}"

    return {
        "id": p.get("paperId", ""),
        "title": p.get("title", ""),
        "authors": parse_authors(p.get("authors", [])),
        "year": p.get("year"),
        "venue": p.get("venue", ""),
        "citation_count": p.get("citationCount", 0),
        "doi": doi,
        "source_url": f"https://doi.org/{doi}" if doi else "",
        "oa_url": oa_url,
        "pdf_path": "",
    }


def main():
    print("Fetching Vision Transformer papers from Semantic Scholar...")
    all_papers: list[dict] = []
    token: str | None = None
    page_num = 0

    while True:
        page_num += 1
        print(f"  Page {page_num} (token={'...' if token else 'start'})...")
        data = fetch_bulk_page(token)
        results = data.get("data", [])
        if not results:
            print("  No more results.")
            break

        all_papers.extend(results)
        token = data.get("token")

        oa_count = sum(1 for p in all_papers if (p.get("openAccessPdf") or {}).get("url") or p.get("externalIds", {}).get("ArXiv"))
        print(f"  Total so far: {len(all_papers)} papers, {oa_count} with OA PDF")

        if oa_count >= CANDIDATE_TARGET or not token:
            break
        time.sleep(SLEEP)

    print(f"\nProcessing {len(all_papers)} candidates...")
    rows = [to_row(p) for p in all_papers]
    df = pd.DataFrame(rows).drop_duplicates(subset="id")

    oa_df = df[df["oa_url"] != ""].copy()
    print(f"  {len(oa_df)} papers have PDF URLs.")

    oa_df = oa_df.sort_values("citation_count", ascending=False).head(TARGET)

    out_path = Path(__file__).parent.parent / "data" / "manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    oa_df.to_csv(out_path, index=False)

    print(f"\nDone. Wrote {len(oa_df)} papers to {out_path}")
    print(f"Citation range: {oa_df['citation_count'].max()} - {oa_df['citation_count'].min()}")
    print(f"Year range: {oa_df['year'].min()} - {oa_df['year'].max()}")
    print("\nTop 15:")
    print(oa_df[["title", "year", "citation_count"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
