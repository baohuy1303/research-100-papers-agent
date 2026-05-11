import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException

router = APIRouter(prefix="/papers", tags=["papers"])

MANIFEST_PATH = Path(__file__).parent.parent.parent / "data" / "manifest.csv"
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    df = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    # Restore numeric types for json-friendly output
    for col in ("citation_count", "year"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df.to_dict(orient="records")


def _run_script(script_name: str):
    script = SCRIPTS_DIR / script_name
    subprocess.run([sys.executable, str(script)], check=True)


@router.get("")
def list_papers(
    year: Optional[int] = None,
    min_citations: Optional[int] = None,
    limit: int = 100,
):
    papers = _load_manifest()
    if year is not None:
        papers = [p for p in papers if p.get("year") == year]
    if min_citations is not None:
        papers = [p for p in papers if (p.get("citation_count") or 0) >= min_citations]
    return papers[:limit]


@router.get("/{paper_id}")
def get_paper(paper_id: str):
    papers = _load_manifest()
    for p in papers:
        if p.get("id") == paper_id or p.get("openalex_id", "").endswith(paper_id):
            return p
    raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")


@router.post("/fetch")
def fetch_papers(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_script, "fetch_papers.py")
    return {"status": "started", "message": "Fetching papers from OpenAlex in background"}


@router.post("/download")
def download_pdfs(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_script, "download_pdfs.py")
    return {"status": "started", "message": "Downloading PDFs in background"}
