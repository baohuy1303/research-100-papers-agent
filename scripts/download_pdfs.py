"""
Downloads open-access PDFs for papers in data/manifest.csv.
Updates the pdf_path column in the manifest after each successful download.
"""
import sys
import time
import httpx
import pandas as pd
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "manifest.csv"
PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"
DELAY_SECONDS = 0.5
TIMEOUT = 60


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", errors="replace").decode("ascii"))


def download_pdf(url: str, dest: Path) -> bool:
    try:
        with httpx.stream("GET", url, timeout=TIMEOUT, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        size = dest.stat().st_size
        if size < 10_000:  # suspiciously small — likely an error page
            dest.unlink()
            return False
        return True
    except Exception as e:
        safe_print(f"    Error: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found at {MANIFEST_PATH}. Run fetch_papers.py first.")
        sys.exit(1)

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST_PATH, dtype={"pdf_path": str, "doi": str, "source_url": str})

    total = len(df)
    success = 0
    skip = 0

    for i, row in df.iterrows():
        paper_id = row["id"]
        oa_url = row.get("oa_url", "")
        dest = PDF_DIR / f"{paper_id}.pdf"

        if not oa_url:
            safe_print(f"[{i+1}/{total}] SKIP (no OA URL): {row['title'][:60]}")
            skip += 1
            continue

        if dest.exists() and dest.stat().st_size > 10_000:
            safe_print(f"[{i+1}/{total}] ALREADY EXISTS: {paper_id}.pdf")
            df.at[i, "pdf_path"] = str(dest.relative_to(Path(__file__).parent.parent))
            success += 1
            continue

        safe_print(f"[{i+1}/{total}] Downloading {paper_id[:16]}...: {row['title'][:55]}")
        ok = download_pdf(oa_url, dest)
        if ok:
            rel_path = str(dest.relative_to(Path(__file__).parent.parent))
            df.at[i, "pdf_path"] = rel_path
            size_kb = dest.stat().st_size // 1024
            safe_print(f"    OK ({size_kb} KB) -> {rel_path}")
            success += 1
        else:
            safe_print(f"    FAILED: {oa_url}")

        time.sleep(DELAY_SECONDS)

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nDone. {success} downloaded, {skip} skipped, {total - success - skip} failed.")
    print(f"Manifest updated at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
