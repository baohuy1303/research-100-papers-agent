"""
Phase 3a: Normalize numeric surface forms across all 100 extracted JSONs.

Reads  data/extractions/*.json
Writes data/normalized/{paper_id}.json  (same structure + _canonical fields)

No LLM — pure regex. Adds:
  model_variants[].param_count_millions   float | null
  benchmark_results[].value_canonical     float | null
  training_details.batch_size             already int from extraction
  training_details.epochs                 already int from extraction

Run:
    python scripts/normalize_numbers.py
    python scripts/normalize_numbers.py --report   # print coverage stats
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
IN_DIR  = ROOT / "data" / "extractions"
OUT_DIR = ROOT / "data" / "normalized"


# ── Number parsers ────────────────────────────────────────────────────────────

def parse_params_to_millions(s: str | None) -> float | None:
    """Parse param-count surface string → float in millions.

    Handles: "86M", "86M parameters", "1.2B", "175 billion parameters",
             "86 M", "1,024M", bare integers assumed to be millions.
    Returns None when the string is unrecognisable or implausible.
    """
    if not s:
        return None
    orig = s
    s = s.strip().lower()

    # Strip leading ~, ≈, ~
    s = re.sub(r"^[~≈~]\s*", "", s)

    # Strip trailing label words
    s = re.sub(
        r"\s*(billion|million)?\s*(model\s+)?params?(eters?)?\s*$", "", s
    ).strip()
    s = re.sub(r"\s+parameter\s+network$", "", s).strip()

    # Remove commas used as thousand-separators
    s = s.replace(",", "")

    # Parenthetical suffix like "(1.43%)" — strip it
    s = re.sub(r"\s*\(.*?\)", "", s).strip()

    # Billions  e.g. "1.2b", "14.8b", "12b"
    m = re.fullmatch(r"([\d.]+)\s*b(?:illion)?", s)
    if m:
        val = float(m.group(1)) * 1_000
        return val if val <= 10_000_000 else None  # >10T params → reject

    # Millions  e.g. "86m", "307m", "14705.1m"
    m = re.fullmatch(r"([\d.]+)\s*m(?:illion)?", s)
    if m:
        val = float(m.group(1))
        return val if val <= 10_000_000 else None

    # Bare number: treat as millions if plausible range (0.01 … 200 000 M)
    m = re.fullmatch(r"[\d.]+", s)
    if m:
        val = float(s)
        if 0.001 <= val <= 200_000:
            return val
        return None

    return None  # unrecognised pattern


def parse_metric_value(s: str | None) -> float | None:
    """Parse a benchmark result value surface → float.

    Handles:
      "85.3"        → 85.3
      "85.3%"       → 85.3
      "0.32 ± 0.16" → 0.32  (primary only)
      "0.226/0.324" → 0.226 (first of range)
      "-"           → None
      "N/A"         → None
    """
    if not s:
        return None
    s = s.strip()

    # Explicit missing markers
    if s in {"-", "–", "—", "N/A", "n/a", "NA", "na", "-", "*"}:
        return None

    # Strip HTML bold tags
    s = re.sub(r"</?b>", "", s).strip()

    # Strip LaTeX/unicode junk  e.g. "77.8_{\x00b1 0.4}"
    s = re.sub(r"[_\{].*$", "", s).strip()

    # Strip trailing %
    s = s.rstrip("%").strip()

    # Take primary of  "value ± error"  (±, ±, ±, Unicode replacement chars)
    s = re.split(r"\s*[±±±\?]\s*", s)[0].strip()

    # Take first of  "value/value2"
    s = s.split("/")[0].strip()

    # Strip parenthetical  "value (†)"  "value (SOTA)"
    s = re.sub(r"\s*\(.*?\)", "", s).strip()

    # Strip trailing non-numeric junk like "*", "†", "‡", "§"
    s = s.rstrip("*†‡§").strip()

    try:
        return float(s)
    except ValueError:
        return None


# ── Per-paper normalizer ──────────────────────────────────────────────────────

def normalize_paper(data: dict) -> dict:
    """Add _canonical fields in-place (mutates and returns data)."""
    # model_variants
    for mv in data.get("model_variants", []):
        mv["param_count_millions"] = parse_params_to_millions(
            mv.get("param_count_surface")
        )

    # benchmark_results
    for br in data.get("benchmark_results", []):
        br["value_canonical"] = parse_metric_value(br.get("value_surface"))

    return data


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true", help="print coverage stats after run")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IN_DIR.glob("*.json"))
    if not files:
        print("No extraction JSONs found in", IN_DIR)
        sys.exit(1)

    param_total = param_parsed = 0
    value_total = value_parsed = 0

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        normalize_paper(data)

        # Stats
        for mv in data.get("model_variants", []):
            if mv.get("param_count_surface"):
                param_total += 1
                if mv["param_count_millions"] is not None:
                    param_parsed += 1
        for br in data.get("benchmark_results", []):
            if br.get("value_surface") and br["value_surface"] not in {"-", "N/A"}:
                value_total += 1
                if br["value_canonical"] is not None:
                    value_parsed += 1

        out = OUT_DIR / f.name
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Normalized {len(files)} papers -> {OUT_DIR}")
    if args.report:
        print(f"\nParam count coverage : {param_parsed}/{param_total} "
              f"({100*param_parsed//max(param_total,1)}%)")
        print(f"Metric value coverage: {value_parsed}/{value_total} "
              f"({100*value_parsed//max(value_total,1)}%)")


if __name__ == "__main__":
    main()
