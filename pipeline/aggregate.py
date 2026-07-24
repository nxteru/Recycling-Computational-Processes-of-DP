"""CLI: python -m pipeline.aggregate [--force-full]

Scans data/store/40-experiment/**/*.result.jsonl and (re)builds
data/store/results_index.parquet, the single tidy table pipeline/plot.py
queries. Only re-parses files whose mtime changed since the last build
(tracked in results_index.sources.json next to the parquet).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.store import STORE_ROOT

INDEX_PATH = STORE_ROOT / "results_index.parquet"
SOURCES_PATH = STORE_ROOT / "results_index.sources.json"


def _result_files() -> list[Path]:
    return sorted((STORE_ROOT / "40-experiment").glob("**/*.result.jsonl"))


def _valid_cells(value: list, U: int) -> list:
    """Flatten a *_by_cell metric down to just its structurally valid
    entries. Some by_cell metrics are a flat U*D list (every entry real:
    run_prefix_value_flat's value_accuracy/kappa_by_cell,
    run_prefix_value's reconstruction_accuracy_by_cell); others are a
    nested (D, U*D) table zero-padded past each row's own (i+1)*U valid
    width (run_prefix_binary's accuracy_by_cell, run_prefix_value's
    value_accuracy/kappa_by_cell) -- naively min()-ing the padded form
    would always return 0 from the padding, not the true worst *valid*
    cell."""
    if value and isinstance(value[0], list):
        return [value[i][j] for i in range(len(value)) for j in range((i + 1) * U)]
    return list(value)


def rebuild_index(force_full: bool = False) -> pd.DataFrame:
    STORE_ROOT.mkdir(parents=True, exist_ok=True)
    files = _result_files()
    current_mtimes = {str(f.relative_to(STORE_ROOT)): f.stat().st_mtime for f in files}

    prev_sources = {}
    existing_df = pd.DataFrame()
    if not force_full and SOURCES_PATH.exists() and INDEX_PATH.exists():
        prev_sources = json.loads(SOURCES_PATH.read_text())
        existing_df = pd.read_parquet(INDEX_PATH)

    unchanged = {k for k, v in current_mtimes.items() if prev_sources.get(k) == v}
    if not existing_df.empty:
        existing_df = existing_df[existing_df["_source"].isin(unchanged)]

    new_rows = []
    for relpath, mtime in current_mtimes.items():
        if relpath in unchanged:
            continue
        with open(STORE_ROOT / relpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row["metric_value"], list):
                    # Per-cell breakdowns (e.g. accuracy_by_cell) are kept
                    # only in the raw *.result.jsonl; the index is a tidy
                    # scalar-per-row table so it stays a valid parquet
                    # schema. Their minimum is still a useful scalar (e.g.
                    # for a "worst cell" heatmap), cheap to derive once here.
                    cells = _valid_cells(row["metric_value"], row["U"])
                    if cells:
                        min_row = dict(row)
                        min_row["metric_name"] = row["metric_name"] + "_min"
                        min_row["metric_value"] = min(cells)
                        min_row["_source"] = relpath
                        new_rows.append(min_row)
                    continue
                row["_source"] = relpath
                new_rows.append(row)

    new_df = pd.DataFrame(new_rows)
    if existing_df.empty:
        combined = new_df
    elif new_df.empty:
        combined = existing_df
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    if combined.empty:
        print("no experiment results found; nothing to index")
        return combined

    combined.to_parquet(INDEX_PATH, index=False)
    SOURCES_PATH.write_text(json.dumps(current_mtimes))
    print(f"indexed {len(combined)} rows from {len(files)} file(s) ({len(new_rows)} new/changed) -> {INDEX_PATH}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-full", action="store_true", help="ignore mtimes and re-read every result file")
    args = parser.parse_args()
    rebuild_index(force_full=args.force_full)


if __name__ == "__main__":
    main()
