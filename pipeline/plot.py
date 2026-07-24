"""CLI: python -m pipeline.plot configs/plots/<name>.yaml
       python -m pipeline.plot configs/plots/            (renders every spec in the directory)

Reads a plot spec and calls the matching src/50-visualize/renderers/<type>.py.
Plot specs are independent of any campaign or run_id -- they just query by
parameter filter -- so a figure can be regenerated (or a new one added)
without re-running any experiment.

data.source selects where the renderer's input comes from:
- "store" (default): queries data/store/results_index.parquet, the tidy
  scalar-metric table aggregate.py builds. Renderer gets a DataFrame.
- "raw_result": reads a single *.result.jsonl directly (identified by the
  full param set: target_method, reservoir_method, U, D, reservoir_size,
  use_size) and pulls out one metric's raw value. Needed for *_by_cell
  metrics (per-n, per-(i,j) breakdowns), which aggregate.py deliberately
  drops from the parquet index since they're list-valued. Renderer gets
  that raw list/nested-list instead of a DataFrame.
- "raw_result_multi": like raw_result, but for several reservoir_method
  panels (each with its own reservoir_size/use_size) sharing one
  target_method/U/D/metric, e.g. INPUT/ELM/NGRC/DECSSP side by side in
  one figure. Renderer gets a list of (reservoir_method, value) tuples,
  in data.panels' order.

For "store", data.filter does plain column equality/isin; data.query (a
pandas DataFrame.query expression) additionally filters on computed
relationships between columns, e.g. "use_size == 10 * reservoir_size" to
isolate a campaign's use_size directive from other campaigns that swept
use_size at the same reservoir_size.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml

from pipeline import store
from pipeline.store import PLOTS_ROOT, REPO_ROOT, STORE_ROOT

RENDERERS_DIR = REPO_ROOT / "src" / "50-visualize" / "renderers"
INDEX_PATH = STORE_ROOT / "results_index.parquet"


def _load_renderer(plot_type: str):
    path = RENDERERS_DIR / f"{plot_type}.py"
    if not path.exists():
        raise ValueError(f"no renderer for plot type {plot_type!r} (expected {path})")
    spec = importlib.util.spec_from_file_location(f"renderer_{plot_type}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_filter(df: pd.DataFrame, filt: dict) -> pd.DataFrame:
    for key, value in filt.items():
        if isinstance(value, list):
            df = df[df[key].isin(value)]
        else:
            df = df[df[key] == value]
    return df


def _load_result_value(target_method: str, reservoir_method: str, U: int, D: int, reservoir_size: int, use_size: int, metric: str):
    result_path = store.experiment_path(target_method, reservoir_method, U, D, reservoir_size, use_size)
    if not result_path.exists():
        raise FileNotFoundError(f"{result_path} not found; run pipeline.runner for this campaign first")
    return store.read_result_metric(result_path, metric)


def _load_raw_result_metric(data_spec: dict):
    return _load_result_value(
        data_spec["target_method"],
        data_spec["reservoir_method"],
        data_spec["U"],
        data_spec["D"],
        data_spec["reservoir_size"],
        data_spec["use_size"],
        data_spec["metric"],
    )


def _load_raw_result_multi(data_spec: dict):
    """Like raw_result, but for several reservoir_method "panels" sharing
    one target_method/U/D/metric -- each method needs its own
    reservoir_size/use_size, so data.panels lists one {reservoir_method,
    reservoir_size, use_size} dict per panel, in draw order (e.g. INPUT,
    ELM, NGRC, DECSSP side by side in one figure)."""
    return [
        (
            p["reservoir_method"],
            _load_result_value(
                data_spec["target_method"], p["reservoir_method"], data_spec["U"], data_spec["D"],
                p["reservoir_size"], p["use_size"], data_spec["metric"],
            ),
        )
        for p in data_spec["panels"]
    ]


def render(spec_path: Path) -> Path:
    spec = yaml.safe_load(Path(spec_path).read_text())
    data_spec = spec["data"]
    source = data_spec.get("source", "store")

    renderer = _load_renderer(spec["plot"]["type"])
    output_path = Path(spec.get("output", PLOTS_ROOT / f"{spec['name']}.png"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source == "store":
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"{INDEX_PATH} not found; run pipeline.aggregate first")
        df = pd.read_parquet(INDEX_PATH)
        df = _apply_filter(df, data_spec.get("filter", {}))
        if "metric" in data_spec:
            df = df[df["metric_name"] == data_spec["metric"]]
        if "query" in data_spec:
            df = df.query(data_spec["query"])
        if df.empty:
            raise ValueError(f"plot spec {spec['name']}: filter matched 0 rows")
        renderer.render(df, spec["plot"], output_path)
    elif source == "raw_result":
        value = _load_raw_result_metric(data_spec)
        renderer.render(value, spec["plot"], output_path)
    elif source == "raw_result_multi":
        panels = _load_raw_result_multi(data_spec)
        renderer.render(panels, spec["plot"], output_path)
    else:
        raise ValueError(f"unknown data source: {source!r}")

    print(f"wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="a plot spec yaml, or a directory of them")
    args = parser.parse_args()

    paths = sorted(args.path.glob("*.yaml")) if args.path.is_dir() else [args.path]
    for p in paths:
        render(p)


if __name__ == "__main__":
    main()
