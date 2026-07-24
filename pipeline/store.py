"""Parameter -> path mapping, meta.json bookkeeping, and cache checks for data/store."""
import functools
import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STORE_ROOT = REPO_ROOT / "data" / "store"
RUNS_ROOT = REPO_ROOT / "data" / "runs"
PLOTS_ROOT = REPO_ROOT / "data" / "plots"


@functools.lru_cache(maxsize=1)
def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _stem(U: int, D: int, size: int, split: str) -> str:
    return f"U{U}_D{D}_N{size}_{split}"


def input_path(input_method: str, U: int, D: int, size: int, split: str) -> Path:
    return STORE_ROOT / "00-input" / input_method / f"{_stem(U, D, size, split)}.bin"


def target_path(target_method: str, U: int, D: int, size: int, split: str) -> Path:
    return STORE_ROOT / "10-target" / target_method / f"{_stem(U, D, size, split)}.bin"


def reservoir_path(reservoir_method: str, U: int, D: int, size: int, split: str, variant: int | None = None) -> Path:
    """`variant` distinguishes reservoir generations whose content isn't a
    pure function of (reservoir_method, U, D, size, split) -- currently
    only ELM, whose num_features is a per-leaf parameter and whose random
    weights are drawn row-major, so a wider generation isn't a superset of
    a narrower one. Different ELM widths need distinct files, or the
    second one generated would silently overwrite the first under the
    same path with different column values."""
    suffix = f"_V{variant}" if variant is not None else ""
    return STORE_ROOT / "20-reservoir" / reservoir_method / f"{_stem(U, D, size, split)}{suffix}.bin"


def reservoir_size_path(reservoir_method: str, U: int, D: int, variant: int | None = None) -> Path:
    """One size.json per (reservoir_method, U, D[, variant]), shared by
    train/test splits. `variant` mirrors reservoir_path's."""
    suffix = f"_V{variant}" if variant is not None else ""
    return STORE_ROOT / "20-reservoir" / reservoir_method / f"U{U}_D{D}{suffix}.size.json"


def ngrc_degree_sizes_path(U: int, D: int) -> Path:
    """Cumulative feature count per degree for NGRC at (U, D), so a campaign
    can request reservoir_size as {ngrc_degree: N} instead of a raw column
    count."""
    return STORE_ROOT / "20-reservoir" / "NGRC" / f"U{U}_D{D}.degree_sizes.json"


def regression_path(target_method: str, reservoir_method: str, U: int, D: int, reservoir_size: int, use_size: int) -> Path:
    stem = f"U{U}_D{D}_R{reservoir_size}_N{use_size}"
    return STORE_ROOT / "30-regression" / target_method / reservoir_method / f"{stem}.wout"


def experiment_path(target_method: str, reservoir_method: str, U: int, D: int, reservoir_size: int, use_size: int) -> Path:
    stem = f"U{U}_D{D}_R{reservoir_size}_N{use_size}"
    return STORE_ROOT / "40-experiment" / target_method / reservoir_method / f"{stem}.result.jsonl"


def meta_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".meta.json")


def write_meta(
    output_path: Path,
    params: dict,
    duration_sec: float | None = None,
    io_sec: float | None = None,
    compute_sec: float | None = None,
) -> None:
    """duration_sec is wall-clock time around the subprocess call;
    io_sec/compute_sec are the task script's own I/O-vs-compute breakdown,
    parsed by runner.py from its "TIMING ..." stderr line (see
    src/_stage_timing.py)."""
    meta = {
        "params": params,
        "git_commit": git_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if duration_sec is not None:
        meta["duration_sec"] = round(duration_sec, 3)
    if io_sec is not None:
        meta["io_sec"] = round(io_sec, 3)
    if compute_sec is not None:
        meta["compute_sec"] = round(compute_sec, 3)
    meta_path(output_path).write_text(json.dumps(meta, indent=2, sort_keys=True))


def read_meta(output_path: Path) -> dict | None:
    mp = meta_path(output_path)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except Exception:
        return None


def is_cached(output_path: Path, params: dict) -> bool:
    if not output_path.exists():
        return False
    meta = read_meta(output_path)
    if meta is None:
        return False
    return meta.get("params") == params


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def atomic_output(final_path: Path):
    """Yield a temp path to write to; renamed into place only on success, so
    a crashed/killed task never leaves a corrupt file for is_cached() to
    trust later. The tmp filename includes pid+thread id so two concurrent
    writers targeting the same final_path (e.g. a shared sidecar file)
    never collide on one tmp file."""
    ensure_parent(final_path)
    tmp_path = final_path.with_name(f"{final_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        yield tmp_path
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def record_feature_count(reservoir_method: str, U: int, D: int, count: int, variant: int | None = None) -> None:
    """Record (or verify) the feature width a reservoir method produces for
    (U, D[, variant]). Train and test splits both call this and can run
    concurrently, so the write must be atomic -- a plain write_text()
    truncates before writing, and a concurrent reader could see an empty,
    unparseable file."""
    path = reservoir_size_path(reservoir_method, U, D, variant)
    if path.exists():
        try:
            recorded = json.loads(path.read_text())["feature_count"]
        except json.JSONDecodeError:
            recorded = None  # lost a race with a concurrent writer; just (re)write below
        if recorded is not None:
            if recorded != count:
                raise RuntimeError(
                    f"reservoir {reservoir_method} U={U} D={D}: train/test feature count mismatch "
                    f"(recorded {recorded}, got {count})"
                )
            return
    with atomic_output(path) as tmp:
        tmp.write_text(json.dumps({"feature_count": count}))


def read_feature_count(reservoir_method: str, U: int, D: int, variant: int | None = None) -> int:
    path = reservoir_size_path(reservoir_method, U, D, variant)
    return json.loads(path.read_text())["feature_count"]


def write_ngrc_degree_sizes(U: int, D: int, sizes: dict[int, int]) -> None:
    """Same atomicity concern as record_feature_count, but no conflict risk:
    a degree's cumulative count never changes regardless of which
    max_degree a generation run stopped at, so concurrent writers can't
    disagree on any shared degree."""
    path = ngrc_degree_sizes_path(U, D)
    payload = json.dumps({str(k): v for k, v in sizes.items()})
    if path.exists() and path.read_text() == payload:
        return
    with atomic_output(path) as tmp:
        tmp.write_text(payload)


def read_ngrc_degree_sizes(U: int, D: int) -> dict[int, int]:
    path = ngrc_degree_sizes_path(U, D)
    data = json.loads(path.read_text())
    return {int(k): v for k, v in data.items()}


def read_ngrc_degree_size(U: int, D: int, degree: int) -> int:
    path = ngrc_degree_sizes_path(U, D)
    sizes = json.loads(path.read_text())
    key = str(degree)
    if key not in sizes:
        raise ValueError(f"NGRC degree {degree} not available at U={U} D={D} (max generated: {max(map(int, sizes))})")
    return sizes[key]


def read_result_metric(result_path: Path, metric_name: str):
    """Pull a single metric's value (scalar or *_by_cell list/nested list)
    out of a *.result.jsonl, for plot specs that need per-cell breakdowns
    the aggregated results_index.parquet drops (see aggregate.py)."""
    with open(result_path) as f:
        for line in f:
            row = json.loads(line)
            if row["metric_name"] == metric_name:
                return row["metric_value"]
    raise ValueError(f"metric {metric_name!r} not found in {result_path}")
