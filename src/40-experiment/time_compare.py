"""CLI: python src/40-experiment/time_compare.py --use-size N [--test-size N] [--repeats R]

Compares two ways to get MAXSSP/MINSSP's answer, per test row (U=D=10,
DECSSP reservoir):

- direct dp: MAXSSP's/MINSSP's own O(U*D^2) dp solve() -- the whole table
  comes out together, so there's no cheaper way to get one answer.
- reservoir + regression: compute DECSSP's reservoir features once per
  row, then reconstruct the subset achieving each target partial sum n via
  this directory's own _common.py reconstruct() (existence + value
  predictions at each of up to D backward steps).

Every (row, n) pair is timed individually and averaged; each reconstructed
subset is checked against the direct-dp table's own ground truth.

Requires the campaign's regression stage to have already produced
data/store/30-regression/{DECSSP,MAXSSP,MINSSP}/DECSSP/U10_D10_R551_N<use_size>.wout
(run configs/campaigns/experiment4_u10d10_maxssp_minssp.yaml first).
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
from pipeline import store  # noqa: E402
import _binstore  # noqa: E402
from _common import prefix_index_map, reconstruct, stack_wout, load_wout  # noqa: E402


def _load_module(stage_dir: str, method: str):
    path = REPO_ROOT / "src" / stage_dir / method / "main.py"
    spec = importlib.util.spec_from_file_location(f"{stage_dir}_{method}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_correct(use, n: int, values: list[int], ground_truth_count: float) -> bool:
    """Matches this directory's _common.py's run_prefix_value correctness
    check exactly: reachability must agree with the direct-dp ground truth
    (ground_truth_count == 0 means n is unreachable), and when reachable,
    the reconstructed subset must both sum to n *and* use exactly
    ground_truth_count elements -- not just any subset summing to n, the
    max/min-cardinality one specifically."""
    if use is None:
        return ground_truth_count == 0
    if ground_truth_count == 0:
        return False
    total = sum(v for v, u in zip(values, use) if u)
    return total == n and sum(use) == ground_truth_count


def _time_reconstruct(direction: str, wout_stacked, index, U: int, D: int, dp, reservoir, values, n: int) -> tuple[float, bool]:
    """Time one reconstruct() call and check it against dp's ground truth
    at (D-1, n). Returns (elapsed_seconds, was_correct)."""
    ground_truth_count = dp[index[D - 1][n]]
    start = time.perf_counter()
    use = reconstruct(direction, wout_stacked, index, U, D, reservoir, values, n)
    elapsed = time.perf_counter() - start
    return elapsed, _check_correct(use, n, values, ground_truth_count)


def _stats(name: str, samples: list[float]) -> str:
    arr = np.array(samples) * 1000
    return (
        f"  {name:28s} mean={arr.mean():8.4f}ms  median={np.median(arr):8.4f}ms  "
        f"min={arr.min():8.4f}ms  max={arr.max():8.4f}ms  std={arr.std():8.4f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--U", type=int, default=10)
    parser.add_argument("--D", type=int, default=10)
    parser.add_argument("--reservoir-size", type=int, default=551)
    parser.add_argument("--use-size", type=int, required=True)
    parser.add_argument("--test-size", type=int, default=2000, help="how many test rows to average over")
    parser.add_argument("--repeats", type=int, default=1, help="repeats per (row, n) pair, for extra noise reduction")
    args = parser.parse_args()

    U, D, R, N = args.U, args.D, args.reservoir_size, args.use_size
    UD = U * D

    input_path = store.input_path("UNIFORM", U, D, 2000, "test")
    with open(input_path, "rb") as f:
        _u, _n, _t = _binstore.read_header(f)
        rows = [_binstore.read_row(f, D, _binstore.INPUT_DTYPE).tolist() for _ in range(args.test_size)]

    maxssp = _load_module("10-target", "MAXSSP")
    minssp = _load_module("10-target", "MINSSP")
    decssp_reservoir = _load_module("20-reservoir", "DECSSP")

    n_cols = U * D * (D + 1) // 2  # DECSSP/MAXSSP/MINSSP all use the full nested (i, j) layout
    wout_exist = load_wout(store.regression_path("DECSSP", "DECSSP", U, D, R, N), n_cols)
    wout_max = load_wout(store.regression_path("MAXSSP", "DECSSP", U, D, R, N), n_cols)
    wout_min = load_wout(store.regression_path("MINSSP", "DECSSP", U, D, R, N), n_cols)
    wout_max_stacked = stack_wout(wout_exist, wout_max)
    wout_min_stacked = stack_wout(wout_exist, wout_min)
    index = prefix_index_map(U, D)

    dp_max_times, dp_min_times, reservoir_times = [], [], []
    recon_max_times, recon_min_times = [], []
    per_n_max_times = [[] for _ in range(UD)]
    per_n_min_times = [[] for _ in range(UD)]
    correct_max = correct_min = 0
    total_checked = 0

    t_report = time.time()
    for row_idx, values in enumerate(rows):
        for _ in range(args.repeats):
            start = time.perf_counter()
            dp_max = maxssp.solve(U, D, values)
            dp_max_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            dp_min = minssp.solve(U, D, values)
            dp_min_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            reservoir = decssp_reservoir.solve(U, D, values)[:R].astype(np.float32)  # match wout_stacked's float32
            reservoir_times.append(time.perf_counter() - start)

            for n in range(1, UD + 1):
                dt, ok = _time_reconstruct("max", wout_max_stacked, index, U, D, dp_max, reservoir, values, n)
                recon_max_times.append(dt)
                per_n_max_times[n - 1].append(dt)
                correct_max += ok

                dt, ok = _time_reconstruct("min", wout_min_stacked, index, U, D, dp_min, reservoir, values, n)
                recon_min_times.append(dt)
                per_n_min_times[n - 1].append(dt)
                correct_min += ok

                total_checked += 1

        if time.time() - t_report > 5:
            print(f"  ... {row_idx + 1}/{len(rows)} rows", flush=True)
            t_report = time.time()

    t_dp_max = sum(dp_max_times) / len(dp_max_times)
    t_dp_min = sum(dp_min_times) / len(dp_min_times)
    t_reservoir = sum(reservoir_times) / len(reservoir_times)
    t_recon_max = sum(recon_max_times) / len(recon_max_times)
    t_recon_min = sum(recon_min_times) / len(recon_min_times)

    t_total_max = t_reservoir + t_recon_max
    t_total_min = t_reservoir + t_recon_min
    # Reservoir features are shared work: once computed for a row, they
    # back a MAXSSP reconstruction, a MINSSP reconstruction, or both for
    # free. Direct dp gets no such sharing -- MAXSSP's and MINSSP's tables
    # are two unrelated O(U*D^2) computations.
    t_total_both = t_reservoir + t_recon_max + t_recon_min
    t_dp_both = t_dp_max + t_dp_min

    print()
    print(f"U={U} D={D} reservoir_size={R} use_size={N}  rows={len(rows)}  n=1..{UD} per row  repeats/pair={args.repeats}")
    print(f"  ({len(recon_max_times)} reconstruction samples per target)")
    print()
    print("=== summary (means) ===")
    print("direct dp (whole table computed per row):")
    print(f"  MAXSSP solve(): {t_dp_max * 1000:.4f} ms")
    print(f"  MINSSP solve(): {t_dp_min * 1000:.4f} ms")
    print()
    print("reservoir + regression (reconstruct the subset for one n, averaged over all rows x n):")
    print(f"  DECSSP reservoir features:      {t_reservoir * 1000:.4f} ms")
    print(f"  MAXSSP reconstruction (avg):    {t_recon_max * 1000:.4f} ms")
    print(f"  MINSSP reconstruction (avg):    {t_recon_min * 1000:.4f} ms")
    print(f"  MAXSSP total:                   {t_total_max * 1000:.4f} ms  ({t_dp_max / t_total_max:.2f}x faster than direct dp)")
    print(f"  MINSSP total:                   {t_total_min * 1000:.4f} ms  ({t_dp_min / t_total_min:.2f}x faster than direct dp)")
    print()
    print("both MAXSSP and MINSSP for the same row (reservoir shared):")
    print(f"  direct dp (MAXSSP + MINSSP tables): {t_dp_both * 1000:.4f} ms")
    print(f"  reservoir + regression (shared):    {t_total_both * 1000:.4f} ms  ({t_dp_both / t_total_both:.2f}x faster than direct dp)")

    print()
    print("=== distribution (mean/median/min/max/std) ===")
    print(_stats("MAXSSP direct dp", dp_max_times))
    print(_stats("MINSSP direct dp", dp_min_times))
    print(_stats("DECSSP reservoir", reservoir_times))
    print(_stats("MAXSSP reconstruction", recon_max_times))
    print(_stats("MINSSP reconstruction", recon_min_times))

    print()
    print("=== correctness (reconstructed subset vs. direct-dp ground truth) ===")
    print(f"  MAXSSP: {correct_max}/{total_checked} correct ({correct_max / total_checked * 100:.2f}%)")
    print(f"  MINSSP: {correct_min}/{total_checked} correct ({correct_min / total_checked * 100:.2f}%)")

    print()
    print("=== reconstruction time by target partial sum n (mean over all rows) ===")
    print("  n     MAXSSP(ms)   MINSSP(ms)")
    step = max(1, UD // 20)  # keep the printout to ~20 rows regardless of U*D
    for n in range(1, UD + 1, step):
        m = np.mean(per_n_max_times[n - 1]) * 1000
        mn = np.mean(per_n_min_times[n - 1]) * 1000
        print(f"  {n:4d}  {m:10.4f}   {mn:10.4f}")


if __name__ == "__main__":
    main()
