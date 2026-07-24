"""Shared scoring logic for all 40-experiment/{target}/main.py entry points.
There are 4 algorithmic shapes among the 6 target methods:

- "prefix_binary": nested (i, j) prefix-sum membership classification
  (DECSSP).
- "prefix_binary_flat": same idea but the target only stores the flat last
  row (DECSSPLAST).
- "prefix_value": nested (i, j) value regression + existence classification,
  plus a full greedy tour reconstruction check (MAXSSP/MINSSP).
- "prefix_value_flat": flat value regression + existence classification,
  no tour reconstruction (MAXSSPLAST/MINSSPLAST).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import store
import _binstore
from _stage_timing import StageTiming


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's kappa, without sklearn's hard requirement that both label
    sets contain >1 unique value: many cells here are trivially all-0 (e.g.
    no valid subset exists at that size), where kappa is conventionally
    undefined (0/0) -- reported as 1.0 when that trivial agreement is
    total (consistent with accuracy=1.0 there), else via the standard
    formula."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(labels) < 2:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    return float(cohen_kappa_score(y_true, y_pred, labels=labels))


def load_matrix(path: str, dtype) -> np.ndarray:
    return _binstore.load_matrix(path, dtype)


def load_wout(path: str, n_cols: int) -> np.ndarray:
    """wout files are stored as float32 already (see 30-regression/
    _common.py's run()): every value only ever feeds a dot product that
    immediately gets round()'d or thresholded at 0.5 (see reconstruct()
    and the predict_bin/predict closures below), so float64 precision was
    never observed -- storing (and here, loading) at half the width halves
    both memory traffic and every downstream dot product's cost."""
    return _binstore.load_flat(path, _binstore.WOUT_DTYPE).reshape(n_cols, -1)


def load_reservoir_test_matrix(path: str, reservoir_size: int) -> np.ndarray:
    """Load a reservoir test file and slice the first reservoir_size
    columns, same convention as regression/_common.py's fit(). Validates the
    slice actually fits -- numpy slicing past an array's width silently
    returns fewer columns instead of raising, which would otherwise let a
    misconfigured reservoir_size train/evaluate on a truncated feature set
    without any error. Reservoir files are stored as float32 (see
    _binstore.RESERVOIR_DTYPE): reservoir features are 0/1 dp cells,
    exactly representable in float32, and only ever feed dot products
    against float32 wout rows."""
    full = _binstore.load_matrix(path, _binstore.RESERVOIR_DTYPE)
    if reservoir_size > full.shape[1]:
        raise ValueError(f"reservoir_size={reservoir_size} exceeds available columns {full.shape[1]} in {path}")
    return full[:, :reservoir_size]


def prefix_index_map(U: int, D: int) -> np.ndarray:
    """index[i][j] = flat column of the j-th prefix-sum value using the
    first i+1 raw values, for j in 1..(i+1)*U. Matches the nested layout
    that target/{DECSSP,MAXSSP,MINSSP}/main.py print in."""
    index = np.zeros((D, U * D + 1), dtype=int)
    cnt = 0
    for i in range(D):
        for j in range(1, (i + 1) * U + 1):
            index[i][j] = cnt
            cnt += 1
    return index


def stack_wout(wout_bin: np.ndarray, wout_val: np.ndarray) -> np.ndarray:
    """Stack the existence and value weight matrices into one (n_cols, 2,
    reservoir_size) array, so a single query can fetch both the existence
    score and the value score in one np.dot call instead of two --
    reconstruct() needs both scores against the same reservoir vector up
    to twice per backward step, and a dot product's fixed dispatch
    overhead dominates its actual flops at reservoir_size ~500.

    The .astype(np.float32) is a no-op when load_wout already produced
    float32 (the common case) and a safety net otherwise. Callers should
    also pass a float32 reservoir (see load_reservoir_test_matrix), or the
    dot product upcasts back to float64."""
    return np.stack([wout_bin, wout_val], axis=1).astype(np.float32)


def reconstruct(direction: str, wout_stacked: np.ndarray, index: np.ndarray, U: int, D: int, reservoir, values, n: int):
    """Greedy backward reconstruction of the subset achieving target
    partial sum n, using only *predicted* existence/value at each step
    (direction is "max" or "min"). Returns the `use` 0/1 array (length D),
    or None if the existence check predicts n isn't reachable at all --
    callers that need the ground-truth-agreement case (n truly
    unreachable) handle that themselves, since it depends on data reconstruct()
    doesn't have."""

    def predict_val(i, j):
        if j == 0:
            return 0
        # index[i, j], not index[i][j]: one fancy-index op instead of two
        # chained __getitem__ calls (the second builds an intermediate view).
        exists_score, val_score = wout_stacked[index[i, j]] @ reservoir
        return int(round(val_score)) if exists_score > 0.5 else 0

    exists_score = wout_stacked[index[D - 1, n]][0] @ reservoir
    if not (exists_score > 0.5):
        return None

    now = n
    use = np.zeros(D)
    for i in range(D - 2, -1, -1):
        if now == 0:
            break
        no = predict_val(i, now)
        if now - values[i + 1] >= 0:
            yes = predict_val(i, now - values[i + 1])
            better = (yes + 1 > no) if direction == "max" else (yes + 1 < no)
            if yes != 0 or now == values[i + 1]:
                if no == 0 or better:
                    now = now - values[i + 1]
                    use[i + 1] = 1
    if values[0] == now:
        use[0] = 1
    return use


def common_args(parser: argparse.ArgumentParser, needs_existence: bool = False, needs_input: bool = False) -> None:
    parser.add_argument("--U", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--target-test-file", required=True)
    parser.add_argument("--reservoir-test-file", required=True)
    parser.add_argument("--wout-file", required=True)
    parser.add_argument("--reservoir-size", type=int, required=True)
    parser.add_argument("--use-size", type=int, required=True)
    parser.add_argument("--train-pool-size", type=int, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--input-method", required=True)
    parser.add_argument("--target-method", required=True)
    parser.add_argument("--reservoir-method", required=True)
    parser.add_argument("--result-out", required=True)
    if needs_existence:
        parser.add_argument("--existence-wout-file", required=True)
    if needs_input:
        parser.add_argument("--input-test-file", required=True)


def write_results(args: argparse.Namespace, metrics: list[tuple[str, object]], timing: StageTiming) -> None:
    params = {
        "input_method": args.input_method,
        "target_method": args.target_method,
        "reservoir_method": args.reservoir_method,
        "U": args.U,
        "D": args.D,
        "train_pool_size": args.train_pool_size,
        "use_size": args.use_size,
        "test_size": args.test_size,
        "reservoir_size": args.reservoir_size,
    }
    computed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    git_commit = store.git_commit()
    with timing.io():
        with open(args.result_out, "w") as f:
            for metric_name, metric_value in metrics:
                row = dict(params)
                row["metric_name"] = metric_name
                row["metric_value"] = metric_value
                row["computed_at"] = computed_at
                row["git_commit"] = git_commit
                f.write(json.dumps(row) + "\n")
    timing.report()


# ---- prefix_binary (DECSSP) -------------------------------------------------

def run_prefix_binary(args: argparse.Namespace) -> None:
    timing = StageTiming()
    U, D = args.U, args.D
    with timing.io():
        Y = load_matrix(args.target_test_file, _binstore.TARGET_DTYPE).astype(np.int64)
        dp = load_reservoir_test_matrix(args.reservoir_test_file, args.reservoir_size)
        wout = load_wout(args.wout_file, Y.shape[1])
    index = prefix_index_map(U, D)

    def predict_bin(i, j, reservoir):
        return np.dot(wout[index[i][j]], reservoir) > 0.5

    test_size = Y.shape[0]
    with timing.compute():
        score_table = np.zeros((D, U * D))
        for t in range(test_size):
            for i in range(D):
                for j in range(1, (i + 1) * U + 1):
                    score_table[i][j - 1] += Y[t][index[i][j]] == predict_bin(i, j, dp[t])
        score_table /= test_size
        cells = [score_table[i][j - 1] for i in range(D) for j in range(1, (i + 1) * U + 1)]
        accuracy_mean = float(np.mean(cells))

    write_results(
        args,
        [
            ("accuracy_mean", accuracy_mean),
            ("accuracy_by_cell", score_table.tolist()),
        ],
        timing,
    )


# ---- prefix_binary_flat (DECSSPLAST) ---------------------------------------

def run_prefix_binary_flat(args: argparse.Namespace) -> None:
    timing = StageTiming()
    U, D = args.U, args.D
    with timing.io():
        Y = load_matrix(args.target_test_file, _binstore.TARGET_DTYPE).astype(np.int64)
        dp = load_reservoir_test_matrix(args.reservoir_test_file, args.reservoir_size)
        wout = load_wout(args.wout_file, Y.shape[1])

    def predict_bin(ind, reservoir):
        return np.dot(wout[ind], reservoir) > 0.5

    test_size = Y.shape[0]
    with timing.compute():
        score_table = np.zeros(U * D)
        for t in range(test_size):
            for j in range(1, D * U + 1):
                score_table[j - 1] += Y[t][j - 1] == predict_bin(j - 1, dp[t])
        score_table /= test_size
        accuracy_mean = float(np.mean(score_table))

    write_results(
        args,
        [
            ("accuracy_mean", accuracy_mean),
            ("accuracy_by_cell", score_table.tolist()),
        ],
        timing,
    )


# ---- prefix_value (MAXSSP/MINSSP) ------------------------------------------

def run_prefix_value(args: argparse.Namespace, direction: str) -> None:
    timing = StageTiming()
    U, D = args.U, args.D
    with timing.io():
        raw_values = load_matrix(args.input_test_file, _binstore.INPUT_DTYPE).astype(np.int64)
        Y = load_matrix(args.target_test_file, _binstore.TARGET_DTYPE).astype(np.int64)
        dp = load_reservoir_test_matrix(args.reservoir_test_file, args.reservoir_size)
        wout = load_wout(args.wout_file, Y.shape[1])
        wout_bin = load_wout(args.existence_wout_file, Y.shape[1])
    index = prefix_index_map(U, D)

    def predict(i, j, reservoir):
        return np.round(np.dot(wout[index[i][j]], reservoir)).astype(int)

    def predict_bin(i, j, reservoir):
        return np.dot(wout_bin[index[i][j]], reservoir) > 0.5

    def predict_val(t, i, j):
        if j == 0:
            return 0
        return predict_bin(i, j, dp[t]) * predict(i, j, dp[t])

    test_size = Y.shape[0]
    score_table = np.zeros((D, U * D))
    kappa_table = np.zeros((D, U * D))
    for i in range(D):
        for j in range(1, (i + 1) * U + 1):
            y_true = np.array([Y[t][index[i][j]] for t in range(test_size)])
            y_pred = np.array([predict_val(t, i, j) for t in range(test_size)])
            score_table[i][j - 1] = np.mean(y_true == y_pred)
            kappa_table[i][j - 1] = cohens_kappa(y_true, y_pred)
    value_cells = [score_table[i][j - 1] for i in range(D) for j in range(1, (i + 1) * U + 1)]
    kappa_cells = [kappa_table[i][j - 1] for i in range(D) for j in range(1, (i + 1) * U + 1)]
    wout_stacked = stack_wout(wout_bin, wout)
    with timing.compute():
        recon_table = np.zeros(U * D)
        for t in range(test_size):
            for n in range(1, U * D + 1):
                use = reconstruct(direction, wout_stacked, index, U, D, dp[t], raw_values[t], n)
                if use is None:
                    recon_table[n - 1] += Y[t][index[D - 1][n]] == 0
                    continue
                recon_table[n - 1] += np.dot(use, raw_values[t]) == n and np.sum(use) == Y[t][index[D - 1][n]]
        recon_table /= test_size

    value_accuracy_mean = float(np.mean(value_cells))
    value_kappa_mean = float(np.mean(kappa_cells))
    reconstruction_accuracy_mean = float(np.mean(recon_table))

    write_results(
        args,
        [
            ("value_accuracy_mean", value_accuracy_mean),
            ("value_accuracy_by_cell", score_table.tolist()),
            ("value_kappa_mean", value_kappa_mean),
            ("value_kappa_by_cell", kappa_table.tolist()),
            ("reconstruction_accuracy_mean", reconstruction_accuracy_mean),
            ("reconstruction_accuracy_by_cell", recon_table.tolist()),
        ],
        timing,
    )


# ---- prefix_value_flat (MAXSSPLAST/MINSSPLAST) -----------------------------

def run_prefix_value_flat(args: argparse.Namespace) -> None:
    timing = StageTiming()
    U, D = args.U, args.D
    with timing.io():
        Y = load_matrix(args.target_test_file, _binstore.TARGET_DTYPE).astype(np.int64)
        dp = load_reservoir_test_matrix(args.reservoir_test_file, args.reservoir_size)
        wout = load_wout(args.wout_file, Y.shape[1])
        wout_bin = load_wout(args.existence_wout_file, Y.shape[1])

    def predict(ind, reservoir):
        return np.round(np.dot(wout[ind], reservoir)).astype(int)

    def predict_bin(ind, reservoir):
        return np.dot(wout_bin[ind], reservoir) > 0.5

    def predict_val(t, j):
        return predict_bin(j, dp[t]) * predict(j, dp[t])

    test_size = Y.shape[0]
    with timing.compute():
        score_table = np.zeros(U * D)
        kappa_table = np.zeros(U * D)
        for j in range(1, D * U + 1):
            y_true = Y[:, j - 1]
            y_pred = np.array([predict_val(t, j - 1) for t in range(test_size)])
            score_table[j - 1] = np.mean(y_true == y_pred)
            kappa_table[j - 1] = cohens_kappa(y_true, y_pred)
        value_accuracy_mean = float(np.mean(score_table))
        value_kappa_mean = float(np.mean(kappa_table))

    write_results(
        args,
        [
            ("value_accuracy_mean", value_accuracy_mean),
            ("value_accuracy_by_cell", score_table.tolist()),
            ("value_kappa_mean", value_kappa_mean),
            ("value_kappa_by_cell", kappa_table.tolist()),
        ],
        timing,
    )
