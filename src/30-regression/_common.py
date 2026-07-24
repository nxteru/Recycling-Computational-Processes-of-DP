"""Shared ridge-regression fitting logic for all 30-regression/{target}/LR.py
entry points. There are two algorithmic shapes among the 6 target methods:

- "direct": fit every target column directly (DECSSP, DECSSPLAST).
- "value": drop rows where the target column is 0 before fitting, since 0
  means "no valid answer for this (i, j)" rather than a real value to
  regress on (MAXSSP, MINSSP, MAXSSPLAST, MINSSPLAST).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _binstore
from _stage_timing import StageTiming


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ridge_tilde(X_gpu: torch.Tensor, rhs_gpu: torch.Tensor, identity: torch.Tensor) -> tuple:
    """Stack the sqrt(lambda)*I ridge-regularization block below X (and the
    matching zero block below the right-hand side), for a shared X_gpu/
    identity -- shape-checked so it works whether rhs_gpu is a single
    column (y) or many (Y)."""
    lambda_reg = 1e-2
    X_tilde = torch.vstack([X_gpu, lambda_reg * identity])
    zeros_shape = (identity.shape[0],) if rhs_gpu.dim() == 1 else (identity.shape[0], rhs_gpu.shape[1])
    rhs_tilde = torch.cat([rhs_gpu, torch.zeros(*zeros_shape, device=rhs_gpu.device)], dim=0)
    return X_tilde, rhs_tilde


def solve_batch(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Ridge-regression solve for every column of Y against the *same* X
    (mode == "direct", where no per-column row masking is needed): one
    factorization shared across all target columns via torch.linalg.lstsq's
    native multi-column right-hand side, instead of Y.shape[1] separate
    factorizations of an identical matrix."""
    device = _device()
    X_gpu = torch.from_numpy(X.astype(np.float32)).to(device)
    Y_gpu = torch.from_numpy(Y.astype(np.float32)).to(device)
    identity = torch.eye(X.shape[1], device=device)
    X_tilde, Y_tilde = _ridge_tilde(X_gpu, Y_gpu, identity)
    solution = torch.linalg.lstsq(X_tilde, Y_tilde, driver="gels").solution
    if device.type == "cuda":
        torch.cuda.synchronize()
    return solution.T.cpu().numpy()


def fit(mode: str, target_file: str, reservoir_file: str, use_size: int, reservoir_size: int, timing: StageTiming) -> np.ndarray:
    # load_matrix memory-maps rather than reading the whole file: opening
    # the map is essentially free, and slicing to [:use_size, :reservoir_size]
    # below means only the bytes actually used ever get paged in from disk
    # (deferred to wherever a caller first touches them, typically the
    # .astype(np.float32) calls under timing.compute() further down).
    with timing.io():
        Y_full = _binstore.load_matrix(target_file, _binstore.TARGET_DTYPE)
        X_full = _binstore.load_matrix(reservoir_file, _binstore.RESERVOIR_DTYPE)
    if use_size > Y_full.shape[0]:
        raise ValueError(f"use_size={use_size} exceeds available rows {Y_full.shape[0]} in {target_file}")
    if reservoir_size > X_full.shape[1]:
        raise ValueError(f"reservoir_size={reservoir_size} exceeds available columns {X_full.shape[1]} in {reservoir_file}")
    Y = Y_full[:use_size]
    X = X_full[:use_size, :reservoir_size]

    if mode == "direct":
        # Same X for every column -- one batched factorization instead of
        # Y.shape[1] redundant ones (see solve_batch).
        with timing.compute():
            return solve_batch(X, Y)

    if mode != "value":
        raise ValueError(f"unknown mode: {mode}")

    # "value" mode: each column masks out its own zero rows, so X isn't
    # shared across columns and can't go through one factorization -- but
    # moving X to the GPU once (instead of re-transferring a masked copy
    # every column) and masking there still cuts real overhead.
    device = _device()
    with timing.compute():
        X_gpu_full = torch.from_numpy(X.astype(np.float32)).to(device)
        identity = torch.eye(X.shape[1], device=device)
    rows = []
    start = time.time()
    for i in range(Y.shape[1]):
        with timing.compute():
            y_col = Y[:, i]
            mask = y_col != 0
            X_gpu = X_gpu_full[torch.from_numpy(mask).to(device)]
            y_gpu = torch.from_numpy(y_col[mask].astype(np.float32)).to(device)
            X_tilde, y_tilde = _ridge_tilde(X_gpu, y_gpu, identity)
            solution = torch.linalg.lstsq(X_tilde, y_tilde, driver="gels").solution
            rows.append(solution.cpu().numpy())
        elapsed = time.time() - start
        print(
            f"{i + 1}/{Y.shape[1]} elapsed={elapsed:.1f}s eta={elapsed / (i + 1) * (Y.shape[1] - i - 1):.1f}s",
            file=sys.stderr,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.array(rows)


def run(mode: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--reservoir-file", required=True)
    parser.add_argument("--use-size", type=int, required=True)
    parser.add_argument("--reservoir-size", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    timing = StageTiming()
    wout = fit(mode, args.target_file, args.reservoir_file, args.use_size, args.reservoir_size, timing)
    with timing.io():
        np.asarray(wout, dtype=_binstore.WOUT_DTYPE).tofile(args.out)
    timing.report()
