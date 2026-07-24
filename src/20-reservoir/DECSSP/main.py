"""Deterministic reservoir feature map: for each row, dp[i][j] = 1 iff some
prefix of the first i+1 values sums to j (identical dp table to
10-target/DECSSP, with the leading constant "1" term folded into dp[0][0],
which is unconditionally true -- see solve()).

dp is a genuinely binary (0/1) table, unlike MAXSSP/MINSSP's (which track
integer counts), so the whole row can go through one vectorized numpy
bool-array slice-OR instead of a Python-level loop over every j: dp[i] =
dp[i-1] | (dp[i-1] shifted by x[i]).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _binstore
from _stage_timing import StageTiming


def solve(u: int, n: int, values: list[int]) -> np.ndarray:
    dp = np.zeros(u * n + 1, dtype=bool)
    dp[0] = True
    dp[values[0]] = True
    rows = [dp[: u + 1].copy()]
    for i in range(1, n):
        dp[values[i] :] |= dp[: -values[i]]
        rows.append(dp[1 : u * (i + 1) + 1].copy())
    return np.concatenate(rows).astype(np.float64)


def main() -> None:
    timing = StageTiming()
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    with timing.io():
        u, n, t = _binstore.read_header(stdin)
        _binstore.write_header(stdout, u, n, t)
    for _ in range(t):
        with timing.io():
            values = _binstore.read_row(stdin, n, _binstore.INPUT_DTYPE).tolist()
        with timing.compute():
            result = solve(u, n, values)
        with timing.io():
            _binstore.write_row(stdout, result, _binstore.RESERVOIR_DTYPE)
    timing.report()


if __name__ == "__main__":
    main()
