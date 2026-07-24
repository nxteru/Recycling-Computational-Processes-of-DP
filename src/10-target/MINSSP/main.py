"""MINSSP target: dp[i][j] = min count of elements (among the first i+1
raw values) that sum to j, or 0 if unreachable.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _binstore
from _stage_timing import StageTiming


def solve(u: int, n: int, values: list[int]) -> np.ndarray:
    dp = np.zeros((n, u * n + 1))
    dp[0][values[0]] = 1
    res = [dp[0][1 : u + 1].copy()]
    for i in range(1, n):
        for j in range((i + 1) * u + 1):
            min_val = n + 1
            if dp[i - 1][j] != 0:
                min_val = min(min_val, dp[i - 1][j])
            if j - values[i] >= 0:
                if j == values[i] or dp[i - 1][j - values[i]] != 0:
                    min_val = min(min_val, dp[i - 1][j - values[i]] + 1)
            if min_val != n + 1:
                dp[i][j] = min_val
        res.append(dp[i][1 : u * (i + 1) + 1].copy())
    return np.concatenate(res)


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
            _binstore.write_row(stdout, result, _binstore.TARGET_DTYPE)
    timing.report()


if __name__ == "__main__":
    main()
