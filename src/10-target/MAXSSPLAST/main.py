"""MAXSSPLAST target: same dp table as MAXSSP, but only the flat last row
(i = n - 1) is emitted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _binstore
from _stage_timing import StageTiming


def solve(u: int, n: int, values: list[int]) -> list[int]:
    dp = [[0] * (n * u + 1) for _ in range(n)]
    dp[0][values[0]] = 1
    for i in range(1, n):
        for j in range((i + 1) * u + 1):
            max_val = -1
            if dp[i - 1][j] != 0:
                max_val = max(max_val, dp[i - 1][j])
            if j - values[i] >= 0 and (j == values[i] or dp[i - 1][j - values[i]] != 0):
                max_val = max(max_val, dp[i - 1][j - values[i]] + 1)
            if max_val != -1:
                dp[i][j] = max_val
    i = n - 1
    return [dp[i][j] for j in range(1, (i + 1) * u + 1)]


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
