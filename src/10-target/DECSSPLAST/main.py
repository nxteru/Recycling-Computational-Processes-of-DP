"""DECSSPLAST target: same dp table as DECSSP, but only the flat last row
(i = n - 1) is emitted.

dp is a genuinely binary (0/1) table, so a whole row packs into a single
Python bignum bitset: bit j set iff dp[i][j] == 1, and dp[i] = dp[i-1] |
(dp[i-1] << x[i]). Only the last row is needed here, so this stays a
bignum instead of the numpy bool array 10-target/DECSSP/main.py uses.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _binstore
from _stage_timing import StageTiming


def solve(u: int, n: int, values: list[int]) -> list[int]:
    bits = (1 << values[0]) | 1  # dp[0][values[0]] = 1, dp[0][0] = 1
    for i in range(1, n):
        bits |= bits << values[i]

    width = n * u
    reversed_bits = format(bits, f"0{width + 1}b")[::-1]
    return [int(c) for c in reversed_bits[1 : width + 1]]


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
