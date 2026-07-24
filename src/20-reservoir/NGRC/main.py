"""NGRC-style reservoir feature map: degree-d monomials of the raw input
values (products of subsets with repetition). Feature 0 is always the
constant "1" term; degrees 1..max_degree are appended after it.

Usage:
  main.py --max-degree D          generate (reads U D T header + T rows of
                                   D input values from stdin, as usual)
  main.py --degree-sizes D        report, for degree 0..D, the cumulative
                                   feature count up to and including that
                                   degree (one "<degree> <count>" line
                                   each) -- only needs the U D T header
                                   line from stdin, no row data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _binstore
from _stage_timing import StageTiming


def dfs(idx: int, n: int, deg: int, product: int, values: list[int], out: list[int]) -> None:
    if deg == 0:
        out.append(product)
        return
    if idx == n:
        return
    dfs(idx, n, deg - 1, product * values[idx], values, out)
    dfs(idx + 1, n, deg, product, values, out)


def solve(n: int, max_degree: int, values: list[int]) -> list[int]:
    res = [1]
    for d in range(1, max_degree + 1):
        degree_terms: list[int] = []
        dfs(0, len(values), d, 1, values, degree_terms)
        res.extend(degree_terms)
    return res


def print_degree_sizes(n: int, max_degree: int) -> None:
    placeholder_values = [1] * n
    cumulative = 1
    print(0, cumulative)
    for d in range(1, max_degree + 1):
        degree_terms: list[int] = []
        dfs(0, n, d, 1, placeholder_values, degree_terms)
        cumulative += len(degree_terms)
        print(d, cumulative)


def main() -> None:
    args = sys.argv
    if len(args) == 3 and args[1] == "--degree-sizes":
        max_degree = int(args[2])
        # Binary-body input file (see _binstore) -- read only the ASCII
        # header line via the raw buffer, not text-mode stdin, since
        # text-mode readline() may read ahead into the binary row data and
        # fail to decode it as UTF-8.
        _u, n, _t = _binstore.read_header(sys.stdin.buffer)
        print_degree_sizes(n, max_degree)
        return
    if len(args) == 3 and args[1] == "--max-degree":
        max_degree = int(args[2])
        timing = StageTiming()
        stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
        with timing.io():
            u, n, t = _binstore.read_header(stdin)
            _binstore.write_header(stdout, u, n, t)
        for _ in range(t):
            with timing.io():
                values = _binstore.read_row(stdin, n, _binstore.INPUT_DTYPE).tolist()
            with timing.compute():
                result = solve(n, max_degree, values)
            with timing.io():
                _binstore.write_row(stdout, result, _binstore.RESERVOIR_DTYPE)
        timing.report()
        return
    print(f"usage: {args[0]} --max-degree D   (generate)", file=sys.stderr)
    print(f"       {args[0]} --degree-sizes D  (report cumulative feature counts per degree)", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
