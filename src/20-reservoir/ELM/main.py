"""Extreme Learning Machine reservoir: random projection + sigmoid.
Usage: main.py --num-features N

The weight matrix is regenerated deterministically from a fixed seed, so
train and test invocations agree automatically as long as they are given
the same N.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _binstore
from _stage_timing import StageTiming


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "--num-features":
        print(f"usage: {sys.argv[0]} --num-features N", file=sys.stderr)
        sys.exit(1)
    num_features = int(sys.argv[2])
    if num_features <= 0:
        print(f"num-features out of range: {num_features}", file=sys.stderr)
        sys.exit(1)

    timing = StageTiming()
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    with timing.io():
        u, n, t = _binstore.read_header(stdin)
        _binstore.write_header(stdout, u, n, t)

    with timing.compute():
        rng = random.Random(2)
        w = [[rng.uniform(-1.0, 1.0) for _ in range(num_features)] for _ in range(n)]
        bias = [rng.uniform(-1.0, 1.0) for _ in range(num_features)]

    for _ in range(t):
        with timing.io():
            values = _binstore.read_row(stdin, n, _binstore.INPUT_DTYPE).tolist()
        with timing.compute():
            x_normalized = [2.0 * v - 1.0 for v in values]  # [1,u] -> [-1,1]
            out = [1.0]
            for node_idx in range(num_features):
                node = bias[node_idx]
                for i in range(n):
                    node += w[i][node_idx] * x_normalized[i]
                out.append(sigmoid(node))
        with timing.io():
            _binstore.write_row(stdout, out, _binstore.RESERVOIR_DTYPE)
    timing.report()


if __name__ == "__main__":
    main()
