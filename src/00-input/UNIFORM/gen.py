"""Generates SIZE rows of D uniformly random integers in [1, U], each row
sorted descending. Usage: gen.py U D SIZE SEED
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _binstore
from _stage_timing import StageTiming


def main() -> None:
    if len(sys.argv) != 5:
        print(f"usage: {sys.argv[0]} U D SIZE SEED", file=sys.stderr)
        sys.exit(1)
    u = int(sys.argv[1])
    d = int(sys.argv[2])
    size = int(sys.argv[3])
    seed = int(sys.argv[4])

    rng = random.Random(seed)
    timing = StageTiming()
    stdout = sys.stdout.buffer
    with timing.io():
        _binstore.write_header(stdout, u, d, size)
    for _ in range(size):
        with timing.compute():
            row = sorted((rng.randint(1, u) for _ in range(d)), reverse=True)
        with timing.io():
            _binstore.write_row(stdout, row, _binstore.INPUT_DTYPE)
    timing.report()


if __name__ == "__main__":
    main()
