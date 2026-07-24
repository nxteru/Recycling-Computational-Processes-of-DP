"""Trivial reservoir: use the raw input values directly as features (plus a
leading constant "1" term, matching the bias-column convention every other
reservoir method uses).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _binstore
from _stage_timing import StageTiming


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
            out = [1] + values
        with timing.io():
            _binstore.write_row(stdout, out, _binstore.RESERVOIR_DTYPE)
    timing.report()


if __name__ == "__main__":
    main()
