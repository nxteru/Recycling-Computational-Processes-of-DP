"""Tiny io/compute timing accumulator shared by every stage script (src/
00-input, 10-target, 20-reservoir, 30-regression, 40-experiment). Each
script reports its own io_sec/compute_sec breakdown as a final stderr line,
which pipeline/runner.py parses and records in meta.json/manifest.jsonl
alongside duration_sec (measured by the runner around the whole subprocess
call, so it still includes process startup and this reporting itself)."""
import sys
import time
from contextlib import contextmanager


class StageTiming:
    def __init__(self) -> None:
        self.io_sec = 0.0
        self.compute_sec = 0.0

    @contextmanager
    def io(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.io_sec += time.perf_counter() - start

    @contextmanager
    def compute(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.compute_sec += time.perf_counter() - start

    def report(self) -> None:
        print(f"TIMING io_sec={self.io_sec:.6f} compute_sec={self.compute_sec:.6f}", file=sys.stderr)
