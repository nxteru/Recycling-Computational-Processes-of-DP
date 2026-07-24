import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import common_args, run_prefix_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    common_args(parser, needs_existence=True, needs_input=True)
    run_prefix_value(parser.parse_args(), direction="min")
