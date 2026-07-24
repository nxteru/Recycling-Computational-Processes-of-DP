import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import common_args, run_prefix_binary_flat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    common_args(parser)
    run_prefix_binary_flat(parser.parse_args())
