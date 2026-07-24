"""Binary row-streaming format shared by every stage script in src/{00-input,
10-target,20-reservoir} and the loaders in src/30-regression, src/40-experiment.

File layout: one ASCII header line "U N T\n", followed by T*W raw
little-endian values of a fixed dtype (row-major, no delimiters). W (row
width) isn't stored -- it's recovered from (file size - header length) /
T / itemsize, the binary equivalent of the old text format's
`.reshape(size, -1)`.

Replaces the previous whitespace-text format (`f.read().split()` +
np.fromiter), whose Python-level str->float parsing dominated pipeline
wall time (regression tasks spent up to ~90% of their time re-parsing
multi-hundred-MB text files -- see the io_sec/compute_sec split each
stage script reports via _stage_timing.StageTiming). load_matrix/load_flat
return memmaps, so a caller that only needs a slice (e.g. regression's
use_size/reservoir_size) only pages in the bytes it actually touches,
instead of the whole file.
"""
import os

import numpy as np

INPUT_DTYPE = np.dtype("int32")
TARGET_DTYPE = np.dtype("float64")
RESERVOIR_DTYPE = np.dtype("float32")
WOUT_DTYPE = np.dtype("float32")


def read_header(stream) -> tuple[int, int, int]:
    u, n, t = (int(v) for v in stream.readline().split())
    return u, n, t


def write_header(stream, u: int, n: int, t: int) -> None:
    stream.write(f"{u} {n} {t}\n".encode("ascii"))


def read_row(stream, width: int, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    return np.frombuffer(stream.read(width * dtype.itemsize), dtype=dtype)


def write_row(stream, row, dtype) -> None:
    stream.write(np.asarray(row, dtype=dtype).tobytes())


def _header_len_and_count(path) -> tuple[int, int]:
    with open(path, "rb") as f:
        header = f.readline()
    return len(header), int(header.split()[2])


def row_width(path, dtype) -> int:
    """Row width of a header+binary-body store file, without reading any
    of the bulk data -- just the header line plus a stat() call."""
    dtype = np.dtype(dtype)
    offset, t = _header_len_and_count(path)
    return (os.path.getsize(path) - offset) // dtype.itemsize // t


def load_matrix(path, dtype) -> np.memmap:
    """Memory-map a header+binary-body store file as a (T, W) array."""
    dtype = np.dtype(dtype)
    offset, t = _header_len_and_count(path)
    width = (os.path.getsize(path) - offset) // dtype.itemsize // t
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(t, width))


def load_flat(path, dtype) -> np.memmap:
    """Memory-map a header-less binary file (regression's .wout output),
    whose shape only the caller knows (see 40-experiment/_common.py's
    load_wout, which reshapes by the n_cols it already has on hand)."""
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")
