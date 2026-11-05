# Copyright (C) 2026 Ujjwal Sharma and Omar Shahbaz Khan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for FaissIndex.incremental_search's skip_ids handling, and for
ZarrIndex's parallel brute-force scan.

Covers the cases where the first search round comes up short. 
Results must not repeat across rounds, fewer than k available unskipped 
items must return what exists rather than widening the scope without bound,
and widening must not leak into the shared index's own nprobe/efSearch.

The indexes are built in a separate process and loaded through
FaissIndex.load_index, the same way the app gets them. 
Building in this process would crash on macOS, since torch 
(imported while pytest collects the model tests) and faiss each 
bundle their own OpenMP runtime, and faiss's index building is 
what starts its OpenMP threads.

ZarrIndex scores its rows in batches on a thread pool sized from the
process's CPU count. Its results must match a sequential scan for any
core count, including the 1 or 2 cores of a small CI runner and a count
that cannot be determined.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from app.core import indexes
from app.core.indexes import FaissIndex, ZarrIndex


DIM = 8
N = 5000
# Above load_index's default nprobe of 64, so IVF has room to widen
NLIST = 128

BUILD_SCRIPT = """
import sys
from pathlib import Path

import faiss
import numpy as np

out, dim, n, nlist = Path(sys.argv[1]), *map(int, sys.argv[2:])
x = np.random.default_rng(0).standard_normal((n, dim)).astype("f4")

hnsw = faiss.IndexHNSWFlat(dim, 16)
hnsw.add(x)
faiss.write_index(hnsw, str(out / "hnsw.faiss"))

ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist)
ivf.train(x)
ivf.add(x)
faiss.write_index(ivf, str(out / "ivf.faiss"))

flat = faiss.IndexFlatL2(dim)
flat.add(x)
faiss.write_index(flat, str(out / "unknown.faiss"))
"""


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("faiss")
    subprocess.run(
        [sys.executable, "-c", BUILD_SCRIPT, str(out), str(DIM), str(N), str(NLIST)],
        check=True,
    )
    return out


@pytest.fixture(params=["hnsw", "ivf", "unknown"])
def index(request, index_dir) -> FaissIndex:
    index = FaissIndex()
    index.load_index(index_dir / f"{request.param}.faiss")
    assert index.index_type == request.param
    return index


@pytest.fixture
def query():
    return np.random.default_rng(1).standard_normal(DIM).astype("f4")


def _scope(index: FaissIndex):
    if index.index_type == "hnsw":
        return index.index.hnsw.efSearch
    if index.index_type == "ivf":
        return index.index.nprobe
    return None


def test_skipped_top_hits_are_replaced_without_duplicates(index, query):
    _, top, _ = index.search(query, 10)
    skip = set(top[:5])

    _, result, _ = index.incremental_search(query, 10, skip)

    assert len(result) == 10
    assert len(set(result)) == 10
    assert skip.isdisjoint(result)


def test_fewer_unskipped_than_k_returns_what_exists(index, query):
    skip = set(range(3, N))

    _, result, _ = index.incremental_search(query, 10, skip)

    assert sorted(result) == [0, 1, 2]


def test_everything_skipped_returns_nothing(index, query):
    _, result, _ = index.incremental_search(query, 5, set(range(N)))
    assert result == []


def test_widening_leaves_the_index_scope_untouched(index, query):
    before = _scope(index)

    index.incremental_search(query, 10, set(range(3, N)))

    assert _scope(index) == before


# ZarrIndex.search scores its rows in batches of `batch_size`. 11 rows in
# batches of 3 give four batches (3, 3, 3 and 2) for the thread pool.
ZARR_ROWS = 11
ZARR_BATCH_SIZE = 3


def _zarr_index(path: Path, n_rows: int) -> tuple[ZarrIndex, np.ndarray]:
    """A real Zarr store of `n_rows` rows, loaded the way the app loads it."""
    rows = np.random.default_rng(2).standard_normal((n_rows, DIM)).astype("f4")
    root = zarr.open_group(str(path), mode="w")
    # One chunk per batch, so a test can break a single batch
    root.create_array(
        "embeddings", shape=rows.shape, dtype="f4",
        chunks=(ZARR_BATCH_SIZE, DIM),
    )
    root["embeddings"][:] = rows

    index = ZarrIndex()
    index.load_index(path)
    index.batch_size = ZARR_BATCH_SIZE
    return index, rows


def _assert_matches_a_sequential_scan(index, rows, query, k):
    _, top, distances = index.search(query, k)

    scores = rows @ query
    expected = np.argsort(scores)[::-1][:k]
    assert top == expected.tolist()
    assert np.allclose(distances, scores[expected])


@pytest.mark.parametrize("cores", [None, 1, 2, 8])
def test_scan_matches_a_sequential_scan_for_any_core_count(
    tmp_path, query, monkeypatch, cores
):
    # process_cpu_count() returns None when the count cannot be determined
    monkeypatch.setattr(indexes, "process_cpu_count", lambda: cores)
    index, rows = _zarr_index(tmp_path / "embeddings.zarr", ZARR_ROWS)

    _assert_matches_a_sequential_scan(index, rows, query, 5)


@pytest.mark.parametrize(
    "n_rows",
    [
        10,  # batches of 3, 3, 3 and 1
        1,  # a single batch of 1
    ],
)
def test_scan_handles_a_last_batch_of_one_row(tmp_path, query, n_rows):
    index, rows = _zarr_index(tmp_path / "embeddings.zarr", n_rows)

    _assert_matches_a_sequential_scan(index, rows, query, 5)


def test_scan_raises_the_error_of_a_failed_batch(tmp_path, query):
    path = tmp_path / "embeddings.zarr"
    index, _ = _zarr_index(path, ZARR_ROWS)
    # Corrupt the chunk of rows 3 to 5, so only the second batch fails
    (path / "embeddings" / "c" / "1" / "0").write_bytes(b"not a chunk")

    # The batch's own read error, not a ValueError from np.concatenate
    with pytest.raises(RuntimeError, match="decompression"):
        index.search(query, 5)
