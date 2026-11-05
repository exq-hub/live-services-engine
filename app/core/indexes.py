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


"""Vector index abstractions for nearest-neighbour search.

This module defines the `BaseIndex` abstract interface and two concrete
implementations used by the search strategies:

`FaissIndex`
    Wraps a FAISS index file (IVF or HNSW). Supports incremental search
    with automatic probe/efSearch expansion when the initial search scope
    is insufficient to satisfy the requested *k* results after filtering.

`ZarrIndex`
    Performs brute-force dot-product search over Zarr-stored embeddings,
    parallelised across CPU cores. Useful when the dataset fits in chunked
    array storage and an approximate index is not available.

Both implementations accept a ``skip_ids`` set so that already-seen,
excluded, or filter-rejected items can be skipped without post-filtering.
Only ``VectorSearchMixin._search_with_expansion`` ever builds one; every
``search``/``incremental_search`` here only reads it, hence the shared
``frozenset()`` default.
"""

from abc import ABC, abstractmethod
from collections.abc import Set
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from os import process_cpu_count
from itertools import repeat

import zarr
from zarr.storage import ZipStore

import faiss


class BaseIndex(ABC):
    """Abstract base class for all vector index backends.

    Subclasses must implement `load_index`, `search`, and
    `incremental_search`. The optional `is_query_in_state` hook enables
    stateful/resumable search when the backend supports it.
    """

    def __init__(self):
        self.index = None
        """The loaded index object (FAISS index, Zarr array, etc.), or ``None`` if not yet loaded."""

        self.query_state_support: bool = False
        """Whether this backend supports stateful/resumable queries."""

    def __del__(self):
        self.close()

    @abstractmethod
    def load_index(self, model_path: Path) -> None:
        """Load the data model from the given path"""
        pass

    @abstractmethod
    def search(
        self, query: np.ndarray | int, k: int,
        skip_ids: Set[int] = frozenset()
    ) -> Tuple[int, List[int], List[float]]:
        """
        Search for the top k items to the given query vector or query id if using query states.
        Returns a query id and list of the top k items.
        If returned query id is -1, stateful search is not supported.
        """
        pass

    @abstractmethod
    def incremental_search(
        self,
        query: np.ndarray | int,
        k: int,
        skip_ids: Set[int] = frozenset(),
        q_id: int = -1,
        resume: bool = False,
    ) -> Tuple[int, List[int], List[float]]:
        """
        Search for the top k items to the given query vector or query id if using query states.
        Returns a query id and list of the top k items.
        If returned query id is -1, stateful search is not supported.
        """
        pass

    def is_query_in_state(self, query_id: int) -> bool:
        """
        Check if a query state exists for a given query id.
        Override in subclasses if stateful search is supported.
        """
        return False

    def close(self) -> None:
        """Close the index and release resources."""
        self.index = None


def open_zarr_array(path: Path) -> zarr.Array:
    """Open a Zarr array from a `.zip`/`.zipstore` archive or a `.zarr` directory store.

    Unwraps to the `embeddings` array if the store root is a `zarr.Group`.
    Used both for the brute-force `ZarrIndex` search path and for reading
    raw embeddings for relevance feedback (`IndexRepository.get_embeddings_array`)
    """
    suffix = path.suffix.lower()
    if suffix in (".zip", ".zipstore"):
        node = zarr.open(ZipStore(path, mode="r"), mode="r")
    elif suffix == ".zarr":
        node = zarr.open(path, mode="r")
    else:
        raise ValueError(f"Invalid file extension for the data model: {path}")

    if isinstance(node, zarr.Group):
        node = node["embeddings"]
    return node


class ZarrIndex(BaseIndex):
    """Brute-force dot-product search over Zarr-stored embeddings.

    Unlike FAISS, this does not use an approximate nearest-neighbour
    structure. Instead, it loads embeddings from Zarr storage in batches 
    and computes dot products in parallel using a thread pool.
    The results are then globally sorted to return the top-*k* items.

    Supports ``.zarr`` directories, ``.zip`` / ``.zipstore`` archives, and
    Zarr groups containing an ``embeddings`` dataset.
    """

    def __init__(self):
        super().__init__()
        self.batch_size = 500_000

    def load_index(self, model_path: Path):
        self.index = open_zarr_array(model_path)

        if isinstance(self.index, zarr.Group):
            self.index = self.index["embeddings"]

    def search(
        self, query: np.ndarray, k: int,
        skip_ids: Set[int] = frozenset(),
    ) -> Tuple[int, List[int], List[float]]:
        if isinstance(query, int):
            raise ValueError("ZarrIndex does not support query by state id.")

        def process_batch(
            q: np.ndarray,
            start_idx: int,
            end_idx: int,
        ) -> np.ndarray:
            embeddings_batch = self.index[start_idx:end_idx]
            distances = np.dot(embeddings_batch, q.T).reshape(-1)
            return distances

        total_items = self.index.shape[0]
        # Calculate distances in batches and keep top k
        starts = range(0, total_items, self.batch_size)
        ends = [min(start + self.batch_size, total_items) for start in starts]
        mask = np.isin(np.arange(total_items), list(skip_ids), invert=True)
        max_workers = max((process_cpu_count() or 1) - 1, 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # A batch that fails re-raises its error here and fails the search
            results = list(
                executor.map(
                    process_batch,
                    repeat(query),
                    starts,
                    ends,
                )
            )

        total_top_dists = np.concatenate(results)
        arg_sorted_dists = np.argsort(total_top_dists)[::-1]
        top_k = arg_sorted_dists[mask[arg_sorted_dists]][:k].tolist()

        return -1, top_k, total_top_dists[top_k].tolist()

    def incremental_search(
        self,
        query: np.ndarray,
        k: int,
        skip_ids: Set[int] = frozenset(),
        q_id: int = -1,
        resume: bool = False,
    ) -> Tuple[int, List[int], List[float]]:
        return self.search(query, k, skip_ids)


class FaissIndex(BaseIndex):
    """FAISS-backed approximate nearest-neighbour index.

    Automatically detects whether the loaded index is IVF or HNSW and
    sets reasonable default search parameters (``nprobe = 64`` for IVF,
    ``efSearch = 100`` for HNSW). During incremental search, these
    parameters are doubled progressively if the initial search scope
    cannot return enough valid results after ``skip_ids`` filtering.
    """

    def __init__(self):
        super().__init__()
        self.index_type: Optional[str] = None
        """Detected FAISS index type: ``'ivf'``, ``'hnsw'``, or ``'unknown'``."""

    def load_index(self, model_path: Path):
        self.index = faiss.read_index(str(model_path))
        if isinstance(self.index, faiss.IndexIVF): 
            self.index_type = "ivf"
        elif isinstance(self.index, faiss.IndexHNSW):
            self.index_type = "hnsw"
        else:
            self.index_type = "unknown"

        if self.index_type == "ivf":
            self.index.nprobe = 64  # set a reasonable default since faiss default is 1
        elif self.index_type == "hnsw":
            if self.index.hnsw.efSearch < 100:
                self.index.hnsw.efSearch = (
                    100  # set a reasonable default since faiss default is 16
                )

    def search(
        self, query: np.ndarray, k: int, skip_ids: Set[int] = frozenset()
    ) -> Tuple[int, List[int], List[float]]:
        if isinstance(query, int):
            raise ValueError("FaissIndex does not support query by state id.")

        if skip_ids:
            return self.incremental_search(query, k, skip_ids)

        distances, top_k = self.index.search(query.reshape(1, -1), k)
        return -1, top_k[0].tolist(), distances[0].tolist()

    def _search_params(
        self, nprobe: Optional[int], ef_search: Optional[int]
    ) -> Optional[faiss.SearchParameters]:
        """Per-call scope parameters, so widening never mutates the shared index."""
        if self.index_type == "ivf":
            return faiss.SearchParametersIVF(nprobe=nprobe)
        if self.index_type == "hnsw":
            return faiss.SearchParametersHNSW(efSearch=ef_search)
        return None

    def incremental_search(
        self,
        query: np.ndarray,
        k: int,
        skip_ids: Set[int] = frozenset(),
        q_id: int = -1,
        resume: bool = False,
    ):
        ntotal = self.index.ntotal
        if k <= 0 or ntotal == 0:
            return -1, [], []

        query = query.reshape(1, -1)
        curr_k = min(k, ntotal)
        nprobe = self.index.nprobe if self.index_type == "ivf" else None
        ef_search = self.index.hnsw.efSearch if self.index_type == "hnsw" else None

        while True:
            distances, top = self.index.search(
                query, curr_k, params=self._search_params(nprobe, ef_search)
            )
            distances, top = distances[0], top[0]  # [0] since there is only one query
            found = top != -1
            n_found = int(found.sum())

            # Fewer results than asked for means the search scope is too narrow,
            # so widen it until it covers the whole index
            if n_found < curr_k:
                if self.index_type == "ivf" and nprobe < self.index.nlist:
                    nprobe = min(nprobe * 2, self.index.nlist)
                    continue
                if self.index_type == "hnsw" and ef_search < ntotal:
                    ef_search = min(ef_search * 2, ntotal)
                    continue

            # Rebuilt from the full result list each round, since a wider
            # scope can reorder it
            res = [
                (t, d)
                for t, d in zip(top[found].tolist(), distances[found].tolist())
                if t not in skip_ids
            ][:k]

            # Stop once k are found, or once everything has been retrieved,
            # or once the scope is at its widest and still comes up short
            if len(res) == k or curr_k >= ntotal or n_found < curr_k:
                break
            curr_k = min(curr_k * 2, ntotal)

        top_k = [r[0] for r in res]
        distances = [r[1] for r in res]
        return -1, top_k, distances
