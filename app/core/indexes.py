from abc import ABC, abstractmethod
from collections.abc import Set
from pathlib import Path
from typing import List, Tuple
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from itertools import repeat

import zarr
from zarr.storage import ZipStore

import faiss

class BaseIndex(ABC):
    def __init__(self):
        self.index = None
        self.query_state_support = False

    def __del__(self):
        self.close()

    @abstractmethod
    def load_index(self, model_path: Path) -> None:
        """Load the data model from the given path"""
        pass

    @abstractmethod
    def search(
        self, query: np.ndarray | int, k: int, skip_ids: Set[int] = set()
    ) -> Tuple[int, List[int], List[float]]:
        """
        Search for the top k items to the given query vector or query id if using query states.
        Returns a query id and list of the top k items.
        If returned query id is -1, stateful search is not supported.
        """
        pass

    @abstractmethod
    def incremental_search(
        self, query: np.ndarray | int, k: int, skip_ids: Set[int] = set(), resume: bool = False
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


class ZarrIndex(BaseIndex):
    """Not actually an index, just a simple dot-product search over zarr-stored embeddings."""

    def __init__(self):
        super().__init__()

    def load_index(self, model_path: Path):
        if (
            model_path.suffix.lower() == ".zip"
            or model_path.suffix.lower() == ".zipstore"
        ):
            self.index = zarr.open(ZipStore(model_path, mode="r"), mode="r")
        elif model_path.suffix.lower() == ".zarr":
            self.index = zarr.open(model_path, mode="r")
        else:
            raise ValueError("Invalid file extension for the data model")
        
        if isinstance(self.index, zarr.Group):
            self.index = self.index["embeddings"]

    def search(self, query: np.ndarray, k: int, skip_ids: Set[int] = set()) -> Tuple[int, List[int], List[float]]:
        if isinstance(query, int):
            raise ValueError("ZarrIndex does not support query by state id.")

        def process_chunk(
            q : np.ndarray, 
            start_idx: int, end_idx: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
            embeddings_chunk = self.index[start_idx:end_idx]
            distances = np.dot(embeddings_chunk, q.T).squeeze()
            return distances


        total_items = self.index.shape[0]
        # Calculate distances in chunks and keep top k
        chunk_size = 500_000
        starts = range(0, total_items, chunk_size)
        ends = [min(start + chunk_size, total_items) for start in starts]
        results = []
        mask = np.isin(np.arange(total_items), list(skip_ids), invert=True)
        with ThreadPoolExecutor(max_workers=cpu_count()-2) as executor:
            try:
                results = list(
                    executor.map(
                        process_chunk,
                        repeat(query),
                        starts,
                        ends,
                    )
                )
            except Exception as e:
                print(f"Error during parallel search: {e}")

        total_top_dists = np.concatenate(results)
        arg_sorted_dists = np.argsort(total_top_dists)[::-1]
        top_k = arg_sorted_dists[mask[arg_sorted_dists]][:k].tolist()

        return -1, top_k, total_top_dists[top_k].tolist()
        
    def incremental_search(
        self, 
        query: np.ndarray,
        k: int,
        skip_ids: Set[int] = set(),
        resume: bool = False,
        q_id: int = 0,
    ) -> Tuple[int, List[int]]:
        return self.search(query, k, skip_ids, q_id)
    

class FaissIndex(BaseIndex):
    def __init__(self):
        super().__init__()
        self.index_type = None

    def load_index(self, model_path: Path):
        self.index = faiss.read_index(str(model_path))
        if isinstance(self.index, faiss.IndexIVF) or (
            isinstance(self.index, faiss.IndexPreTransform) and 
            isinstance(faiss.downcast_index(self.index.index), faiss.IndexIVF)
        ):
            self.index_type = 'ivf'
        elif isinstance(self.index, faiss.IndexHNSW):
            self.index_type = 'hnsw'
        else:
            self.index_type = 'unknown'

        if self.index_type == 'ivf':
            self.index.nprobe = 64 # set a reasonable default since faiss default is 1
        elif self.index_type == 'hnsw':
            if self.index.hnsw.efSearch < 100:
                self.index.hnsw.efSearch = 100 # set a reasonable default since faiss default is 16

    def search(self, query: np.ndarray, k: int, skip_ids: Set[int] = set()) -> Tuple[int, List[int], List[float]]:
        if isinstance(query, int):
            raise ValueError("FaissIndex does not support query by state id.")

        if skip_ids:
            return self.incremental_search(query, k, skip_ids)

        distances, top_k = self.index.search(query.reshape(1, -1), k)
        return -1, top_k[0].tolist(), distances[0].tolist()

    def incremental_search(self, query: np.ndarray, k: int, skip_ids: Set[int] = set(), resume: bool = False):
        cnt = 0
        res = []
        curr_k = k
        while cnt < k:
            distances, top = self.index.search(query.reshape(1, -1), curr_k)
            # If the top-k results are less than the current k,
            # the current search scope is not sufficient
            if np.where(top != -1)[0].size < curr_k:
                if self.index_type == 'ivf':
                    self.index.nprobe = self.index.nprobe * 2
                elif self.index_type == 'hnsw':
                    self.index.hnsw.efSearch = self.index.hnsw.efSearch * 2
                continue
            distances, top = distances[0], top[0]
            for idx, t in enumerate(top):  # [0] since there is only one query
                if t not in skip_ids:
                    res.append((t, distances[idx]))
                    cnt += 1
                    if cnt == k:
                        break
            curr_k = curr_k * 2
        top_k = [r[0] for r in res]
        distances = [r[1] for r in res]
        return -1, top_k, distances
