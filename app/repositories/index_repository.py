"""Repository for managing vector search indices and embedding arrays.

`IndexRepository` owns the lifecycle of all per-collection vector indices
(FAISS or Zarr) and embedding stores used by the search strategies. It
provides a uniform interface for:

- Loading and caching CLIP indices from disk.
- Executing nearest-neighbour searches with ``skip_ids`` filtering.
- Opening Zarr embedding arrays for use by the relevance-feedback strategy.
- Checking query-state support for resumable searches (future capability).
"""

import zarr
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from app.core.indexes import BaseIndex, FaissIndex, ZarrIndex

from ..core.exceptions import IndexError


class IndexRepository:
    """Repository for managing vector indices and embeddings."""

    def __init__(self):
        self._clip_indices: Dict[str, BaseIndex] = {}
        """Per-collection CLIP vector indices keyed by collection name."""

        self._embeddings_zarr: Dict[str, str] = {}
        """Per-collection file paths to Zarr embedding arrays for relevance feedback."""

    def load_clip_index(self, collection: str, index_path: str, index_type = "faiss") -> BaseIndex:
        """Load CLIP index for a collection."""
        try:
            if collection in self._clip_indices:
                return self._clip_indices[collection]

            index_file = Path(index_path)
            if not index_file.exists():
                raise IndexError(f"CLIP index file not found: {index_path}")

            if index_type == "faiss":
                self._clip_indices[collection] = FaissIndex()
            elif index_type == "zarr":
                self._clip_indices[collection] = ZarrIndex()
            elif index_type == "ecp":
                raise IndexError("eCP is not currently supported for indices.")
            else:
                raise IndexError(f"Unsupported index type: {index_type}")

            self._clip_indices[collection].load_index(index_file)

            return self._clip_indices[collection]

        except Exception as e:
            raise IndexError(f"Failed to load CLIP index from {index_path}: {e}")

    def set_embeddings_zarr_path(self, collection: str, embeddings_path: str):
        """Set the path to embeddings zarr file for a collection."""
        embeddings_file = Path(embeddings_path)
        if not embeddings_file.exists():
            raise IndexError(f"Embeddings file not found: {embeddings_path}")

        self._embeddings_zarr[collection] = str(embeddings_file)

    def get_clip_index(self, collection: str) -> Optional[BaseIndex]:
        """Get CLIP index for a collection."""
        return self._clip_indices.get(collection)

    def get_embeddings_zarr_path(self, collection: str) -> Optional[str]:
        """Get embeddings zarr path for a collection."""
        return self._embeddings_zarr.get(collection)

    def is_query_in_state_clip(self, collection: str, state: int) -> bool:
        """Check if a query state exists for a collection."""

        index = self.get_clip_index(collection)
        if index is None:
            raise IndexError(f"CLIP index not loaded for collection: {collection}")

        if self._clip_indices[collection].query_state_support:
            return self._clip_indices[collection].is_query_in_state(state)

        return False

    def search_clip(
        self, collection: str, query_vector: np.ndarray, k: int,
        skip_ids: set[int] = set()
        #, q_id: int = -1, resume: bool = False
    ) -> Tuple[int, np.ndarray]:
        """Search CLIP index."""
        index = self.get_clip_index(collection)
        if index is None:
            raise IndexError(f"CLIP index not loaded for collection: {collection}")

        try:
            _, indices, distances = index.search(query_vector, k, skip_ids=skip_ids)
            return indices, distances
        except Exception as e:
            raise IndexError(f"CLIP search failed for collection {collection}: {e}")

    def get_embeddings_array(self, collection: str) -> zarr.Array:
        """Get zarr embeddings array for a collection."""
        zarr_path = self.get_embeddings_zarr_path(collection)
        if zarr_path is None:
            raise IndexError(
                f"No embeddings configured for collection: {collection}"
            )
        store = zarr.storage.ZipStore(zarr_path, mode="r")

        try:
            emb_arr = zarr.open(store, mode="r")["embeddings"]
            return emb_arr
        except Exception as e:
            raise IndexError(
                f"Failed to open embeddings array for collection {collection}: {e}"
            )

    def clear_cache(self, collection: Optional[str] = None):
        """Clear cached indices for a collection or all collections."""
        if collection:
            self._clip_indices.pop(collection, None)
            self._embeddings_zarr.pop(collection, None)
        else:
            self._clip_indices.clear()
            self._embeddings_zarr.clear()
