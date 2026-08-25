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


"""Repository for managing vector search indices and embedding arrays.

`IndexRepository` owns the lifecycle of all per-collection, per-index vector
indices (FAISS or Zarr) and embedding stores used by the search strategies.
It holds the validated `LSEConfig` and resolves/loads a given
(collection, index_name) pair straight from it on a cache miss -- there is
no separate imperative "load" step. A collection's default index is loaded
eagerly at startup (see `ApplicationContainer.initialize`); any other index
loads lazily the first time something asks for it (e.g. `search`,
`get_embeddings_array`), or eagerly at startup too if the collection sets
`preload_all_indexes`.

It provides a uniform interface for:

- Resolving and loading indices from config, cached by (collection, index_name).
- Executing nearest-neighbour searches with ``skip_ids`` filtering.
- Opening Zarr embedding arrays for use by the relevance-feedback strategy.
- Checking query-state support for resumable searches (future capability).
"""

from collections.abc import Set
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import zarr

from app.core.config import IndexConfig, LSEConfig
from app.core.indexes import BaseIndex, FaissIndex, ZarrIndex, open_zarr_array

from ..core.exceptions import IndexError


class IndexRepository:
    """Repository for managing vector indices and embeddings."""

    def __init__(self, config: LSEConfig):
        self.config = config
        """Validated LSE configuration, used to resolve and load indexes on demand."""

        self._indices: Dict[Tuple[str, str], BaseIndex] = {}
        """Loaded ANN indexes keyed by (collection, index_name)."""

        self._embeddings_zarr: Dict[Tuple[str, str], str] = {}
        """Loaded embeddings file paths keyed by (collection, index_name), for relevance feedback."""

    def _resolve_index_name(self, collection: str, index_name: Optional[str]) -> str:
        """Default to the collection's default index when none is given."""
        if index_name is not None:
            return index_name
        try:
            collection_config = self.config.collection_configs[collection]
        except KeyError:
            raise IndexError(f"Unknown collection: {collection!r}")
        return collection_config.default_index.name

    def _get_index_config(self, collection: str, index_name: str) -> IndexConfig:
        try:
            collection_config = self.config.collection_configs[collection]
        except KeyError:
            raise IndexError(f"Unknown collection: {collection!r}")

        try:
            return collection_config.get_index(index_name)
        except ValueError as e:
            raise IndexError(f"{e} (collection {collection!r})")

    def get_index(
        self, collection: str, index_name: Optional[str] = None
    ) -> BaseIndex:
        """Get the ANN index for collection/index_name, loading it on demand."""
        index_name = self._resolve_index_name(collection, index_name)
        key = (collection, index_name)
        if key in self._indices:
            return self._indices[key]

        index_config = self._get_index_config(collection, index_name)
        try:
            if index_config.index_type == "faiss":
                index_path = index_config.index_file
                index_obj: BaseIndex = FaissIndex()
            elif index_config.index_type == "zarr":
                index_path = index_config.embeddings_file
                index_obj = ZarrIndex()
            elif index_config.index_type == "ecp":
                raise IndexError("eCP is not currently supported for indices.")
            else:
                raise IndexError(f"Unsupported index type: {index_config.index_type}")

            index_file = Path(index_path)
            if not index_file.exists():
                raise IndexError(f"Index file not found: {index_path}")

            index_obj.load_index(index_file)

        except Exception as e:
            raise IndexError(
                f"Failed to load index {index_name!r} for collection {collection!r}: {e}"
            )

        self._indices[key] = index_obj
        return index_obj

    def get_embeddings_zarr_path(
        self, collection: str, index_name: Optional[str] = None
    ) -> str:
        """Get the embeddings file path for collection/index_name, resolving it on demand."""
        index_name = self._resolve_index_name(collection, index_name)
        key = (collection, index_name)
        if key not in self._embeddings_zarr:
            index_config = self._get_index_config(collection, index_name)
            embeddings_file = Path(index_config.embeddings_file)
            if not embeddings_file.exists():
                raise IndexError(
                    f"Embeddings file not found: {index_config.embeddings_file}"
                )
            self._embeddings_zarr[key] = str(embeddings_file)

        return self._embeddings_zarr[key]

    def preload(self, collection: str, index_name: Optional[str] = None) -> None:
        """Eagerly load the ANN index and embeddings path for collection/index_name."""
        self.get_index(collection, index_name)
        self.get_embeddings_zarr_path(collection, index_name)

    def is_query_in_state(
        self, collection: str, state: int, index_name: Optional[str] = None
    ) -> bool:
        """Check if a query state exists for collection/index_name."""
        index = self.get_index(collection, index_name)

        if index.query_state_support:
            return index.is_query_in_state(state)

        return False

    def search(
        self,
        collection: str,
        query_vector: np.ndarray,
        k: int,
        skip_ids: Set[int] = set(),
        index_name: Optional[str] = None,
        # , q_id: int = -1, resume: bool = False
    ) -> Tuple[int, np.ndarray]:
        """Search the ANN index for collection/index_name."""
        index = self.get_index(collection, index_name)

        try:
            _, indices, distances = index.search(query_vector, k, skip_ids=skip_ids)
            return indices, distances
        except Exception as e:
            raise IndexError(f"Search failed for collection {collection}: {e}")

    def get_embeddings_array(
        self, collection: str, index_name: Optional[str] = None
    ) -> zarr.Array:
        """Get the raw Zarr embeddings array for collection/index_name."""
        zarr_path = self.get_embeddings_zarr_path(collection, index_name)

        try:
            emb_arr = open_zarr_array(Path(zarr_path))
            return emb_arr
        except Exception as e:
            raise IndexError(
                f"Failed to open embeddings array for collection {collection}: {e}"
            )

    def clear_cache(self, collection: Optional[str] = None):
        """Clear cached indices for a collection or all collections."""
        if collection:
            for cache in (self._indices, self._embeddings_zarr):
                for key in [k for k in cache if k[0] == collection]:
                    cache.pop(key, None)
        else:
            self._indices.clear()
            self._embeddings_zarr.clear()
