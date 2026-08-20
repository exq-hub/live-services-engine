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


"""Shared vector-search machinery, plus the text-to-vector search strategy
built on top of it.

`VectorSearchMixin` is modality-agnostic: given a query vector (however
it was produced -- a text encoder today), it resolves which of a
collection's indexes belongs to a given embedding family, builds the
skip-id set from seen/excluded/filtered media, searches the ANN index,
and maps index positions back to media IDs. It doesn't inherit
`TextSearchStrategy`, so a future strategy whose query isn't `text`
(e.g. search-by-example-image against a CLIP vision encoder) can reuse
it directly without being forced into a text-taking contract it can't
satisfy.

`EmbeddingSearchStrategy` builds the text-query search on top of that:
subclasses (`CLIPSearchStrategy`, `TextEmbeddingSearchStrategy`) supply
`embedding_type` and `_encode_text`, since that's the one part that
genuinely differs by family -- open_clip's tokenize-then-forward vs.
sentence-transformers' `.encode()`.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional

import numpy as np

from app.core.config import IndexConfig
from app.core.models import ModelManager
from app.repositories.database_repository import DatabaseRepository
from app.repositories.index_repository import IndexRepository

from .base import TextSearchStrategy
from ..schemas import ActiveFilters
from ..core.exceptions import SearchError


class VectorSearchMixin:
    """Expanding-radius ANN search + ID mapping for an already-resolved
    index, independent of how the query vector was produced or which
    embedding family that index belongs to.
    """

    index_repo: IndexRepository
    database_repo: DatabaseRepository

    def _build_excluded_set(self, collection: str, excluded: List[int]) -> set:
        """Build set of excluded items including related items."""
        if not excluded:
            return set()

        excluded_set = set(excluded)
        database_repo = self.database_repo
        for exc in excluded:
            item = database_repo.get_item(collection, exc)
            related = database_repo.get_related_items(collection, item["group"])
            excluded_set.update(related)

        return excluded_set

    async def _search_with_expansion(
        self,
        collection: str,
        index_name: str,
        query_features: np.ndarray,
        n: int,
        seen_set: set,
        excluded_set: set,
        filters: Optional[ActiveFilters] = None,
    ) -> List[int]:
        """Search with expanding radius until sufficient results."""
        active_n = n
        total_items = self.database_repo.get_total_items(collection, index_name)
        skip_ids = set()
        if len(seen_set) != 0:
            skip_ids.update(
                self.database_repo.get_index_ids(collection, list(seen_set), index_name)
            )
        if len(excluded_set) != 0:
            skip_ids.update(
                self.database_repo.get_index_ids(
                    collection, list(excluded_set), index_name
                )
            )

        if filters:
            passed_ids = self.database_repo.get_filtered_media_ids(collection, filters)
            # NOTE: Can use the size of passed_ids to determine if index search is needed
            #       If it is lower than a certain threshold we can search through the subset with
            #       the zarr embeddings array directly
            index_passed_ids = self.database_repo.get_index_ids(
                collection, passed_ids, index_name
            )
            index_skip_ids = set(range(total_items)) - set(index_passed_ids)
            skip_ids.update(index_skip_ids)

        indices, _ = self.index_repo.search_clip(
            collection, query_features, active_n, skip_ids=skip_ids, index_name=index_name
        )
        suggestions = self.database_repo.get_media_ids(collection, indices, index_name)

        return suggestions


class EmbeddingSearchStrategy(VectorSearchMixin, TextSearchStrategy, ABC):
    """Base for text-to-vector search strategies, one per embedding family."""

    embedding_type: ClassVar[str]
    """Which `IndexConfig.embedding_type` this strategy searches."""

    def __init__(
        self,
        model_manager: ModelManager,
        index_repository: IndexRepository,
        database_repository: DatabaseRepository,
    ):
        self.model_manager: ModelManager = model_manager
        """Model manager providing this family's text encoders and the device."""

        self.index_repo: IndexRepository = index_repository
        """Index repository for executing nearest-neighbour vector searches."""

        self.database_repo: DatabaseRepository = database_repository
        """Database repository for ID mapping, filters, and exclusion lookups."""

    def _resolve_index(
        self, collection: str, index_name: Optional[str] = None
    ) -> IndexConfig:
        """Which of the collection's indexes belongs to this family.

        An explicit `index_name` must itself be of this strategy's
        `embedding_type` -- e.g. /clip can't be pointed at a Text index.
        Without one, falls back to `resolve_index_for_embedding_type`.
        """
        collection_config = self.model_manager.config.collection_configs[collection]
        if index_name is not None:
            index_config = collection_config.get_index(index_name)
            if index_config.embedding_type != self.embedding_type:
                raise ValueError(
                    f"Index {index_name!r} is {index_config.embedding_type!r}-type, "
                    f"not {self.embedding_type!r}"
                )
            return index_config
        return collection_config.resolve_index_for_embedding_type(self.embedding_type)

    async def search(
        self,
        collection: str,
        text: str,
        n: int,
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFilters] = None,
        index_name: Optional[str] = None,
    ) -> List[int]:
        """Execute a text-to-vector search against this family's index."""
        try:
            index_config = self._resolve_index(collection, index_name)
            text_features = await self._encode_text(index_config.model_name, text)

            excluded_set = self._build_excluded_set(collection, excluded)
            seen_set = set(seen)

            return await self._search_with_expansion(
                collection,
                index_config.name,
                text_features,
                n,
                seen_set,
                excluded_set,
                filters,
            )

        except Exception as e:
            raise SearchError(
                f"{self.embedding_type} search failed: {e}",
                {"collection": collection, "text": text},
            )

    @abstractmethod
    async def _encode_text(self, model_name: str, text: str) -> np.ndarray:
        """Encode a text query into this family's embedding space. Implemented per family."""
        ...
