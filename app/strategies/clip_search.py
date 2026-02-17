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


"""CLIP text-to-image search strategy.

Encodes a natural-language text query into the CLIP embedding space using
the ``ViT-SO400M-14-SigLIP-384`` text encoder, then retrieves the *n*
most similar media items from the collection's vector index.

The search pipeline:

1. **Text encoding** -- tokenize and forward through the CLIP text model
   (FP16 on CUDA when available), L2-normalise the output.
2. **Skip-set construction** -- translate ``seen``, ``excluded``, and
   filter-rejected media IDs into index-space IDs so the index can skip
   them during search rather than requiring post-filtering.
3. **Index search** -- delegate to `IndexRepository.search_clip` which
   dispatches to the appropriate `BaseIndex` implementation.
4. **ID mapping** -- translate the resulting index positions back to media
   IDs via `DatabaseRepository.get_media_ids`.
"""

import contextlib
from typing import List, Optional

import torch
import numpy as np

from app.core.models import ModelManager
from app.repositories.database_repository import DatabaseRepository
from app.repositories.index_repository import IndexRepository

from .base import TextSearchStrategy
from ..schemas import ActiveFilters
from ..core.exceptions import SearchError


class CLIPSearchStrategy(TextSearchStrategy):
    """Search strategy using CLIP text embeddings."""

    def __init__(
        self,
        model_manager: ModelManager,
        index_repository: IndexRepository,
        database_repository: DatabaseRepository,
    ):
        self.model_manager: ModelManager = model_manager
        """Model manager providing the CLIP text encoder and device."""

        self.index_repo: IndexRepository = index_repository
        """Index repository for executing nearest-neighbour vector searches."""

        self.database_repo: DatabaseRepository = database_repository
        """Database repository for ID mapping, filters, and exclusion lookups."""

    def get_strategy_name(self) -> str:
        return "CLIP Search"

    async def search(
        self,
        collection: str,
        text: str,
        n: int,
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFilters] = None,
        # state: Optional[int] = None,
    ) -> List[int]:
        """Execute CLIP text search."""
        try:
            # if state and self.index_repo.is_query_in_state_clip(collection, state):
            # TODO: Implement stateful search and resuming functionality
            # Process exclusions
            # excluded_set = self._build_excluded_set(collection, excluded)
            # seen_set = set(seen)
            # pass

            # Encode text using CLIP
            text_features = await self._encode_text(text)

            # Process exclusions
            excluded_set = self._build_excluded_set(collection, excluded)
            seen_set = set(seen)

            # if state:
            #     # TODO: Store query vector in state for resuming
            #     pass

            # Search with expanding radius until we have enough results
            return await self._search_with_expansion(
                collection, text_features, n, seen_set, excluded_set, filters
            )

        except Exception as e:
            raise SearchError(
                f"CLIP search failed: {e}", {"collection": collection, "text": text}
            )

    async def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using CLIP model."""
        device = self.model_manager.device

        with (
            torch.inference_mode(),
            (
                torch.amp.autocast("cuda")
                if torch.cuda.is_available()
                else contextlib.nullcontext()
            ),
        ):
            tokenized_text = self.model_manager.clip_text_tokenizer([text]).to(device)
            text_features = self.model_manager.clip_text_model(tokenized_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.detach().cpu().numpy()

    def _build_excluded_set(self, collection: str, excluded: List[int]) -> set:
        """Build set of excluded items including related items."""
        if not excluded:
            return set()

        excluded_set = set(excluded)
        database_repo: DatabaseRepository = self.database_repo
        for exc in excluded:
            item = database_repo.get_item(collection, exc)
            related = database_repo.get_related_items(collection, item["group"])
            excluded_set.update(related)

        return excluded_set

    async def _search_with_expansion(
        self,
        collection: str,
        text_features: np.ndarray,
        n: int,
        seen_set: set,
        excluded_set: set,
        filters: Optional[ActiveFilters] = None,
    ) -> List[int]:
        """Search with expanding radius until sufficient results."""
        active_n = n
        total_items = self.database_repo.get_total_items(collection)
        skip_ids = set()
        if len(seen_set) != 0:
            skip_ids.update(
                self.database_repo.get_index_ids(
                    collection, list(seen_set), index="clip"
                )
            )
        if len(excluded_set) != 0:
            skip_ids.update(
                self.database_repo.get_index_ids(
                    collection, list(excluded_set), index="clip"
                )
            )

        if filters:
            passed_ids = []
            passed_ids = self.database_repo.get_filtered_media_ids(collection, filters)
            # NOTE: Can use the size of passed_ids to determine if index search is needed
            #       If it is lower than a certain threshold we can search through the subset with
            #       the zarr embeddings array directly
            index_passed_ids = self.database_repo.get_index_ids(
                collection, passed_ids, index="clip"
            )
            index_skip_ids = set(range(total_items)) - set(index_passed_ids)
            skip_ids.update(index_skip_ids)

        indices, _ = self.index_repo.search_clip(
            collection, text_features, active_n, skip_ids=skip_ids
        )
        suggestions = self.database_repo.get_media_ids(collection, indices)

        return suggestions
