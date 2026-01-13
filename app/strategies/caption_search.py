"""Caption-based search strategy."""

from typing import Any, Dict, List, Optional

import numpy as np

from app.core.models import ModelManager
from app.repositories.database_repository import DatabaseRepository
from app.repositories.index_repository import IndexRepository
from app.repositories.metadata_repository import MetadataRepository

from .base import TextSearchStrategy
from ..schemas import ActiveFilters, ActiveFiltersDB
from ..search_utils import check_active_filters
from ..core.exceptions import SearchError


class CaptionSearchStrategy(TextSearchStrategy):
    """Search strategy using caption embeddings."""

    def __init__(
        self, 
        model_manager: ModelManager,
        index_repository: IndexRepository, 
        metadata_repository: MetadataRepository | DatabaseRepository):
        self.model_manager = model_manager
        self.index_repo = index_repository
        self.metadata_repo = metadata_repository

    def get_strategy_name(self) -> str:
        return "Caption Search"

    async def search(
        self,
        collection: str,
        text: str,
        n: int,
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFilters] = None,
    ) -> List[Dict[str, Any]]:
        """Execute caption-based text search."""
        try:
            # Check if caption search is available
            if not self.index_repo.get_caption_index(collection):
                return []  # Return empty if caption search not available

            # Encode text using sentence transformer
            text_features = await self._encode_text(text)

            # Process exclusions
            excluded_set = self._build_excluded_set(collection, excluded)
            seen_set = set(seen)

            # Search with expanding radius until we have enough results
            return await self._search_with_expansion(
                collection, text_features, n, seen_set, excluded_set, filters
            )

        except Exception as e:
            raise SearchError(
                f"Caption search failed: {e}", {"collection": collection, "text": text}
            )

    async def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using sentence transformer."""
        text_features = self.model_manager.caption_embedding_model.encode(text)

        # Normalize features
        text_features = text_features / np.linalg.norm(
            text_features, axis=-1, keepdims=True
        )
        return text_features.reshape(1, -1)

    def _build_excluded_set(self, collection: str, excluded: List[int]) -> set:
        """Build set of excluded items including related items."""
        if not excluded:
            return set()

        excluded_set = set()
        metadata = self.metadata_repo.get_metadata(collection)
        related_items = self.metadata_repo.get_related_items(collection)

        if metadata and related_items:
            for exc in excluded:
                if exc < len(metadata["items"]):
                    item_id = metadata["items"][exc]["item_id"].split("_")[0]
                    excluded_set.update(related_items.get(item_id, []))

        return excluded_set

    async def _search_with_expansion(
        self,
        collection: str,
        text_features: np.ndarray,
        n: int,
        seen_set: set,
        excluded_set: set,
        filters: Optional[ActiveFilters | ActiveFiltersDB],
    ) -> List[Dict[str, Any]]:
        """Search with expanding radius until sufficient results."""
        active_n = n
        total_items = self.metadata_repo.get_total_items(collection, index='caption')
        skip_ids = set()
        if len(seen_set) != 0:
            skip_ids.update(self.metadata_repo.get_index_ids(collection, list(seen_set), index='caption'))
        if len(excluded_set) != 0:
            skip_ids.update(self.metadata_repo.get_index_ids(collection, list(excluded_set), index='caption'))

        if filters:
            passed_ids = []
            passed_ids = self.metadata_repo.get_filtered_media_ids(
                collection, filters
            )
            # NOTE: Can use the size of passed_ids to determine if index search is needed
            #       If it is lower than a certain threshold we can search through the subset with
            #       the zarr embeddings array directly
            index_passed_ids = self.metadata_repo.get_index_ids(collection, passed_ids, index='caption')
            index_skip_ids = set(range(total_items)) - set(index_passed_ids)
            skip_ids.update(index_skip_ids)

        indices = self.index_repo.search_caption(
            collection, text_features, active_n, skip_ids=skip_ids
        )
        suggestions = self.metadata_repo.get_media_ids(collection, indices, index='caption')

        return self.metadata_repo.get_text_source_with_nearest_keyframes(collection, 'caption', suggestions)