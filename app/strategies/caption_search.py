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
        # if filters and isinstance(self.metadata_repo, DatabaseRepository):
        #     passed_ids = []
        #     passed_ids = self.metadata_repo.get_filtered_media_ids(
        #         collection, filters
        #     )
        # elif isinstance(self.metadata_repo, MetadataRepository):
        #     metadata = self.metadata_repo.get_metadata(collection)
        #     collection_filters = self.metadata_repo.get_filters(collection)


        # Get shot mapping data
        # shot_overlap_mapper = self.model_manager.get_shot_overlap_mapper(collection)
        # caption_shot_ids_list = self.model_manager.get_caption_shot_ids_list(collection)
        # item_to_datapoint = self.metadata_repo.create_item_to_datapoint_mapping(
        #     collection
        # )

        while True:
            last = active_n >= total_items

            # Search caption index
            _, indices = self.index_repo.search_caption(
                collection, text_features, active_n
            )

            # mapped_indices = indices[0].tolist()
            # if isinstance(self.metadata_repo, DatabaseRepository):
            #     mapped_indices = self.metadata_repo.get_media_ids(collection, mapped_indices, index='caption')
            
            # Map caption indices to shot indices
            # caption_ids = [caption_shot_ids_list[idx] for idx in indices[0].tolist()]
            # base_shots = [shot_overlap_mapper[cap_id][0] for cap_id in caption_ids]
            # mapped_indices = [item_to_datapoint[shot] for shot in base_shots]

            # Filter results
            suggestions = []
            for idx in indices[0].tolist():
                if idx not in seen_set and idx not in excluded_set:
                    # if (
                    #     filters is None
                    #     or (isinstance(self.metadata_repo, DatabaseRepository) and idx in passed_ids)
                    #     or (isinstance(self.metadata_repo, MetadataRepository)
                    #         and check_active_filters(metadata["items"][idx], filters, collection_filters)
                    #     )
                    # ):
                    suggestions.append(idx)
            

            # Check if we have enough results
            if len(suggestions) >= n:
                suggestions = suggestions[:n]
                break
            elif last:
                break

            # Expand search radius
            active_n = min(active_n * 2, total_items)

        if isinstance(self.metadata_repo, DatabaseRepository):
            return self.metadata_repo.get_captions_with_nearest_keyframes(collection, suggestions)
