"""CLIP-based search strategy."""

import contextlib
from typing import List, Optional

import torch
import numpy as np

from app.repositories.database_repository import DatabaseRepository
from app.repositories.metadata_repository import MetadataRepository

from .base import TextSearchStrategy
from ..schemas import ActiveFilters, ActiveFiltersDB
from ..search_utils import check_active_filters
from ..core.exceptions import SearchError


class CLIPSearchStrategy(TextSearchStrategy):
    """Search strategy using CLIP text embeddings."""

    def __init__(self, model_manager, index_repository, metadata_repository):
        self.model_manager = model_manager
        self.index_repo = index_repository
        self.metadata_repo = metadata_repository

    def get_strategy_name(self) -> str:
        return "CLIP Search"

    async def search(
        self,
        collection: str,
        text: str,
        n: int,
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFilters | ActiveFiltersDB] = None,
    ) -> List[int]:
        """Execute CLIP text search."""
        try:
            # Encode text using CLIP
            text_features = await self._encode_text(text)

            # Apply PCA transformation if available
            text_features = self._apply_pca_transform(collection, text_features)

            # Process exclusions
            excluded_set = self._build_excluded_set(collection, excluded)
            seen_set = set(seen)

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

    def _apply_pca_transform(self, collection: str, features: np.ndarray) -> np.ndarray:
        """Apply PCA transformation if available."""
        pca_data = self.index_repo.get_pca_data(collection)
        if pca_data:
            scaler = pca_data["scaler"]
            pca_model = pca_data["model"]
            scaled_features = scaler.transform(features)
            return pca_model.transform(scaled_features)
        return features

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
        filters: Optional[ActiveFilters | ActiveFiltersDB] = None,
    ) -> List[int]:
        """Search with expanding radius until sufficient results."""
        active_n = n
        total_items = self.metadata_repo.get_total_items(collection)
        if filters and isinstance(self.metadata_repo, DatabaseRepository):
            passed_ids = []
            passed_ids = self.metadata_repo.get_filtered_media_ids(
                collection, filters
            )
        elif isinstance(self.metadata_repo, MetadataRepository):
            metadata = self.metadata_repo.get_metadata(collection)
            collection_filters = self.metadata_repo.get_filters(collection)

        while True:
            last = active_n >= total_items

            # Search index
            _, indices = self.index_repo.search_clip(
                collection, text_features, active_n
            )

            indices = indices[0].tolist()
            if isinstance(self.metadata_repo, DatabaseRepository):
                indices = self.metadata_repo.get_media_ids(collection, indices)
            
            # Filter results
            suggestions = []
            for idx in indices:
                if idx not in seen_set and idx not in excluded_set:
                    if (
                        filters is None
                        or (isinstance(self.metadata_repo, DatabaseRepository) and idx in passed_ids)
                        or (isinstance(self.metadata_repo, MetadataRepository)
                            and check_active_filters(metadata["items"][idx], filters, collection_filters)
                        )
                    ):
                        suggestions.append(idx)

            # Check if we have enough results
            if len(suggestions) >= n:
                return suggestions[:n]
            elif last:
                return suggestions

            # Expand search radius
            active_n = min(active_n * 2, total_items)
