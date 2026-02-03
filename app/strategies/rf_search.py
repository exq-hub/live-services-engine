"""Relevance feedback search strategy using SVM."""

from typing import List, Optional

import numpy as np
from sklearn.linear_model import SGDClassifier
from numpy.random import default_rng

from app.repositories.database_repository import DatabaseRepository

from .base import RFSearchStrategy
from .clip_search import CLIPSearchStrategy
from ..schemas import ActiveFiltersDB
from ..core.exceptions import SearchError


class RFSearchStrategy(RFSearchStrategy):
    """Relevance feedback search using Linear SVM."""

    def __init__(self, model_manager, index_repository, metadata_repository):
        self.model_manager = model_manager
        self.index_repo = index_repository
        self.metadata_repo = metadata_repository
        # Initialize CLIP search for pseudo RF
        self.clip_search = CLIPSearchStrategy(
            model_manager, index_repository, metadata_repository
        )

    def get_strategy_name(self) -> str:
        return "RF Search"

    async def search(
        self,
        collection: str,
        pos: List[int],
        neg: List[int],
        n: int,
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFiltersDB] = None,
        query: Optional[str] = None,
    ) -> List[int]:
        """Execute relevance feedback search using SVM."""
        try:
            # Get embeddings array
            emb_arr = self.index_repo.get_embeddings_array(collection)
            total_items = self.metadata_repo.get_total_items(collection)

            # Prepare positive samples
            pos_samples = await self._prepare_positive_samples(
                collection, pos, query, seen, excluded, filters
            )

            # Prepare negative samples
            neg_samples = self._prepare_negative_samples(collection, neg, total_items)

            # Train SVM classifier
            if len(pos_samples) == 0:
                return []  # No positive samples to work with

            samples = emb_arr[np.concatenate((pos_samples, neg_samples))]
            labels = np.concatenate(
                ([1.0] * len(pos_samples), [-1.0] * len(neg_samples))
            )

            clf = SGDClassifier(random_state=42)
            clf.fit(samples, labels)

            # Use hyperplane for search
            hyperplane = clf.coef_

            # Process exclusions
            excluded_set = self._build_excluded_set(collection, excluded)
            seen_set = set(seen)

            # Search with expanding radius
            return await self._search_with_expansion(
                collection, hyperplane, n, seen_set, excluded_set, filters
            )

        except Exception as e:
            raise SearchError(
                f"RF search failed: {e}",
                {
                    "collection": collection,
                    "pos_count": len(pos),
                    "neg_count": len(neg),
                },
            )

    async def _prepare_positive_samples(
        self,
        collection: str,
        pos: List[int],
        query: Optional[str],
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFiltersDB],
    ) -> np.ndarray:
        """Prepare positive samples, including pseudo RF from query if available."""
        positive_samples = list(pos)

        # Add pseudo RF samples if query is provided
        if query is not None:
            try:
                pseudo_rf = await self.clip_search.search(
                    collection=collection,
                    text=query,
                    n=10,
                    seen=seen,
                    excluded=excluded,
                    filters=filters,
                )
                positive_samples.extend(pseudo_rf)
            except Exception:
                # If pseudo RF fails, continue with just the provided positive samples
                pass

        # If no positive samples and no query, add random samples
        if not positive_samples and query is None:
            rng = default_rng()
            total_items = self.metadata_repo.get_total_items(collection)
            positive_samples = rng.choice(total_items, size=5, replace=False).tolist()

        if isinstance(self.metadata_repo, DatabaseRepository):
            positive_samples = self.metadata_repo.get_index_ids(collection, positive_samples)

        return np.asarray(positive_samples)

    def _prepare_negative_samples(self, collection, neg: List[int], total_items: int) -> np.ndarray:
        """Prepare negative samples."""
        if neg:

            if isinstance(self.metadata_repo, DatabaseRepository):
                neg = self.metadata_repo.get_index_ids(collection, neg)
            return np.asarray(neg)
        else:
            # Add random negative samples if none provided
            rng = default_rng()
            if isinstance(self.metadata_repo, DatabaseRepository):
                neg = self.metadata_repo.get_index_ids(
                    collection,
                    rng.choice(total_items, size=5, replace=False).tolist()
                )
                return np.asarray(neg)
            return rng.choice(total_items, size=5, replace=False)

    def _build_excluded_set(self, collection: str, excluded: List[int]) -> set:
        """Build set of excluded items including related items."""
        if not excluded:
            return set()

        excluded_set = set(excluded)
        metadata_repo: DatabaseRepository = self.metadata_repo
        for exc in excluded:
            item = metadata_repo.get_item(collection, exc)
            related = metadata_repo.get_related_items(collection, item['group'])
            excluded_set.update(related)

        return excluded_set

    async def _search_with_expansion(
        self,
        collection: str,
        hyperplane: np.ndarray,
        n: int,
        seen_set: set,
        excluded_set: set,
        filters: Optional[ActiveFiltersDB],
    ) -> List[int]:
        """Search with expanding radius until sufficient results."""
        active_n = n
        total_items = self.metadata_repo.get_total_items(collection)
        skip_ids = set()
        if len(seen_set) != 0:
            skip_ids.update(self.metadata_repo.get_index_ids(collection, list(seen_set), index='clip'))
        if len(excluded_set) != 0:
            skip_ids.update(self.metadata_repo.get_index_ids(collection, list(excluded_set), index='clip'))

        if filters:
            passed_ids = []
            passed_ids = self.metadata_repo.get_filtered_media_ids(
                collection, filters
            )
            # NOTE: Can use the size of passed_ids to determine if index search is needed
            #       If it is lower than a certain threshold we can search through the subset with
            #       the zarr embeddings array directly
            index_passed_ids = self.metadata_repo.get_index_ids(collection, passed_ids, index='clip')
            index_skip_ids = set(range(total_items)) - set(index_passed_ids)
            skip_ids.update(index_skip_ids)

        indices, _ = self.index_repo.search_clip(
            collection, hyperplane, active_n, skip_ids=skip_ids
        )
        suggestions = self.metadata_repo.get_media_ids(collection, indices)

        return suggestions
