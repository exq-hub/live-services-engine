"""Aggregate search strategy using Reciprocal Rank Fusion."""

from collections import defaultdict
from typing import List, Optional

from .base import TextSearchStrategy
from .clip_search import CLIPSearchStrategy
from .caption_search import CaptionSearchStrategy
from ..schemas import ActiveFilters
from ..core.exceptions import SearchError


class AggregateSearchStrategy(TextSearchStrategy):
    """Search strategy that aggregates multiple search methods using RRF."""

    def __init__(self, model_manager, index_repository, database_repository):
        self.clip_search = CLIPSearchStrategy(
            model_manager, index_repository, database_repository
        )
        self.caption_search = CaptionSearchStrategy(
            model_manager, index_repository, database_repository
        )
        self.rrf_k = 60  # RRF constant

    def get_strategy_name(self) -> str:
        return "Aggregate Text Search (RRF)"

    async def search(
        self,
        collection: str,
        text: str,
        n: int,
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFilters] = None,
    ) -> List[int]:
        """Execute aggregated search using Reciprocal Rank Fusion."""
        try:
            # Execute both search strategies
            clip_results = await self.clip_search.search(
                collection, text, n, seen, excluded, filters
            )

            caption_results = await self.caption_search.search(
                collection, text, n, seen, excluded, filters
            )

            # Apply RRF to combine results
            rrf_scores = self._calculate_rrf_scores(clip_results, caption_results)

            if not rrf_scores:
                return []

            # Sort by RRF score (descending - higher scores are better)
            aggregated_results = sorted(
                rrf_scores.items(), key=lambda x: x[1], reverse=True
            )

            # Return the list of items sorted by their RRF score
            return [item_id for item_id, _ in aggregated_results]

        except Exception as e:
            raise SearchError(
                f"Aggregate search failed: {e}",
                {"collection": collection, "text": text},
            )

    def _calculate_rrf_scores(
        self, clip_results: List[int], caption_results: List[int]
    ) -> dict:
        """Calculate Reciprocal Rank Fusion scores."""
        rrf_scores = defaultdict(float)

        # Process CLIP results
        for rank, item_id in enumerate(clip_results):
            rrf_scores[item_id] += 1 / (self.rrf_k + rank + 1)

        # Process caption results
        for rank, item_id in enumerate(caption_results):
            rrf_scores[item_id] += 1 / (self.rrf_k + rank + 1)

        return dict(rrf_scores)
