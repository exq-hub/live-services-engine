"""Faceted search strategy"""

from typing import List

from .base import FacetedSearchStrategy
from ..schemas import ActiveFilters, ActiveFiltersDB
from ..core.exceptions import SearchError


class FacetedSearchStrategy(FacetedSearchStrategy):
    """
    Search strategy using faceted filters.
    """

    def __init__(self, metadata_repository):
        self.metadata_repo = metadata_repository

    def get_strategy_name(self) -> str:
        return "Faceted Search"

    async def search(
        self,
        collection: str,
        n: int,
        filters: ActiveFilters | ActiveFiltersDB,
    ) -> List[int]:
        """Execute aggregated search using Reciprocal Rank Fusion."""
        try:
            passed_ids = self.metadata_repo.get_filtered_media_ids(collection, filters)
            return passed_ids[:n]

        except Exception as e:
            raise SearchError(
                f"Faceted search failed: {e}",
                {"collection": collection, "filters": filters},
            )