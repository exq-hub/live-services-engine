"""Faceted search strategy"""

from typing import List

from app.repositories.database_repository import DatabaseRepository

from .base import FacetedSearchStrategy
from ..schemas import ActiveFilters
from ..core.exceptions import SearchError


class FacetedSearchStrategy(FacetedSearchStrategy):
    """
    Search strategy using faceted filters.
    """

    def __init__(self, metadata_repository: DatabaseRepository):
        self.metadata_repo = metadata_repository

    def get_strategy_name(self) -> str:
        return "Faceted Search"

    async def search(
        self,
        collection: str,
        n: int,
        filters: ActiveFilters,
    ) -> List[int]:
        """Execute faceted search using filters."""
        try:
            passed_ids = self.metadata_repo.get_filtered_media_ids(collection, filters)
            return passed_ids[:n]

        except Exception as e:
            raise SearchError(
                f"Faceted search failed: {e}",
                {"collection": collection, "filters": filters},
            )