"""Faceted (filter-only) search strategy.

Unlike the CLIP and RF strategies, this strategy performs no vector
similarity search at all. It compiles the user's `ActiveFilters` tree into
a SQL query (via `db_helper.compile_active_filters`), executes it against
the collection's SQLite database, and returns the first *n* matching media
IDs. This is useful when the user wants to browse items by metadata
attributes without providing a text query or relevance feedback.
"""

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
        self.metadata_repo: DatabaseRepository = metadata_repository
        """Database repository for compiling and executing filter queries."""

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