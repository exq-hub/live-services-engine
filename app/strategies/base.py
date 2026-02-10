"""Base classes for search strategies."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..schemas import ActiveFilters


class SearchStrategy(ABC):
    """Abstract base class defining the interface for all search strategies.

    This class establishes the common interface that all search strategy
    implementations must follow, ensuring consistency across different
    search algorithms used in the LSE.
    """

    @abstractmethod
    async def search(
        self,
        collection: str,
        n: int,
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFilters] = None,
        **kwargs,
    ) -> List[int]:
        """Execute a search operation and return matching item indices.

        Args:
            collection: Name of the data collection to search
            n: Maximum number of results to return
            seen: List of item IDs that have already been seen
            excluded: List of item IDs to exclude from results
            filters: Optional filters to apply to the search
            **kwargs: Additional strategy-specific parameters

        Returns:
            List of item indices matching the search criteria
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name identifier for this search strategy.

        Returns:
            String identifier for the strategy (e.g., 'clip', 'caption', 'rf')
        """
        pass


class TextSearchStrategy(SearchStrategy):
    """Abstract base class for text query-based search strategies.

    This class specializes the SearchStrategy interface for strategies
    that accept text queries as input, such as CLIP text-to-image search
    or caption-based text search.
    """

    @abstractmethod
    async def search(
        self,
        collection: str,
        text: str,
        n: int,
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFilters] = None,
    ) -> List[int]:
        """Execute a text-based search and return matching item indices.

        Args:
            collection: Name of the data collection to search
            text: Text query to search for
            n: Maximum number of results to return
            seen: List of item IDs that have already been seen
            excluded: List of item IDs to exclude from results
            filters: Optional filters to apply to the search

        Returns:
            List of item indices matching the text query
        """
        pass


class RFSearchStrategy(SearchStrategy):
    """Abstract base class for relevance feedback search strategies.

    This class specializes the SearchStrategy interface for strategies
    that use positive and negative example items to refine search results
    through machine learning-based relevance feedback.
    """

    @abstractmethod
    async def search(
        self,
        collection: str,
        pos: List[int],
        neg: List[int],
        n: int,
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFilters] = None,
        query: Optional[str] = None,
    ) -> List[int]:
        """Execute a relevance feedback search and return matching item indices.

        Args:
            collection: Name of the data collection to search
            pos: List of positive example item IDs
            neg: List of negative example item IDs
            n: Maximum number of results to return
            seen: List of item IDs that have already been seen
            excluded: List of item IDs to exclude from results
            filters: Optional filters to apply to the search
            query: Optional text query to combine with feedback

        Returns:
            List of item indices matching the relevance feedback criteria
        """
        pass


class FacetedSearchStrategy(SearchStrategy):
    """Abstract base class for faceted search strategies.

    This class specializes the SearchStrategy interface for strategies
    that use filters fetch items.
    """

    @abstractmethod
    async def search(
        self,
        collection: str,
        filters: ActiveFilters,
        n: int,
    ) -> List[int]:
        """Execute a relevance feedback search and return matching item indices.

        Args:
            collection: Name of the data collection to search
            n: Maximum number of results to return
            filters: Optional filters to apply to the search

        Returns:
            List of item indices matching the relevance feedback criteria
        """
        pass