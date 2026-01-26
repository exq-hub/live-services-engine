"""Search service that coordinates different search strategies."""

import time
from typing import Dict, List

from ..strategies.base import SearchStrategy, TextSearchStrategy, RFSearchStrategy, FacetedSearchStrategy
from ..strategies.clip_search import CLIPSearchStrategy
from ..strategies.caption_search import CaptionSearchStrategy
from ..strategies.rf_search import RFSearchStrategy as RFSearchImpl
from ..strategies.faceted_search import FacetedSearchStrategy as FacetedSearchImpl
from ..strategies.aggregate_search import AggregateSearchStrategy
from ..schemas import FacetedSearchRequest, TextSearchRequest, RFSearchRequest
from ..core.exceptions import SearchError


class SearchService:
    """Service for managing different search strategies."""

    def __init__(self, model_manager, index_repository, metadata_repository):
        self.model_manager = model_manager
        self.index_repo = index_repository
        self.metadata_repo = metadata_repository

        # Initialize strategies
        self.strategies: Dict[str, SearchStrategy] = {
            "clip": CLIPSearchStrategy(
                model_manager, index_repository, metadata_repository
            ),
            "caption": CaptionSearchStrategy(
                model_manager, index_repository, metadata_repository
            ),
            "rf": RFSearchImpl(model_manager, index_repository, metadata_repository),
            "aggregate": AggregateSearchStrategy(
                model_manager, index_repository, metadata_repository
            ),
            "faceted": FacetedSearchImpl(
                metadata_repository
            )
        }

    async def search_text(self, strategy_name: str, request: TextSearchRequest) -> Dict:
        """Execute text-based search using specified strategy."""
        if strategy_name not in self.strategies:
            raise SearchError(f"Unknown search strategy: {strategy_name}")

        strategy = self.strategies[strategy_name]
        if not isinstance(strategy, TextSearchStrategy):
            raise SearchError(f"Strategy {strategy_name} does not support text search")

        start_time = int(time.time())

        try:
            suggestions = await strategy.search(
                collection=request.session_info.collection,
                text=request.text,
                n=request.n,
                seen=request.seen or [],
                excluded=request.excluded or [],
                filters=request.filters,
            )

            completion_time = int(time.time()) - start_time

            return {
                "suggestions": suggestions,
                "request_timestamp": start_time,
                "completion_time": completion_time,
                "strategy": strategy_name,
            }

        except Exception as e:
            raise SearchError(
                f"Text search failed with strategy {strategy_name}: {e}",
                {
                    "strategy": strategy_name,
                    "collection": request.session_info.collection,
                    "text": request.text,
                },
            )

    async def search_rf(self, request: RFSearchRequest) -> Dict:
        """Execute relevance feedback search."""
        strategy = self.strategies["rf"]
        if not isinstance(strategy, RFSearchStrategy):
            raise SearchError("RF strategy not properly configured")

        start_time = int(time.time())

        try:
            suggestions = await strategy.search(
                collection=request.session_info.collection,
                pos=request.pos,
                neg=request.neg,
                n=request.n,
                seen=request.seen,
                excluded=request.excluded,
                filters=request.filters,
                query=request.query,
            )

            completion_time = int(time.time()) - start_time

            return {
                "suggestions": suggestions,
                "request_timestamp": start_time,
                "completion_time": completion_time,
                "strategy": "rf",
            }

        except Exception as e:
            raise SearchError(
                f"RF search failed: {e}",
                {
                    "collection": request.session_info.collection,
                    "pos_count": len(request.pos),
                    "neg_count": len(request.neg),
                },
            )
    
    async def search_faceted(self, request: FacetedSearchRequest) -> Dict:
        """Execute faceted search"""
        strategy = self.strategies["faceted"]
        if not isinstance(strategy, FacetedSearchStrategy):
            raise SearchError("Faceted strategy not properly configured")
        
        start_time = int(time.time())

        try:
            suggestions = await strategy.search(
                collection=request.session_info.collection,
                n=request.n,
                filters=request.filters
            )

            completion_time = int(time.time()) - start_time

            return {
                "suggestions": suggestions,
                "request_timestamp": start_time,
                "completion_time": completion_time,
                "strategy": "faceted"
            }
        except Exception as e:
            raise SearchError(
                f"Faceted search failed: {e}",
                {
                    "collection": request.session_info.collection,
                    "filters": request.filters
                }
            )
        

    def get_available_strategies(self) -> List[str]:
        """Get list of available search strategies."""
        return list(self.strategies.keys())

    def is_strategy_available(self, strategy_name: str, collection: str) -> bool:
        """Check if a strategy is available for a collection."""
        if strategy_name not in self.strategies:
            return False

        # Check strategy-specific availability
        if strategy_name == "caption":
            return self.index_repo.get_caption_index(collection) is not None

        return True
