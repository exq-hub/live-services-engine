# Copyright (C) 2026 Ujjwal Sharma and Omar Shahbaz Khan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Search service -- strategy dispatcher and performance tracker.

`SearchService` is the entry point for all search operations. It initialises
the three available strategies at construction time and dispatches incoming
requests to the correct one:

- ``"clip"`` -- `CLIPSearchStrategy` for text-to-image similarity search.
- ``"rf"`` -- `RFSearchStrategy` for SVM-based relevance feedback.
- ``"faceted"`` -- `FacetedSearchStrategy` for filter-only retrieval.

Each search call is timed (wall-clock seconds) and the result dict includes
``request_timestamp``, ``completion_time``, and ``strategy`` metadata used
by the route layer for audit logging.
"""

import time
from typing import Dict, List

from ..strategies.base import SearchStrategy, TextSearchStrategy, RFSearchStrategy, FacetedSearchStrategy
from ..strategies.clip_search import CLIPSearchStrategy
from ..strategies.rf_search import RFSearchStrategy as RFSearchImpl
from ..strategies.faceted_search import FacetedSearchStrategy as FacetedSearchImpl
from ..schemas import FacetedSearchRequest, TextSearchRequest, RFSearchRequest
from ..core.exceptions import SearchError


class SearchService:
    """Service for managing different search strategies."""

    def __init__(self, model_manager, index_repository, metadata_repository):
        self.model_manager = model_manager
        """Shared `ModelManager` providing the CLIP text encoder and device."""

        self.index_repo = index_repository
        """Shared `IndexRepository` for vector nearest-neighbour lookups."""

        self.metadata_repo = metadata_repository
        """Shared `DatabaseRepository` for metadata and ID-mapping queries."""

        self.strategies: Dict[str, SearchStrategy] = {
            "clip": CLIPSearchStrategy(
                model_manager, index_repository, metadata_repository
            ),
            "rf": RFSearchImpl(model_manager, index_repository, metadata_repository),
            "faceted": FacetedSearchImpl(
                metadata_repository
            )
        }
        """Registry of available search strategies keyed by name (``clip``, ``rf``, ``faceted``)."""

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
        return strategy_name in self.strategies
