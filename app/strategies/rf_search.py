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


"""Relevance-feedback search strategy using a linear SVM.

Unlike CLIPSearchStrategy/TextEmbeddingSearchStrategy (each tied to one
embedding family), RF isn't tied to one: it resolves the target index by
an explicit `index_name` (or the collection's overall default), and
dispatches its optional pseudo-RF text-query blending to whichever
internal strategy matches that *resolved* index's embedding_type.

The pipeline:

1. **Index resolution** -- explicit index_name, or the collection's default.
2. **Positive sample preparation** -- collects user-provided positive IDs.
   If a text query is also provided, pseudo-RF is performed by encoding
   it and searching the resolved index directly (not by re-resolving
   through the internal strategy's own search(), which could otherwise
   pick a different same-family index), treating the top-10 results as
   additional positives. If no positives and no query are given, 5
   random items are sampled.
2. **Negative sample preparation** -- uses user-provided negatives, or
   falls back to 5 random items.
3. **SVM training** -- fits a `SGDClassifier` (linear SVM via SGD) on the
   embeddings of the positive (+1) and negative (-1) samples.
4. **Hyperplane search** -- uses the learned weight vector (hyperplane
   normal) as a query vector for the resolved index.
5. **Skip-set & filter handling** -- shared with CLIP/Text search via
   `VectorSearchMixin`.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import SGDClassifier
from numpy.random import default_rng

from app.core.config import IndexConfig, LSEConfig
from app.core.models import CLIPModelManager, TextModelManager
from app.repositories.database_repository import DatabaseRepository
from app.repositories.index_repository import IndexRepository

from .base import RFSearchStrategyABC
from .clip_search import CLIPSearchStrategy
from .embedding_search import EmbeddingSearchStrategy, VectorSearchMixin
from .text_search import TextEmbeddingSearchStrategy
from ..schemas import ActiveFilters
from ..core.exceptions import SearchError


class RFSearchStrategy(VectorSearchMixin, RFSearchStrategyABC):
    """Relevance feedback search using Linear SVM."""

    def __init__(
        self,
        clip_model_manager: CLIPModelManager,
        text_model_manager: TextModelManager,
        index_repository: IndexRepository,
        metadata_repository: DatabaseRepository,
    ):
        self.config: LSEConfig = clip_model_manager.config
        """Validated LSE configuration, used to resolve the target index."""

        self.index_repo: IndexRepository = index_repository
        """Index repository for executing nearest-neighbour vector searches."""

        self.database_repo: DatabaseRepository = metadata_repository
        """Database repository for ID mapping, filters, and item lookups."""

        self.clip_search: CLIPSearchStrategy = CLIPSearchStrategy(
            clip_model_manager, index_repository, metadata_repository
        )
        """Internal CLIP search strategy used for pseudo relevance-feedback queries."""

        self.text_search: TextEmbeddingSearchStrategy = TextEmbeddingSearchStrategy(
            text_model_manager, index_repository, metadata_repository
        )
        """Internal Text search strategy used for pseudo relevance-feedback queries."""

        self._query_strategies: Dict[str, EmbeddingSearchStrategy] = {
            "CLIP": self.clip_search,
            "Text": self.text_search,
        }
        """Text-query-blending strategy per embedding_type, keyed the same
        way SearchService keys its own strategy registry."""

    def get_strategy_name(self) -> str:
        return "RF Search"

    def _resolve_index(self, collection: str, index_name: Optional[str]) -> IndexConfig:
        """Explicit index_name, or the collection's overall default."""
        collection_config = self.config.collection_configs[collection]
        return collection_config.resolve_index(index_name)

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
        index_name: Optional[str] = None,
    ) -> List[int]:
        """Execute relevance feedback search using SVM."""
        try:
            index_config = self._resolve_index(collection, index_name)

            # Get embeddings array
            emb_arr = self.index_repo.get_embeddings_array(collection, index_config.name)
            total_items = self.database_repo.get_total_items(collection, index_config.name)

            # Prepare positive samples
            pos_samples = await self._prepare_positive_samples(
                collection, index_config, pos, query, seen, excluded, filters
            )

            # Prepare negative samples
            neg_samples = self._prepare_negative_samples(
                collection, index_config.name, neg, total_items
            )

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
                collection, index_config.name, hyperplane, n, seen_set, excluded_set, filters
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
        index_config: IndexConfig,
        pos: List[int],
        query: Optional[str],
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFilters],
    ) -> np.ndarray:
        """Prepare positive samples, including pseudo RF from query if available."""
        positive_samples = list(pos)

        # Add pseudo RF samples if query is provided
        if query is not None:
            try:
                query_strategy = self._query_strategies.get(index_config.embedding_type)
                if query_strategy is None:
                    raise SearchError(
                        f"No text-query strategy available for embedding_type "
                        f"{index_config.embedding_type!r}"
                    )
                # Encode/search against index_config directly rather than
                # query_strategy.search(), which re-resolves the index for
                # its own family and could pick a different same-family one.
                text_features = await query_strategy._encode_text(
                    index_config.model_name, query
                )
                excluded_set = self._build_excluded_set(collection, excluded)
                seen_set = set(seen)
                pseudo_rf = await query_strategy._search_with_expansion(
                    collection,
                    index_config.name,
                    text_features,
                    10,
                    seen_set,
                    excluded_set,
                    filters,
                )
                positive_samples.extend(pseudo_rf)
            except Exception:
                # If pseudo RF fails, continue with just the provided positive samples
                pass

        # If no positive samples and no query, add random samples
        if not positive_samples and query is None:
            rng = default_rng()
            total_items = self.database_repo.get_total_items(collection, index_config.name)
            positive_samples = rng.choice(total_items, size=5, replace=False).tolist()

        positive_samples = self.database_repo.get_index_ids(
            collection, positive_samples, index_config.name
        )

        return np.asarray(positive_samples)

    def _prepare_negative_samples(
        self, collection: str, index_name: str, neg: List[int], total_items: int
    ) -> np.ndarray:
        """Prepare negative samples."""
        if neg:
            neg = self.database_repo.get_index_ids(collection, neg, index_name)
            return np.asarray(neg)
        else:
            # Add random negative samples if none provided
            rng = default_rng()
            neg = self.database_repo.get_index_ids(
                collection,
                rng.choice(total_items, size=5, replace=False).tolist(),
                index_name,
            )
            return np.asarray(neg)
