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


"""Temporal search over an ordered sequence of queries (hops).

The method is described in "Exquisitor at the Video Browser Showdown 2026:
Temporal Queries Revisited" (https://doi.org/10.1007/978-981-95-6963-2_27).

The pipeline is as follows:

1. **Anchors** (S_R). The first hop retrieves its ``top_r`` segments.
2. **Pool** (S_C). Every segment of the anchors' videos that starts
   strictly after that video's earliest anchor, timed from the start and
   end time tagsets, of the source type each later hop's index maps.
3. **Scoring**. Later hops sharing an index are scored together in one
   exact pass over their pool, giving each pool segment a rank per hop.
4. **Chains**. From each anchor, every later hop picks its best ranked
   segment of the same video starting after the previously chosen one,
   until a hop has no such segment.
5. **Ranking**. Chains rank by length, then by RRF over their hop ranks.
6. **Selection**. Within a video, a chain overlapping a better ranked one
   by IoU above the threshold is dropped. Up to ``chains_per_video``
   chains are kept for each of the ``n`` best videos.

Steps 4 to 6 are the pure functions and classes at the top of this
module. `TemporalSearchStrategy` gathers their inputs from the repositories.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.config import IndexConfig, LSEConfig
from app.repositories.database_repository import DatabaseRepository
from app.repositories.index_repository import IndexRepository

from .base import TemporalSearchStrategyABC
from .embedding_search import EmbeddingSearchStrategy, VectorSearchMixin
from .rf_search import RFSearchStrategy
from ..schemas import ActiveFilters, FilterGroup, TemporalChain, TemporalHop
from ..core.exceptions import SearchError


@dataclass(frozen=True)
class Segment:
    """A timed segment with its rank for one hop."""

    media_id: int
    group: int
    start_ms: float
    end_ms: float
    rank: int


@dataclass
class Chain:
    """Segments of one video matched to consecutive hops, first hop first."""

    group: int
    segments: List[Segment]
    score: float

    @property
    def length(self) -> int:
        return len(self.segments)

    @property
    def start_ms(self) -> float:
        return self.segments[0].start_ms

    @property
    def end_ms(self) -> float:
        return self.segments[-1].end_ms


class HopTimeline:
    """One hop's ranked pool, answering "best ranked segment of a video
    starting strictly after t" with a binary search.

    Per video, segments are sorted by start time alongside the suffix
    minimum of their ranks, so position i holds the best rank among every
    segment starting at or after the i-th start.
    """

    def __init__(self, segments: Sequence[Segment]):
        self._by_rank: Dict[int, Segment] = {s.rank: s for s in segments}
        self._groups: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        by_group: Dict[int, List[Segment]] = {}
        for segment in segments:
            by_group.setdefault(segment.group, []).append(segment)
        for group, group_segments in by_group.items():
            group_segments.sort(key=lambda s: s.start_ms)
            starts = np.array([s.start_ms for s in group_segments])
            ranks = np.array([s.rank for s in group_segments])
            best_ranks = np.minimum.accumulate(ranks[::-1])[::-1]
            self._groups[group] = (starts, best_ranks)

    def best_after(self, group: int, t: float) -> Optional[Segment]:
        """Best ranked segment of the video starting strictly after t, if any."""
        if group not in self._groups:
            return None
        starts, best_ranks = self._groups[group]
        i = int(np.searchsorted(starts, t, side="right"))
        if i == len(starts):
            return None
        return self._by_rank[int(best_ranks[i])]


def rrf_score(ranks: Sequence[int], weights: Sequence[float], k: float) -> float:
    """Reciprocal Rank Fusion of a chain's hop ranks, the sum of w_i / (k + r_i)."""
    return sum(w / (k + r) for r, w in zip(ranks, weights))


def assemble_chains(
    anchors: Sequence[Segment],
    timelines: Sequence[HopTimeline],
    weights: Sequence[float],
    k: float,
) -> List[Chain]:
    """Walk forward in time from each anchor through the later hops.

    Args:
        anchors: First-hop segments, each starting one chain
        timelines: One `HopTimeline` per later hop, in hop order
        weights: Per-hop RRF weights, first hop first
        k: RRF smoothing constant

    Returns:
        One chain per anchor, ending at the first hop without a match
    """
    chains = []
    for anchor in anchors:
        segments = [anchor]
        for timeline in timelines:
            segment = timeline.best_after(anchor.group, segments[-1].start_ms)
            if segment is None:
                break
            segments.append(segment)
        score = rrf_score([s.rank for s in segments], weights, k)
        chains.append(Chain(group=anchor.group, segments=segments, score=score))
    return chains


def interval_iou(a: Chain, b: Chain) -> float:
    """Intersection over union of two chains' time spans.

    Two zero-length spans at the same time count as fully overlapping.
    """
    intersection = max(0.0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))
    union = max(a.end_ms, b.end_ms) - min(a.start_ms, b.start_ms)
    if union == 0:
        return 1.0
    return intersection / union


def select_chains(
    chains: Sequence[Chain],
    n: int,
    chains_per_video: int,
    iou_threshold: float,
) -> List[Chain]:
    """Rank chains, suppress overlaps per video, and keep the best videos.

    Chains rank by length, then RRF score, then their anchor's rank. Within
    a video, a chain whose IoU with an already kept chain is above
    `iou_threshold` is dropped, so of two overlapping chains the longer is
    kept, and on equal length the higher scoring one.

    Args:
        chains: Candidate chains of any videos
        n: Number of distinct videos to keep
        chains_per_video: Maximum chains kept per video, or -1 for all
        iou_threshold: IoU above which two chains of a video overlap

    Returns:
        Kept chains grouped by video, videos ordered by their best chain
    """
    ranked = sorted(
        chains, key=lambda c: (-c.length, -c.score, c.segments[0].rank)
    )

    kept: Dict[int, List[Chain]] = {}
    for chain in ranked:
        video_chains = kept.get(chain.group)
        if video_chains is None:
            if len(kept) == n:
                continue
            kept[chain.group] = [chain]
            continue
        if chains_per_video != -1 and len(video_chains) >= chains_per_video:
            continue
        if any(interval_iou(chain, other) > iou_threshold for other in video_chains):
            continue
        video_chains.append(chain)

    # Dicts keep insertion order, which is each video's best chain order
    return [chain for video_chains in kept.values() for chain in video_chains]


def _combine_filters(*filters: Optional[ActiveFilters]) -> Optional[ActiveFilters]:
    """AND together the given filters, ignoring missing ones."""
    present = [f for f in filters if f is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return ActiveFilters(
        root=FilterGroup(operator="AND", children=[f.root for f in present])
    )


class TemporalSearchStrategy(VectorSearchMixin, TemporalSearchStrategyABC):
    """Temporal search over hops of text or RF queries."""

    def __init__(
        self,
        config: LSEConfig,
        clip_search: EmbeddingSearchStrategy,
        text_search: EmbeddingSearchStrategy,
        rf_search: RFSearchStrategy,
        index_repository: IndexRepository,
        database_repository: DatabaseRepository,
    ):
        self.config: LSEConfig = config
        """Validated LSE configuration, used to resolve each query hop's index."""

        self.query_strategies: Dict[str, EmbeddingSearchStrategy] = {
            strategy.embedding_type: strategy for strategy in (clip_search, text_search)
        }
        """Strategy encoding query hops, keyed by their index's embedding_type."""

        self.rf_search: RFSearchStrategy = rf_search
        """Strategy resolving RF hops and training their hyperplanes."""

        self.index_repo: IndexRepository = index_repository
        """Index repository for anchor retrieval and pool scoring."""

        self.database_repo: DatabaseRepository = database_repository
        """Database repository for segment times, filters and ID mapping."""

    def get_strategy_name(self) -> str:
        return "Temporal Search"

    async def _build_hop_query(
        self,
        collection: str,
        hop: TemporalHop,
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFilters],
    ) -> Tuple[IndexConfig, np.ndarray]:
        """Resolve a hop's index and its query vector."""
        if hop.kind == "rf":
            index_config, hyperplane = await self.rf_search.build_query(
                collection,
                hop.pos,
                hop.neg,
                seen,
                excluded,
                filters,
                hop.query,
                hop.index_name,
            )
            if hyperplane is None:
                raise SearchError("RF hop has no positive samples")
            return index_config, hyperplane

        index_config = self.config.collection_configs[collection].resolve_index(
            hop.index_name
        )
        strategy = self.query_strategies.get(index_config.embedding_type)
        if strategy is None:
            raise SearchError(
                f"No query strategy for embedding_type {index_config.embedding_type!r}"
            )
        return await strategy.build_query(collection, hop.text, index_config.name)

    async def search(
        self,
        collection: str,
        hops: List[TemporalHop],
        n: int,
        seen: List[int],
        excluded: List[int],
        filters: Optional[ActiveFilters] = None,
        top_r: int = 1000,
        rrf_k: float = 60.0,
        chains_per_video: int = 1,
        iou_threshold: float = 0.5,
    ) -> List[TemporalChain]:
        """Execute a temporal search and return the ranked chains."""
        try:
            hop_filters = [_combine_filters(filters, hop.filters) for hop in hops]
            queries = [
                await self._build_hop_query(collection, hop, seen, excluded, f)
                for hop, f in zip(hops, hop_filters)
            ]
            excluded_set = self._build_excluded_set(collection, excluded)

            anchor_index, anchor_vector = queries[0]
            anchor_ids = await self._search_with_expansion(
                collection,
                anchor_index.name,
                anchor_vector,
                top_r,
                set(seen),
                excluded_set,
                hop_filters[0],
            )
            if not anchor_ids:
                return []

            timeline = _SegmentTimes(
                self.database_repo.get_group_segment_times(
                    collection,
                    anchor_ids,
                    sorted({index_config.source_type for index_config, _ in queries}),
                )
            )
            anchors = [
                timeline.segment(media_id, rank)
                for rank, media_id in enumerate(anchor_ids, start=1)
                if media_id in timeline
            ]
            if not anchors:
                return []

            timelines = self._rank_later_hops(
                collection, queries[1:], hop_filters[1:], anchors, timeline, excluded_set
            )
            chains = assemble_chains(
                anchors, timelines, [hop.weight for hop in hops], rrf_k
            )
            selected = select_chains(chains, n, chains_per_video, iou_threshold)

            return [
                TemporalChain(
                    group=chain.group,
                    items=[s.media_id for s in chain.segments],
                    length=chain.length,
                    score=chain.score,
                    start_ms=chain.start_ms,
                    end_ms=chain.end_ms,
                )
                for chain in selected
            ]

        except Exception as e:
            raise SearchError(
                f"Temporal search failed: {e}",
                {"collection": collection, "hops": len(hops)},
            )

    def _rank_later_hops(
        self,
        collection: str,
        queries: List[Tuple[IndexConfig, np.ndarray]],
        hop_filters: List[Optional[ActiveFilters]],
        anchors: List[Segment],
        timeline: "_SegmentTimes",
        excluded_set: set,
    ) -> List[HopTimeline]:
        """Rank each later hop's pool, scoring hops that share an index together."""
        earliest_anchor: Dict[int, float] = {}
        for anchor in anchors:
            earliest = earliest_anchor.get(anchor.group, anchor.start_ms)
            earliest_anchor[anchor.group] = min(earliest, anchor.start_ms)

        # Pool positions per later hop, in its own index
        pools: List[np.ndarray] = []
        passed_by_filter: Dict[str, set] = {}
        for (index_config, _), hop_filter in zip(queries, hop_filters):
            media_ids = timeline.after(earliest_anchor, index_config.source_type)
            media_ids = [m for m in media_ids if m not in excluded_set]
            if hop_filter is not None:
                # Hops often share a filter, so each distinct one is evaluated once
                key = hop_filter.model_dump_json()
                if key not in passed_by_filter:
                    passed_by_filter[key] = set(
                        self.database_repo.get_filtered_media_ids(collection, hop_filter)
                    )
                passed = passed_by_filter[key]
                media_ids = [m for m in media_ids if m in passed]
            pools.append(
                np.asarray(
                    self.database_repo.get_index_ids(
                        collection, media_ids, index_config.name, skip_unmapped=True
                    ),
                    dtype=np.int64,
                )
            )

        hops_by_index: Dict[str, List[int]] = {}
        for hop, (index_config, _) in enumerate(queries):
            hops_by_index.setdefault(index_config.name, []).append(hop)

        timelines: List[Optional[HopTimeline]] = [None] * len(queries)
        for index_name, index_hops in hops_by_index.items():
            candidates = set(np.concatenate([pools[h] for h in index_hops]).tolist())
            ids, scores = self.index_repo.score_subset(
                collection,
                np.vstack([queries[h][1].reshape(1, -1) for h in index_hops]),
                candidates,
                index_name=index_name,
            )
            for row, hop in enumerate(index_hops):
                in_pool = np.isin(ids, pools[hop])
                hop_ids = ids[in_pool]
                order = np.argsort(-scores[row][in_pool], kind="stable")
                media_ids = self.database_repo.get_media_ids(
                    collection, hop_ids[order].tolist(), index_name
                )
                timelines[hop] = HopTimeline(
                    [
                        timeline.segment(media_id, rank)
                        for rank, media_id in enumerate(media_ids, start=1)
                    ]
                )
        return timelines


class _SegmentTimes:
    """Lookup over the rows of `DatabaseRepository.get_group_segment_times`."""

    def __init__(self, rows: List[Tuple[int, int, str, float, float]]):
        self._rows: Dict[int, Tuple[int, str, float, float]] = {
            media_id: (group, source_type, start_ms, end_ms)
            for media_id, group, source_type, start_ms, end_ms in rows
        }

    def __contains__(self, media_id: int) -> bool:
        return media_id in self._rows

    def segment(self, media_id: int, rank: int) -> Segment:
        group, _, start_ms, end_ms = self._rows[media_id]
        return Segment(media_id, group, start_ms, end_ms, rank)

    def after(self, earliest: Dict[int, float], source_type: str) -> List[int]:
        """Media IDs of the source type starting strictly after their group's earliest time."""
        return [
            media_id
            for media_id, (group, row_source_type, start_ms, _) in self._rows.items()
            if row_source_type == source_type
            and group in earliest
            and start_ms > earliest[group]
        ]
