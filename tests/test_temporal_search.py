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

"""Tests for temporal search.

The chain logic (timeline lookup, chain walking, RRF, overlap suppression)
is tested directly as pure functions. `TemporalSearchStrategy` is tested
with mocked repositories and hop strategies, checking how it builds each
hop's pool and batches the scoring of hops that share an index.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from app.core.exceptions import SearchError
from app.schemas import (
    ActiveFilters,
    DBFilter,
    DBValueConstraint,
    FilterLeaf,
    QueryHop,
    RFHop,
    TemporalSearchRequest,
)
from app.strategies.temporal_search import (
    Chain,
    HopTimeline,
    Segment,
    TemporalSearchStrategy,
    assemble_chains,
    interval_iou,
    rrf_score,
    select_chains,
)


def _seg(media_id, start, rank, group=1, end=None):
    return Segment(media_id, group, start, start if end is None else end, rank)


def _chain(group, spans, score, first_rank=1):
    """A chain over (start, end) spans, the first one ranked first_rank."""
    segments = [
        Segment(i, group, start, end, first_rank if i == 0 else 1)
        for i, (start, end) in enumerate(spans)
    ]
    return Chain(group=group, segments=segments, score=score)


class TestHopTimeline:
    timeline = HopTimeline([_seg(1, 0, 3), _seg(2, 10, 1), _seg(3, 20, 2)])

    def test_picks_the_best_rank_among_later_segments(self):
        assert self.timeline.best_after(1, -1).media_id == 2

    def test_is_strictly_after(self):
        assert self.timeline.best_after(1, 10).media_id == 3

    def test_none_after_the_last_segment(self):
        assert self.timeline.best_after(1, 20) is None

    def test_none_for_an_unknown_video(self):
        assert self.timeline.best_after(2, 0) is None


class TestAssembleChains:
    def test_walks_forward_through_every_hop(self):
        timelines = [
            HopTimeline([_seg(20, 15, 1), _seg(21, 5, 2)]),
            HopTimeline([_seg(30, 10, 1), _seg(31, 25, 2)]),
        ]
        (chain,) = assemble_chains([_seg(10, 0, 1)], timelines, [1, 1, 1], 60)
        assert [s.media_id for s in chain.segments] == [10, 20, 31]

    def test_stops_at_the_first_hop_without_a_match(self):
        timelines = [
            HopTimeline([_seg(20, 15, 1)]),
            HopTimeline([_seg(30, 5, 1)]),
            HopTimeline([_seg(40, 50, 1)]),
        ]
        (chain,) = assemble_chains([_seg(10, 0, 1)], timelines, [1, 1, 1, 1], 60)
        assert [s.media_id for s in chain.segments] == [10, 20]

    def test_scores_matched_hops_with_rrf(self):
        timelines = [HopTimeline([_seg(20, 15, 4)]), HopTimeline([])]
        (chain,) = assemble_chains([_seg(10, 0, 2)], timelines, [1, 2, 3], 60)
        assert chain.score == pytest.approx(1 / 62 + 2 / 64)


def test_rrf_score_applies_weights_and_k():
    assert rrf_score([1, 2], [1.0, 2.0], 60) == pytest.approx(1 / 61 + 2 / 62)


class TestIntervalIoU:
    def test_partial_overlap(self):
        assert interval_iou(_chain(1, [(0, 10)], 0), _chain(1, [(5, 15)], 0)) == (
            pytest.approx(5 / 15)
        )

    def test_identical_zero_length_spans_fully_overlap(self):
        assert interval_iou(_chain(1, [(5, 5)], 0), _chain(1, [(5, 5)], 0)) == 1.0

    def test_disjoint_zero_length_spans_do_not_overlap(self):
        assert interval_iou(_chain(1, [(5, 5)], 0), _chain(1, [(6, 6)], 0)) == 0.0


class TestSelectChains:
    def test_length_ranks_before_score(self):
        short = _chain(1, [(0, 1)], score=1.0)
        long = _chain(2, [(0, 1), (2, 3)], score=0.1)
        assert select_chains([short, long], 2, 1, 0.5) == [long, short]

    def test_overlap_keeps_the_longer_chain(self):
        short = _chain(1, [(0, 10)], score=1.0)
        long = _chain(1, [(0, 4), (5, 10)], score=0.1)
        assert select_chains([short, long], 1, -1, 0.5) == [long]

    def test_overlap_of_equal_length_keeps_the_higher_score(self):
        low = _chain(1, [(0, 10)], score=0.1)
        high = _chain(1, [(1, 10)], score=0.2)
        assert select_chains([low, high], 1, -1, 0.5) == [high]

    def test_overlap_equal_to_the_threshold_is_kept(self):
        a = _chain(1, [(0, 10)], score=0.2)
        b = _chain(1, [(5, 15)], score=0.1)
        assert select_chains([a, b], 1, -1, 5 / 15) == [a, b]

    def test_chains_per_video_limits_each_video(self):
        chains = [_chain(1, [(i * 10, i * 10 + 1)], score=1 / (i + 1)) for i in range(3)]
        assert len(select_chains(chains, 1, 1, 0.5)) == 1
        assert len(select_chains(chains, 1, 2, 0.5)) == 2
        assert len(select_chains(chains, 1, -1, 0.5)) == 3

    def test_n_limits_videos_and_groups_chains_by_video(self):
        a1 = _chain(1, [(0, 1)], score=0.9)
        b1 = _chain(2, [(0, 1)], score=0.8)
        a2 = _chain(1, [(5, 6)], score=0.7)
        c1 = _chain(3, [(0, 1)], score=0.6)
        assert select_chains([c1, a2, b1, a1], 2, -1, 0.5) == [a1, a2, b1]


class TestTemporalSearchRequest:
    session = {"session": "s", "modelId": 1, "collection": "c"}

    def test_hops_are_discriminated_by_kind(self):
        request = TemporalSearchRequest(
            session_info=self.session,
            hops=[{"kind": "query", "text": "a"}, {"kind": "rf", "pos": [1]}],
        )
        assert [type(h) for h in request.hops] == [QueryHop, RFHop]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"hops": []},
            {"hops": [{"kind": "clip", "text": "a"}]},
            {"chains_per_video": 0},
            {"chains_per_video": -2},
            {"iou_threshold": 1.5},
            {"rrf_k": 0},
            {"top_r": 0},
        ],
    )
    def test_rejects_invalid_parameters(self, overrides):
        fields = {"session_info": self.session, "hops": [{"kind": "query", "text": "a"}]}
        with pytest.raises(ValidationError):
            TemporalSearchRequest(**{**fields, **overrides})

    def test_chains_per_video_accepts_all(self):
        request = TemporalSearchRequest(
            session_info=self.session,
            hops=[{"kind": "query", "text": "a"}],
            chains_per_video=-1,
        )
        assert request.chains_per_video == -1


CLIP_INDEX = SimpleNamespace(
    name="CLIP", source_type="Image", embedding_type="CLIP", model_name="m"
)
TEXT_INDEX = SimpleNamespace(
    name="Text", source_type="Text", embedding_type="Text", model_name="m"
)
INDEXES = {"CLIP": CLIP_INDEX, "Text": TEXT_INDEX}
"""The collection's indexes by name, CLIP being the default."""

# Video 100 has Image segments 10..13 and Text segments 20, 21. Video 200
# has Image segments 30, 31. Rows are (media_id, group, source_type, start, end).
SEGMENT_ROWS = [
    (10, 100, "Image", 0, 0),
    (11, 100, "Image", 1000, 1000),
    (12, 100, "Image", 2000, 2000),
    (13, 100, "Image", 3000, 3000),
    (20, 100, "Text", 1500, 1500),
    (21, 100, "Text", 2500, 2500),
    (30, 200, "Image", 0, 0),
    (31, 200, "Image", 1000, 1000),
]

# Score of each index position (equal to its media ID) per query text.
SCORES = {
    "b": {20: 0.9, 21: 0.5},
    "c": {11: 0.1, 12: 0.8, 13: 0.9, 31: 0.7},
    "d": {11: 0.9, 12: 0.1, 13: 0.1, 31: 0.1},
}
QUERY_TEXTS = ["a", "b", "c", "d"]
"""Query texts, encoded as their position here so the fake scores can find them."""


def _score_subset(collection, query_vectors, candidate_ids, index_name=None):
    ids = np.array(sorted(candidate_ids), dtype=np.int64)
    scores = np.array(
        [
            [SCORES[QUERY_TEXTS[int(q[0])]].get(int(i), 0.0) for i in ids]
            for q in query_vectors
        ]
    ).reshape(len(query_vectors), len(ids))
    return ids, scores


def _encoder(index_config):
    """build_query of an embedding strategy, encoding a text as its QUERY_TEXTS position."""

    def build_query(collection, text, index_name):
        assert index_name == index_config.name
        return index_config, np.array([[float(QUERY_TEXTS.index(text))]])

    return AsyncMock(side_effect=build_query)


@pytest.fixture
def strategy():
    config = MagicMock()
    config.collection_configs = {"testcol": MagicMock()}
    config.collection_configs["testcol"].resolve_index.side_effect = (
        lambda name=None: INDEXES[name or "CLIP"]
    )
    clip = MagicMock(embedding_type="CLIP", build_query=_encoder(CLIP_INDEX))
    text = MagicMock(embedding_type="Text", build_query=_encoder(TEXT_INDEX))
    rf = MagicMock()
    index_repo, database_repo = MagicMock(), MagicMock()
    # Media IDs and index positions coincide, to keep the pools readable.
    database_repo.get_index_ids.side_effect = (
        lambda collection, ids, index_name, skip_unmapped=False: list(ids)
    )
    database_repo.get_media_ids.side_effect = (
        lambda collection, ids, index_name: list(ids)
    )
    database_repo.get_group_segment_times.return_value = SEGMENT_ROWS
    index_repo.score_subset.side_effect = _score_subset

    s = TemporalSearchStrategy(config, clip, text, rf, index_repo, database_repo)
    s._search_with_expansion = AsyncMock(return_value=[10, 30])
    return s


def _search(strategy, hops, **kwargs):
    return asyncio.run(
        strategy.search(
            collection="testcol", hops=hops, n=10, seen=[], excluded=[], **kwargs
        )
    )


class TestTemporalSearchStrategy:
    def test_chains_cross_indexes_in_time_order(self, strategy):
        chains = _search(
            strategy,
            [
                QueryHop(kind="query", text="a"),
                QueryHop(kind="query", text="b", index_name="Text"),
                QueryHop(kind="query", text="c", index_name="CLIP"),
            ],
        )

        assert [(c.group, c.items) for c in chains] == [
            (100, [10, 20, 13]),
            (200, [30]),
        ]
        assert chains[0].score == pytest.approx(3 / 61)
        assert (chains[0].start_ms, chains[0].end_ms) == (0, 3000)
        _, _, source_types = strategy.database_repo.get_group_segment_times.call_args.args
        assert source_types == ["Image", "Text"]

    def test_query_hop_is_encoded_by_its_index_family(self, strategy):
        _search(
            strategy,
            [
                QueryHop(kind="query", text="a"),
                QueryHop(kind="query", text="b", index_name="Text"),
            ],
        )

        clip_call = strategy.query_strategies["CLIP"].build_query.call_args
        text_call = strategy.query_strategies["Text"].build_query.call_args
        assert clip_call.args == ("testcol", "a", "CLIP")
        assert text_call.args == ("testcol", "b", "Text")

    def test_unknown_embedding_type_fails(self, strategy):
        resolve_index = strategy.config.collection_configs["testcol"].resolve_index
        resolve_index.side_effect = lambda name=None: SimpleNamespace(
            name="SIFT", embedding_type="SIFT"
        )
        with pytest.raises(SearchError):
            _search(strategy, [QueryHop(kind="query", text="a")])

    def test_text_hop_pool_only_holds_its_own_source_type(self, strategy):
        _search(
            strategy,
            [
                QueryHop(kind="query", text="a"),
                QueryHop(kind="query", text="b", index_name="Text"),
            ],
        )

        (call,) = strategy.index_repo.score_subset.call_args_list
        assert call.args[2] == {20, 21}
        assert call.kwargs["index_name"] == "Text"

    def test_hops_sharing_an_index_are_scored_together(self, strategy):
        chains = _search(
            strategy,
            [
                QueryHop(kind="query", text="a"),
                QueryHop(kind="query", text="d"),
                QueryHop(kind="query", text="c"),
            ],
        )

        (call,) = strategy.index_repo.score_subset.call_args_list
        assert call.args[1].shape == (2, 1)
        assert call.args[2] == {11, 12, 13, 31}
        assert [c.items for c in chains] == [[10, 11, 13], [30, 31]]

    def test_rf_hop_without_positives_fails(self, strategy):
        strategy.rf_search.build_query = AsyncMock(return_value=(CLIP_INDEX, None))
        with pytest.raises(SearchError):
            _search(strategy, [RFHop(kind="rf")])

    def test_no_anchors_returns_nothing(self, strategy):
        strategy._search_with_expansion = AsyncMock(return_value=[])
        assert _search(strategy, [QueryHop(kind="query", text="a")]) == []

    def test_a_filter_shared_by_later_hops_is_evaluated_once(self, strategy):
        strategy.database_repo.get_filtered_media_ids.return_value = [11, 13, 31]
        chains = _search(
            strategy,
            [
                QueryHop(kind="query", text="a"),
                QueryHop(kind="query", text="d"),
                QueryHop(kind="query", text="c"),
            ],
            filters=_tag_filter(1),
        )

        strategy.database_repo.get_filtered_media_ids.assert_called_once()
        assert [c.items for c in chains] == [[10, 11, 13], [30, 31]]

    def test_each_distinct_hop_filter_is_evaluated(self, strategy):
        strategy.database_repo.get_filtered_media_ids.return_value = [11, 13, 31]
        _search(
            strategy,
            [
                QueryHop(kind="query", text="a"),
                QueryHop(kind="query", text="d", filters=_tag_filter(2)),
                QueryHop(kind="query", text="c"),
            ],
            filters=_tag_filter(1),
        )

        assert strategy.database_repo.get_filtered_media_ids.call_count == 2


def _tag_filter(tag_id):
    return ActiveFilters(
        root=FilterLeaf(
            filter=DBFilter(
                id=1, tagtype_id=1, constraint=DBValueConstraint(value_ids=[tag_id])
            )
        )
    )
