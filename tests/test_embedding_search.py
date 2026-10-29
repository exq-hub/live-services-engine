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

"""Tests for VectorSearchMixin's choice between exact subset search and
ANN search when filters are active.

A filter whose (seen/excluded-pruned) candidate set fits within
EXACT_SEARCH_THRESHOLD is scored directly against the embeddings via
`IndexRepository.search_subset`. A larger one falls back to the ANN index
with everything outside the candidates added to skip_ids.
"""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.strategies import embedding_search
from app.strategies.embedding_search import VectorSearchMixin


class _Searcher(VectorSearchMixin):
    def __init__(self):
        self.index_repo = MagicMock()
        self.database_repo = MagicMock()


@pytest.fixture
def searcher():
    s = _Searcher()
    # Media 100 + p is at index position p, so a missed conversion fails a test
    s.database_repo.get_index_ids.side_effect = (
        lambda collection, ids, index_name, skip_unmapped=False: [m - 100 for m in ids]
    )
    s.database_repo.get_media_ids.side_effect = (
        lambda collection, ids, index_name: [p + 100 for p in ids]
    )
    s.database_repo.get_filtered_media_ids.return_value = {101, 102, 103, 104}
    s.database_repo.get_total_items.return_value = 10
    s.index_repo.search_subset.return_value = ([4, 1], [0.9, 0.5])
    s.index_repo.search.return_value = ([4, 1], [0.9, 0.5])
    return s


def _run(searcher, filters):
    return asyncio.run(
        searcher._search_with_expansion(
            "testcol",
            "idx",
            np.zeros((1, 3)),
            2,
            seen_set={102},
            excluded_set={103, 107},
            filters=filters,
        )
    )


def test_small_candidate_set_uses_exact_subset_search(searcher):
    result = _run(searcher, filters=MagicMock())

    searcher.index_repo.search.assert_not_called()
    _, _, k, candidates = searcher.index_repo.search_subset.call_args.args
    assert k == 2
    assert candidates == {1, 4}
    assert result == [104, 101]


def test_large_candidate_set_falls_back_to_ann_search(searcher, monkeypatch):
    monkeypatch.setattr(embedding_search, "EXACT_SEARCH_THRESHOLD", 1)

    result = _run(searcher, filters=MagicMock())

    searcher.index_repo.search_subset.assert_not_called()
    skip_ids = searcher.index_repo.search.call_args.kwargs["skip_ids"]
    assert skip_ids == {0, 2, 3, 5, 6, 7, 8, 9}
    assert result == [104, 101]


def test_no_filters_uses_ann_search_with_seen_and_excluded_skipped(searcher):
    _run(searcher, filters=None)

    searcher.index_repo.search_subset.assert_not_called()
    assert searcher.index_repo.search.call_args.kwargs["skip_ids"] == {2, 3, 7}
