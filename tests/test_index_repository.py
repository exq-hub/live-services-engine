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

"""Tests for IndexRepository's lazy, config-driven loading.

IndexRepository holds an `LSEConfig` and resolves/loads a collection's
indexes on demand: a cache miss on `get_index`/`get_embeddings_zarr_path`
triggers loading straight from config, defaulting to the collection's
default index when no index_name is given. `preload()` is the eager path
ApplicationContainer uses at startup for the default (and, for
preload_all_indexes collections, every) index.

Real (tiny) Zarr stores are used here rather than empty touched files,
since ZarrIndex.load_index actually opens and reads them.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr

from app.core.config import ConfigManager
from app.core.exceptions import IndexError
from app.repositories.index_repository import IndexRepository

from .conftest import collection_toml, index_toml


def _make_zarr_store(path: Path) -> str:
    """Create a real, minimal Zarr directory store with an 'embeddings' array."""
    root = zarr.open_group(str(path), mode="w")
    root.create_array("embeddings", shape=(4, 3), dtype="f4")
    root["embeddings"][:] = np.arange(12).reshape(4, 3)
    return str(path)


@pytest.fixture
def multi_index_config(write_config, dummy_files, tmp_path):
    """One collection with two real zarr indexes: one default, one not."""
    default_store = _make_zarr_store(tmp_path / "default.zarr")
    extra_store = _make_zarr_store(tmp_path / "extra.zarr")

    indexes = "\n".join(
        [
            index_toml(
                name="default_idx",
                index_type="zarr",
                embeddings_file=default_store,
                default=True,
            ),
            index_toml(
                name="extra_idx",
                index_type="zarr",
                embeddings_file=extra_store,
            ),
        ]
    )
    collection = collection_toml(
        name="testcol",
        database_file=dummy_files["database_file"],
        thumbnail_media_url="https://localhost:5000/testcol",
        original_media_url="https://localhost:5000/testcol",
        indexes=indexes,
    )
    path = write_config(collection)
    return ConfigManager(str(path)).load_config()


class TestGetIndex:
    def test_loads_default_index_when_name_omitted(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        index = repo.get_index("testcol")
        assert index is not None

    def test_caches_across_calls(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        first = repo.get_index("testcol")
        second = repo.get_index("testcol")
        assert first is second

    def test_omitted_name_and_explicit_default_name_share_the_same_cache_entry(
        self, multi_index_config
    ):
        repo = IndexRepository(multi_index_config)
        implicit = repo.get_index("testcol")
        explicit = repo.get_index("testcol", "default_idx")
        assert implicit is explicit

    def test_loads_named_non_default_index(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        default_index = repo.get_index("testcol")
        extra_index = repo.get_index("testcol", "extra_idx")
        assert extra_index is not None
        assert extra_index is not default_index

    def test_unknown_collection_raises(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        with pytest.raises(IndexError):
            repo.get_index("does-not-exist")

    def test_unknown_index_name_raises(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        with pytest.raises(IndexError):
            repo.get_index("testcol", "does-not-exist")


class TestGetEmbeddingsZarrPath:
    def test_resolves_default_index_path(self, multi_index_config, tmp_path):
        repo = IndexRepository(multi_index_config)
        path = repo.get_embeddings_zarr_path("testcol")
        assert path == str(tmp_path / "default.zarr")

    def test_resolves_named_index_path(self, multi_index_config, tmp_path):
        repo = IndexRepository(multi_index_config)
        path = repo.get_embeddings_zarr_path("testcol", "extra_idx")
        assert path == str(tmp_path / "extra.zarr")


class TestPreload:
    def test_loads_both_index_and_embeddings_path(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        repo.preload("testcol", "extra_idx")

        assert ("testcol", "extra_idx") in repo._indices
        assert ("testcol", "extra_idx") in repo._embeddings_zarr


class TestSearchSubset:
    # Fixture rows are [0,1,2], [3,4,5], [6,7,8], [9,10,11], so against
    # [1,0,0] each row scores its first element: 0, 3, 6, 9.
    query = np.array([[1.0, 0.0, 0.0]], dtype="f4")

    def test_returns_top_k_in_descending_score_order(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        indices, distances = repo.search_subset("testcol", self.query, 2, {0, 1, 2, 3})
        assert indices == [3, 2]
        assert distances == [9.0, 6.0]

    def test_only_returns_candidates(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        indices, _ = repo.search_subset("testcol", self.query, 2, {0, 2})
        assert indices == [2, 0]

    def test_k_larger_than_candidates_returns_all_sorted(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        indices, _ = repo.search_subset("testcol", self.query, 10, {1, 3, 0})
        assert indices == [3, 1, 0]

    def test_empty_candidates_returns_nothing(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        assert repo.search_subset("testcol", self.query, 5, set()) == ([], [])

    def test_uses_named_index_embeddings(self, multi_index_config, tmp_path):
        extra = zarr.open_group(str(tmp_path / "extra.zarr"), mode="r+")
        extra["embeddings"][:] = -np.arange(12).reshape(4, 3)

        repo = IndexRepository(multi_index_config)
        indices, _ = repo.search_subset(
            "testcol", self.query, 2, {0, 1, 2, 3}, index_name="extra_idx"
        )
        assert indices == [0, 1]


class TestScoreSubset:
    # Same fixture rows as TestSearchSubset, scored against two queries
    # at once, [1,0,0] (first element) and [0,0,1] (last element).
    queries = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype="f4")

    def test_scores_every_query_aligned_with_sorted_ids(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        ids, scores = repo.score_subset("testcol", self.queries, {3, 0, 2})
        assert ids.tolist() == [0, 2, 3]
        assert scores.tolist() == [[0.0, 6.0, 9.0], [2.0, 8.0, 11.0]]

    def test_scores_across_blocks(self, multi_index_config, monkeypatch):
        monkeypatch.setattr(
            "app.repositories.index_repository.SCORE_BLOCK_ROWS", 3
        )
        repo = IndexRepository(multi_index_config)
        ids, scores = repo.score_subset("testcol", self.queries, {0, 1, 2, 3})
        assert ids.tolist() == [0, 1, 2, 3]
        assert scores.tolist() == [[0.0, 3.0, 6.0, 9.0], [2.0, 5.0, 8.0, 11.0]]

    def test_empty_candidates_returns_an_empty_score_row_per_query(
        self, multi_index_config
    ):
        repo = IndexRepository(multi_index_config)
        ids, scores = repo.score_subset("testcol", self.queries, set())
        assert ids.shape == (0,)
        assert scores.shape == (2, 0)


class TestClearCache:
    def test_clear_cache_for_one_collection_leaves_others_intact(
        self, write_config, dummy_files, tmp_path
    ):
        store_a = _make_zarr_store(tmp_path / "a.zarr")
        store_b = _make_zarr_store(tmp_path / "b.zarr")
        collection_a = collection_toml(
            name="col_a",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/a",
            original_media_url="https://localhost:5000/a",
            indexes=index_toml(name="idx", index_type="zarr", embeddings_file=store_a),
        )
        collection_b = collection_toml(
            name="col_b",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/b",
            original_media_url="https://localhost:5000/b",
            indexes=index_toml(name="idx", index_type="zarr", embeddings_file=store_b),
        )
        path = write_config(collection_a + "\n\n" + collection_b)
        config = ConfigManager(str(path)).load_config()

        repo = IndexRepository(config)
        repo.preload("col_a")
        repo.preload("col_b")

        repo.clear_cache("col_a")

        assert ("col_a", "idx") not in repo._indices
        assert ("col_b", "idx") in repo._indices

    def test_clear_cache_with_no_argument_clears_everything(self, multi_index_config):
        repo = IndexRepository(multi_index_config)
        repo.preload("testcol")

        repo.clear_cache()

        assert repo._indices == {}
        assert repo._embeddings_zarr == {}
