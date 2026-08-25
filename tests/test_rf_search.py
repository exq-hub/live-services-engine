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

"""Tests for RFSearchStrategy's index resolution and family dispatch.

Unlike CLIPSearchStrategy/TextEmbeddingSearchStrategy (each scoped to
one embedding_type), RF isn't tied to one family: it resolves by an
explicit index_name (or the collection's overall default), and its
optional pseudo-RF text-query blending dispatches to whichever internal
strategy (self.clip_search / self.text_search) matches the *resolved*
index's embedding_type -- not a re-resolved one, so it can't drift to a
different same-family index.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from app.core.config import ConfigManager
from app.core.models import CLIPModelManager, TextModelManager
from app.strategies.rf_search import RFSearchStrategy

from .conftest import collection_toml, index_toml


def _make_strategy(write_config, dummy_files, indexes):
    collection = collection_toml(
        name="testcol",
        database_file=dummy_files["database_file"],
        thumbnail_media_url="https://localhost:5000/testcol",
        original_media_url="https://localhost:5000/testcol",
        indexes=indexes,
    )
    path = write_config(collection)
    config = ConfigManager(str(path)).load_config()

    strategy = RFSearchStrategy(
        CLIPModelManager(config),
        TextModelManager(config),
        MagicMock(),
        MagicMock(),
    )
    return strategy


class TestQueryStrategyRegistry:
    def test_registry_has_one_entry_per_family(self, write_config, dummy_files):
        strategy = _make_strategy(
            write_config,
            dummy_files,
            index_toml(
                name="clip_idx",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
        )
        assert set(strategy._query_strategies.keys()) == {"CLIP", "Text"}
        assert strategy._query_strategies["CLIP"] is strategy.clip_search
        assert strategy._query_strategies["Text"] is strategy.text_search


class TestResolveIndex:
    def _two_family_indexes(self, dummy_files):
        return "\n".join(
            [
                index_toml(
                    name="clip_idx",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    model_name="clip-model",
                    default=True,
                ),
                index_toml(
                    name="text_idx",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    embedding_type="Text",
                    model_name="text-model",
                ),
            ]
        )

    def test_no_index_name_falls_back_to_collection_default(
        self, write_config, dummy_files
    ):
        strategy = _make_strategy(
            write_config, dummy_files, self._two_family_indexes(dummy_files)
        )
        resolved = strategy._resolve_index("testcol", None)
        assert resolved.name == "clip_idx"

    def test_explicit_index_name_overrides_the_default(self, write_config, dummy_files):
        strategy = _make_strategy(
            write_config, dummy_files, self._two_family_indexes(dummy_files)
        )
        resolved = strategy._resolve_index("testcol", "text_idx")
        assert resolved.name == "text_idx"
        assert resolved.embedding_type == "Text"

    def test_unknown_index_name_raises(self, write_config, dummy_files):
        strategy = _make_strategy(
            write_config, dummy_files, self._two_family_indexes(dummy_files)
        )
        with pytest.raises(ValueError):
            strategy._resolve_index("testcol", "does-not-exist")


class TestRandomFallbackSamples:
    """Random pos/neg fallback samples are already index positions, not
    media IDs, so they must bypass get_index_ids (which expects the
    latter) -- unlike explicitly-provided pos/neg, which are media IDs
    and do need that conversion.
    """

    def _strategy_with_index(self, write_config, dummy_files):
        return _make_strategy(
            write_config,
            dummy_files,
            index_toml(
                name="idx", index_type="zarr", embeddings_file=dummy_files["embeddings_file"]
            ),
        )

    def test_negative_fallback_skips_get_index_ids(self, write_config, dummy_files):
        strategy = self._strategy_with_index(write_config, dummy_files)

        result = strategy._prepare_negative_samples("testcol", "idx", [], total_items=10)

        strategy.database_repo.get_index_ids.assert_not_called()
        assert len(result) == 5
        assert all(0 <= v < 10 for v in result)

    def test_positive_fallback_skips_get_index_ids(self, write_config, dummy_files):
        strategy = self._strategy_with_index(write_config, dummy_files)
        strategy.database_repo.get_total_items.return_value = 10
        index_config = strategy.config.collection_configs["testcol"].get_index("idx")

        result = asyncio.run(
            strategy._prepare_positive_samples(
                "testcol", index_config, [], None, [], [], None
            )
        )

        strategy.database_repo.get_index_ids.assert_not_called()
        assert len(result) == 5
        assert all(0 <= v < 10 for v in result)
