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

"""Tests for CLIPSearchStrategy's index resolution.

_resolve_index (inherited from EmbeddingSearchStrategy) must resolve to
a CLIP-type index specifically -- not just "the collection's default
index" -- since a collection's overall default could belong to a
different family. index_repository/database_repository aren't touched
by this method, so they're stubbed out.
"""

from unittest.mock import MagicMock

import pytest

from app.core.config import ConfigManager
from app.core.models import CLIPModelManager
from app.strategies.clip_search import CLIPSearchStrategy

from .conftest import collection_toml, index_toml


class TestResolveIndex:
    def test_uses_collection_default_index_when_it_is_clip_type(
        self, write_config, dummy_files
    ):
        indexes = "\n".join(
            [
                index_toml(
                    name="a",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    model_name="model-one",
                    default=True,
                ),
                index_toml(
                    name="b",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    model_name="model-two",
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
        config = ConfigManager(str(path)).load_config()

        model_manager = CLIPModelManager(config)
        strategy = CLIPSearchStrategy(model_manager, MagicMock(), MagicMock())

        resolved = strategy._resolve_index("testcol")
        assert resolved.name == "a"
        assert resolved.model_name == "model-one"

    def test_single_index_collection_resolves_to_its_own_index(
        self, write_config, dummy_files
    ):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="CLIP",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                model_name="only-model",
            ),
        )
        path = write_config(collection)
        config = ConfigManager(str(path)).load_config()

        model_manager = CLIPModelManager(config)
        strategy = CLIPSearchStrategy(model_manager, MagicMock(), MagicMock())

        assert strategy._resolve_index("testcol").model_name == "only-model"

    def test_falls_back_to_the_clip_type_index_when_overall_default_is_text(
        self, write_config, dummy_files
    ):
        # The collection's *overall* default is the Text index; CLIPSearchStrategy
        # must still resolve to the CLIP-type one, not blindly use the default.
        indexes = "\n".join(
            [
                index_toml(
                    name="text_default",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    embedding_type="Text",
                    model_name="text-model",
                    default=True,
                ),
                index_toml(
                    name="clip_idx",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    model_name="clip-model",
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
        config = ConfigManager(str(path)).load_config()

        model_manager = CLIPModelManager(config)
        strategy = CLIPSearchStrategy(model_manager, MagicMock(), MagicMock())

        resolved = strategy._resolve_index("testcol")
        assert resolved.name == "clip_idx"
        assert resolved.model_name == "clip-model"

    def test_raises_when_collection_has_no_clip_type_index(
        self, write_config, dummy_files
    ):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="text_only",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                embedding_type="Text",
            ),
        )
        path = write_config(collection)
        config = ConfigManager(str(path)).load_config()

        model_manager = CLIPModelManager(config)
        strategy = CLIPSearchStrategy(model_manager, MagicMock(), MagicMock())

        with pytest.raises(ValueError):
            strategy._resolve_index("testcol")

    def test_explicit_index_name_overrides_the_family_default(
        self, write_config, dummy_files
    ):
        # Two CLIP-type indexes -- without an explicit index_name there'd be
        # no way to reach "b" at all, since it isn't the collection default.
        indexes = "\n".join(
            [
                index_toml(
                    name="a",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    model_name="model-one",
                    default=True,
                ),
                index_toml(
                    name="b",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    model_name="model-two",
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
        config = ConfigManager(str(path)).load_config()

        model_manager = CLIPModelManager(config)
        strategy = CLIPSearchStrategy(model_manager, MagicMock(), MagicMock())

        resolved = strategy._resolve_index("testcol", index_name="b")
        assert resolved.name == "b"
        assert resolved.model_name == "model-two"

    def test_explicit_index_name_of_the_wrong_family_raises(
        self, write_config, dummy_files
    ):
        indexes = "\n".join(
            [
                index_toml(
                    name="clip_idx",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    default=True,
                ),
                index_toml(
                    name="text_idx",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    embedding_type="Text",
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
        config = ConfigManager(str(path)).load_config()

        model_manager = CLIPModelManager(config)
        strategy = CLIPSearchStrategy(model_manager, MagicMock(), MagicMock())

        with pytest.raises(ValueError):
            strategy._resolve_index("testcol", index_name="text_idx")
