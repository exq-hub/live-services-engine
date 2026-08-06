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

"""Tests for CLIPSearchStrategy's text-encoder model-name resolution.

Only covers _resolve_model_name -- the hook that will later switch from
"the collection's default index" to "whichever index the request
selected" once index selection lands. index_repository/database_repository
aren't touched by this method, so they're stubbed out.
"""

from unittest.mock import MagicMock

from app.core.config import ConfigManager
from app.core.models import CLIPModelManager
from app.strategies.clip_search import CLIPSearchStrategy

from .conftest import collection_toml, index_toml


class TestResolveModelName:
    def test_uses_collection_default_index_model_name(self, write_config, dummy_files):
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

        assert strategy._resolve_model_name("testcol") == "model-one"

    def test_single_index_collection_resolves_to_its_own_model(
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

        assert strategy._resolve_model_name("testcol") == "only-model"
