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

"""Tests for how ItemService shapes item info responses."""

from unittest.mock import MagicMock

import pytest

from app.core.config import ConfigManager
from app.services.item_service import ItemService

from .conftest import collection_toml, index_toml


@pytest.fixture
def collection_config(write_config, dummy_files):
    collection = collection_toml(
        name="testcol",
        database_file=dummy_files["database_file"],
        thumbnail_media_url="https://localhost:5000/thumbs",
        original_media_url="https://localhost:5000/originals",
        indexes=index_toml(
            name="primary",
            index_type="zarr",
            embeddings_file=dummy_files["embeddings_file"],
        ),
        thumbnail_media_params={"token": "tok"},
        original_media_params={"token": "tok"},
    )
    path = write_config(collection)
    return ConfigManager(str(path)).load_config().collection_configs["testcol"]


def _item(thumbnail_uri):
    return {
        "item_id": 7,
        "source_type": 2,
        "thumbnail_uri": thumbnail_uri,
        "media_uri": "videos/00001.mp4",
        "group": None,
    }


class TestFormatItem:
    def test_media_without_a_thumbnail_has_a_null_thumb_path(self, collection_config):
        service = ItemService(MagicMock(), MagicMock())

        result = service._format_item(7, _item(None), collection_config, False)

        assert result["thumbPath"] is None
        assert (
            result["srcPath"]
            == "https://localhost:5000/originals/videos/00001.mp4?token=tok"
        )

    def test_media_with_a_thumbnail_gets_the_full_url(self, collection_config):
        service = ItemService(MagicMock(), MagicMock())

        result = service._format_item(7, _item("a/b.jpg"), collection_config, False)

        assert result["thumbPath"] == "https://localhost:5000/thumbs/a/b.jpg?token=tok"
        assert (
            result["srcPath"]
            == "https://localhost:5000/originals/videos/00001.mp4?token=tok"
        )
