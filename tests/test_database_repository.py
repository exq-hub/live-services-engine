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

"""Tests for DatabaseRepository's per-index, config-driven id mapping.

`create_item_to_datapoint_mapping` reads an index's `tagset_name`
(defaulting to `<index name> Index ID`) from a real (minimal)
M3-DB-schema SQLite database -- these tests build one directly, matching
https://github.com/Ok2610/Simplified-M3-DB, rather than mocking the
database layer, since the actual SQL joins are exactly what's under test.

Like IndexRepository, the default index's mapping loads eagerly in
load_database; any other index's mapping loads lazily on first use,
or eagerly too if the collection sets preload_all_indexes.
"""

import sqlite3
from pathlib import Path

import pytest

from app.core.config import ConfigManager
from app.core.exceptions import DatabaseError
from app.repositories.database_repository import DatabaseRepository

from .conftest import collection_toml, index_toml


def _build_sqlite_db(path: Path) -> None:
    """A minimal M3-DB-schema database with two index-mapping tagsets.

    - source_types: Image=1, Video=2, Audio=3, Text=4, Other=5.
    - medias 10, 11 are Image keyframes mapped by 'CLIP Index ID' (positions 0, 1).
    - media 20 is a Text transcript mapped by 'Text Index ID' (position 0).
    """
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE tag_types (id INTEGER PRIMARY KEY, description TEXT);
        CREATE TABLE source_types (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE tagsets (id INTEGER PRIMARY KEY, name TEXT, tagtype_id INTEGER);
        CREATE TABLE medias (
            id INTEGER PRIMARY KEY,
            source TEXT,
            source_type INTEGER,
            thumbnail_uri TEXT,
            group_id INTEGER
        );
        CREATE TABLE taggings (media_id INTEGER, tag_id INTEGER);
        CREATE TABLE numerical_int_tags (
            id INTEGER PRIMARY KEY,
            value INTEGER,
            tagset_id INTEGER
        );

        INSERT INTO source_types (id, name) VALUES
            (1, 'Image'), (2, 'Video'), (3, 'Audio'), (4, 'Text'), (5, 'Other');
        INSERT INTO tag_types (id, description) VALUES (1, 'numerical_int');
        INSERT INTO tagsets (id, name, tagtype_id) VALUES
            (1, 'CLIP Index ID', 1), (2, 'Text Index ID', 1);

        INSERT INTO medias (id, source, source_type, thumbnail_uri, group_id) VALUES
            (10, 'img1.jpg', 1, 'thumb1.jpg', 100),
            (11, 'img2.jpg', 1, 'thumb2.jpg', 100),
            (20, 'transcript1.txt', 4, 'thumb3.jpg', 200);

        INSERT INTO numerical_int_tags (id, value, tagset_id) VALUES
            (1000, 0, 1), (1001, 1, 1), (2000, 0, 2);
        INSERT INTO taggings (media_id, tag_id) VALUES
            (10, 1000), (11, 1001), (20, 2000);
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture
def multi_index_db_config(write_config, dummy_files, tmp_path):
    """One collection: CLIP/Image index (default), Text index (not default)."""
    db_path = tmp_path / "collection.db"
    _build_sqlite_db(db_path)

    indexes = "\n".join(
        [
            index_toml(
                name="CLIP",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                default=True,
            ),
            index_toml(
                name="Text",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                embedding_type="Text",
                source_type="Text",
            ),
        ]
    )
    collection = collection_toml(
        name="testcol",
        database_file=str(db_path),
        thumbnail_media_url="https://localhost:5000/testcol",
        original_media_url="https://localhost:5000/testcol",
        indexes=indexes,
    )
    path = write_config(collection)
    return ConfigManager(str(path)).load_config()


def _load(config):
    repo = DatabaseRepository(config)
    db_file = config.collection_configs["testcol"].database_file
    repo.load_database("testcol", db_file)
    return repo


class TestLoadDatabaseEagerLoading:
    def test_eagerly_builds_default_index_mapping(self, multi_index_db_config):
        repo = _load(multi_index_db_config)
        assert repo.get_total_items("testcol", "CLIP") == 2

    def test_does_not_eagerly_build_non_default_index_mapping(
        self, multi_index_db_config
    ):
        repo = _load(multi_index_db_config)
        assert "Text" not in repo._item_datapoint_mapping_cache["testcol"]

    def test_preload_all_indexes_eagerly_builds_every_mapping(
        self, write_config, dummy_files, tmp_path
    ):
        db_path = tmp_path / "collection.db"
        _build_sqlite_db(db_path)
        indexes = "\n".join(
            [
                index_toml(
                    name="CLIP",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    default=True,
                ),
                index_toml(
                    name="Text",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    embedding_type="Text",
                    source_type="Text",
                ),
            ]
        )
        collection = collection_toml(
            name="testcol",
            database_file=str(db_path),
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=indexes,
            preload_all_indexes=True,
        )
        path = write_config(collection)
        config = ConfigManager(str(path)).load_config()

        repo = _load(config)

        assert "Text" in repo._item_datapoint_mapping_cache["testcol"]


class TestLazyLoad:
    def test_get_total_items_lazily_builds_non_default_index_mapping(
        self, multi_index_db_config
    ):
        repo = _load(multi_index_db_config)
        assert "Text" not in repo._item_datapoint_mapping_cache["testcol"]

        assert repo.get_total_items("testcol", "Text") == 1

        assert "Text" in repo._item_datapoint_mapping_cache["testcol"]


class TestGetMediaIdsAndIndexIds:
    def test_get_media_ids_defaults_to_the_collection_default_index(
        self, multi_index_db_config
    ):
        repo = _load(multi_index_db_config)
        assert repo.get_media_ids("testcol", [0, 1]) == [10, 11]

    def test_get_media_ids_with_explicit_non_default_index(
        self, multi_index_db_config
    ):
        repo = _load(multi_index_db_config)
        assert repo.get_media_ids("testcol", [0], index="Text") == [20]

    def test_get_index_ids_roundtrips_get_media_ids(self, multi_index_db_config):
        repo = _load(multi_index_db_config)
        assert repo.get_index_ids("testcol", [10, 11]) == [0, 1]

    def test_get_index_ids_with_explicit_non_default_index(
        self, multi_index_db_config
    ):
        repo = _load(multi_index_db_config)
        assert repo.get_index_ids("testcol", [20], index="Text") == [0]


class TestSkipUnmappedIndexIds:
    def test_default_still_raises_for_a_media_id_outside_the_index(
        self, multi_index_db_config
    ):
        repo = _load(multi_index_db_config)
        with pytest.raises(DatabaseError):
            repo.get_index_ids("testcol", [10, 20], index="CLIP")

    def test_skip_unmapped_drops_a_media_id_outside_the_index(
        self, multi_index_db_config
    ):
        repo = _load(multi_index_db_config)
        result = repo.get_index_ids(
            "testcol", [10, 20], index="CLIP", skip_unmapped=True
        )
        assert result == [0]

    def test_skip_unmapped_drops_a_position_outside_the_index_range(
        self, write_config, dummy_files, tmp_path
    ):
        db_path = tmp_path / "collection.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE tag_types (id INTEGER PRIMARY KEY, description TEXT);
            CREATE TABLE source_types (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE tagsets (id INTEGER PRIMARY KEY, name TEXT, tagtype_id INTEGER);
            CREATE TABLE medias (
                id INTEGER PRIMARY KEY,
                source TEXT,
                source_type INTEGER,
                thumbnail_uri TEXT,
                group_id INTEGER
            );
            CREATE TABLE taggings (media_id INTEGER, tag_id INTEGER);
            CREATE TABLE numerical_int_tags (
                id INTEGER PRIMARY KEY,
                value INTEGER,
                tagset_id INTEGER
            );

            INSERT INTO source_types (id, name) VALUES (1, 'Image');
            INSERT INTO tag_types (id, description) VALUES (1, 'numerical_int');
            INSERT INTO tagsets (id, name, tagtype_id) VALUES (1, 'CLIP Index ID', 1);

            INSERT INTO medias (id, source, source_type, thumbnail_uri, group_id) VALUES
                (10, 'img1.jpg', 1, 'thumb1.jpg', 100),
                (99, 'bad.jpg', 1, 'thumb_bad.jpg', 999);

            INSERT INTO numerical_int_tags (id, value, tagset_id) VALUES
                (1000, 0, 1), (1001, 5, 1);
            INSERT INTO taggings (media_id, tag_id) VALUES
                (10, 1000), (99, 1001);
            """
        )
        conn.commit()
        conn.close()

        collection = collection_toml(
            name="testcol",
            database_file=str(db_path),
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="CLIP",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
        )
        path = write_config(collection)
        config = ConfigManager(str(path)).load_config()
        repo = _load(config)

        assert repo.get_total_items("testcol", "CLIP") == 2

        result = repo.get_index_ids("testcol", [10, 99], index="CLIP", skip_unmapped=True)
        assert result == [0]


class TestSourceTypeFiltering:
    def test_each_index_only_maps_medias_matching_its_own_source_type(
        self, multi_index_db_config
    ):
        repo = _load(multi_index_db_config)
        assert repo.get_total_items("testcol", "CLIP") == 2
        assert repo.get_total_items("testcol", "Text") == 1


class TestErrors:
    def test_unknown_index_name_raises(self, multi_index_db_config):
        repo = _load(multi_index_db_config)
        with pytest.raises(DatabaseError):
            repo.get_total_items("testcol", "does-not-exist")
