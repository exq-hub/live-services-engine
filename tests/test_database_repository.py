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
from app.schemas import ActiveFilters, DBFilter, DBValueConstraint, FilterLeaf

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


@pytest.fixture
def related_db_config(multi_index_db_config):
    """Extends the base database with sortable tagsets for group 100.

    - media 12 joins group 100 (no index mapping, no sort values).
    - 'Frame' (numerical_int): 10 -> 50, 11 -> 20 and 60 (multi-valued).
    - 'Label' (alphanumerical): 10 -> 'b', 11 -> 'a', 12 -> 'c'.
    """
    db_file = multi_index_db_config.collection_configs["testcol"].database_file
    conn = sqlite3.connect(str(db_file))
    conn.executescript(
        """
        CREATE TABLE alphanumerical_tags (
            id INTEGER PRIMARY KEY,
            value TEXT,
            tagset_id INTEGER
        );

        INSERT INTO tag_types (id, description) VALUES (2, 'alphanumerical');
        INSERT INTO tagsets (id, name, tagtype_id) VALUES
            (3, 'Frame', 1), (4, 'Label', 2);

        INSERT INTO medias (id, source, source_type, thumbnail_uri, group_id) VALUES
            (12, 'img3.jpg', 1, 'thumb4.jpg', 100);

        INSERT INTO numerical_int_tags (id, value, tagset_id) VALUES
            (3000, 50, 3), (3001, 20, 3), (3002, 60, 3);
        INSERT INTO alphanumerical_tags (id, value, tagset_id) VALUES
            (4000, 'b', 4), (4001, 'a', 4), (4002, 'c', 4);
        INSERT INTO taggings (media_id, tag_id) VALUES
            (10, 3000), (11, 3001), (11, 3002),
            (10, 4000), (11, 4001), (12, 4002);
        """
    )
    conn.commit()
    conn.close()
    return multi_index_db_config


class TestGetRelatedItems:
    def test_unsorted_returns_every_group_member(self, related_db_config):
        repo = _load(related_db_config)
        assert sorted(repo.get_related_items("testcol", 100)) == [10, 11, 12]

    def test_sorts_ascending_by_smallest_value(self, related_db_config):
        repo = _load(related_db_config)
        assert repo.get_related_items("testcol", 100, sort_by=3) == [11, 10, 12]

    def test_sorts_descending_by_largest_value(self, related_db_config):
        repo = _load(related_db_config)
        assert repo.get_related_items(
            "testcol", 100, sort_by=3, sort_order="desc"
        ) == [11, 10, 12]

    def test_resolves_the_tagsets_own_tagtype_table(self, related_db_config):
        repo = _load(related_db_config)
        assert repo.get_related_items("testcol", 100, sort_by=4) == [11, 10, 12]
        assert repo.get_related_items(
            "testcol", 100, sort_by=4, sort_order="desc"
        ) == [12, 10, 11]

    def test_items_without_a_value_come_last(self, related_db_config):
        repo = _load(related_db_config)
        assert repo.get_related_items("testcol", 100, sort_by=3)[-1] == 12
        assert repo.get_related_items(
            "testcol", 100, sort_by=3, sort_order="desc"
        )[-1] == 12

    def test_unknown_tagset_raises(self, related_db_config):
        repo = _load(related_db_config)
        with pytest.raises(DatabaseError):
            repo.get_related_items("testcol", 100, sort_by=999)

    def test_invalid_sort_order_raises(self, related_db_config):
        repo = _load(related_db_config)
        with pytest.raises(DatabaseError):
            repo.get_related_items("testcol", 100, sort_by=3, sort_order="up")


def _add_group_media(path: Path, media_id: int, tag_id: int) -> None:
    """Add a Video group media (no group_id of its own) carrying one tag."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "INSERT INTO medias (id, source, source_type, thumbnail_uri, group_id) "
        "VALUES (?, ?, 2, NULL, NULL)",
        [media_id, f"video{media_id}.mp4"],
    )
    conn.execute(
        "INSERT INTO numerical_int_tags (id, value, tagset_id) VALUES (?, 1, 1)",
        [tag_id],
    )
    conn.execute(
        "INSERT INTO taggings (media_id, tag_id) VALUES (?, ?)", [media_id, tag_id]
    )
    conn.commit()
    conn.close()


@pytest.fixture
def two_collection_config(write_config, dummy_files, tmp_path):
    """Two collections whose group medias differ.

    - testcol has group media 100 (tag 5000), parent of medias 10 and 11.
    - othercol has group media 300 (tag 6000) and no media 100.
    """
    collections = []
    for name, group_id, tag_id in (("testcol", 100, 5000), ("othercol", 300, 6000)):
        db_path = tmp_path / f"{name}.db"
        _build_sqlite_db(db_path)
        _add_group_media(db_path, group_id, tag_id)
        indexes = index_toml(
            name="CLIP",
            index_type="zarr",
            embeddings_file=dummy_files["embeddings_file"],
        )
        collections.append(
            collection_toml(
                name=name,
                database_file=str(db_path),
                thumbnail_media_url=f"https://localhost:5000/{name}",
                original_media_url=f"https://localhost:5000/{name}",
                indexes=indexes,
            )
        )
    path = write_config("\n\n".join(collections))
    return ConfigManager(str(path)).load_config()


def _tag_filter(tag_id: int) -> ActiveFilters:
    return ActiveFilters(
        root=FilterLeaf(
            filter=DBFilter(
                id=1,
                tagtype_id=1,
                constraint=DBValueConstraint(value_ids=[tag_id]),
            )
        )
    )


class TestGroupMediaIdsPerCollection:
    def test_group_filter_expands_after_loading_another_collection(
        self, two_collection_config
    ):
        repo = DatabaseRepository(two_collection_config)
        for name in ("testcol", "othercol"):
            db_file = two_collection_config.collection_configs[name].database_file
            repo.load_database(name, db_file)

        passed = repo.get_filtered_media_ids("testcol", _tag_filter(5000))

        assert sorted(passed) == [10, 11]

    def test_each_collection_keeps_its_own_group_media_ids(
        self, two_collection_config
    ):
        repo = DatabaseRepository(two_collection_config)
        for name in ("testcol", "othercol"):
            db_file = two_collection_config.collection_configs[name].database_file
            repo.load_database(name, db_file)

        assert repo._group_media_ids["testcol"] == {100}
        assert repo._group_media_ids["othercol"] == {300}


@pytest.fixture
def timed_db_config(multi_index_db_config):
    """Extends the base database with segment times.

    - group 100 (Image): 10 has Start 1000 and End 2000, 11 has Start 3000
      and no End, 12 has only Keyframe 5000, 13 has no time at all.
    - group 200 (Text): 20 has Start 1500 and End 2500.
    - group 300 (Image): 30 has Start 0, in a group no anchor belongs to.
    """
    db_file = multi_index_db_config.collection_configs["testcol"].database_file
    conn = sqlite3.connect(str(db_file))
    conn.executescript(
        """
        INSERT INTO tagsets (id, name, tagtype_id) VALUES
            (5, 'Start (ms)', 1), (6, 'Keyframe (ms)', 1), (7, 'End (ms)', 1);

        INSERT INTO medias (id, source, source_type, thumbnail_uri, group_id) VALUES
            (12, 'img3.jpg', 1, 'thumb4.jpg', 100),
            (13, 'img4.jpg', 1, 'thumb5.jpg', 100),
            (30, 'img5.jpg', 1, 'thumb6.jpg', 300);

        INSERT INTO numerical_int_tags (id, value, tagset_id) VALUES
            (5000, 1000, 5), (5001, 3000, 5), (5002, 1500, 5), (5003, 0, 5),
            (6000, 5000, 6),
            (7000, 2000, 7), (7001, 2500, 7);
        INSERT INTO taggings (media_id, tag_id) VALUES
            (10, 5000), (10, 7000), (11, 5001), (12, 6000),
            (20, 5002), (20, 7001), (30, 5003);
        """
    )
    conn.commit()
    conn.close()
    return multi_index_db_config


class TestGetGroupSegmentTimes:
    def test_times_every_segment_in_the_anchors_groups(self, timed_db_config):
        repo = _load(timed_db_config)
        rows = repo.get_group_segment_times("testcol", [10, 20], ["Image", "Text"])
        assert sorted(rows) == [
            (10, 100, "Image", 1000, 2000),
            (11, 100, "Image", 3000, 3000),
            (12, 100, "Image", 5000, 5000),
            (20, 200, "Text", 1500, 2500),
        ]

    def test_only_returns_the_requested_source_types(self, timed_db_config):
        repo = _load(timed_db_config)
        rows = repo.get_group_segment_times("testcol", [10, 20], ["Text"])
        assert [row[0] for row in rows] == [20]

    def test_start_prefers_start_over_keyframe(self, timed_db_config):
        db_file = timed_db_config.collection_configs["testcol"].database_file
        conn = sqlite3.connect(str(db_file))
        conn.executescript(
            """
            INSERT INTO numerical_int_tags (id, value, tagset_id) VALUES (6001, 9999, 6);
            INSERT INTO taggings (media_id, tag_id) VALUES (10, 6001);
            """
        )
        conn.commit()
        conn.close()

        repo = _load(timed_db_config)
        rows = repo.get_group_segment_times("testcol", [10], ["Image"])
        assert dict((row[0], row[3]) for row in rows)[10] == 1000

    def test_single_start_tagset_without_coalesce(self, timed_db_config):
        db_file = timed_db_config.collection_configs["testcol"].database_file
        conn = sqlite3.connect(str(db_file))
        conn.execute("DELETE FROM tagsets WHERE name = 'Keyframe (ms)'")
        conn.commit()
        conn.close()

        repo = _load(timed_db_config)
        rows = repo.get_group_segment_times("testcol", [10], ["Image"])
        assert sorted(row[0] for row in rows) == [10, 11]

    def test_anchors_beyond_the_sqlite_parameter_limit(self, timed_db_config):
        repo = _load(timed_db_config)
        anchors = [10] + list(range(100_000, 101_500))
        rows = repo.get_group_segment_times("testcol", anchors, ["Image"])
        assert sorted(row[0] for row in rows) == [10, 11, 12]

    def test_no_time_tagsets_raises(self, multi_index_db_config):
        repo = _load(multi_index_db_config)
        with pytest.raises(DatabaseError):
            repo.get_group_segment_times("testcol", [10], ["Image"])
