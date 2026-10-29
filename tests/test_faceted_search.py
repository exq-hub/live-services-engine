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

"""Tests for faceted search returning only the media of one index."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

from app.core.config import ConfigManager
from app.core.exceptions import SearchError
from app.repositories.database_repository import DatabaseRepository
from app.schemas import ActiveFilters, DBFilter, DBRangeConstraint, FilterLeaf
from app.strategies.faceted_search import FacetedSearchStrategy

from .conftest import collection_toml, index_toml


def _build_sqlite_db(path: Path) -> None:
    """A video with keyframes and transcripts, all with a start time.

    - Video 100 has keyframes 10, 11 (Image) and transcripts 20, 21 (Text).
    - 10 and 20 start at 0 ms, 11 and 21 at 90000 ms.
    - 'CLIP Index ID' maps the keyframes and 'Text Index ID' the transcripts.
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
            (1, 'Image'), (2, 'Video'), (4, 'Text');
        INSERT INTO tag_types (id, description) VALUES (1, 'numerical_int');
        INSERT INTO tagsets (id, name, tagtype_id) VALUES
            (1, 'CLIP Index ID', 1), (2, 'Text Index ID', 1), (3, 'Start (ms)', 1);

        INSERT INTO medias (id, source, source_type, thumbnail_uri, group_id) VALUES
            (100, 'video1.mp4', 2, NULL, NULL),
            (10, 'keyframe1.jpg', 1, 'keyframe1.jpg', 100),
            (11, 'keyframe2.jpg', 1, 'keyframe2.jpg', 100),
            (20, 'transcript1.txt', 4, NULL, 100),
            (21, 'transcript2.txt', 4, NULL, 100);

        INSERT INTO numerical_int_tags (id, value, tagset_id) VALUES
            (1000, 0, 1), (1001, 1, 1),
            (2000, 0, 2), (2001, 1, 2),
            (3000, 0, 3), (3001, 90000, 3);
        INSERT INTO taggings (media_id, tag_id) VALUES
            (10, 1000), (11, 1001),
            (20, 2000), (21, 2001),
            (10, 3000), (20, 3000),
            (11, 3001), (21, 3001);
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture
def strategy(write_config, dummy_files, tmp_path):
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
    config = ConfigManager(str(write_config(collection))).load_config()

    repo = DatabaseRepository(config)
    repo.load_database("testcol", str(db_path))
    return FacetedSearchStrategy(repo)


def _starts_between(lower: int, upper: int) -> ActiveFilters:
    return ActiveFilters(
        root=FilterLeaf(
            filter=DBFilter(
                id=3,  # Start (ms) tagset
                tagtype_id=1,  # numerical_int
                constraint=DBRangeConstraint(lower_bound=lower, upper_bound=upper),
            )
        )
    )


FIRST_MINUTE = _starts_between(0, 60000)  # keyframe 10 and transcript 20


def _search(strategy, filters, n=10, **kwargs):
    return asyncio.run(strategy.search("testcol", n=n, filters=filters, **kwargs))


class TestFacetedSearch:
    def test_clip_index_returns_only_keyframes(self, strategy):
        assert _search(strategy, FIRST_MINUTE, index_name="CLIP") == [10]

    def test_text_index_returns_only_transcripts(self, strategy):
        assert _search(strategy, FIRST_MINUTE, index_name="Text") == [20]

    def test_without_index_name_uses_the_default_index(self, strategy):
        assert _search(strategy, FIRST_MINUTE) == [10]  # CLIP is the default

    def test_returns_at_most_n_items_of_the_index(self, strategy):
        everything = _starts_between(0, 90000)
        result = _search(strategy, everything, n=1, index_name="Text")
        assert len(result) == 1
        assert result[0] in (20, 21)

    def test_unknown_index_raises_search_error(self, strategy):
        with pytest.raises(SearchError):
            _search(strategy, FIRST_MINUTE, index_name="nope")
