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

"""Tests for compile_active_filters, run on the worked example of the docs.

One test checks the generated SQL itself. The others run it on the example
in an in-memory SQLite database.
"""

import sqlite3

import pytest

from app.repositories.db_helper import compile_active_filters
from app.schemas import (
    ActiveFilters,
    DBFilter,
    DBRangeConstraint,
    DBValueConstraint,
    FilterGroup,
    FilterLeaf,
)

TAGTYPES = {1: "alphanumerical", 5: "numerical_int"}
"""Tag type IDs as the compiler gets them from the tag_types table."""

CATEGORY, START_MS = 1, 2  # tagset IDs
DOCUMENTARY, TRAVEL = 100, 101  # tag IDs of two Category values


@pytest.fixture
def conn():
    """The worked example.

    - Media 10 is a group with Category documentary. Media 11 and 12 have
      group_id 10, with Start (ms) 10000 and 2000.
    - Media 40 is a group with Category travel. Media 41 has group_id 40 and
      no tags. Media 42 and 43 have group_id 41, 42 with Start (ms) 15000 and
      43 without tags.
    - Media 50 is a group with Category drama. Media 51 and 52 have
      group_id 50, with Start (ms) 8000 and 9000.
    - Media 20 has no group_id, Category travel and Start (ms) 30000.
    - Media 30 has no group_id and no tags.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE medias (id INTEGER PRIMARY KEY, group_id INTEGER);
        CREATE TABLE taggings (media_id INTEGER, tag_id INTEGER);
        CREATE TABLE alphanumerical_tags (id INTEGER PRIMARY KEY, value TEXT, tagset_id INTEGER);
        CREATE TABLE numerical_int_tags (id INTEGER PRIMARY KEY, value INTEGER, tagset_id INTEGER);

        INSERT INTO medias (id, group_id) VALUES
            (10, NULL), (11, 10), (12, 10),
            (40, NULL), (41, 40), (42, 41), (43, 41),
            (50, NULL), (51, 50), (52, 50),
            (20, NULL), (30, NULL);

        INSERT INTO alphanumerical_tags (id, value, tagset_id) VALUES
            (100, 'documentary', 1), (101, 'travel', 1), (102, 'drama', 1);
        INSERT INTO numerical_int_tags (id, value, tagset_id) VALUES
            (200, 10000, 2), (201, 2000, 2), (202, 15000, 2),
            (203, 8000, 2), (204, 9000, 2), (205, 30000, 2);
        INSERT INTO taggings (media_id, tag_id) VALUES
            (10, 100), (11, 200), (12, 201),
            (40, 101), (42, 202),
            (50, 102), (51, 203), (52, 204),
            (20, 101), (20, 205);
        """
    )
    yield conn
    conn.close()


def _category(tag_ids, not_=False) -> FilterLeaf:
    return FilterLeaf(
        not_=not_,
        filter=DBFilter(
            id=CATEGORY, tagtype_id=1, constraint=DBValueConstraint(value_ids=tag_ids)
        ),
    )


def _start_in_range(not_=False) -> FilterLeaf:
    # Start (ms) from 5000 to 60000
    return FilterLeaf(
        not_=not_,
        filter=DBFilter(
            id=START_MS,
            tagtype_id=5,
            constraint=DBRangeConstraint(lower_bound=5000, upper_bound=60000),
        ),
    )


# Category is documentary or travel AND Start (ms) from 5000 to 60000
EXAMPLE = FilterGroup(
    operator="AND", children=[_category([DOCUMENTARY, TRAVEL]), _start_in_range()]
)


def _passed(conn, root) -> list:
    sql, params = compile_active_filters(ActiveFilters(root=root), TAGTYPES)
    return sorted(r[0] for r in conn.execute(sql, params))


def test_generated_query():
    sql, params = compile_active_filters(ActiveFilters(root=EXAMPLE), TAGTYPES)
    expected = """
        WITH RECURSIVE
        f AS (
            SELECT media_id,
                   MAX(CASE WHEN flag = 0 THEN 1 ELSE 0 END) AS l0,
                   MAX(CASE WHEN flag = 1 THEN 1 ELSE 0 END) AS l1
            FROM (SELECT x.media_id, 0 AS flag FROM taggings x WHERE x.tag_id IN (?,?)
                  UNION ALL
                  SELECT x.media_id, 1 AS flag FROM taggings x
                  JOIN numerical_int_tags r ON r.id = x.tag_id
                  WHERE r.tagset_id = ? AND r.value BETWEEN ? AND ?)
            GROUP BY media_id
        ),
        down (media_id, source_id, depth) AS (
            SELECT media_id, media_id, 0 FROM f
            UNION ALL
            SELECT ch.id, d.source_id, d.depth + 1
            FROM down d
            JOIN medias ch ON ch.group_id = d.media_id
            WHERE d.depth < 4
        ),
        flags AS (
            SELECT d.media_id, MAX(f.l0) AS l0, MAX(f.l1) AS l1
            FROM down d
            JOIN f ON f.media_id = d.source_id
            GROUP BY d.media_id
        )
        SELECT m.id
        FROM medias m
        LEFT JOIN flags fl ON fl.media_id = m.id
        WHERE (COALESCE(fl.l0, 0) = 1) AND (COALESCE(fl.l1, 0) = 1)
    """
    # Whitespace doesn't matter, only the query
    assert " ".join(sql.split()) == " ".join(expected.split())
    assert params == [DOCUMENTARY, TRAVEL, START_MS, 5000, 60000]


def test_example_filter(conn):
    # Media 11 gets Category from group 10, media 42 from group 40 through
    # group 41, and media 20 has both tags itself
    assert _passed(conn, EXAMPLE) == [11, 20, 42]


def test_negated_category_also_excludes_the_group_media(conn):
    # Media 11 and 12 get documentary from group 10, and media 30 passes without tags
    assert _passed(conn, _category([DOCUMENTARY], not_=True)) == [
        20, 30, 40, 41, 42, 43, 50, 51, 52
    ]


def test_negated_range_keeps_media_without_a_start_in_range(conn):
    assert _passed(conn, _start_in_range(not_=True)) == [10, 12, 30, 40, 41, 43, 50]


def test_tags_of_an_and_leaf_can_come_from_the_media_and_its_group():
    # Media 1 is a group with tag 300. Media 2 and 3 have group_id 1, and
    # only media 2 has tag 301
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE medias (id INTEGER PRIMARY KEY, group_id INTEGER);
        CREATE TABLE taggings (media_id INTEGER, tag_id INTEGER);
        INSERT INTO medias (id, group_id) VALUES (1, NULL), (2, 1), (3, 1);
        INSERT INTO taggings (media_id, tag_id) VALUES (1, 300), (2, 301);
        """
    )
    both = DBValueConstraint(value_ids=[300, 301], operator="AND")
    root = FilterLeaf(filter=DBFilter(id=3, tagtype_id=1, constraint=both))
    assert _passed(conn, root) == [2]
    conn.close()
