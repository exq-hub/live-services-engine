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


"""SQLite-backed repository for collection metadata and filter evaluation.

`DatabaseRepository` is the single point of access for all relational data
in the LSE. Each media collection is backed by its own SQLite database that
contains:

- **medias** -- the master table of media items (source URI, thumbnail URI,
  group ID, source type).
- **tags / taggings / tagsets / tag_types** -- a flexible tagging system
  where tag values are stored in type-specific tables (e.g.
  ``categorical_tags``, ``numerical_int_tags``) and linked to media items
  via the ``taggings`` join table.
- **Index mapping** -- each configured index reads its item-to-position
  mapping from a tagset in ``numerical_int_tags`` (its ``tagset_name``,
  which defaults to ``<index name> Index ID``) that maps each media item
  to its position in that index, enabling bidirectional translation
  between media IDs and index IDs.

The repository caches database connections and index mappings per
collection and provides methods for item retrieval, filter evaluation
(via `db_helper.compile_active_filters`), and related-item lookups.

Id mappings load lazily from `config` on first use, same as
`IndexRepository`: a collection's default index loads eagerly in
`load_database`; any other index loads the first time something asks
for it (or eagerly too, for a `preload_all_indexes` collection).
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.repositories import db_helper
from app.schemas import ActiveFilters

from ..core.config import IndexConfig, LSEConfig
from ..core.exceptions import DatabaseError

import pandas as pd


START_TIME_TAGSETS = ("Start (ms)", "Keyframe (ms)", "Time (ms)")
"""Tagsets giving a segment's start time, in priority order. A segment's
start is the first of these it has a value for."""

END_TIME_TAGSET = "End (ms)"
"""Tagset giving a segment's end time. Segments without one end at their start."""


class DatabaseRepository:
    """Repository class for managing metadata, filters, and related item data.

    This repository provides access to collection metadata, filters for search
    operations, and related item mappings.
    """

    def __init__(self, config: LSEConfig):
        """Initialize the metadata repository with empty caches."""
        self.config = config
        """Validated LSE configuration, used to resolve and build id mappings on demand."""

        self._db_connection: Dict[str, sqlite3.Connection] = {}
        """Per-collection SQLite connections keyed by collection name."""

        self._db_path: Dict[str, str] = {}
        """Per-collection database file paths."""

        self._db_type: Dict[str, str] = {}
        """Per-collection database backend type (e.g. ``'sqlite'``, ``'duckdb'``, ``'postgres'``)."""

        self._item_datapoint_mapping_cache: Dict[str, Dict[str, Dict[int, int]]] = {}
        """Nested mapping: ``collection -> index_name -> index_id -> media_id``."""

        self._rev_item_datapoint_mapping_cache: Dict[
            str, Dict[str, Dict[int, int]]
        ] = {}
        """Reverse mapping: ``collection -> index_name -> media_id -> index_id``."""

        self._group_media_ids: Dict[str, Set[int]] = {}
        """Per-collection ids of the medias that represent a group (e.g., a video)."""

        self._tagtype_cache: Dict[str, Dict[int, str]] = {}
        """Per-collection tag-type lookup: ``collection -> tagtype_id -> tagtype_name``."""

        self._time_tagsets_cache: Dict[
            str, Tuple[List[Tuple[int, str]], Optional[Tuple[int, str]]]
        ] = {}
        """Per-collection ``(tagset_id, tagtype_name)`` of the start time
        tagsets present, in `START_TIME_TAGSETS` order, and of the end time
        tagset if present."""

        self._sqlite_limit = 999
        """SQLite has a default limit of 999 parameters per query, so we may need to batch queries that exceed this limit."""

    def __del__(self):
        """Close all database connections on deletion."""
        for conn in self._db_connection.values():
            try:
                conn.close()
            except Exception:
                pass

    def load_database(
        self,
        collection: str,
        database_file: str,
        database: str = "sqlite",
    ) -> None:
        """
        Open connection to the database file and build the id mapping for
        the collection's default index (and, for a preload_all_indexes
        collection, every other index too).

        Args:
            collection: Name of the collection to load metadata for
            database_file: Path to the collection's database file

        Raises:
            DatabaseError: If file doesn't exist or connection fails
        """
        try:
            db_path = Path(database_file)
            if not db_path.exists():
                raise DatabaseError(f"Database file not found: {database_file}")
            self._db_path[collection] = db_path
            self._db_type[collection] = database
            if database == "sqlite":
                self._db_connection[collection] = sqlite3.connect(
                    db_path, autocommit=False
                )
                rows = (
                    self._db_connection[collection]
                    .execute(
                        """
                        SELECT id, description as name
                        FROM tag_types
                        """
                    )
                    .fetchall()
                )
                self._tagtype_cache[collection] = {row[0]: row[1] for row in rows}
            self._item_datapoint_mapping_cache[collection] = {}
            self._rev_item_datapoint_mapping_cache[collection] = {}
            self._group_media_ids[collection] = set([
                row[0] for row in self._db_connection[collection]
                .execute(
                    """
                    SELECT id FROM medias WHERE group_id IS NULL
                    """
                ).fetchall()
            ])

            collection_config = self.config.collection_configs[collection]
            default_index = collection_config.default_index
            self._ensure_index_mapping(collection, default_index.name)

            if collection_config.preload_all_indexes:
                for index in collection_config.indexes:
                    if index.name != default_index.name:
                        self._ensure_index_mapping(collection, index.name)

        except Exception as e:
            raise DatabaseError(f"Failed to load database from {database_file}: {e}")

    def _resolve_index_name(self, collection: str, index_name: Optional[str]) -> str:
        """Default to the collection's default index when none is given."""
        if index_name is not None:
            return index_name
        try:
            collection_config = self.config.collection_configs[collection]
        except KeyError:
            raise DatabaseError(f"Unknown collection: {collection!r}")
        return collection_config.default_index.name

    def _ensure_index_mapping(self, collection: str, index_name: str) -> None:
        """Build the id mapping for collection/index_name on demand if not already cached."""
        if index_name in self._item_datapoint_mapping_cache.get(collection, {}):
            return

        try:
            collection_config = self.config.collection_configs[collection]
        except KeyError:
            raise DatabaseError(f"Unknown collection: {collection!r}")

        try:
            index_config = collection_config.get_index(index_name)
        except ValueError as e:
            raise DatabaseError(f"{e} (collection {collection!r})")

        mapping, rev_mapping = self.create_item_to_datapoint_mapping(
            collection, index_config
        )
        self._item_datapoint_mapping_cache.setdefault(collection, {})[index_name] = (
            mapping
        )
        self._rev_item_datapoint_mapping_cache.setdefault(collection, {})[
            index_name
        ] = rev_mapping

    def is_loaded(self, collection: str) -> bool:
        """Check if the database for the specified collection is loaded.

        Args:
            collection: Name of the collection
        Returns:
            True if the database is loaded, False otherwise
        """
        if collection not in self._db_connection:
            return False

        if self._db_connection[collection] is None:
            return False

        return True

    def get_tagtypes(self, collection: str) -> Optional[Dict[int, str]]:
        """Retrieve cached tagtypes for the specified collection.

        Args:
            collection: Name of the collection
        Returns:
            Cached tagtypes dictionary or None if not loaded
        """
        return self._tagtype_cache.get(collection)

    def get_filters(self, collection: str) -> List[Dict[str, Any]]:
        """Retrieve cached filter definitions for the specified collection.

        Args:
            collection: Name of the collection

        Returns:
            Cached filter definitions dictionary or None if not loaded
        """
        cursor = None
        try:
            filters = []
            cursor = self._db_connection[collection].cursor()
            rows = cursor.execute(
                """
                SELECT ts.id, ts.name, tt.id as tagtype_id, tt.description as tagtype
                FROM tagsets ts 
                JOIN tag_types tt ON ts.tagtype_id = tt.id
                """
            ).fetchall()
            filters = [
                {"id": f[0], "name": f[1], "tagtypeId": f[2], "tagtype": f[3]}
                for f in rows
            ]
            return filters
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve filters from {collection}: {e}")
        finally:
            if cursor:
                cursor.close()

    def get_filter_values(
        self, collection: str, filter_id: int, tagtype_id: int
    ) -> Optional[List[Any]]:
        """Retrieve values for a specific filter by its ID and tagtype

        Args:
            collection: Name of the collection
            filter_id: ID of the filter (tagset)
            tagtype: Tagtype ID of the filter (tagset)

        Returns:
            List of filter values or None if not found

        Raises:
            DatabaseError: If retrieval fails
        """
        cursor = None
        try:
            cursor = self._db_connection[collection].cursor()
            if tagtype_id not in self._tagtype_cache[collection]:
                raise DatabaseError(
                    f"Tagtype with id {tagtype_id} not found in collection {collection}"
                )

            rows = cursor.execute(
                f"""
                SELECT t.id, t.value 
                FROM {self._tagtype_cache[collection][tagtype_id]}_tags t
                JOIN tagsets ts ON t.tagset_id = ts.id
                WHERE ts.tagtype_id = ? AND ts.id = ?
                """,
                [tagtype_id, filter_id],
            ).fetchall()
            columns = [col[0] for col in cursor.description]
            filter_values = [dict(zip(columns, row)) for row in rows]
            return filter_values
        except Exception as e:
            raise DatabaseError(
                f"Failed to retrieve filter values of {filter_id} from {collection}: {e}"
            )
        finally:
            if cursor:
                cursor.close()

    def get_media_ids(
        self, collection, index_ids: list[int], index: Optional[str] = None
    ) -> list[int]:
        index = self._resolve_index_name(collection, index)
        self._ensure_index_mapping(collection, index)
        try:
            return [
                self._item_datapoint_mapping_cache[collection][index][idx]
                for idx in index_ids
                if idx != -1
            ]
        except Exception as e:
            raise DatabaseError(
                f"Failed to map index IDs ({index_ids}) to media IDs for collection {collection}: {e}"
            )

    def get_index_ids(
        self,
        collection,
        media_ids: list[int],
        index: Optional[str] = None,
        skip_unmapped: bool = False,
    ) -> list[int]:
        """Map media IDs to their positions in collection/index.

        Args:
            collection: Name of the collection
            media_ids: Media IDs to map
            index: Name of the index; defaults to the collection's default index
            skip_unmapped: If True, silently drop any media ID that doesn't
                belong to this index instead of raising for callers like
                "seen"/"excluded"/filtered ids, where an unrelated id is a
                no-op, not an error. Explicit RF pos/neg samples should stay
                strict (the default), since an out-of-index id there is more
                likely a real caller bug.
        """
        index = self._resolve_index_name(collection, index)
        self._ensure_index_mapping(collection, index)
        mapping = self._rev_item_datapoint_mapping_cache[collection][index]
        try:
            if skip_unmapped:
                total_items = len(self._item_datapoint_mapping_cache[collection][index])
                return [
                    position
                    for idx in media_ids
                    if idx != -1
                    and (position := mapping.get(idx)) is not None
                    and 0 <= position < total_items
                ]
            return [mapping[idx] for idx in media_ids if idx != -1]
        except Exception as e:
            raise DatabaseError(
                f"Failed to map media IDs ({media_ids}) to index IDs for collection {collection}, index {index}: {e}"
            )

    def get_media_metadata(
        self,
        cursor: sqlite3.Connection,
        media_id: int,
        collection: str,
        filters: List[int] = [],
    ) -> Dict[str, Any]:
        """Retrieve metadata for a specific media item.

        Args:
            cursor: Database cursor
            media_id: ID of the media item
            filters: List of metadata fields to include, if empty include all

        Returns:
            Metadata dictionary for the media item
        """
        try:
            # Get metadata from tags
            metadata = {}
            query = """
                    SELECT tgs.tag_id, 
                        ts.id as tagset_id,
                        ts.name as tagset_name, 
                        tt.description as tagtype
                    FROM medias m
                    JOIN taggings tgs ON m.id = tgs.media_id
                    JOIN tags t ON tgs.tag_id = t.id
                    JOIN tagsets ts ON t.tagset_id = ts.id
                    JOIN tag_types tt ON ts.tagtype_id = tt.id
                    WHERE m.id = ?
                    """
            tag_info = None
            if filters:
                query += f""" AND ts.id IN ({",".join(["?"] * len(filters))}) 
                            ORDER BY ts.id"""
                if self._db_type[collection] == "sqlite":
                    tag_info = pd.read_sql_query(
                        query,
                        self._db_connection[collection],
                        params=[media_id] + filters,
                    )
                elif self._db_type[collection] == "duckdb":
                    tag_info = cursor.execute(query, [media_id] + filters).fetchdf()
            else:
                query += " ORDER BY ts.id"
                if self._db_type[collection] == "sqlite":
                    tag_info = pd.read_sql_query(
                        query, self._db_connection[collection], params=[media_id]
                    )
                elif self._db_type[collection] == "duckdb":
                    tag_info = cursor.execute(query, [media_id]).fetchdf()

            # Collapse rows into one per tagset, collecting tag IDs and metadata
            grouped = (
                tag_info.groupby("tagset_id")
                .agg(
                    tag_ids=("tag_id", list),
                    tagtype=("tagtype", "first"),
                    tagset_name=("tagset_name", "first"),
                )
                .reset_index()
            )
            for row in grouped.itertuples():
                tag_ids_placeholder = ",".join(["?"] * len(row.tag_ids))
                tag_values = cursor.execute(
                    f"""
                    SELECT value 
                    FROM {row.tagtype}_tags
                    WHERE id IN ({tag_ids_placeholder})
                    """,
                    row.tag_ids,
                ).fetchall()
                if len(tag_values) > 1:
                    metadata[row.tagset_name] = [value[0] for value in tag_values]
                else:
                    metadata[row.tagset_name] = tag_values[0][0]
            return metadata
        except Exception as e:
            raise DatabaseError(
                f"Failed to retrieve media metadata for media {media_id}: {e}"
            )

    @staticmethod
    def _tagset_value_subquery(tagtype: str, agg: str) -> str:
        """Scalar subquery aggregating one tagset's values for the outer ``m``.

        Correlated on ``m.id`` and taking the tagset ID as its single bind
        parameter. Resolves through the covering taggings primary key and
        the tag table's primary key, so it costs a few index seeks per media.
        """
        return f"""(SELECT {agg}(t.value)
                    FROM taggings tgs
                    JOIN {tagtype}_tags t ON t.id = tgs.tag_id
                    WHERE tgs.media_id = m.id AND t.tagset_id = ?)"""

    def get_related_items(
        self,
        collection: str,
        group_id: int,
        sort_by: Optional[int] = None,
        sort_order: str = "asc",
    ) -> Optional[List[int]]:
        """Retrieve related items for the specified collection.

        Args:
            collection: Name of the collection
            group_id: Media ID of the group whose members to return
            sort_by: Optional tagset ID whose values order the result. For
                multi-valued tagsets the smallest value is used ascending and
                the largest descending. Items without a value come last.
            sort_order: ``"asc"`` or ``"desc"``

        Returns:
            Media IDs of the items in the group
        """
        cursor = None
        try:
            cursor = self._db_connection[collection].cursor()
            related_items = []
            if self._db_type[collection] == "sqlite":
                if sort_by is None:
                    rows = cursor.execute(
                        """
                        SELECT id
                        FROM medias
                        WHERE group_id = ?
                        """,
                        [group_id],
                    ).fetchall()
                else:
                    if sort_order not in ("asc", "desc"):
                        raise DatabaseError(f"Invalid sort order: {sort_order!r}")
                    tagtype_row = cursor.execute(
                        "SELECT tagtype_id FROM tagsets WHERE id = ?", [sort_by]
                    ).fetchone()
                    if tagtype_row is None:
                        raise DatabaseError(f"Tagset {sort_by} not found")
                    tagtype = self._tagtype_cache[collection][tagtype_row[0]]
                    agg, direction = (
                        ("MIN", "ASC") if sort_order == "asc" else ("MAX", "DESC")
                    )
                    rows = cursor.execute(
                        f"""
                        SELECT m.id,
                               {self._tagset_value_subquery(tagtype, agg)} AS sort_value
                        FROM medias m
                        WHERE m.group_id = ?
                        ORDER BY sort_value IS NULL, sort_value {direction}
                        """,
                        [sort_by, group_id],
                    ).fetchall()
                if rows:
                    related_items = [row[0] for row in rows]
            return related_items
        except Exception as e:
            raise DatabaseError(
                f"Failed to retrieve related items for {group_id} from {collection}: {e}"
            )
        finally:
            if cursor:
                cursor.close()

    def _resolve_time_tagsets(
        self, collection: str, cursor: sqlite3.Cursor
    ) -> Tuple[List[Tuple[int, str]], Optional[Tuple[int, str]]]:
        """Resolve the collection's start and end time tagsets, once per collection."""
        if collection not in self._time_tagsets_cache:
            names = (*START_TIME_TAGSETS, END_TIME_TAGSET)
            rows = cursor.execute(
                f"""
                SELECT name, id, tagtype_id
                FROM tagsets
                WHERE name IN ({db_helper.placeholders(len(names))})
                """,
                list(names),
            ).fetchall()
            found = {
                name: (tagset_id, self._tagtype_cache[collection][tagtype_id])
                for name, tagset_id, tagtype_id in rows
            }
            starts = [found[name] for name in START_TIME_TAGSETS if name in found]
            if not starts:
                raise DatabaseError(
                    f"Collection {collection!r} has none of the start time "
                    f"tagsets {list(START_TIME_TAGSETS)}"
                )
            self._time_tagsets_cache[collection] = (starts, found.get(END_TIME_TAGSET))
        return self._time_tagsets_cache[collection]

    def get_group_segment_times(
        self,
        collection: str,
        anchor_ids: List[int],
        source_types: List[str],
    ) -> List[Tuple[int, int, str, float, float]]:
        """Time every segment of the given source types in the anchors' groups.

        Driven from the anchors, so the cost scales with the groups they
        belong to rather than with the collection. Segments without a start
        time are left out, and segments without an end time end at their
        start.

        Args:
            collection: Name of the collection
            anchor_ids: Media IDs whose groups (videos) to time
            source_types: Source type names ('Image', 'Text', ...) to include

        Returns:
            ``(media_id, group_id, source_type, start_ms, end_ms)`` per segment
        """
        if not anchor_ids or not source_types:
            return []

        cursor = None
        batch_conn = None
        try:
            cursor = self._db_connection[collection].cursor()
            starts, end = self._resolve_time_tagsets(collection, cursor)
            source_type_names = dict(
                cursor.execute(
                    f"""
                    SELECT id, name
                    FROM source_types
                    WHERE name IN ({db_helper.placeholders(len(source_types))})
                    """,
                    list(source_types),
                ).fetchall()
            )
            if not source_type_names:
                return []

            start_subqueries = [
                self._tagset_value_subquery(tagtype, "MIN") for _, tagtype in starts
            ]
            # SQLite's coalesce() needs at least two arguments
            start_sql = (
                start_subqueries[0]
                if len(start_subqueries) == 1
                else f"COALESCE({', '.join(start_subqueries)})"
            )
            end_sql = self._tagset_value_subquery(end[1], "MAX") if end else "NULL"
            params = (
                [tagset_id for tagset_id, _ in starts]
                + ([end[0]] if end else [])
                + list(source_type_names)
            )

            # Anchors can exceed the SQLite parameter limit, so they go in a
            # temp table on a dedicated connection, as in get_filtered_media_ids
            batch_conn = sqlite3.connect(self._db_path[collection], autocommit=False)
            batch_conn.execute(
                "CREATE TEMPORARY TABLE temp_anchor (id INTEGER PRIMARY KEY)"
            )
            batch_conn.executemany(
                "INSERT OR IGNORE INTO temp_anchor (id) VALUES (?)",
                [(media_id,) for media_id in anchor_ids],
            )
            rows = batch_conn.execute(
                f"""
                SELECT m.id,
                       m.group_id,
                       m.source_type,
                       {start_sql} AS start_ms,
                       {end_sql} AS end_ms
                FROM medias m
                WHERE m.group_id IN (
                    SELECT a.group_id
                    FROM medias a
                    WHERE a.id IN (SELECT id FROM temp_anchor)
                )
                AND m.source_type IN ({db_helper.placeholders(len(source_type_names))})
                """,
                params,
            ).fetchall()
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(
                f"Failed to time the anchors' group segments in {collection}: {e}"
            )
        finally:
            if cursor:
                cursor.close()
            if batch_conn:
                batch_conn.close()

        return [
            (
                media_id,
                group_id,
                source_type_names[source_type],
                start_ms,
                start_ms if end_ms is None else end_ms,
            )
            for media_id, group_id, source_type, start_ms, end_ms in rows
            if start_ms is not None
        ]

    def get_item(
        self, collection: str, media_id: int, filters: List[int] = []
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific item's metadata by its ID.

        Args:
            collection: Name of the collection
            item_id: Index of the item in the collection
            filters: List of metadata fields to include, if empty include all

        Returns:
            Item metadata dictionary or None if not found or invalid ID
        """
        cursor = None
        try:
            item = {}
            item["item_id"] = media_id
            cursor = self._db_connection[collection].cursor()
            media_info = cursor.execute(
                """
                SELECT m.source,
                       m.source_type,
                       m.thumbnail_uri,
                       m.group_id
                FROM medias m
                WHERE m.id = ?
                """,
                [media_id],
            ).fetchone()
            if media_info is None:
                return None
            # TODO Consider renaming fields for clarity
            item["media_uri"] = media_info[0]
            item["source_type"] = media_info[1]
            item["thumbnail_uri"] = media_info[2]
            item["group"] = media_info[3]
            if filters:
                item["metadata"] = self.get_media_metadata(
                    cursor, media_id, collection, filters
                )
            return item
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve item data ({media_id}): {e}")
        finally:
            if cursor:
                cursor.close()

    def get_items(
        self, collection: str, media_ids: List[int], filters: List[int] = []
    ) -> Dict[int, Dict[str, Any]]:
        """Retrieve many items' data in a fixed number of queries.

        Args:
            collection: Name of the collection
            media_ids: Media IDs of the items to retrieve
            filters: Tagset IDs whose values to include under ``metadata``.
                If empty, ``metadata`` is left out, as in `get_item`

        Returns:
            Item dictionaries shaped like `get_item`'s, keyed by media ID.
            IDs not found in the collection are absent.
        """
        if not media_ids:
            return {}

        cursor = None
        batch_conn = None
        try:
            cursor = self._db_connection[collection].cursor()
            # The IDs can exceed the SQLite parameter limit, so they go in a
            # temp table on a dedicated connection, as in get_group_segment_times
            batch_conn = sqlite3.connect(self._db_path[collection], autocommit=False)
            batch_conn.execute(
                "CREATE TEMPORARY TABLE temp_item (id INTEGER PRIMARY KEY)"
            )
            batch_conn.executemany(
                "INSERT OR IGNORE INTO temp_item (id) VALUES (?)",
                [(media_id,) for media_id in media_ids],
            )
            rows = batch_conn.execute(
                """
                SELECT m.id,
                       m.source,
                       m.source_type,
                       m.thumbnail_uri,
                       m.group_id
                FROM medias m
                WHERE m.id IN (SELECT id FROM temp_item)
                """
            ).fetchall()
            items = {
                media_id: {
                    "item_id": media_id,
                    "media_uri": source,
                    "source_type": source_type,
                    "thumbnail_uri": thumbnail_uri,
                    "group": group_id,
                }
                for media_id, source, source_type, thumbnail_uri, group_id in rows
            }
            if not filters or not items:
                return items

            tagsets = cursor.execute(
                f"""
                SELECT id, name, tagtype_id
                FROM tagsets
                WHERE id IN ({db_helper.placeholders(len(filters))})
                ORDER BY id
                """,
                list(filters),
            ).fetchall()
            tagset_ids_by_tagtype: Dict[str, List[int]] = {}
            for tagset_id, _, tagtype_id in tagsets:
                tagtype = self._tagtype_cache[collection][tagtype_id]
                tagset_ids_by_tagtype.setdefault(tagtype, []).append(tagset_id)

            # One query per tag type table, covering every item and tagset in it
            values: Dict[Tuple[int, int], List[Any]] = {}
            for tagtype, tagset_ids in tagset_ids_by_tagtype.items():
                value_rows = batch_conn.execute(
                    f"""
                    SELECT tgs.media_id, t.tagset_id, t.value
                    FROM taggings tgs
                    JOIN {tagtype}_tags t ON t.id = tgs.tag_id
                    WHERE tgs.media_id IN (SELECT id FROM temp_item)
                    AND t.tagset_id IN ({db_helper.placeholders(len(tagset_ids))})
                    ORDER BY t.id
                    """,
                    tagset_ids,
                ).fetchall()
                for media_id, tagset_id, value in value_rows:
                    values.setdefault((media_id, tagset_id), []).append(value)

            # Keys follow tagset ID order and a lone value stays unwrapped,
            # matching get_media_metadata
            for media_id, item in items.items():
                item["metadata"] = {}
                for tagset_id, tagset_name, _ in tagsets:
                    tagset_values = values.get((media_id, tagset_id))
                    if tagset_values:
                        item["metadata"][tagset_name] = (
                            tagset_values
                            if len(tagset_values) > 1
                            else tagset_values[0]
                        )
            return items
        except Exception as e:
            raise DatabaseError(
                f"Failed to retrieve item data for {len(media_ids)} items from {collection}: {e}"
            )
        finally:
            if cursor:
                cursor.close()
            if batch_conn:
                batch_conn.close()

    def get_total_items(self, collection: str, index: Optional[str] = None) -> int:
        """Get the total count of items in the specified collection/index.

        Args:
            collection: Name of the collection
            index: Name of the index; defaults to the collection's default index

        Returns:
            Total number of items in the index

        Raises:
            DatabaseError: If the collection or index is unknown
        """
        index = self._resolve_index_name(collection, index)
        self._ensure_index_mapping(collection, index)
        try:
            return len(self._item_datapoint_mapping_cache[collection][index])
        except Exception as e:
            raise DatabaseError(
                f"Failed to get total items for collection {collection}: {e}"
            )

    def get_filtered_media_ids(self, collection: str, filters: ActiveFilters) -> set:
        """Retrieve item IDs that pass the specified active filters.

        Args:
            collection: Name of the collection
            filters: ActiveFilters object specifying filter criteria
        Returns:
            Set of item IDs that pass the filters
        """
        cursor = None
        try:
            query, params = db_helper.compile_active_filters(
                active=filters, tagtype_map=self._tagtype_cache[collection]
            )
            cursor = self._db_connection[collection].cursor()
            passed_ids: Set[int] = {r[0] for r in cursor.execute(query, params).fetchall()}
            group_ids = passed_ids & self._group_media_ids.get(collection, set())
            group_medias = set()
            if len(group_ids) > self._sqlite_limit:
                # Use a dedicated connection so the temp table stays off the shared one.
                # With autocommit=False the transaction is already open on connect, and
                # closing the connection rolls it back and discards the temp table
                batch_conn = sqlite3.connect(self._db_path[collection], autocommit=False)
                try:
                    batch_conn.execute("CREATE TEMPORARY TABLE temp_passed_gid (id INTEGER PRIMARY KEY)")
                    batch_conn.executemany("INSERT INTO temp_passed_gid (id) VALUES (?)", [(gid,) for gid in group_ids])
                    group_medias = batch_conn.execute(
                        """
                        SELECT id
                        FROM medias m
                        WHERE m.group_id IN (SELECT id FROM temp_passed_gid)
                        """
                    ).fetchall()
                finally:
                    batch_conn.close()
            else:
                group_medias = cursor.execute(
                    f"""
                    SELECT id
                    FROM medias m
                    WHERE m.group_id IN ({",".join(["?"] * len(group_ids))})
                    """,
                    list(group_ids)
                ).fetchall()
            group_medias = set([r[0] for r in group_medias])
            passed_none_group_ids = list((passed_ids - group_ids) | group_medias)
            return list(passed_none_group_ids)
        except Exception as e:
            raise DatabaseError(
                f"Failed to retrieve filtered item IDs from {collection}: {e}"
            )
        finally:
            if cursor:
                cursor.close()

    def create_item_to_datapoint_mapping(
        self, collection: str, index_config: IndexConfig
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Create a mapping from item IDs to their datapoint indices for one index.

        Reads `index_config.tagset_name`, resolving `index_config.source_type`
        (e.g. "Image") against the database's own source_types table
        rather than assuming fixed IDs, and using it to filter the mapped
        medias to the expected kind.

        Args:
            collection: Name of the collection
            index_config: The index whose id mapping to build

        Returns:
            (mapping, rev_mapping): index_id -> media_id and media_id -> index_id
        """
        cursor = None
        try:
            cursor = self._db_connection[collection].cursor()
            if self._db_type[collection] == "sqlite":
                source_type_row = cursor.execute(
                    "SELECT id FROM source_types WHERE name = ?",
                    [index_config.source_type],
                ).fetchone()
                if source_type_row is None:
                    raise DatabaseError(
                        f"Unknown source_type {index_config.source_type!r} for collection {collection!r}"
                    )
                source_type_id = source_type_row[0]

                mapping = {}
                rev_mapping = {}
                ts_id = cursor.execute(
                    "SELECT id FROM tagsets WHERE name = ?",
                    [index_config.tagset_name],
                ).fetchone()

                if ts_id is not None:
                    # Fetch mapping from numerical_int_tags
                    ts_id = ts_id[0]
                    res = cursor.execute(
                        """
                        SELECT m.id, nit.value
                        FROM medias m
                        JOIN taggings tgs ON m.id = tgs.media_id
                        JOIN numerical_int_tags nit ON tgs.tag_id = nit.id
                        WHERE nit.tagset_id = ?
                        AND m.source_type = ?
                        """,
                        [ts_id, source_type_id],
                    ).fetchall()
                    for media_id, index_id in res:
                        mapping[index_id] = media_id
                        rev_mapping[media_id] = index_id

                if mapping == {}:
                    raise DatabaseError(
                        f"No valid item to datapoint mapping could be created for "
                        f"collection {collection}, index {index_config.name!r}"
                    )

                return mapping, rev_mapping
        except Exception as e:
            raise DatabaseError(
                f"Failed to create item to datapoint mapping for collection "
                f"{collection}, index {index_config.name!r}: {e}"
            )
        finally:
            if cursor:
                cursor.close()

    def clear_cache(self, collection: Optional[str] = None):
        """Clear cached data for the specified collection or all collections.

        This method is useful for memory management or when data needs to be
        reloaded from disk.

        Args:
            collection: Name of the collection to clear, or None to clear all
        """
        if collection:
            self._item_datapoint_mapping_cache.pop(collection, None)
            self._rev_item_datapoint_mapping_cache.pop(collection, None)
            self._time_tagsets_cache.pop(collection, None)
        else:
            self._item_datapoint_mapping_cache.clear()
            self._rev_item_datapoint_mapping_cache.clear()
            self._time_tagsets_cache.clear()
