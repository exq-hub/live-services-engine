import duckdb
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import FilePath

from app.repositories import db_helper
from app.schemas import ActiveFiltersDB

from ..core.exceptions import DatabaseError

class DatabaseRepository:
    """Repository class for managing metadata, filters, and related item data.

    This repository provides access to collection metadata, filters for search
    operations, and related item mappings. 
    """

    def __init__(self):
        """Initialize the metadata repository with empty caches."""
        self._db_connection: Dict[str, duckdb.DuckDBPyConnection] = {}
        self._db_type: Dict[str, str] = {}
        self._item_datapoint_mapping_cache: Dict[str, Dict[int, int]] = {}
        self._rev_item_datapoint_mapping_cache: Dict[str, Dict[int, int]] = {}
        self._tagtype_cache: Dict[str, Dict[int, str]] = {}

    def load_database(
        self,
        collection: str,
        database_file: str,
        database: str='duckdb',
    ) -> None:
        """
        Open connection to the database file and create item to datapoint mapping
        with the provided manifest file. If manifest_file is not provided, the mapping
        is based on the database insertion order (not guaranteed).

        Args:
            collection: Name of the collection to load metadata for
            metadata_file: Path to the JSON file containing metadata
            manifest_file: Text file listing file paths of datapoints in order

        Raises:
            DatabaseError: If file doesn't exist or connection fails
        """
        try:
            db_path = Path(database_file)
            if not db_path.exists():
                raise DatabaseError(f"Database file not found: {database_file}")
            self._db_type[collection] = database
            if database == 'duckdb':
                self._db_connection[collection] = duckdb.connect(db_path, read_only=True)
                rows = self._db_connection[collection].execute(
                    """
                    SELECT id, description as name 
                    FROM tag_types
                    """
                ).fetchall()
                self._tagtype_cache[collection] = {
                    row[0]: row[1] for row in rows
                }
            self._item_datapoint_mapping_cache[collection] = {}
            self._rev_item_datapoint_mapping_cache[collection] = {}
        except Exception as e:
            raise DatabaseError(f"Failed to load database from {db_path}: {e}")


    def map_manifest_to_db(
        self,
        collection: str,
        manifest_file: str,
        index: str='clip'
    ) -> None:
        try:
            if index == 'clip':
                file_type = 1 
            elif index == 'caption':
                file_type = 4
            (
                self._item_datapoint_mapping_cache[collection][index],
                self._rev_item_datapoint_mapping_cache[collection][index]
            ) = \
                self.create_item_to_datapoint_mapping(collection, manifest_file, file_type)
        except Exception as e:
            raise DatabaseError(f"Failed to map manifest {manifest_file}: {e}")

    
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


    def get_filters(self, collection: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached filter definitions for the specified collection.

        Args:
            collection: Name of the collection

        Returns:
            Cached filter definitions dictionary or None if not loaded
        """
        try:
            filters = {}
            with self._db_connection[collection].cursor() as cursor:
                if self._db_type[collection] == 'duckdb':
                    rows = cursor.execute(
                        """
                        SELECT ts.id, ts.name, tt.id as tagtype_id, tt.description as tagtype
                        FROM tagsets ts 
                        JOIN tag_types tt ON ts.tagtype_id = tt.id
                        """
                        ).fetchall()
                    columns = [col[0] for col in cursor.description]
                    filters = [dict(zip(columns,row)) for row in rows]
            return filters
        
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve filters from {collection}: {e}")


    def get_filter_values(
        self, 
        collection: str, 
        filter_id: int, 
        tagtype_id: int
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
        try:
            with self._db_connection[collection].cursor() as cursor:
                if self._db_type[collection] == 'duckdb':
                    if tagtype_id not in self._tagtype_cache[collection]:
                        raise DatabaseError(
                            f"Tagtype with id {tagtype_id} not found in collection {collection}: {e}"
                        )
                    
                    rows = cursor.execute(
                        f"""
                        SELECT t.id, t.name 
                        FROM {self._tagtype_cache[collection][tagtype_id]}_tags t
                        JOIN tagsets ts ON t.tagset_id = ts.id
                        WHERE ts.tagtype_id = ? AND ts.id = ?
                        """,
                        [tagtype_id, filter_id]
                    ).fetchall()
                    columns = [col[0] for col in cursor.description]
                    filter_values = [dict(zip(columns,row)) for row in rows]
                return filter_values
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve filter values of {filter_id} from {collection}: {e}")


    def get_media_ids(self, collection, index_ids: list[int], index='clip') -> list[int]:
        return [
            self._item_datapoint_mapping_cache[collection][index][idx]
            for idx in index_ids if idx != -1
        ]
    
    def get_index_ids(self, collection, index_ids: list[int], index='clip') -> list[int]:
        return [
            self._rev_item_datapoint_mapping_cache[collection][index][idx]
            for idx in index_ids if idx != -1
        ]


    def get_media_metadata(
        self, 
        cursor: duckdb.DuckDBPyConnection,
        media_id: int, 
        filters: List[str]=[],
        index: str='clip'
    ) -> Dict[str, Any]:
        """Retrieve metadata for a specific media item.

        Args:
            cursor: Database cursor
            media_id: ID of the media item
            filters: List of metadata fields to include, if empty include all
            mapped: Whether the item_id is already mapped to media_id
        
        Returns:
            Metadata dictionary for the media item
        """

        # Get metadata from tags
        metadata = {}
        query = """
                SELECT tgs.tag_id, 
                    ts.id as tagset_id,
                    ts.name as tagset_name, 
                    tt.description as tagtype,
                FROM medias m
                JOIN taggings tgs ON m.id = tgs.media_id
                JOIN tags t ON tgs.tag_id = t.id
                JOIN tagsets ts ON t.tagset_id = ts.id
                JOIN tag_types tt ON ts.tagtype_id = tt.id
                WHERE m.id = ?
                """
        tag_info = None
        if filters:
            query += f""" AND ts.id IN ({','.join(['?'] * len(filters))}) 
                        ORDER BY ts.id"""
            tag_info = cursor.execute(query, [media_id] + filters).fetchdf()
        else:
            query += " ORDER BY ts.id"
            tag_info = cursor.execute(query, [media_id]).fetchdf()

        # Group tags by tagtype and tagset
        grouped = (
            tag_info.groupby('tagset_id')
                .agg(
                    tag_ids = ('tag_id', list),
                    tagtype = ('tagtype', 'first'),
                    tagset_name = ('tagset_name', 'first'),
                )
                .reset_index()
        )
        for row in grouped.itertuples():
            tag_ids_placeholder = ",".join(['?'] * len(row.tag_ids))
            tag_names = cursor.execute(f"""
                SELECT name 
                FROM {row.tagtype}_tags
                WHERE id IN ({tag_ids_placeholder})
                """,
                row.tag_ids
            ).fetchall()
            if len(tag_names) > 1:
                metadata[row.tagset_name] = [name[0] for name in tag_names]
            else:
                metadata[row.tagset_name] = tag_names[0][0]
        return metadata


    def get_related_items(self, collection: str, item_id: int) -> Optional[List[int]]:
        """Retrieve related items for the specified collection.

        Args:
            collection: Name of the collection

        Returns:
            Cached related items for a specific item
        """
        try:
            with self._db_connection[collection].cursor() as cursor:
                related_items = []
                if self._db_type[collection] == 'duckdb':
                    rows = cursor.execute(
                        """
                        SELECT id
                        FROM medias
                        WHERE group_id = (SELECT group_id FROM medias WHERE id = ?)
                        """,
                        [item_id]
                    ).fetchall()
                    if rows:
                        related_items = [row[0] for row in rows]
                return related_items
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve related items for {item_id} from {collection}: {e}")


    def get_item(
        self,
        collection: str,
        media_id: int,
        filters: list[int]=[]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific item's metadata by its ID.

        Args:
            collection: Name of the collection
            item_id: Index of the item in the collection
            filters: List of metadata fields to include, if empty include all

        Returns:
            Item metadata dictionary or None if not found or invalid ID
        """
        try:
            item = {}
            item['item_id'] = media_id
            with self._db_connection[collection].cursor() as cursor:
                if self._db_type[collection] == 'duckdb':
                    media_info = cursor.execute("""
                            SELECT m.file_uri,
                                    m.group_id,
                                    m2.file_uri as group_uri
                            FROM medias m
                            JOIN medias m2 ON m2.id = m.group_id
                            WHERE m.id = ?
                            """, [media_id]).fetchone()
                    # TODO Consider renaming fields for clarity 
                    item['thumbnail_uri'] = media_info[0]
                    item['group'] = media_info[1]
                    item['media_uri'] = media_info[2]
                    item['metadata'] = \
                        self.get_media_metadata(
                            cursor, 
                            media_id,
                            filters,
                        )
                return item
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve metadata for item {media_id}: {e}")


    def get_group(
        self,
        collection: str,
        group_id: int,
        filters: list[int]=[]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific group's metadata by its ID.

        Args:
            collection: Name of the collection
            group_id: Media ID of the group

        Returns:
            Group metadata dictionary or None if not found
        """
        try:
            with self._db_connection[collection].cursor() as cursor:
                if self._db_type[collection] == 'duckdb':
                    group_info = \
                        self.get_media_metadata(
                            cursor, 
                            group_id,
                            filters,
                        )
                    return group_info
            return None
    
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve group {group_id} from {collection}: {e}")


    def get_total_items(self, collection: str, index='clip') -> int:
        """Get the total count of items in the specified collection.

        Args:
            collection: Name of the collection

        Returns:
            Total number of items, or 0 if collection not loaded
        """
        try:
            if index == 'clip':
                count = self._db_connection[collection].execute(
                    "SELECT COUNT(*) FROM medias WHERE group_id IS NOT NULL"
                ).fetchone()[0]
                return count
            elif index == 'caption':
                count = self._db_connection[collection].execute(
                    "SELECT COUNT(*) FROM temp_table"
                ).fetchone()[0]
                return count
        except Exception as e:
            raise DatabaseError(f"Failed to get total items for collection {collection}: {e}")

    def get_captions_with_nearest_keyframes(self, collection:str, suggestions: List[int]) -> List[Dict[str, Any]]:
        """
        Get transcripts of suggestions along with their nearest keyframe
        """
        try:
            with self._db_connection[collection].cursor() as cursor:
                ph = ",".join("?" * len(suggestions))
                if self._db_type[collection] == 'duckdb':
                    caption_rows = cursor.execute(
                        f"""
                        SELECT text, start_sec, day_id, hour_id, camera_id FROM transcript_table WHERE id IN ({ph})
                        """,
                        suggestions
                    ).fetchall()
                    start_sec_tagset_id = cursor.execute("SELECT id FROM tagsets WHERE name = 'Start (sec)'").fetchone()[0]
                    results = []
                    for row in caption_rows:
                        relevant_media_ids = cursor.execute(
                            """
                            SELECT m.id, m.file_uri 
                            FROM medias m
                            JOIN taggings tgs ON tgs.media_id = m.id
                            WHERE file_type = 1
                            GROUP BY m.id, m.file_uri
                            HAVING COUNT(DISTINCT CASE WHEN tgs.tag_id IN (?, ?, ?) THEN tgs.tag_id END) = 3
                            ORDER BY m.id
                            """,
                            [row[2], row[3], row[4]]
                        ).fetchall()
                        relevant_media_ids = [r[0] for r in relevant_media_ids]
                        ph = ",".join("?" * len(relevant_media_ids))
                        closest_media_keyframe = cursor.execute(
                            f"""
                            SELECT m.id, m.file_uri, ndt.name
                            FROM medias m
                            JOIN taggings tgs ON tgs.media_id = m.id
                            JOIN numerical_dec_tags ndt ON tgs.tag_id = ndt.id
                            WHERE ndt.tagset_id = ?
                            AND   ndt.name <= ?
                            AND   m.id IN ({ph})
                            ORDER BY ndt.name DESC
                            """,
                            [start_sec_tagset_id, row[1]] + relevant_media_ids
                        ).fetchone()
                        if closest_media_keyframe is None:
                            closest_media_keyframe = cursor.execute(
                                f"""
                                SELECT m.id, m.file_uri, ndt.name
                                FROM medias m
                                JOIN taggings tgs ON tgs.media_id = m.id
                                JOIN numerical_dec_tags ndt ON tgs.tag_id = ndt.id
                                WHERE ndt.tagset_id = ?
                                AND   ndt.name >= ?
                                AND   m.id IN ({ph})
                                ORDER BY ndt.name ASC
                                """,
                                [start_sec_tagset_id, row[1]] + relevant_media_ids
                            ).fetchone()
                        results.append({'text': row[0], 'media_id': closest_media_keyframe[0]})
                    return results
        except Exception as e:
            raise DatabaseError(f"Failed to get captions from suggestions {collection, suggestions}: {e}")

    def get_filtered_media_ids(self, collection: str, filters: ActiveFiltersDB) -> set:
        """Retrieve item IDs that pass the specified active filters.
        
        Args:
            collection: Name of the collection
            filters: ActiveFiltersDB object specifying filter criteria
        Returns:
            Set of item IDs that pass the filters
        """
        try:
            query, params = \
                db_helper.compile_active_filters(
                    active=filters, tagtype_map=self._tagtype_cache[collection]
                )
            with self._db_connection[collection].cursor() as cursor:
                passed_ids = [r[0] for r in cursor.execute(query, params).fetchall()]
                return passed_ids
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve filtered item IDs from {collection}: {e}")


    def create_item_to_datapoint_mapping(self, collection: str, manifest_file: str, file_type: int=1) -> Dict[str, int]:
        """Create a mapping from item IDs to their datapoint indices.

        This is useful for converting between item identifiers and their
        positions in embedding matrices or other indexed data structures.

        Args:
            collection: Name of the collection
            manifest_file: Text file listing file paths of datapoints in order

        Returns:
            Dictionary mapping index ids to
        """
        try:
            cursor = self._db_connection[collection].cursor()
            if self._db_type[collection] == 'duckdb':
                rows = cursor.execute(
                    """
                    SELECT id, file_uri
                    FROM medias
                    WHERE group_id IS NOT NULL
                    AND file_type = ?
                    ORDER BY id
                    """,
                    [file_type])
                mapping = {}
                rev_mapping = {}
                if Path(manifest_file).exists():
                    with open(manifest_file, 'r') as f:
                        file_paths = {}
                        for idx, line in enumerate(f):
                            file_paths[line.strip()] = idx
                        for media_id, file_uri in rows.fetchall():
                            if file_uri in file_paths:
                                mapping[file_paths[file_uri]] = media_id
                                rev_mapping[media_id] = file_paths[file_uri]
                else:
                    for idx, (media_id, _) in enumerate(rows.fetchall()):
                        mapping[idx] = media_id    
                        rev_mapping[media_id] = idx
                return mapping, rev_mapping
        except Exception as e:
            raise DatabaseError(f"Failed to create item to datapoint mapping for collection {collection}: {e}")


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
        else:
            self._item_datapoint_mapping_cache.clear()
            self._rev_item_datapoint_mapping_cache.clear()
