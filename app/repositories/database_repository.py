import duckdb
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        self._metadata_cache: Dict[str, Dict] = {}
        self._filters_cache: Dict[str, Dict] = {}
        self._item_datapoint_mapping_cache: Dict[str, Dict[str, int]] = {}
        self._tagtype_cache: Dict[str, Dict[int, str]] = {}

    def load_database(
            self, collection: str,
            database_file: str,
            manifest_file: str='',
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

            
            self._item_datapoint_mapping_cache[collection] = \
                self.create_item_to_datapoint_mapping(collection, manifest_file)

        except Exception as e:
            raise DatabaseError(f"Failed to load database from {db_path}: {e}")
    
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
                        SELECT ts.id, ts.name, tt.id as tagtype_id, tt.description as tagtype"
                        FROM tagset ts 
                        JOIN tag_types tt ON ts.tagtype_id = tt.id
                        """
                        ).fetchall()
                    columns = [col[0] for col in cursor.description]
                    filters = [dict(zip(columns,row)) for row in rows]
            return filters
        
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve filters from {collection}: {e}")


    def get_filter_values(self, collection: str, filter_id: int, tagtype_id: int) -> Optional[List[Any]]:
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


    def get_media_metadata(
            self, 
            cursor: duckdb.DuckDBPyCursor,
            media_id: int, 
            filters: List[str]
    ) -> Dict[str, Any]:
        """Retrieve metadata for a specific media item.

        Args:
            cursor: Database cursor
            media_id: ID of the media item
            filters: List of metadata fields to include, if empty include all
        
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
                JOIN tags ON tgs.tag_id = t.id
                JOIN tagsets ts ON t.tagset_id = ts.id
                JOIN tagtypes tt ON ts.tagtype_id = tt.id
                WHERE media_id = ?
                """
        tag_info = None
        if filters:
            query += f""" AND ts.name IN ({','.join(['?'] * len(filters))}) 
                        ORDER BY ts.id"""
            tag_info = cursor.execute(query, [media_id, filters]).fetchdf()
        else:
            query += " ORDER BY ts.id"
            tag_info = cursor.execute(query).fetchdf()

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
                [row.tag_ids]
            ).fetchall()
            metadata[row.tagset_name] = [name[0] for name in tag_names]
        return metadata


    def get_related_items(self, collection: str, item_id: int) -> Optional[List[int]]:
        """Retrieve cached related items mapping for the specified collection.

        Args:
            collection: Name of the collection

        Returns:
            Cached related items mapping for a specific item
        """
        try:
            with self._db_connection[collection].cursor() as cursor:
                related_items = []
                if self._db_type[collection] == 'duckdb':
                    rows = cursor.execute(
                        """
                        SELECT media_id
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


    def get_item(self, collection: str, item_id: int, filters=[]) -> Optional[Dict[str, Any]]:
        """Retrieve a specific item's metadata by its ID.

        Args:
            collection: Name of the collection
            item_id: Index of the item in the collection
            filters: List of metadata fields to include, if empty include all

        Returns:
            Item metadata dictionary or None if not found or invalid ID
        """
        if item_id not in self._metadata_cache:
            try:
                mapped_item_id = self._item_datapoint_mapping_cache[collection].get(str(item_id))
                item = {}
                item['item_id'] = str(item_id)
                    
                with self._db_connection[collection].cursor() as cursor:
                    if self._db_type[collection] == 'duckdb':
                        media_info = cursor.execute("""
                                SELECT m.file_uri,
                                       m.group_id,
                                       m2.file_uri as group_uri
                                FROM medias m
                                JOIN medias m2 ON m.id = m2.group_id
                                WHERE m.id = ?
                                """, [mapped_item_id]).fetchone()
                        # TODO Consider renaming fields for clarity 
                        item['thumbnail_uri'] = media_info[0]
                        item['group'] = media_info[1]
                        item['media_uri'] = media_info[2]
                        item['metadata'] = self.get_media_metadata(cursor, mapped_item_id, filters)
                    self._metadata_cache[item_id] = item
            except Exception as e:
                raise DatabaseError(f"Failed to retrieve metadata for item {item_id}: {e}")
        else:
            return self._metadata_cache[item_id]


    def get_group(self, collection: str, group_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific group's metadata by its ID.

        Args:
            collection: Name of the collection
            group_id: Media ID of the group

        Returns:
            Group metadata dictionary or None if not found
        """
        if group_id not in self._metadata_cache: 
            try:
                with self._db_connection[collection].cursor() as cursor:
                    if self._db_type[collection] == 'duckdb':
                        group_info = self.get_media_metadata(cursor, group_id, [])
                        return group_info
                return None
        
            except Exception as e:
                raise DatabaseError(f"Failed to retrieve group {group_id} from {collection}: {e}")
        else:
            return self._metadata_cache[int(group_id)]


    def get_total_items(self, collection: str) -> int:
        """Get the total count of items in the specified collection.

        Args:
            collection: Name of the collection

        Returns:
            Total number of items, or 0 if collection not loaded
        """
        try:
            count = self._db_connection[collection].execute(
                "SELECT COUNT(*) FROM medias WHERE group_id IS NOT NULL"
            ).fetchone()[0]
            return count
        except Exception as e:
            raise DatabaseError(f"Failed to get total items for collection {collection}: {e}")


    def create_item_to_datapoint_mapping(self, collection: str, manifest_file: str) -> Dict[str, int]:
        """Create a mapping from item IDs to their datapoint indices.

        This is useful for converting between item identifiers and their
        positions in embedding matrices or other indexed data structures.

        Args:
            collection: Name of the collection
            manifest_file: Text file listing file paths of datapoints in order

        Returns:
            Dictionary mapping item_id strings to integer indices
        """
        try:
            cursor = self._db_connection[collection].cursor()
            if self._db_type[collection] == 'duckdb':
                rows = cursor.execute(
                    """
                    SELECT id, file_uri
                    FROM medias
                    WHERE group_id IS NOT NULL
                    ORDER BY id
                    """)
                mapping = {}
                if Path(manifest_file).exists():
                    with open(manifest_file, 'r') as f:
                        file_paths = {}
                        for idx, line in enumerate(f):
                            file_paths[line.strip()] = idx
                        for media_id, file_uri in rows.fetchall():
                            if file_uri in file_paths:
                                mapping[file_paths[file_uri]] = media_id
                else:
                    for idx, (media_id, _) in enumerate(rows.fetchall()):
                        mapping[media_id] = idx    
                return mapping
        except Exception as e:
            raise DatabaseError(f"Failed to create item to datapoint mapping for collection {collection}: {e}")
        metadata = self.get_metadata(collection)
        if not metadata or "items" not in metadata:
            return {}

        return {item["item_id"]: idx for idx, item in enumerate(metadata["items"])}

    def clear_cache(self, collection: Optional[str] = None):
        """Clear cached data for the specified collection or all collections.

        This method is useful for memory management or when data needs to be
        reloaded from disk.

        Args:
            collection: Name of the collection to clear, or None to clear all
        """
        if collection:
            self._metadata_cache.pop(collection, None)
            self._filters_cache.pop(collection, None)
            self._item_datapoint_mapping_cache.pop(collection, None)
        else:
            self._metadata_cache.clear()
            self._filters_cache.clear()
            self._item_datapoint_mapping_cache.clear()
