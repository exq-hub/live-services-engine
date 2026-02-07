import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.repositories import db_helper
from app.schemas import ActiveFiltersDB

from ..core.exceptions import DatabaseError

import pandas as pd

class DatabaseRepository:
    """Repository class for managing metadata, filters, and related item data.

    This repository provides access to collection metadata, filters for search
    operations, and related item mappings. 
    """

    def __init__(self):
        """Initialize the metadata repository with empty caches."""
        self._db_connection: Dict[str, sqlite3.Connection] = {}
        self._db_type: Dict[str, str] = {}
        self._item_datapoint_mapping_cache: Dict[str, Dict[str, Dict[int, int]]] = {}
        self._rev_item_datapoint_mapping_cache: Dict[str, Dict[str, Dict[int, int]]] = {}
        self._tagtype_cache: Dict[str, Dict[int, str]] = {}
            
    
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
        database: str='sqlite',
    ) -> None:
        """
        Open connection to the database file and create item to datapoint mapping
        with the provided manifest file.

        Args:
            collection: Name of the collection to load metadata for
            metadata_file: Path to the JSON file containing metadata

        Raises:
            DatabaseError: If file doesn't exist or connection fails
        """
        try:
            db_path = Path(database_file)
            if not db_path.exists():
                raise DatabaseError(f"Database file not found: {database_file}")
            self._db_type[collection] = database
            if database == 'sqlite':
                self._db_connection[collection] = sqlite3.connect(db_path, autocommit=False)
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
            (
                self._item_datapoint_mapping_cache[collection]['clip'],
                self._rev_item_datapoint_mapping_cache[collection]['clip']
            ) = \
                self.create_item_to_datapoint_mapping(collection, source_type=1)
            
            # TODO: Check for other index types

        except Exception as e:
            raise DatabaseError(f"Failed to load database from {db_path}: {e}")


    def map_manifest_to_db(
        self,
        collection: str,
        manifest_file: str=None,
        index: str='clip'
    ) -> None:
        """Create item to datapoint mapping with the provided manifest file.

        Args:
            collection: Name of the collection to load metadata for
            manifest_file: Text file listing file paths of datapoints in order
        Raises:
            DatabaseError: If mapping fails
        """
        try:
            if index == 'clip':
                source_type = 1 
            elif index == 'caption':
                source_type = 4
            (
                self._item_datapoint_mapping_cache[collection][index],
                self._rev_item_datapoint_mapping_cache[collection][index]
            ) = \
                self.create_item_to_datapoint_mapping(collection, manifest_file, source_type)
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
                {
                    "id": f[0],
                    "name": f[1],
                    "tagtype_id": f[2],
                    "tagtype": f[3] 
                }
                for f in rows
            ]
            return filters
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve filters from {collection}: {e}")
        finally:
            if cursor:
                cursor.close()


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
        cursor = None
        try:
            cursor = self._db_connection[collection].cursor()
            if tagtype_id not in self._tagtype_cache[collection]:
                raise DatabaseError(
                    f"Tagtype with id {tagtype_id} not found in collection {collection}: {e}"
                )
            
            rows = cursor.execute(
                f"""
                SELECT t.id, t.value 
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
        finally:
            if cursor:
                cursor.close()


    def get_media_ids(self, collection, index_ids: list[int], index='clip') -> list[int]:
        try:
            return [
                self._item_datapoint_mapping_cache[collection][index][idx]
                for idx in index_ids if idx != -1
            ]
        except Exception as e:
            raise DatabaseError(
                f"Failed to map index IDs ({index_ids}) to media IDs for collection {collection}: {e}"
            )
    
    def get_index_ids(self, collection, media_ids: list[int], index='clip') -> list[int]:
        try:
            return [
                self._rev_item_datapoint_mapping_cache[collection][index][idx]
                for idx in media_ids if idx != -1
            ]
        except Exception as e:
            raise DatabaseError(
                f"Failed to map media IDs ({media_ids}) to index IDs for collection {collection}: {e}"
            )

    def get_media_metadata(
        self, 
        cursor: sqlite3.Connection,
        media_id: int, 
        collection: str,
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
                query += f""" AND ts.id IN ({','.join(['?'] * len(filters))}) 
                            ORDER BY ts.id"""
                if self._db_type[collection] == 'sqlite':
                    tag_info = pd.read_sql_query(
                        query,
                        self._db_connection[collection],
                        params=[media_id] + filters
                    )
                elif self._db_type[collection] == 'duckdb':
                    tag_info = cursor.execute(query, [media_id] + filters).fetchdf()
            else:
                query += " ORDER BY ts.id"
                if self._db_type[collection] == 'sqlite':
                    tag_info = pd.read_sql_query(
                        query,
                        self._db_connection[collection],
                        params=[media_id]
                    )
                elif self._db_type[collection] == 'duckdb':
                    tag_info = cursor.execute(query, [media_id]).fetchdf()

            # Collapse rows into one per tagset, collecting tag IDs and metadata
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
                tag_values = cursor.execute(f"""
                    SELECT value 
                    FROM {row.tagtype}_tags
                    WHERE id IN ({tag_ids_placeholder})
                    """,
                    row.tag_ids
                ).fetchall()
                if len(tag_values) > 1:
                    metadata[row.tagset_name] = [value[0] for value in tag_values]
                else:
                    metadata[row.tagset_name] = tag_values[0][0]
            return metadata
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve metadata for media {media_id}: {e}")


    def get_related_items(self, collection: str, item_id: int) -> Optional[List[int]]:
        """Retrieve related items for the specified collection.

        Args:
            collection: Name of the collection

        Returns:
            Cached related items for a specific item
        """
        cursor = None
        try:
            cursor = self._db_connection[collection].cursor() 
            related_items = []
            if self._db_type[collection] == 'sqlite':
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
        finally:
            if cursor:
                cursor.close()


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
        cursor = None
        try:
            item = {}
            item['item_id'] = media_id
            cursor = self._db_connection[collection].cursor() 
            media_info = cursor.execute(
                """
                SELECT m.source,
                       m.group_id,
                       m2.source as group_src
                FROM medias m
                JOIN medias m2 ON m2.id = m.group_id
                WHERE m.id = ?
                """, 
                [media_id]
            ).fetchone()
            # TODO Consider renaming fields for clarity 
            item['thumbnail_uri'] = media_info[0]
            item['group'] = media_info[1]
            item['media_uri'] = media_info[2]
            item['metadata'] = \
                self.get_media_metadata(
                    cursor, 
                    media_id,
                    collection,
                    filters
                )
            return item
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve metadata for item {media_id}: {e}")
        finally:
            if cursor:
                cursor.close()


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
        cursor = None
        try:
            cursor = self._db_connection[collection].cursor()
            group_info = \
                self.get_media_metadata(
                    cursor, 
                    group_id,
                    collection,
                    filters
                )
            return group_info
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
            return len(self._item_datapoint_mapping_cache[collection][index])
        except Exception as e:
            raise DatabaseError(f"Failed to get total items for collection {collection}: {e}")

    def get_text_source_with_nearest_keyframes(self, collection:str, index:str, suggestions: List[int]) -> List[Dict[str, Any]]:
        """
        Get source for suggestions along with their nearest keyframe
        """
        cursor = None
        try:
            if index not in ['caption', 'transcript']:
                raise DatabaseError(f"Index must be 'caption' or 'transcript', got {index}")

            media_ids = self.get_media_ids(collection, suggestions, index)
            if len(media_ids) != len(suggestions):
                raise DatabaseError(f"Returned media_ids do not match the amount of provided list")

            cursor = self._db_connection[collection].cursor()
            ph = ",".join("?" * len(suggestions))
            closest_kf_ts_id = cursor.execute(
                "SELECT id FROM tagsets WHERE name = 'Closest Keyframe'"
            ).fetchone()
            if closest_kf_ts_id is not None:
                closest_kf_ts_id = closest_kf_ts_id[0]
                texts = cursor.execute(
                    f"""
                    SELECT tgs.media_id, at.value as text
                    FROM taggings tgs
                    JOIN alphanumerical_tags at ON at.id = tgs.tag_id
                    WHERE at.tagset_id = (SELECT id FROM tagsets WHERE name = ?)
                    AND tgs.media_id IN ({ph})
                    ORDER BY tgs.media_id
                    """,
                    [index.capitalize()] + media_ids
                ).fetchall()
                closest_kf_ids = cursor.execute(
                    f"""
                    SELECT tgs.media_id, nit.value as closest_kf_id
                    FROM taggings tgs
                    JOIN numerical_int_tags nit ON nit.id = tgs.tag_id
                    WHERE nit.tagset_id = ?
                    AND   tgs.media_id IN ({ph})
                    ORDER BY tgs.media_id
                    """,
                    [closest_kf_ts_id] + media_ids
                ).fetchall()

                if len(texts) != len(suggestions) or len(closest_kf_ids) != len(suggestions):
                    raise DatabaseError(f"Returned texts or closest keyframe IDs do not match the amount of provided suggestions")

                results = []
                for text_row, kf_row in zip(texts, closest_kf_ids):
                    if text_row[0] != kf_row[0]:
                        raise DatabaseError(f"Text and closest keyframe ID rows do not match for media IDs {media_ids}")
                    results.append({'text': text_row[1], 'media_id': kf_row[1]})
                return results
                
            # if no closest keyframe tagset exists for index, calculate based on 'Start (sec)' tagset
            start_sec_tagset_id = cursor.execute("SELECT id FROM tagsets WHERE name = 'Start (sec)'").fetchone()[0]
            text_rows = cursor.execute(
                f"""
                SELECT m.id, m.group_id, at.value as text 
                FROM medias m
                JOIN taggings tgs ON m.id = tgs.media_id
                JOIN alphanumerical_tags at ON tgs.tag_id = at.id
                WHERE m.id IN ({ph})
                AND   at.tagset_id = (SELECT id FROM tagsets WHERE name = ?)
                ORDER BY m.id
                """,
                media_ids + [index.capitalize()]
            ).fetchall()
            start_secs = cursor.execute(
                f"""
                SELECT m.id, nit.value as start_sec
                FROM medias m
                JOIN taggings tgs ON m.id = tgs.media_id
                JOIN numerical_int_tags nit ON tgs.tag_id = nit.id
                WHERE m.id IN ({ph})
                AND   nit.tagset_id = ?
                ORDER BY m.id
                """,
                media_ids + [start_sec_tagset_id]
            ).fetchall()

            results = []
            for (text_row, start_row) in zip(text_rows, start_secs):
                if text_row[0] != start_row[0]:
                    raise DatabaseError(f"Text and start time rows do not match for media IDs {media_ids}")
                grp_id = text_row[1]
                text = text_row[2]
                txt_start_sec = start_row[3]
                # Pick the keyframe with the smallest absolute difference to the start time of the text
                closest_keyframe = cursor.execute(
                    f"""
                    SELECT m.id, ABS(nit.value - $1) as abs_difference
                    FROM medias m
                    JOIN taggings tgs ON m.id = tgs.media_id
                    JOIN numerical_int_tags nit ON tgs.tag_id = nit.id
                    WHERE source_type = 1   -- Image
                    AND m.group_id = $2     -- Video
                    AND nit.tagset_id = $3  -- Start (sec) ID
                    ORDER BY abs_difference ASC, nit.value ASC -- On ties choose the earlier time keyframe
                    LIMIT 1
                    """,
                    [txt_start_sec, grp_id, start_sec_tagset_id]
                ).fetchone()

                if closest_keyframe is None:
                    raise DatabaseError("Could not determine closest keyframe through tagsets")

                results.append({'text': text, 'media_id': closest_keyframe[0]})
            return results
        except Exception as e:
            raise DatabaseError(
                f"Failed to get nearest keyframes for text suggestions {collection, suggestions}: {e}"
            )
        finally:
            if cursor:
                cursor.close()

    def get_filtered_media_ids(self, collection: str, filters: ActiveFiltersDB) -> set:
        """Retrieve item IDs that pass the specified active filters.
        
        Args:
            collection: Name of the collection
            filters: ActiveFiltersDB object specifying filter criteria
        Returns:
            Set of item IDs that pass the filters
        """
        cursor = None
        try:
            query, params = \
                db_helper.compile_active_filters(
                    active=filters, tagtype_map=self._tagtype_cache[collection]
                )
            cursor = self._db_connection[collection].cursor()
            passed_ids = [r[0] for r in cursor.execute(query, params).fetchall()]
            return passed_ids
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve filtered item IDs from {collection}: {e}")
        finally:
            if cursor:
                cursor.close()


    def create_item_to_datapoint_mapping(self, collection: str, source_type: int=1, index='clip') -> Dict[str, int]:
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
            if self._db_type[collection] == 'sqlite':
                rows = cursor.execute(
                    """
                    SELECT id, source
                    FROM medias
                    WHERE group_id IS NOT NULL
                    AND source_type = ?
                    ORDER BY id
                    """,
                    [source_type])
                mapping = {}
                rev_mapping = {}
                # Determine tagset ID based on index type
                ts_id = None
                if index == 'clip':
                    ts_id = cursor.execute("SELECT id FROM tagsets WHERE name = 'CLIP Index ID'").fetchone()
                elif index == 'caption':
                    ts_id = cursor.execute("SELECT id FROM tagsets WHERE name = 'Caption Index ID'").fetchone()
                elif index == 'transcript':
                    ts_id = cursor.execute("SELECT id FROM tagsets WHERE name = 'Transcript Index ID'").fetchone()

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
                        """,
                        [ts_id]
                    ).fetchall()
                    for media_id, index_id in res:
                        mapping[index_id] = media_id
                        rev_mapping[media_id] = index_id

                if mapping == {}:
                    raise DatabaseError(f"No valid item to datapoint mapping could be created for collection {collection}")

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
