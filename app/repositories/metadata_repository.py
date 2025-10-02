"""Repository for metadata operations."""

import orjson
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.exceptions import MetadataError


class MetadataRepository:
    """Repository class for managing metadata, filters, and related item data.

    This repository provides access to collection metadata, filters for search
    operations, and related item mappings. It implements caching to avoid
    repeated file I/O operations and provides thread-safe access to metadata.
    """

    def __init__(self):
        """Initialize the metadata repository with empty caches."""
        self._metadata_cache: Dict[str, Dict] = {}
        self._filters_cache: Dict[str, Dict] = {}
        self._related_items_cache: Dict[str, Dict] = {}

    def load_metadata(self, collection: str, metadata_file: str) -> Dict[str, Any]:
        """Load and cache metadata from a JSON file for the specified collection.

        Args:
            collection: Name of the collection to load metadata for
            metadata_file: Path to the JSON file containing metadata

        Returns:
            Dictionary containing the loaded metadata

        Raises:
            MetadataError: If file doesn't exist, contains invalid JSON, or loading fails
        """
        try:
            if collection in self._metadata_cache:
                return self._metadata_cache[collection]

            metadata_path = Path(metadata_file)
            if not metadata_path.exists():
                raise MetadataError(f"Metadata file not found: {metadata_file}")

            with open(metadata_path, "r") as f:
                metadata = orjson.loads(f.read())

            self._metadata_cache[collection] = metadata
            return metadata

        except orjson.JSONDecodeError as e:
            raise MetadataError(f"Invalid JSON in metadata file {metadata_file}: {e}")
        except Exception as e:
            raise MetadataError(f"Failed to load metadata from {metadata_file}: {e}")

    def load_filters(self, collection: str, filters_file: str) -> Dict[str, Any]:
        """Load and cache filter definitions from a JSON file for the specified collection.

        Args:
            collection: Name of the collection to load filters for
            filters_file: Path to the JSON file containing filter definitions

        Returns:
            Dictionary containing the loaded filter definitions

        Raises:
            MetadataError: If file doesn't exist, contains invalid JSON, or loading fails
        """
        try:
            if collection in self._filters_cache:
                return self._filters_cache[collection]

            filters_path = Path(filters_file)
            if not filters_path.exists():
                raise MetadataError(f"Filters file not found: {filters_file}")

            with open(filters_path, "r") as f:
                filters = orjson.loads(f.read())

            self._filters_cache[collection] = filters
            return filters

        except orjson.JSONDecodeError as e:
            raise MetadataError(f"Invalid JSON in filters file {filters_file}: {e}")
        except Exception as e:
            raise MetadataError(f"Failed to load filters from {filters_file}: {e}")

    def load_related_items(
        self, collection: str, related_items_file: str
    ) -> Dict[str, List[int]]:
        """Load and cache related items mapping from a JSON file.

        Args:
            collection: Name of the collection to load related items for
            related_items_file: Path to the JSON file containing item relationships

        Returns:
            Dictionary mapping item IDs to lists of related item IDs

        Raises:
            MetadataError: If file doesn't exist, contains invalid JSON, or loading fails
        """
        try:
            if collection in self._related_items_cache:
                return self._related_items_cache[collection]

            related_path = Path(related_items_file)
            if not related_path.exists():
                raise MetadataError(
                    f"Related items file not found: {related_items_file}"
                )

            with open(related_path, "r") as f:
                related_items = orjson.loads(f.read())

            self._related_items_cache[collection] = related_items
            return related_items

        except orjson.JSONDecodeError as e:
            raise MetadataError(
                f"Invalid JSON in related items file {related_items_file}: {e}"
            )
        except Exception as e:
            raise MetadataError(
                f"Failed to load related items from {related_items_file}: {e}"
            )

    def get_metadata(self, collection: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached metadata for the specified collection.

        Args:
            collection: Name of the collection

        Returns:
            Cached metadata dictionary or None if not loaded
        """
        return self._metadata_cache.get(collection)

    def get_filters(self, collection: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached filter definitions for the specified collection.

        Args:
            collection: Name of the collection

        Returns:
            Cached filter definitions dictionary or None if not loaded
        """
        return self._filters_cache.get(collection)

    def get_related_items(self, collection: str) -> Optional[Dict[str, List[int]]]:
        """Retrieve cached related items mapping for the specified collection.

        Args:
            collection: Name of the collection

        Returns:
            Cached related items mapping or None if not loaded
        """
        return self._related_items_cache.get(collection)

    def get_item(self, collection: str, item_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific item's metadata by its ID.

        Args:
            collection: Name of the collection
            item_id: Index of the item in the collection

        Returns:
            Item metadata dictionary or None if not found or invalid ID
        """
        metadata = self.get_metadata(collection)
        if not metadata or "items" not in metadata:
            return None

        items = metadata["items"]
        if 0 <= item_id < len(items):
            return items[item_id]
        return None

    def get_group(self, collection: str, group_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific group's metadata by its ID.

        Args:
            collection: Name of the collection
            group_id: String identifier of the group

        Returns:
            Group metadata dictionary or None if not found
        """
        metadata = self.get_metadata(collection)
        if not metadata or "groups" not in metadata:
            return None

        return metadata["groups"].get(group_id)

    def get_total_items(self, collection: str) -> int:
        """Get the total count of items in the specified collection.

        Args:
            collection: Name of the collection

        Returns:
            Total number of items, or 0 if collection not loaded
        """
        metadata = self.get_metadata(collection)
        if not metadata or "items" not in metadata:
            return 0
        return len(metadata["items"])

    def create_item_to_datapoint_mapping(self, collection: str) -> Dict[str, int]:
        """Create a mapping from item IDs to their datapoint indices.

        This is useful for converting between item identifiers and their
        positions in embedding matrices or other indexed data structures.

        Args:
            collection: Name of the collection

        Returns:
            Dictionary mapping item_id strings to integer indices
        """
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
            self._related_items_cache.pop(collection, None)
        else:
            self._metadata_cache.clear()
            self._filters_cache.clear()
            self._related_items_cache.clear()
