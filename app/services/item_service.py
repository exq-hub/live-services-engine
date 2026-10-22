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


"""Service for handling item-related operations.

`ItemService` provides a high-level API consumed by the item route handlers.
It delegates to `DatabaseRepository` for data access and enriches raw
database results with collection-specific media URLs from `ConfigManager`.

Key operations:

- **Base info** -- media URI, thumbnail path, source type, group ID, plus
  metadata values for selected tagsets/filters when ``filterIds`` is given.
- **Batch info** returns base info for many items in one call, listing the
  IDs not found separately.
- **Detailed info** -- deprecated; metadata values only, derived from base info.
- **Related items** -- other media items sharing the same ``group_id``,
  optionally sorted by a tagset's values.
- **Exclusion check** -- whether a given item belongs to any excluded group,
  used by the UI to grey-out or hide items during interactive search.
"""

from typing import Dict, List, Optional, Tuple, Any

from app.core.config import CollectionConfig, ConfigManager
from app.repositories.database_repository import DatabaseRepository

from ..schemas import (
    ItemBatchRequest,
    ItemDetailRequest,
    ItemRequest,
    IsExcludedRequest,
)
from ..core.exceptions import DatabaseError


class ItemService:
    """Service for item-related operations."""

    def __init__(self, database_repository, config_manager):
        self.database_repo: DatabaseRepository = database_repository
        """Repository used for all item and metadata lookups."""

        self.config_manager: ConfigManager = config_manager
        """Configuration manager providing collection media URLs."""

    def get_item_info(self, request: ItemRequest) -> Dict[str, Any]:
        """Get basic information for an item.

        If ``request.filterIds`` is non-empty, the values of those
        tagsets/filters are included under ``metadata``.
        """
        collection = request.session_info.collection
        item = self.database_repo.get_item(
            collection, request.mediaId, request.filterIds
        )

        if not item:
            raise DatabaseError(
                f"Item {request.mediaId} not found in collection {collection}"
            )

        # Get collection config for paths
        collection_config = self.config_manager.config.collection_configs[collection]

        return self._format_item(
            request.mediaId, item, collection_config, bool(request.filterIds)
        )

    def get_items_info(self, request: ItemBatchRequest) -> Dict[str, Any]:
        """Get basic information for many items in one call.

        Returns the `get_item_info` result of every found item under
        ``items``, in request order with duplicates collapsed, and the IDs
        not found in the collection under ``missing``.
        """
        collection = request.session_info.collection
        media_ids = list(dict.fromkeys(request.mediaIds))
        items = self.database_repo.get_items(collection, media_ids, request.filterIds)

        # Get collection config for paths
        collection_config = self.config_manager.config.collection_configs[collection]

        return {
            "items": [
                self._format_item(
                    media_id,
                    items[media_id],
                    collection_config,
                    bool(request.filterIds),
                )
                for media_id in media_ids
                if media_id in items
            ],
            "missing": [media_id for media_id in media_ids if media_id not in items],
        }

    def _format_item(
        self,
        media_id: int,
        item: Dict[str, Any],
        collection_config: CollectionConfig,
        include_metadata: bool,
    ) -> Dict[str, Any]:
        """Shape a repository item into the item info response."""
        result = {
            "id": media_id,
            "name": item["item_id"],
            "mediaId": media_id,
            "mediaType": item["source_type"],
            "thumbPath": collection_config.thumbnail_url(item["thumbnail_uri"]),
            "srcPath": collection_config.original_url(item["media_uri"]),
            "groupId": item["group"],
        }

        if include_metadata:
            result["metadata"] = item.get("metadata") or {}

        return result

    def get_item_detailed_info(self, request: ItemDetailRequest) -> Dict[str, Any]:
        """Get detailed information for an item.

        Deprecated: use `get_item_info` with ``filterIds``. This fetches
        the full base info and throws everything away except ``metadata``.
        """
        return self.get_item_info(request).get("metadata", {})

    def get_related_items(
        self,
        request: ItemRequest,
        sort_by: Optional[int] = None,
        sort_order: str = "asc",
    ) -> Dict[str, List[int]]:
        """Get related items for an item, optionally sorted by a tagset's values."""
        collection = request.session_info.collection
        related_items = self.database_repo.get_related_items(
            collection, request.mediaId, sort_by=sort_by, sort_order=sort_order
        )
        return {"related": related_items}

    def is_item_excluded(self, request: IsExcludedRequest) -> Dict[str, bool]:
        """Check if an item is in an excluded group."""

        # Check against all excluded items
        for excluded_id in request.excluded_ids:
            excluded_related = self.get_related_items(
                ItemRequest(mediaId=excluded_id, session_info=request.session_info)
            )["related"]

            if request.itemId in excluded_related:
                return {"excludedOrNot": True}

        return {"excludedOrNot": False}

    def _process_metadata(
        self, metadata: Dict[str, Any]
    ) -> List[Tuple[str, List[str]]]:
        """Process item metadata into display format."""
        info_pairs = []

        for key, value in metadata.items():
            display_name = key.replace("_", " ").capitalize()

            if isinstance(value, list):
                display_values = [str(s).replace("_", " ").capitalize() for s in value]
            else:
                display_values = [str(value).replace("_", " ").capitalize()]

            info_pairs.append((display_name, display_values))

        return info_pairs

    def _process_group_data(self, group: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
        """Process group data into display format."""
        info_pairs = []

        for key, value in group.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], str):
                    display_values = value
                else:
                    display_values = [str(i) for i in value]
            else:
                display_values = [str(value)]

            info_pairs.append((key, display_values))

        return info_pairs
