"""Service for handling item-related operations."""

from typing import Dict, List, Tuple, Any

from app.core.config import ConfigManager
from app.repositories.database_repository import DatabaseRepository

from ..schemas import ItemDetailRequest, ItemRequest, IsExcludedRequest
from ..core.exceptions import DatabaseError


class ItemService:
    """Service for item-related operations."""

    def __init__(self, database_repository, config_manager):
        self.database_repo: DatabaseRepository = database_repository
        self.config_manager: ConfigManager = config_manager

    def get_item_base_info(self, request: ItemRequest) -> Dict[str, Any]:
        """Get basic information for an item."""
        collection = request.session_info.collection
        item = self.database_repo.get_item(collection, request.mediaId)

        if not item:
            raise DatabaseError(
                f"Item {request.mediaId} not found in collection {collection}"
            )

        # Get collection config for paths
        collection_config = self.config_manager.config.collection_configs[collection]

        result = {
            "id": request.mediaId,
            "name": item["item_id"],
            "mediaId": request.mediaId,
            "mediaType": item["source_type"],
            "thumbPath": f"{collection_config.thumbnail_media_url}/{item['thumbnail_uri']}",
            "srcPath": f"{collection_config.original_media_url}/{item['media_uri']}",
            "groupId": item["group"],
        }

        return result

    def get_item_detailed_info(self, request: ItemDetailRequest) -> Dict[str, Any]:
        """Get detailed information for an item."""
        collection = request.session_info.collection
        item = {}
        group = {}
        item = self.database_repo.get_item(collection, request.mediaId, request.filterIds)

        if not item:
            raise DatabaseError(
                f"Item {request.mediaId} not found in collection {collection}"
            )

        if item is None or item.get('metadata') is None:
            return {}

        # Process item metadata
        metadata = {
            k : v
            for k, v in item['metadata'].items()
        }
        # Process group metadata
        metadata.update({
            k : v
            for k, v in group.items()
        })

        return metadata


    def get_related_items(self, request: ItemRequest) -> Dict[str, List[int]]:
        """Get related items for an item."""
        collection = request.session_info.collection
        related_items = self.database_repo.get_related_items(collection, request.mediaId)
        return {"related": related_items}


    def is_item_excluded(self, request: IsExcludedRequest) -> Dict[str, bool]:
        """Check if an item is in an excluded group."""

        # Check against all excluded items
        for excluded_id in request.excluded_ids:
            excluded_related = self.get_related_items(
                ItemRequest(mediaId=excluded_id, session_info=request.session_info)
            )["related"]

            if request.mediaId in excluded_related:
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
