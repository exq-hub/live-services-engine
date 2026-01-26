"""Service for handling item-related operations."""

from typing import Dict, List, Tuple, Any

from app.repositories.database_repository import DatabaseRepository
from app.repositories.metadata_repository import MetadataRepository

from ..schemas import ItemRequest, IsExcludedRequest
from ..core.exceptions import MetadataError


class ItemService:
    """Service for item-related operations."""

    def __init__(self, metadata_repository, config_manager):
        self.metadata_repo: MetadataRepository | DatabaseRepository = metadata_repository
        self.config_manager = config_manager

    def get_item_base_info(self, request: ItemRequest) -> Dict[str, Any]:
        """Get basic information for an item."""
        collection = request.session_info.collection
        item = self.metadata_repo.get_item(collection, request.itemId, request.filter_ids)

        if not item:
            raise MetadataError(
                f"Item {request.itemId} not found in collection {collection}"
            )

        # Determine media type
        media_type = 1 if ".mp4" in item["media_uri"] else 0

        # Get collection config for paths
        collection_config = self.config_manager.config.collection_configs[collection]

        result = {
            "id": request.itemId,
            "name": item["item_id"],
            "mediaId": request.itemId,
            "mediaType": media_type,
            "relatedGroupId": item["group"],
            "thumbPath": f"{collection_config.thumbnail_media_url}/{item['thumbnail_uri']}",
            "srcPath": f"{collection_config.original_media_url}/{item['media_uri']}",
            "metadata": item["metadata"]
        }

        # Add segment info for videos
        if (
            media_type == 1
            and "metadata" in item
            and "segment_info" in item["metadata"]
        ):
            result["segmentInfo"] = item["metadata"]["segment_info"]

        return result

    def get_item_detailed_info(self, request: ItemRequest) -> Dict[str, Any]:
        """Get detailed information for an item."""
        collection = request.session_info.collection
        if isinstance(self.metadata_repo, DatabaseRepository):
            item = self.metadata_repo.get_item(collection, request.itemId, request.filter_ids)
            group = self.metadata_repo.get_group(collection, item["group"], request.filter_ids)
        else:
            item = self.metadata_repo.get_item(collection, request.itemId)
            group = self.metadata_repo.get_group(collection, item["group"])

        if not item:
            raise MetadataError(
                f"Item {request.itemId} not found in collection {collection}"
            )

        # Process item metadata
        info_pairs = {}
        info_pairs['item'] = []
        if "metadata" in item:
            info_pairs['item'].extend(self._process_metadata(item["metadata"]))

        # Process group metadata
        if group:
            info_pairs['group'] = []
            info_pairs['group'].extend(self._process_group_data(group))

        return {"infoPairs": info_pairs}

    def get_related_items(self, request: ItemRequest) -> Dict[str, List[int]]:
        """Get related items for an item."""
        collection = request.session_info.collection
        if isinstance(self.metadata_repo, DatabaseRepository):
            related_items = self.metadata_repo.get_related_items(collection, request.itemId)
            return {"related": related_items}
        item = self.metadata_repo.get_item(collection, request.itemId)

        if not item:
            raise MetadataError(
                f"Item {request.itemId} not found in collection {collection}"
            )

        related_items = self.metadata_repo.get_related_items(collection)
        if not related_items:
            return {"related": []}

        group_items = related_items.get(item["group"], [])
        return {"related": group_items}

    def is_item_excluded(self, request: IsExcludedRequest) -> Dict[str, bool]:
        """Check if an item is in an excluded group."""
        collection = request.session_info.collection

        # Get the item's related items
        item_related = self.get_related_items(
            ItemRequest(itemId=request.itemId, session_info=request.session_info)
        )["related"]
        item_related_set = set(item_related)

        # Check against all excluded items
        for excluded_id in request.excluded_ids:
            excluded_related = self.get_related_items(
                ItemRequest(itemId=excluded_id, session_info=request.session_info)
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
