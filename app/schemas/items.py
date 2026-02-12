from __future__ import annotations
from pydantic import BaseModel

from .session import SessionInfo


class ItemRequest(BaseModel):
    """Request model for retrieving information about a specific item.

    Attributes:
        mediaId: Unique identifier for the item to retrieve
        session_info: Session context information
    """

    mediaId: int
    session_info: SessionInfo

class ItemDetailRequest(BaseModel):
    """
    Request tagset/filter values for a given media_id/item_id and its group.
    Attributes:
        mediaId: Unique identifier for the item to retrieve
        filterIds: List of tagset/filter IDs to apply to the item
        session_info: Session context information
    """
    mediaId: int
    filterIds: list[int] = []
    session_info: SessionInfo


class IsExcludedRequest(BaseModel):
    """Request model for checking if an item is in the excluded list.

    Attributes:
        itemId: Unique identifier for the item to check
        excluded_ids: List of excluded item IDs to check against
        session_info: Session context information
    """

    itemId: int
    excluded_ids: list[int]
    session_info: SessionInfo


class ItemInfo(BaseModel):
    """Response model containing detailed information about a media item.

    Attributes:
        item_id: Unique identifier for the item
        thumbnail_uri: URI for the item's thumbnail image
        media_uri: URI for the full media content
        group: Group or collection identifier the item belongs to
        metadata: Dictionary containing additional item metadata
    """

    item_id: int
    thumbnail_uri: str
    media_uri: str
    group: str
    metadata: dict


class ClearItemSetRequest(BaseModel):
    """Request model for clearing a set of items from a session.

    Attributes:
        session: Unique identifier for the search session
        modelId: Identifier for the model associated with the item set
        name: Name of the item set to clear
    """

    session: str
    modelId: int
    name: str
