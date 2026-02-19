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


"""Item-related request and response schemas.

These models are used by the ``/exq/item/*`` endpoints and `ItemService`.
"""

from __future__ import annotations
from pydantic import BaseModel

from .session import SessionInfo


class ItemRequest(BaseModel):
    """Request model for retrieving information about a specific item."""

    mediaId: int
    """Unique identifier for the media item to retrieve."""

    session_info: SessionInfo
    """Session context information."""


class ItemDetailRequest(BaseModel):
    """Request tagset/filter values for a given media item and its group."""

    mediaId: int
    """Unique identifier for the media item to retrieve."""

    filterIds: list[int] = []
    """Tagset/filter IDs whose values should be included in the response."""

    session_info: SessionInfo
    """Session context information."""


class IsExcludedRequest(BaseModel):
    """Request model for checking if an item belongs to an excluded group."""

    itemId: int
    """Unique identifier for the item to check."""

    excluded_ids: list[int]
    """List of excluded item IDs to check against."""

    session_info: SessionInfo
    """Session context information."""


class ItemInfo(BaseModel):
    """Response model containing detailed information about a media item."""

    item_id: int
    """Unique identifier for the item."""

    thumbnail_uri: str
    """URI for the item's thumbnail image."""

    media_uri: str
    """URI for the full-resolution media content."""

    group: str
    """Group identifier the item belongs to."""

    metadata: dict
    """Additional item metadata (tag values, etc.)."""


class ClearItemSetRequest(BaseModel):
    """Request model for clearing a set of items from a session."""

    session: str
    """Unique identifier for the search session."""

    modelId: int
    """Identifier for the model associated with the item set."""

    name: str
    """Name of the item set to clear (e.g. ``'seen'``, ``'excluded'``)."""
