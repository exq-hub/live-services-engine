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


"""Item metadata and relationship endpoint route handlers.

Defines four POST endpoints under ``/exq/item/``:

- ``/base`` -- basic item info (media URI, thumbnail, source type, group).
- ``/details`` -- detailed metadata for selected tagsets/filters.
- ``/related`` -- IDs of items sharing the same ``group_id``.
- ``/excluded`` -- check whether an item belongs to any excluded group.

All endpoints delegate to `ItemService` and schedule background audit-log
tasks for every request.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.exceptions import DatabaseError

from ...schemas import ItemDetailRequest, ItemRequest, IsExcludedRequest
from ...services.item_service import ItemService
from ..dependencies import get_item_service

router = APIRouter(prefix="/exq", tags=["items"])


@router.post("/item/base")
async def get_item_base_info(
    request: ItemRequest,
    background_tasks: BackgroundTasks,
    item_service: ItemService = Depends(get_item_service),
) -> Dict[str, Any]:
    """Get basic information for an item."""
    try:
        result = item_service.get_item_base_info(request)

        # Log the request
        background_tasks.add_task(
            _log_item_request,
            action="Get Info For Item",
            session=request.session_info.session,
            model_id=request.session_info.modelId,
            collection=request.session_info.collection,
            media_id=request.mediaId,
            item_name=result["name"],
            additional_data={
                "mediaType": result["mediaType"],
            },
        )

        return result

    except DatabaseError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/item/details")
async def get_item_detailed_info(
    request: ItemDetailRequest,
    background_tasks: BackgroundTasks,
    item_service: ItemService = Depends(get_item_service),
) -> Dict[str, Any]:
    """Get detailed information for an item."""
    try:
        result = item_service.get_item_detailed_info(request)

        # Log the request
        background_tasks.add_task(
            _log_item_request,
            action="Get Detailed Info For Item",
            session=request.session_info.session,
            model_id=request.session_info.modelId,
            collection=request.session_info.collection,
            media_id=request.mediaId,
            item_name=request.mediaId
        )

        return result

    except DatabaseError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/item/related")
async def get_related_items(
    request: ItemRequest,
    background_tasks: BackgroundTasks,
    item_service: ItemService = Depends(get_item_service),
) -> Dict[str, List[int]]:
    """Get related items for an item."""
    try:
        result = item_service.get_related_items(request)

        # Get item name for logging
        base_info = item_service.get_item_base_info(request)

        # Log the request
        background_tasks.add_task(
            _log_item_request,
            action="Get Related Info For Item",
            session=request.session_info.session,
            model_id=request.session_info.modelId,
            collection=request.session_info.collection,
            media_id=request.mediaId,
            item_name=base_info["name"],
        )

        return result

    except DatabaseError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/item/excluded")
async def is_item_excluded(
    request: IsExcludedRequest,
    background_tasks: BackgroundTasks,
    item_service: ItemService = Depends(get_item_service),
) -> Dict[str, bool]:
    """Check if item is in an excluded group."""
    try:
        result = item_service.is_item_excluded(request)

        # Get item name for logging
        base_info = item_service.get_item_base_info(
            ItemRequest(mediaId=request.mediaId, session_info=request.session_info)
        )

        # Log the request
        background_tasks.add_task(
            _log_item_request,
            action="Check if item is in excluded group",
            session=request.session_info.session,
            model_id=request.session_info.modelId,
            collection=request.session_info.collection,
            media_id=request.mediaId,
            item_name=base_info["name"],
            additional_data={
                "excluded": result["excludedOrNot"],
            },
        )

        return result

    except DatabaseError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _log_item_request(
    action: str,
    session: str,
    model_id: int,
    collection: str,
    media_id: int,
    item_name: str,
    additional_data: Dict[str, Any] = None,
):
    """Background task to log item requests."""
    from ...utils import dump_log_msgpack, get_current_timestamp
    from ...core.models import container

    config = container.config_manager.config
    collection_config = config.collection_configs.get(collection)
    if collection_config and collection_config.log_directory:
        import uuid

        log_file = f"{collection_config.log_directory}/items_{uuid.uuid4().hex[:8]}.log"
    else:
        log_file = "./logs/items.log"

    data = {
        "session": session,
        "modelId": model_id,
        "collection": collection,
        "mediaId": media_id,
        "mediaName": item_name,
    }
    if additional_data:
        data.update(additional_data)

    log_message = {
        "timestamp": get_current_timestamp(),
        "session": session,
        "action": action,
        "display_attrs": data,
    }

    dump_log_msgpack(log_message, log_file)
