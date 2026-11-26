# SPDX-FileCopyrightText: 2026 Ujjwal Sharma and Omar Shahbaz Khan
# SPDX-License-Identifier: AGPL-3.0-or-later


"""Item metadata and relationship endpoint route handlers.

Defines POST endpoints under ``/exq/item/``:

- ``/info`` -- basic item info (media URI, thumbnail, source type, group),
  plus ``metadata`` for selected tagsets/filters when ``filterIds`` is given.
- ``/info/batch`` returns the ``/info`` result for many items at once, with
  unknown IDs listed under ``missing``.
- ``/base`` -- deprecated alias of ``/info``.
- ``/details`` -- deprecated; ``metadata`` only, derived from ``/info``.
- ``/related`` -- IDs of items sharing the same ``group_id``, optionally
  sorted by a tagset's values (``sortBy``, ``sortOrder``).
- ``/excluded`` -- check whether an item belongs to any excluded group.

All endpoints delegate to `ItemService` and schedule background audit-log
tasks for every request.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.exceptions import DatabaseError

from ...schemas import (
    ItemBatchRequest,
    ItemDetailRequest,
    ItemRequest,
    IsExcludedRequest,
    RelatedItemsRequest,
)
from ...services.item_service import ItemService
from ...services.logging_service import LoggingService
from ..dependencies import get_item_service, get_logging_service

router = APIRouter(prefix="/exq", tags=["items"])


@router.post("/item/info")
@router.post("/item/base", deprecated=True)
async def get_item_base_info(
    request: ItemRequest,
    background_tasks: BackgroundTasks,
    item_service: ItemService = Depends(get_item_service),
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, Any]:
    """Get basic information for an item.

    ``/item/base`` is a deprecated alias of ``/item/info``, kept for
    existing clients.
    """
    try:
        result = item_service.get_item_info(request)

        background_tasks.add_task(
            logging_service.log_item_request,
            "Get Info For Item",
            request.session_info.session,
            request.session_info.modelId,
            request.session_info.collection,
            request.mediaId,
            {"mediaType": result["mediaType"]},
        )

        return result

    except DatabaseError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/item/info/batch")
async def get_items_base_info(
    request: ItemBatchRequest,
    background_tasks: BackgroundTasks,
    item_service: ItemService = Depends(get_item_service),
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, List[Any]]:
    """Get basic information for many items in one call.

    Each entry under ``items`` matches what ``/item/info`` returns for that
    ID. IDs not found in the collection are listed under ``missing``
    instead of failing the request.
    """
    try:
        result = item_service.get_items_info(request)

        background_tasks.add_task(
            logging_service.log_item_request,
            "Get Info For Items",
            request.session_info.session,
            request.session_info.modelId,
            request.session_info.collection,
            request.mediaIds,
            {"missing": result["missing"]},
        )

        return result

    except DatabaseError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/item/details", deprecated=True)
async def get_item_detailed_info(
    request: ItemDetailRequest,
    background_tasks: BackgroundTasks,
    item_service: ItemService = Depends(get_item_service),
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, Any]:
    """Get detailed information for an item.

    Deprecated: use ``/exq/item/info`` with ``filterIds``. This endpoint
    fetches the full base info and throws everything away except
    ``metadata``.
    """
    try:
        info = item_service.get_item_info(request)

        background_tasks.add_task(
            logging_service.log_item_request,
            "Get Detailed Info For Item",
            request.session_info.session,
            request.session_info.modelId,
            request.session_info.collection,
            request.mediaId,
        )

        return info.get("metadata", {})

    except DatabaseError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/item/related")
async def get_related_items(
    request: RelatedItemsRequest,
    background_tasks: BackgroundTasks,
    item_service: ItemService = Depends(get_item_service),
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, List[int]]:
    """Get related items for an item, optionally sorted by a tagset's values."""
    try:
        result = item_service.get_related_items(
            request, sort_by=request.sortBy, sort_order=request.sortOrder
        )

        background_tasks.add_task(
            logging_service.log_item_request,
            "Get Related Info For Item",
            request.session_info.session,
            request.session_info.modelId,
            request.session_info.collection,
            request.mediaId,
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
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, bool]:
    """Check if item is in an excluded group."""
    try:
        result = item_service.is_item_excluded(request)

        background_tasks.add_task(
            logging_service.log_item_request,
            "Check if item is in excluded group",
            request.session_info.session,
            request.session_info.modelId,
            request.session_info.collection,
            request.itemId,
            {"excluded": result["excludedOrNot"]},
        )

        return result

    except DatabaseError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
