"""Item-related API routes."""

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.repositories.database_repository import DatabaseRepository

from ...schemas import ItemRequest, IsExcludedRequest, SessionInfo, Filter
from ...services.item_service import ItemService
from ...core.exceptions import MetadataError
from ..dependencies import get_item_service, get_metadata_repository

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
            item_id=request.itemId,
            item_name=result["name"],
            additional_data={
                "mediaType": result["mediaType"],
            },
        )

        return result

    except MetadataError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/item/details")
async def get_item_detailed_info(
    request: ItemRequest,
    background_tasks: BackgroundTasks,
    item_service: ItemService = Depends(get_item_service),
) -> Dict[str, Any]:
    """Get detailed information for an item."""
    try:
        result = item_service.get_item_detailed_info(request)

        # Get item name for logging
        base_info = item_service.get_item_base_info(request)

        # Log the request
        background_tasks.add_task(
            _log_item_request,
            action="Get Detailed Info For Item",
            session=request.session_info.session,
            model_id=request.session_info.modelId,
            collection=request.session_info.collection,
            item_id=request.itemId,
            item_name=base_info["name"],
        )

        return result

    except MetadataError as e:
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
            item_id=request.itemId,
            item_name=base_info["name"],
        )

        return result

    except MetadataError as e:
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
            ItemRequest(itemId=request.itemId, session_info=request.session_info)
        )

        # Log the request
        background_tasks.add_task(
            _log_item_request,
            action="Check if item is in excluded group",
            session=request.session_info.session,
            model_id=request.session_info.modelId,
            collection=request.session_info.collection,
            item_id=request.itemId,
            item_name=base_info["name"],
            additional_data={
                "excluded": result["excludedOrNot"],
            },
        )

        return result

    except MetadataError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/filters?session={session}&collection={collection}&filterId={filter_id}&tagtype={tagtype}")
async def get_selected_filters_values(
    session: str,
    collection: str,
    filter_id: str,
    tagtype: str,
    background_tasks: BackgroundTasks,
    metadata_repo: DatabaseRepository=Depends(get_metadata_repository),
) -> Dict[str, List]:
    """Get values for selected filters in a collection."""
    try:
        filter_id = int(filter_id)
        filter_values = metadata_repo.get_filter_values(collection, filter_id, tagtype)

        # Log the request
        background_tasks.add_task(
            _log_filter_request,
            action="Log Get Selected Filters",
            session=session,
            collection=collection,
        )

        return {}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid parameter")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/filters/{session}__{collection}")
async def get_filters(
    session: str,
    collection: str,
    background_tasks: BackgroundTasks,
    metadata_repo=Depends(get_metadata_repository),
) -> Dict[str, List[Filter]]:
    """Get all filters and their values for a collection."""
    try:
        filters = metadata_repo.get_filters(collection)
        if not filters:
            raise HTTPException(
                status_code=404, detail=f"No filters found for collection: {collection}"
            )

        filter_objects = []
        for idx, k in enumerate(filters):
            obj = {
                "id": idx,
                "collectionId": 0,
                "name": k.replace("_", " ").capitalize(),
                "filterType": filters[k]["type"],
            }

            if obj["filterType"] in [2, 3]:  # NumberRange / NumberRangeMulti
                obj["values"] = [
                    min(list(filters[k]["values"])),
                    max(list(filters[k]["values"])),
                ]
            elif obj["filterType"] in [4, 5]:  # RangeLabel / RangeLabelMulti
                obj["values"] = [
                    (idx, el) for idx, el in enumerate(filters[k]["values"])
                ]
            elif obj["filterType"] == 6:  # Count
                obj["values"] = []
                for val, cnt_min, cnt_max in filters[k]["values"]:
                    obj["values"].append(val)
                    obj["count"] = (cnt_min, cnt_max)
            else:  # Single / Multi
                obj["values"] = [
                    v.replace("_", " ").capitalize() for v in list(filters[k]["values"])
                ]

            filter_objects.append(obj)

        # Log the request
        background_tasks.add_task(
            _log_filter_request,
            action="Log Get Filters",
            session=session,
            collection=collection,
        )

        return {"filters": filter_objects}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _log_item_request(
    action: str,
    session: str,
    model_id: int,
    collection: str,
    item_id: int,
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
        "item": item_id,
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


async def _log_filter_request(action: str, session: str, collection: str):
    """Background task to log filter requests."""
    from ...utils import dump_log_msgpack, get_current_timestamp
    from ...core.models import container

    config = container.config_manager.config
    collection_config = config.collection_configs.get(collection)
    if collection_config and collection_config.log_directory:
        import uuid

        log_file = (
            f"{collection_config.log_directory}/filters_{uuid.uuid4().hex[:8]}.log"
        )
    else:
        log_file = "./logs/filters.log"

    log_message = {
        "timestamp": get_current_timestamp(),
        "session": session,
        "action": action,
        "display_attrs": {"collection": collection},
        "body": {},
    }

    dump_log_msgpack(log_message, log_file)