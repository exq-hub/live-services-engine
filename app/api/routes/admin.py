"""Administrative and logging API routes."""

from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends

from app.repositories.database_repository import DatabaseRepository
from app.repositories.metadata_repository import MetadataRepository

from ...schemas import (
    SessionInfo,
    AddOrRemoveModelRequest,
    AppliedFilter,
    ClearItemSetRequest,
)
from ..dependencies import get_metadata_repository, get_config_manager

router = APIRouter(prefix="/exq", tags=["admin"])


@router.get("/init/{session}")
async def init_session(
    session: str,
    background_tasks: BackgroundTasks,
    config_manager=Depends(get_config_manager),
) -> Dict[str, Any]:
    """Initialize session and return available collections."""
    try:
        config = config_manager.config
        collections = config.collections

        # Log session initialization
        background_tasks.add_task(
            _log_session_init, session=session, collections=collections
        )

        return {"session": session, "collections": collections}

    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/info/totalItems")
async def get_total_items(
    request: SessionInfo,
    background_tasks: BackgroundTasks,
    metadata_repo: MetadataRepository | DatabaseRepository=Depends(get_metadata_repository),
) -> Dict[str, int]:
    """Get total number of items in a collection."""
    try:
        total_items = metadata_repo.get_total_items(request.collection)

        # Log the request
        background_tasks.add_task(
            _log_total_items_request,
            session=request.session,
            collection=request.collection,
            total_items=total_items,
        )

        return {"total_items": total_items}

    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/info/filters")
async def get_filters(
    request: SessionInfo,
    background_tasks: BackgroundTasks,
    metadata_repo: MetadataRepository | DatabaseRepository=Depends(get_metadata_repository),
) -> Dict[str, Any]:
    """Get available filter definitions for a collection"""
    try:
        filters = metadata_repo.get_filters(request.collection) or {}

        # Log the request
        background_tasks.add_task(
            _log_filters_request,
            session=request.session,
            collection=request.collection,
            filters=filters,
        )

        return {"filters": filters}

    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/log/addModel")
async def log_add_model(
    request: AddOrRemoveModelRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Log model addition."""
    background_tasks.add_task(
        _log_model_operation,
        action="Log Add Model",
        session=request.session,
        model_id=request.modelId,
        collection=request.collection,
        body_json=request.model_dump_json(),
    )
    return {"status": "Logged successfully"}


@router.post("/log/removeModel")
async def log_remove_model(
    request: AddOrRemoveModelRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Log model removal."""
    background_tasks.add_task(
        _log_model_operation,
        action="Log Remove Model",
        session=request.session,
        model_id=request.modelId,
        collection=request.collection,
        body_json=request.model_dump_json(),
    )
    return {"status": "Logged successfully"}


@router.post("/log/applyFilters")
async def log_apply_filters(
    request: AppliedFilter,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Log filter application."""
    background_tasks.add_task(
        _log_filter_operation,
        action="Log Apply Filters",
        session=request.session_info.session,
        model_id=request.session_info.modelId,
        collection=request.session_info.collection,
        filter_data={
            "FilterName": request.name,
            "FilterValues": request.values,
            "body": request.model_dump_json(),
        },
    )
    return {"status": "Logged successfully"}


@router.post("/log/resetFilters")
async def log_reset_filters(
    request: SessionInfo,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Log filter reset."""
    background_tasks.add_task(
        _log_filter_operation,
        action="Log Reset Filters",
        session=request.session,
        model_id=request.modelId,
        collection=request.collection,
        filter_data={"body": request.model_dump_json()},
    )
    return {"status": "Logged successfully"}


@router.post("/log/clearExcludedGroups")
async def clear_all_excluded(
    request: SessionInfo,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Log clearing of excluded groups."""
    background_tasks.add_task(
        _log_clear_operation,
        action="Cleared the excluded groups list",
        session=request.session,
        model_id=request.modelId,
        body_json=request.model_dump_json(),
    )
    return {"status": "Logged successfully"}


@router.post("/log/clearItemSet")
async def clear_item_set(
    request: ClearItemSetRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Log item set clearing."""
    background_tasks.add_task(
        _log_clear_operation,
        action=f"Cleared items from {request.name}",
        session=request.session,
        model_id=request.modelId,
        body_json=request.model_dump_json(),
        additional_data={"name": request.name},
    )
    return {"status": "Logged successfully"}


@router.post("/log/clearRFModel")
async def clear_rf_model(
    request: SessionInfo,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Log RF model clearing."""
    background_tasks.add_task(
        _log_clear_operation,
        action="Cleared RF Model",
        session=request.session,
        model_id=request.modelId,
        body_json=request.model_dump_json(),
    )
    return {"status": "Logged successfully"}


@router.post("/log/clearConversation")
async def clear_conversation(
    request: SessionInfo,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Log conversation clearing."""
    background_tasks.add_task(
        _log_clear_operation,
        action="Cleared Conversation",
        session=request.session,
        model_id=request.modelId,
        body_json=request.model_dump_json(),
    )
    return {"status": "Logged successfully"}


# Background logging functions
async def _log_session_init(session: str, collections: list):
    """Background task to log session initialization."""
    from ...utils import dump_log_msgpack, get_current_timestamp

    log_message = {
        "timestamp": get_current_timestamp(),
        "action": "Initialize Exquisitor LSE Session",
        "data": {"session": session, "collections": collections},
    }

    dump_log_msgpack(log_message, "./logs/admin.log")


async def _log_total_items_request(session: str, collection: str, total_items: int):
    """Background task to log total items request."""
    from ...utils import dump_log_msgpack, get_current_timestamp

    log_message = {
        "timestamp": get_current_timestamp(),
        "action": "Initialize Exquisitor LSE Session",
        "data": {
            "session": session,
            "collection": collection,
            "total_items": total_items,
        },
    }

    dump_log_msgpack(log_message, "./logs/admin.log")


async def _log_filters_request(session: str, collection: str, total_items: int):
    """Background task to log total items request."""
    from ...utils import dump_log_msgpack, get_current_timestamp

    log_message = {
        "timestamp": get_current_timestamp(),
        "action": "Get filter definitions for collection request",
        "data": {
            "session": session,
            "collection": collection,
            "total_items": total_items,
        },
    }

    dump_log_msgpack(log_message, "./logs/admin.log")


async def _log_model_operation(
    action: str, session: str, model_id: int, collection: str, body_json: str
):
    """Background task to log model operations."""
    from ...utils import dump_log_msgpack, get_current_timestamp

    log_message = {
        "timestamp": get_current_timestamp(),
        "session": session,
        "action": action,
        "display_attrs": {
            "session": session,
            "modelId": model_id,
            "collection": collection,
        },
        "body": body_json,
    }

    dump_log_msgpack(log_message, "./logs/admin.log")


async def _log_filter_operation(
    action: str,
    session: str,
    model_id: int,
    collection: str,
    filter_data: Dict[str, Any],
):
    """Background task to log filter operations."""
    from ...utils import dump_log_msgpack, get_current_timestamp

    log_message = {
        "timestamp": get_current_timestamp(),
        "session": session,
        "action": action,
        "display_attrs": {
            "session": session,
            "modelId": model_id,
            "collection": collection,
            **{k: v for k, v in filter_data.items() if k != "body"},
        },
        "body": filter_data.get("body", ""),
    }

    dump_log_msgpack(log_message, "./logs/admin.log")


async def _log_clear_operation(
    action: str,
    session: str,
    model_id: int,
    body_json: str,
    additional_data: Dict[str, Any] = None,
):
    """Background task to log clear operations."""
    from ...utils import dump_log_msgpack, get_current_timestamp

    display_attrs = {
        "session": session,
        "modelId": model_id,
    }
    if additional_data:
        display_attrs.update(additional_data)

    log_message = {
        "timestamp": get_current_timestamp(),
        "session": session,
        "action": action,
        "display_attrs": display_attrs,
        "body": body_json,
    }

    dump_log_msgpack(log_message, "./logs/admin.log")
