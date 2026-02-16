"""Administrative, informational, and client-event logging routes.

Defines endpoints under ``/exq/`` for session management, collection
metadata queries, and audit logging of client-side events:

- ``GET  /init/{session}`` -- initialise a session; returns available collections.
- ``POST /info/totalItems`` -- total item count for a collection.
- ``GET  /info/filters/{session}/{collection}`` -- filter (tagset) definitions.
- ``GET  /info/filters/values/{session}/{collection}/{tagtypeId}/{tagsetId}``
  -- possible values for a specific filter.
- ``POST /log/addModel`` / ``POST /log/removeModel`` -- audit model lifecycle.
- ``POST /log/clientEvent`` -- batch-log arbitrary client-side UI events.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends

from app.repositories.database_repository import DatabaseRepository

from ...schemas import (
    ClientEventRequest,
    SessionInfo,
    AddOrRemoveModelRequest,
)
from ..dependencies import get_database_repository, get_config_manager

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
    database_repo: DatabaseRepository=Depends(get_database_repository),
) -> Dict[str, int]:
    """Get total number of items in a collection."""
    try:
        total_items = database_repo.get_total_items(request.collection)

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

@router.get("/info/filters/{session}/{collection}")
async def get_filters(
    session: str,
    collection: str,
    background_tasks: BackgroundTasks,
    database_repo: DatabaseRepository=Depends(get_database_repository),
) -> List[Dict[str, Any]]:
    """Get available filter definitions for a collection"""
    try:
        filters = database_repo.get_filters(collection) or []

        # Log the request
        background_tasks.add_task(
            _log_filters_request,
            session=session,
            collection=collection,
            filters=filters,
        )

        return filters
    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/info/filters/values/{session}/{collection}/{tagtypeId}/{tagsetId}")
async def get_filter_values(
    session: str,
    collection: str,
    tagtypeId: int,
    tagsetId: int,
    background_tasks: BackgroundTasks,
    database_repo: DatabaseRepository=Depends(get_database_repository),
) -> List[Dict[str, Any]]:
    """Get possible values for a specific filter in a collection."""
    try:
        filter_values = database_repo.get_filter_values(collection, tagsetId, tagtypeId)
        # Log the request
        background_tasks.add_task(
            _log_filters_values_request,
            session=session,
            collection=collection,
            filter_values=filter_values,
            tagtypeId=tagtypeId,
            tagsetId=tagsetId,
        )
        return filter_values

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


@router.post("/log/clientEvent")
async def log_client_event(
    request: List[ClientEventRequest],
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Log a generic client event."""
    background_tasks.add_task(
        _log_client_event,
        events=request
    )
    return {"status": "Logged successfully"}


# Background logging functions
async def _log_session_init(session: str, collections: list):
    """Background task to log session initialization."""
    from ...utils import dump_log_msgpack, get_current_timestamp

    log_message = {
        "timestamp": get_current_timestamp(),
        "action": "Initialize Exquisitor LSE Session",
        "session": session,
        "display_attrs": {"session": session, "collections": collections},
    }

    dump_log_msgpack(log_message, "./logs/admin.log")


async def _log_total_items_request(session: str, collection: str, total_items: int):
    """Background task to log total items request."""
    from ...utils import dump_log_msgpack, get_current_timestamp

    log_message = {
        "timestamp": get_current_timestamp(),
        "action": "Initialize Exquisitor LSE Session",
        "session": session,
        "display_attrs": {
            "session": session,
            "collection": collection,
            "total_items": total_items,
        },
    }

    dump_log_msgpack(log_message, "./logs/admin.log")


async def _log_filters_request(session: str, collection: str, filters: list[Dict[str, Any]]):
    """Background task to log total items request."""
    from ...utils import dump_log_msgpack, get_current_timestamp

    log_message = {
        "timestamp": get_current_timestamp(),
        "action": "Get filter definitions for collection request",
        "session": session,
        "display_attrs": {
            "session": session,
            "collection": collection,
            "filters": filters,
        },
    }

    dump_log_msgpack(log_message, "./logs/admin.log")


async def _log_filters_values_request(session: str, collection: str, filter_values: list[Dict[str, Any]], tagtypeId: int, tagsetId: int):
    """Background task to log total items request."""
    from ...utils import dump_log_msgpack, get_current_timestamp

    log_message = {
        "timestamp": get_current_timestamp(),
        "action": "Get filter values for collection request",
        "session": session,
        "display_attrs": {
            "session": session,
            "collection": collection,
            "filter_values": filter_values,
            "tagtypeId": tagtypeId,
            "tagsetId": tagsetId,
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


async def _log_client_event(events: List[ClientEventRequest]):
    """Background task to log client events."""
    from ...utils import dump_log_msgpack, get_current_timestamp

    for event in events:
        display_attrs = {
            "session": event.session,
            "element_id": event.element_id,
            "route": event.route,
        }
        log_message = {
            "timestamp": get_current_timestamp(),
            "session": event.session,
            "action": f"Client Event: {event.action}",
            "display_attrs": display_attrs,
            "body": event.data,
        }

        dump_log_msgpack(log_message, "./logs/admin.log")