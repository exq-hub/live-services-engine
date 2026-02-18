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
from ...services.logging_service import LoggingService
from ..dependencies import get_database_repository, get_config_manager, get_logging_service

router = APIRouter(prefix="/exq", tags=["admin"])


@router.get("/init/{session}")
async def init_session(
    session: str,
    background_tasks: BackgroundTasks,
    config_manager=Depends(get_config_manager),
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, Any]:
    """Initialize session and return available collections."""
    try:
        config = config_manager.config
        collections = config.collections

        background_tasks.add_task(logging_service.log_session_init, session, collections)

        return {"session": session, "collections": collections}

    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/info/totalItems")
async def get_total_items(
    request: SessionInfo,
    background_tasks: BackgroundTasks,
    database_repo: DatabaseRepository = Depends(get_database_repository),
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, int]:
    """Get total number of items in a collection."""
    try:
        total_items = database_repo.get_total_items(request.collection)

        background_tasks.add_task(
            logging_service.log_total_items_request,
            request.session,
            request.collection,
            total_items,
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
    database_repo: DatabaseRepository = Depends(get_database_repository),
    logging_service: LoggingService = Depends(get_logging_service),
) -> List[Dict[str, Any]]:
    """Get available filter definitions for a collection"""
    try:
        filters = database_repo.get_filters(collection) or []

        background_tasks.add_task(
            logging_service.log_filter_operation,
            "Get filter definitions for collection",
            session,
            0,
            collection,
            {"filters": filters},
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
    database_repo: DatabaseRepository = Depends(get_database_repository),
    logging_service: LoggingService = Depends(get_logging_service),
) -> List[Dict[str, Any]]:
    """Get possible values for a specific filter in a collection."""
    try:
        filter_values = database_repo.get_filter_values(collection, tagsetId, tagtypeId)

        background_tasks.add_task(
            logging_service.log_filter_operation,
            "Get filter values for collection",
            session,
            0,
            collection,
            {"tagtypeId": tagtypeId, "tagsetId": tagsetId},
        )

        return filter_values

    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/log/addModel")
async def log_add_model(
    request: AddOrRemoveModelRequest,
    background_tasks: BackgroundTasks,
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, str]:
    """Log model addition."""
    background_tasks.add_task(
        logging_service.log_model_operation,
        "Log Add Model",
        request.session,
        request.modelId,
        request.collection,
        request.model_dump_json(),
    )
    return {"status": "Logged successfully"}


@router.post("/log/removeModel")
async def log_remove_model(
    request: AddOrRemoveModelRequest,
    background_tasks: BackgroundTasks,
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, str]:
    """Log model removal."""
    background_tasks.add_task(
        logging_service.log_model_operation,
        "Log Remove Model",
        request.session,
        request.modelId,
        request.collection,
        request.model_dump_json(),
    )
    return {"status": "Logged successfully"}


@router.post("/log/clientEvent")
async def log_client_event(
    request: List[ClientEventRequest],
    background_tasks: BackgroundTasks,
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, str]:
    """Log a generic client event."""
    for event in request:
        background_tasks.add_task(
            logging_service.audit_logger.log,
            action=f"Client Event: {event.action}",
            session=event.session,
            data={"element_id": event.element_id, "route": event.route, "body": event.data},
        )
    return {"status": "Logged successfully"}
