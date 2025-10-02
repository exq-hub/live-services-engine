"""Search-related API routes."""

from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ...schemas import TextSearchRequest, RFSearchRequest
from ...services.search_service import SearchService
from ...services.logging_service import LoggingService
from ...core.exceptions import SearchError
from ..dependencies import get_search_service

router = APIRouter(prefix="/exq/search", tags=["search"])


@router.post("/clip")
async def clip_search(
    request: TextSearchRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service),
) -> Dict[str, Any]:
    """Search using CLIP text embeddings."""
    try:
        result = await search_service.search_text("clip", request)

        # Log the search request
        background_tasks.add_task(
            _log_search_request,
            action="CLIP Search",
            session=request.session_info.session,
            model_id=request.session_info.modelId,
            collection=request.session_info.collection,
            query_data={
                "query": request.text,
                "n_seen": len(request.seen or []),
                "filters": request.filters.model_dump_json()
                if request.filters
                else None,
                "excluded": request.excluded or [],
            },
            suggestions=result["suggestions"],
            request_timestamp=result["request_timestamp"],
            completion_time=result["completion_time"],
        )

        return {"suggestions": result["suggestions"]}

    except SearchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/caption")
async def caption_search(
    request: TextSearchRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service),
) -> Dict[str, Any]:
    """Search using caption embeddings."""
    try:
        result = await search_service.search_text("caption", request)

        # Log the search request
        background_tasks.add_task(
            _log_search_request,
            action="Caption Search",
            session=request.session_info.session,
            model_id=request.session_info.modelId,
            collection=request.session_info.collection,
            query_data={
                "query": request.text,
                "n_seen": len(request.seen or []),
                "filters": request.filters.model_dump_json()
                if request.filters
                else None,
                "excluded": request.excluded or [],
            },
            suggestions=result["suggestions"],
            request_timestamp=result["request_timestamp"],
            completion_time=result["completion_time"],
        )

        return {"suggestions": result["suggestions"]}

    except SearchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/aggregate")
async def aggregate_search(
    request: TextSearchRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service),
) -> Dict[str, Any]:
    """Search using aggregated results from multiple strategies."""
    try:
        result = await search_service.search_text("aggregate", request)

        # Log the search request
        background_tasks.add_task(
            _log_search_request,
            action="Aggregate Text Search (RRF)",
            session=request.session_info.session,
            model_id=request.session_info.modelId,
            collection=request.session_info.collection,
            query_data={
                "query": request.text,
                "n_seen": len(request.seen or []),
                "filters": request.filters.model_dump_json()
                if request.filters
                else None,
                "excluded": request.excluded or [],
                "rrf_k": 60,  # RRF constant
            },
            suggestions=result["suggestions"],
            request_timestamp=result["request_timestamp"],
            completion_time=result["completion_time"],
        )

        return {"suggestions": result["suggestions"]}

    except SearchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/rf")
async def rf_search(
    request: RFSearchRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service),
) -> Dict[str, Any]:
    """Search using relevance feedback (SVM)."""
    try:
        result = await search_service.search_rf(request)

        # Log the search request
        background_tasks.add_task(
            _log_search_request,
            action="RF Search",
            session=request.session_info.session,
            model_id=request.session_info.modelId,
            collection=request.session_info.collection,
            query_data={
                "pos": request.pos,
                "neg": request.neg,
                "n_seen": len(request.seen),
                "filters": request.filters.model_dump_json()
                if request.filters
                else None,
                "excluded": request.excluded,
                "query": request.query,
            },
            suggestions=result["suggestions"],
            request_timestamp=result["request_timestamp"],
            completion_time=result["completion_time"],
        )

        return {"suggestions": result["suggestions"]}

    except SearchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _log_search_request(
    action: str,
    session: str,
    model_id: int,
    collection: str,
    query_data: Dict[str, Any],
    suggestions: list,
    request_timestamp: int,
    completion_time: int,
):
    """Background task to log search requests."""
    # This would normally use the logging service from app state
    # For now, we'll use the original logging method
    from ...utils import dump_log_msgpack
    from ...core.models import container

    config = container.config_manager.config
    # Get log file from collection config
    collection_config = config.collection_configs.get(collection)
    if collection_config and collection_config.log_directory:
        import uuid

        log_file = (
            f"{collection_config.log_directory}/search_{uuid.uuid4().hex[:8]}.log"
        )
    else:
        log_file = "./logs/search.log"

    log_message = {
        "request_timestamp": request_timestamp,
        "completion_time": completion_time,
        "session": session,
        "collection": collection,
        "action": action,
        "display_attrs": {
            "session": session,
            "modelId": model_id,
            "collection": collection,
            "suggestions": suggestions,
            **query_data,
        },
    }

    dump_log_msgpack(log_message, log_file)
