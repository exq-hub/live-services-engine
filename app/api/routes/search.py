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


"""Search endpoint route handlers.

Defines three POST endpoints under ``/exq/search/``:

- ``/clip`` -- CLIP text-to-image similarity search.
- ``/rf`` -- relevance-feedback search (SVM-based).
- ``/faceted`` -- filter-only search with no vector similarity.

Each endpoint validates the incoming Pydantic request, delegates to
`SearchService`, schedules a background audit-log task, and returns a
``{"suggestions": [...]}`` response containing the matching media IDs.
"""

from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ...schemas import FacetedSearchRequest, TextSearchRequest, RFSearchRequest
from ...services.search_service import SearchService
from ...services.logging_service import LoggingService
from ...core.exceptions import SearchError
from ..dependencies import get_search_service, get_logging_service

router = APIRouter(prefix="/exq/search", tags=["search"])


@router.post("/clip")
async def clip_search(
    request: TextSearchRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service),
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, Any]:
    """Search using CLIP text embeddings."""
    try:
        result = await search_service.search_text("clip", request)

        background_tasks.add_task(
            logging_service.log_search_request,
            "CLIP Search",
            request.session_info.session,
            request.session_info.modelId,
            request.session_info.collection,
            {
                "query": request.text,
                "n_seen": len(request.seen or []),
                "filters": request.filters.model_dump_json() if request.filters else None,
                "excluded": request.excluded or [],
                "n_requested": request.n,
            },
            result["suggestions"],
            result["request_timestamp"],
            result["completion_time"],
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
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, Any]:
    """Search using relevance feedback (SVM)."""
    try:
        result = await search_service.search_rf(request)

        background_tasks.add_task(
            logging_service.log_search_request,
            "RF Search",
            request.session_info.session,
            request.session_info.modelId,
            request.session_info.collection,
            {
                "pos": request.pos,
                "neg": request.neg,
                "n_seen": len(request.seen),
                "filters": request.filters.model_dump_json() if request.filters else None,
                "excluded": request.excluded,
                "query": request.query,
                "n_requested": request.n,
            },
            result["suggestions"],
            result["request_timestamp"],
            result["completion_time"],
        )

        return {"suggestions": result["suggestions"]}

    except SearchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/faceted")
async def faceted_search(
    request: FacetedSearchRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service),
    logging_service: LoggingService = Depends(get_logging_service),
) -> Dict[str, Any]:
    """Search using filters only"""
    try:
        result = await search_service.search_faceted(request)

        background_tasks.add_task(
            logging_service.log_search_request,
            "Faceted Search",
            request.session_info.session,
            request.session_info.modelId,
            request.session_info.collection,
            {
                "filters": request.filters.model_dump_json(),
                "n_requested": request.n,
            },
            result["suggestions"],
            result["request_timestamp"],
            result["completion_time"],
        )

        return {"suggestions": result["suggestions"]}

    except SearchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
