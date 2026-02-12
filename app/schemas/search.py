from __future__ import annotations
from typing import Optional, Tuple
from pydantic import BaseModel

from .session import SessionInfo
from .filters import ActiveFilters


class TextSearchRequest(BaseModel):
    """Request model for text-based search operations.

    Attributes:
        text: Text query to search for
        n: Number of results to return
        seen: Optional list of item IDs that have already been seen
        filters: Optional active filters to apply
        excluded: Optional list of item IDs to exclude from results
        session_info: Session context information
    """

    text: str
    n: int
    seen: Optional[list[int]] = []
    filters: Optional[ActiveFilters] = None
    excluded: Optional[list[int]] = []
    session_info: SessionInfo


class RFSearchRequest(BaseModel):
    """Request model for Relevance Feedback (RF) search operations.

    Attributes:
        pos: List of positive example item IDs
        neg: List of negative example item IDs
        n: Number of results to return
        seen: List of item IDs that have already been seen
        query: Optional text query to combine with RF
        filters: Optional active filters to apply
        excluded: List of item IDs to exclude from results
        session_info: Session context information
    """

    pos: list[int]
    neg: list[int]
    n: int
    seen: list[int]
    query: Optional[str] = None
    filters: Optional[ActiveFilters] = None
    excluded: list[int]
    session_info: SessionInfo


class FacetedSearchRequest(BaseModel):
    """Request model for faceted search operations.

    Attributes:
        n: Number of results to return
        filters: Active filters to apply
        session_info: Session context information
    """

    n: int
    filters: ActiveFilters
    session_info: SessionInfo


class AggregationSearchRequest(BaseModel):
    """Request model for aggregated search operations across multiple text queries.

    Attributes:
        texts: List of text queries to search for
        n: Number of results to return
        seen: Optional list of item IDs that have already been seen
        filters: Optional active filters to apply
        excluded: Optional list of item IDs to exclude from results
        session_info: Session context information
        RF: Optional tuple containing positive and negative RF examples
    """

    texts: list[str]
    n: int
    seen: Optional[list[int]] = []
    filters: Optional[ActiveFilters] = None
    excluded: Optional[list[int]] = []
    session_info: SessionInfo
    RF: Optional[Tuple[list[int], list[int]]] = None


class SubmitRequest(BaseModel):
    """Request model for submitting search results or feedback.

    Attributes:
        session_info: Session context information
        itemId: Optional identifier for a specific item
        name: Optional name or label for the submission
        text: Text content associated with the submission
        qa: Boolean flag indicating if this is a Q&A submission
        evalId: Evaluation identifier for tracking purposes
        start: Optional start time for temporal data
        end: Optional end time for temporal data
    """

    session_info: SessionInfo
    itemId: Optional[int] = None
    name: Optional[str] = ""
    text: str
    qa: bool
    evalId: str
    start: Optional[float] = None
    end: Optional[float] = None
