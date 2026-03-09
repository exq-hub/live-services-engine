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


"""Search request schemas for all search endpoints.

Each schema carries the parameters needed by one search strategy plus the
common `SessionInfo` context and optional `ActiveFilters` tree.
"""

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
    """Natural-language text query to search for."""

    n: int
    """Maximum number of results to return."""

    seen: Optional[list[int]] = []
    """Media IDs the user has already seen (skipped during search)."""

    filters: Optional[ActiveFilters] = None
    """Optional filter expression tree to constrain results."""

    excluded: Optional[list[int]] = []
    """Media IDs to exclude from results entirely."""

    session_info: SessionInfo
    """Session context (session ID, collection, model ID)."""


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
    """Positive example media IDs provided as relevance feedback."""

    neg: list[int]
    """Negative example media IDs provided as relevance feedback."""

    n: int
    """Maximum number of results to return."""

    seen: list[int]
    """Media IDs the user has already seen."""

    query: Optional[str] = None
    """Optional text query for pseudo relevance-feedback."""

    filters: Optional[ActiveFilters] = None
    """Optional filter expression tree to constrain results."""

    excluded: list[int]
    """Media IDs to exclude from results entirely."""

    session_info: SessionInfo
    """Session context (session ID, collection, model ID)."""


class FacetedSearchRequest(BaseModel):
    """Request model for faceted search operations.

    Attributes:
        n: Number of results to return
        filters: Active filters to apply
        session_info: Session context information
    """

    n: int
    """Maximum number of results to return."""

    filters: ActiveFilters
    """Filter expression tree (required for faceted search)."""

    session_info: SessionInfo
    """Session context (session ID, collection, model ID)."""


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
    """List of text queries to search and aggregate."""

    n: int
    """Maximum number of results to return."""

    seen: Optional[list[int]] = []
    """Media IDs the user has already seen."""

    filters: Optional[ActiveFilters] = None
    """Optional filter expression tree to constrain results."""

    excluded: Optional[list[int]] = []
    """Media IDs to exclude from results entirely."""

    session_info: SessionInfo
    """Session context (session ID, collection, model ID)."""

    RF: Optional[Tuple[list[int], list[int]]] = None
    """Optional ``(positive_ids, negative_ids)`` tuple for relevance feedback."""


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
    """Session context (session ID, collection, model ID)."""

    itemId: Optional[int] = None
    """Optional media ID associated with this submission."""

    name: Optional[str] = ""
    """Optional name or label for the submission."""

    text: str
    """Text content of the submission."""

    qa: bool
    """Whether this is a Q&A-style submission."""

    evalId: str
    """Evaluation identifier for tracking and scoring purposes."""

    start: Optional[float] = None
    """Optional start timestamp for temporal submissions."""

    end: Optional[float] = None
    """Optional end timestamp for temporal submissions."""
