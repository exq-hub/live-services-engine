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
from typing import Annotated, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator

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
        index_name: Optional index to search against
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

    index_name: Optional[str] = None
    """Which of the collection's indexes to search; must be of this
    endpoint's embedding_type (CLIP for /clip, Text for /text). Omitted,
    it falls back to the family's resolved default index."""


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
        index_name: Optional index to run relevance feedback against
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

    index_name: Optional[str] = None
    """Which of the collection's indexes to run relevance feedback against.
    Unlike /clip and /text (implicitly scoped to their own model manager),
    RF isn't tied to one family, so it's named explicitly here; omitted,
    it falls back to the collection's overall default index."""


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


class _TemporalHopBase(BaseModel):
    """Fields shared by every hop of a temporal search."""

    index_name: Optional[str] = None
    """Which of the collection's indexes this hop searches, defaulting to
    the collection's default index."""

    weight: float = Field(1.0, ge=0.0)
    """Weight w_i of this hop's term in the chain's RRF score."""

    filters: Optional[ActiveFilters] = None
    """Optional filters for this hop only, combined with the request's
    top-level filters (both must pass)."""


class QueryHop(_TemporalHopBase):
    """A hop matched by text query similarity.

    The query is encoded by the model family of the hop's index, so the
    same hop searches keyframes through a CLIP index or transcripts
    through a Text index.
    """

    kind: Literal["query"]
    """Discriminator literal identifying a text query hop."""

    text: str
    """Natural-language text query for this hop."""


class RFHop(_TemporalHopBase):
    """A hop matched by a relevance-feedback hyperplane."""

    kind: Literal["rf"]
    """Discriminator literal identifying an RF hop."""

    pos: list[int] = []
    """Positive example media IDs for this hop."""

    neg: list[int] = []
    """Negative example media IDs for this hop."""

    query: Optional[str] = None
    """Optional text query for pseudo relevance-feedback."""


TemporalHop = Annotated[Union[QueryHop, RFHop], Field(discriminator="kind")]
"""One query of a temporal sequence, discriminated by ``kind``."""


class TemporalSearchRequest(BaseModel):
    """Request model for temporal search over an ordered sequence of hops.

    The first hop retrieves ``top_r`` anchor segments. Later hops are
    matched against segments that occur strictly after those anchors in
    the same video, and the resulting chains are ranked by length and
    then by RRF over the hops' ranks.
    """

    session_info: SessionInfo
    """Session context (session ID, collection, model ID)."""

    hops: list[TemporalHop] = Field(..., min_length=1)
    """The ordered sequence of queries, first hop first."""

    n: int = Field(50, ge=1)
    """Number of distinct videos to return."""

    seen: list[int] = []
    """Media IDs the user has already seen, skipped as first-hop anchors."""

    excluded: list[int] = []
    """Media IDs to exclude, along with every segment of their videos."""

    filters: Optional[ActiveFilters] = None
    """Optional filters applied to every hop."""

    top_r: int = Field(1000, ge=1)
    """Number of first-hop anchor segments to retrieve."""

    rrf_k: float = Field(60.0, gt=0.0)
    """Smoothing constant k of the RRF score."""

    chains_per_video: int = 1
    """Maximum chains returned per video, or -1 for every chain that
    survives overlap suppression."""

    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    """Two chains of the same video whose time spans have an IoU above
    this are redundant, and only the better ranked one is kept."""

    @field_validator("chains_per_video")
    @classmethod
    def validate_chains_per_video(cls, v: int) -> int:
        if v != -1 and v < 1:
            raise ValueError("chains_per_video must be -1 (all) or at least 1")
        return v


class TemporalChain(BaseModel):
    """One ranked chain of segments within a video."""

    group: int
    """Media ID of the video (group) the chain belongs to."""

    items: list[int]
    """Media IDs of the chain's segments, one per matched hop, in order."""

    length: int
    """Number of hops matched, counted from the first hop."""

    score: float
    """RRF score over the matched hops' ranks."""

    start_ms: float
    """Start time of the chain's first segment."""

    end_ms: float
    """End time of the chain's last segment."""
