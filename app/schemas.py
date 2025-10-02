"""Pydantic schemas for request/response data models in the LSE API.

This module defines all the data models used for API requests and responses
in the Live Services Engine, including search requests, session management,
filtering, and item information structures.
"""

from typing import List, Optional, Union, Tuple
from pydantic import BaseModel


class AddOrRemoveModelRequest(BaseModel):
    """Request model for adding or removing a model from a session.

    Attributes:
        session: Unique identifier for the search session
        collection: Name of the data collection to operate on
        modelId: Unique identifier for the model being managed
    """

    session: str
    collection: str
    modelId: int


class SessionInfo(BaseModel):
    """Session information containing context for search operations.

    Attributes:
        session: Unique identifier for the search session
        collection: Name of the data collection being searched
        modelId: Identifier for the model associated with this session
    """

    session: str
    collection: str
    modelId: int


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


class AppliedFilter(BaseModel):
    """Model for a single applied filter in a search session.

    Attributes:
        session_info: Session context information
        name: Name of the filter being applied
        values: List of integer values for the filter criteria
    """

    session_info: SessionInfo
    name: str
    values: list[int]


class ActiveFilters(BaseModel):
    """Model for managing multiple active filters in search operations.

    Attributes:
        names: List of filter names that are currently active
        values: List of lists containing integer values for each filter
        treat_values_as_and: Optional list of filter names to treat with AND logic
    """

    names: list[str]
    values: list[list[int]]
    treat_values_as_and: Optional[list[str]] = []


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


class ItemRequest(BaseModel):
    """Request model for retrieving information about a specific item.

    Attributes:
        itemId: Unique identifier for the item to retrieve
        session_info: Session context information
    """

    itemId: int
    session_info: SessionInfo


class IsExcludedRequest(BaseModel):
    """Request model for checking if an item is in the excluded list.

    Attributes:
        itemId: Unique identifier for the item to check
        excluded_ids: List of excluded item IDs to check against
        session_info: Session context information
    """

    itemId: int
    excluded_ids: list[int]
    session_info: SessionInfo


class ItemInfo(BaseModel):
    """Response model containing detailed information about a media item.

    Attributes:
        item_id: Unique identifier for the item
        thumbnail_uri: URI for the item's thumbnail image
        media_uri: URI for the full media content
        group: Group or collection identifier the item belongs to
        metadata: Dictionary containing additional item metadata
    """

    item_id: int
    thumbnail_uri: str
    media_uri: str
    group: str
    metadata: dict


class ClearItemSetRequest(BaseModel):
    """Request model for clearing a set of items from a session.

    Attributes:
        session: Unique identifier for the search session
        modelId: Identifier for the model associated with the item set
        name: Name of the item set to clear
    """

    session: str
    modelId: int
    name: str


class Filter(BaseModel):
    """Model representing a filter that can be applied to search results.

    Filter types are enumerated as:
    - Single (0): Single value selection
    - Multi: Multiple value selection
    - NumberRange: Numeric range filtering
    - NumberMultiRange: Multiple numeric ranges
    - LabelRange: Label-based range filtering
    - LabelMultiRange: Multiple label ranges
    - Count: Count-based filtering
    - MultiCount: Multiple count criteria

    Attributes:
        id: Unique identifier for the filter
        collectionId: ID of the collection this filter applies to
        name: Human-readable name of the filter
        values: List of possible filter values (strings or integers)
        filterType: Integer representing the filter type (see enum above)
        range: Optional range specification as [start, end] or [valueId, label]
        count: Optional list of count constraints as [min, max] pairs
    """

    id: int
    collectionId: int
    name: str
    values: list[str] | list[int]
    filterType: int
    range: Optional[Union[List[int], List[Union[str, int]]]] = []
    count: Optional[list[int]] = []
