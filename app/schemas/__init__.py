# SPDX-FileCopyrightText: 2026 Ujjwal Sharma and Omar Shahbaz Khan
# SPDX-License-Identifier: AGPL-3.0-or-later


"""Pydantic schemas for request/response data models in the LSE API.

This package centralises every Pydantic model used across the API and
service layers, re-exported here for convenient single-import access
(e.g. ``from app.schemas import TextSearchRequest, ActiveFilters``).

Submodules
----------
session
    `SessionInfo`, `AddOrRemoveModelRequest` -- session context carried by
    every request.
filters
    `ActiveFilters`, `FilterExpr`, `FilterGroup`, `FilterLeaf`, `DBFilter`,
    `DBValueConstraint`, `DBRangeConstraint`, `AppliedFilter` -- the
    recursive filter expression tree compiled to SQL by `db_helper`.
search
    `TextSearchRequest`, `RFSearchRequest`, `FacetedSearchRequest`,
    `AggregationSearchRequest` -- search endpoint payloads.
    `TemporalSearchRequest` with its `TemporalHop` variants and its
    `TemporalChain` result are the temporal search payloads.
items
    `ItemRequest`, `ItemBatchRequest`, `ItemDetailRequest`, `RelatedItemsRequest`,
    `IsExcludedRequest`, `ItemInfo`, `ClearItemSetRequest` -- item endpoint
    payloads.
logging
    `ClientEventRequest` -- client-side event logging payload.
"""

from .session import SessionInfo, AddOrRemoveModelRequest
from .filters import (
    DBValueConstraint,
    DBRangeConstraint,
    DBFilter,
    FilterGroup,
    FilterLeaf,
    FilterExpr,
    ActiveFilters,
    AppliedFilter,
)
from .search import (
    TextSearchRequest,
    RFSearchRequest,
    FacetedSearchRequest,
    AggregationSearchRequest,
    QueryHop,
    RFHop,
    TemporalHop,
    TemporalSearchRequest,
    TemporalChain,
)
from .items import (
    ItemRequest,
    ItemBatchRequest,
    ItemDetailRequest,
    RelatedItemsRequest,
    IsExcludedRequest,
    ItemInfo,
    ClearItemSetRequest,
)
from .logging import ClientEventRequest

__all__ = [
    "SessionInfo",
    "AddOrRemoveModelRequest",
    "DBValueConstraint",
    "DBRangeConstraint",
    "DBFilter",
    "FilterGroup",
    "FilterLeaf",
    "FilterExpr",
    "ActiveFilters",
    "AppliedFilter",
    "TextSearchRequest",
    "RFSearchRequest",
    "FacetedSearchRequest",
    "AggregationSearchRequest",
    "QueryHop",
    "RFHop",
    "TemporalHop",
    "TemporalSearchRequest",
    "TemporalChain",
    "ItemRequest",
    "ItemBatchRequest",
    "ItemDetailRequest",
    "RelatedItemsRequest",
    "IsExcludedRequest",
    "ItemInfo",
    "ClearItemSetRequest",
    "ClientEventRequest",
]
