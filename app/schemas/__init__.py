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
