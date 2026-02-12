from __future__ import annotations
from typing import Literal, Union
from pydantic import BaseModel, Field

from .session import SessionInfo


class DBValueConstraint(BaseModel):
    value_ids: list[int]
    operator: Literal["AND", "OR"] = "OR"

class DBRangeConstraint(BaseModel):
    upper_bound: int | float | str
    lower_bound: int | float | str

class DBFilter(BaseModel):
    id: int
    tagtype_id: int
    constraint: DBValueConstraint | DBRangeConstraint

class FilterGroup(BaseModel):
    """A group of filters combined with a logical operator."""
    kind: Literal["group"] = "group"
    operator: Literal["AND", "OR"]
    children: list[FilterExpr] = Field(default_factory=list)
    not_: bool = False

class FilterLeaf(BaseModel):
    """A single filter condition."""
    kind: Literal["leaf"] = "leaf"
    filter: DBFilter
    not_: bool = False

FilterExpr = Union[FilterLeaf, FilterGroup]

class ActiveFilters(BaseModel):
    """Model for managing multiple active filters in search operations."""
    root: FilterExpr


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
