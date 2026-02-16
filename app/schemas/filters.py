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


"""Filter expression schemas for composable search constraints.

These models form a recursive expression tree that represents the user's
active filter selection. The tree is compiled into SQL by
`app.repositories.db_helper.compile_active_filters`.

Structure::

    ActiveFilters
    └── root: FilterExpr
        ├── FilterLeaf  (wraps a single DBFilter + optional NOT)
        └── FilterGroup (AND/OR over child FilterExprs + optional NOT)

    DBFilter
    ├── DBValueConstraint  (tag ID list + AND/OR operator)
    └── DBRangeConstraint  (numeric lower/upper bound)
"""

from __future__ import annotations
from typing import Literal, Union
from pydantic import BaseModel, Field

from .session import SessionInfo


class DBValueConstraint(BaseModel):
    """Constraint matching media items by a set of tag value IDs.

    When ``operator`` is ``"OR"`` (default), a media item matches if it has
    *any* of the listed tags.  When ``"AND"``, it must have *all* of them.
    """

    value_ids: list[int]
    """Tag value IDs to match against."""

    operator: Literal["AND", "OR"] = "OR"
    """Logical operator: ``"OR"`` matches any, ``"AND"`` requires all."""

class DBRangeConstraint(BaseModel):
    """Constraint matching media items whose tag value falls within a numeric range."""

    upper_bound: int | float | str
    """Inclusive upper bound of the range."""

    lower_bound: int | float | str
    """Inclusive lower bound of the range."""

class DBFilter(BaseModel):
    """A single database-level filter targeting one tagset.

    Attributes:
        id: The tagset ID this filter applies to.
        tagtype_id: The tag-type ID (determines the underlying value table).
        constraint: Either a value-set or numeric-range constraint.
    """

    id: int
    """Tagset ID this filter applies to."""

    tagtype_id: int
    """Tag-type ID determining the underlying value table."""

    constraint: DBValueConstraint | DBRangeConstraint
    """Either a value-set or numeric-range constraint."""

class FilterGroup(BaseModel):
    """A group of filters combined with a logical operator."""

    kind: Literal["group"] = "group"
    """Discriminator literal identifying this node as a group."""

    operator: Literal["AND", "OR"]
    """Logical combinator applied to all children."""

    children: list[FilterExpr] = Field(default_factory=list)
    """Child filter expressions (leaves or nested groups)."""

    not_: bool = False
    """If ``True``, the entire group result is logically negated."""

class FilterLeaf(BaseModel):
    """A single filter condition."""

    kind: Literal["leaf"] = "leaf"
    """Discriminator literal identifying this node as a leaf."""

    filter: DBFilter
    """The database-level filter to evaluate."""

    not_: bool = False
    """If ``True``, the filter result is logically negated."""

FilterExpr = Union[FilterLeaf, FilterGroup]
"""Recursive type alias: a filter expression is either a `FilterLeaf` or a `FilterGroup`."""

class ActiveFilters(BaseModel):
    """Model for managing multiple active filters in search operations."""

    root: FilterExpr
    """Top-level filter expression tree root."""


class AppliedFilter(BaseModel):
    """Model for a single applied filter in a search session.

    Attributes:
        session_info: Session context information
        name: Name of the filter being applied
        values: List of integer values for the filter criteria
    """

    session_info: SessionInfo
    """Session context information."""

    name: str
    """Name of the filter being applied."""

    values: list[int]
    """List of tag value IDs for the filter criteria."""
