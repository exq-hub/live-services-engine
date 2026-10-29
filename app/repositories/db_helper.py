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


"""SQL compilation helpers for the filter expression tree.

This module translates the Pydantic filter model (`ActiveFilters`) into a
parameterised SQL query that can be executed against the collection's SQLite
database to retrieve matching media IDs.

The filter model is a recursive tree of `FilterGroup` (AND/OR with optional
NOT) and `FilterLeaf` nodes.  Each leaf carries a `DBFilter` with either a
`DBValueConstraint` (match tag IDs with ANY/ALL semantics) or a
`DBRangeConstraint` (numeric BETWEEN).

The compiler walks the tree recursively and builds one query, here for
"Category is documentary or travel AND Start (ms) is from 5000 to 60000"::

    WITH RECURSIVE
    -- One row per media that itself has a tag of some leaf. lN is 1 when it
    -- has a tag of leaf N. A leaf with operator AND gets one lN per tag.
    f AS (
        SELECT media_id,
               MAX(CASE WHEN flag = 0 THEN 1 ELSE 0 END) AS l0,
               MAX(CASE WHEN flag = 1 THEN 1 ELSE 0 END) AS l1
        FROM (
            -- Leaf 0, Category is documentary or travel
            SELECT x.media_id, 0 AS flag FROM taggings x WHERE x.tag_id IN (?,?)
            UNION ALL
            -- Leaf 1, Start (ms) from 5000 to 60000, through its tag type's table
            SELECT x.media_id, 1 AS flag FROM taggings x
            JOIN numerical_int_tags r ON r.id = x.tag_id
            WHERE r.tagset_id = ? AND r.value BETWEEN ? AND ?
        )
        GROUP BY media_id
    ),
    -- Hands the flags down through group_id. A row means media_id gets the
    -- flags of source_id, which is depth groups above it.
    down (media_id, source_id, depth) AS (
        SELECT media_id, media_id, 0 FROM f
        UNION ALL
        SELECT ch.id, d.source_id, d.depth + 1
        FROM down d
        JOIN medias ch ON ch.group_id = d.media_id
        -- A generous ceiling, which also stops group_id cycles
        WHERE d.depth < 4
    ),
    -- Per media, its own flags and its groups' flags combined
    flags AS (
        SELECT d.media_id, MAX(f.l0) AS l0, MAX(f.l1) AS l1
        FROM down d
        JOIN f ON f.media_id = d.source_id
        GROUP BY d.media_id
    )
    SELECT m.id
    FROM medias m
    -- A media without any flag still counts, with every flag 0, so NOT works
    LEFT JOIN flags fl ON fl.media_id = m.id
    WHERE (COALESCE(fl.l0, 0) = 1) AND (COALESCE(fl.l1, 0) = 1)

Groups that pass are returned too. Callers keep the media of their index
with ``get_index_ids``.
"""

from typing import Any, Dict, List, Tuple, Union, Optional

from app.schemas import (
    ActiveFilters,
    DBRangeConstraint,
    DBValueConstraint,
    FilterExpr,
    FilterGroup,
    FilterLeaf,
)


def placeholders(n: int) -> str:
    """Return a comma-separated string of ``n`` SQL ``?`` placeholders."""
    return ",".join("?" for _ in range(n))


def compile_active_filters(
    active: ActiveFilters, tagtype_map: Dict[int, str]
) -> Tuple[str, List[Any]]:
    """
    Args:
        active: selected filters represented as an ActiveFiltersDB instance (dict) with shape {"root": FilterExpr}
    Returns (sql, params) for querying media_ids matching the filter expression
    """

    # Flags, one per leaf or one per tag of a leaf with operator AND
    flag_selects: List[str] = []
    flag_params: List[Any] = []

    def add_flag(join: str, where: str, p: List[Any]) -> str:
        """Add one UNION ALL branch to `f` and return the SQL of its flag.

        Args:
            join: Join to the tag table of a range leaf, or "" for a list of tag IDs
            where: Condition on the taggings `x`, e.g. "x.tag_id IN (?,?)"
            p: Bind parameters of `where`

        Returns:
            `COALESCE(fl.lN, 0)`, where N is this flag's number. It is 1 when the
            media or one of its groups has a matching tag, else 0.
        """
        k = len(flag_selects)
        flag_selects.append(
            f"SELECT x.media_id, {k} AS flag FROM taggings x {join} WHERE {where}"
        )
        flag_params.extend(p)
        return f"COALESCE(fl.l{k}, 0)"

    # --- helper functions for compiling constraints ---
    def emit_value_any(tag_ids: List[int], negated: bool) -> str:
        if not tag_ids:
            # An empty tag list matches nothing, so NOT of it matches everything
            return "1=1" if negated else "0=1"
        ph = placeholders(len(tag_ids))
        flag = add_flag("", f"x.tag_id IN ({ph})", tag_ids)
        return f"{flag} = {0 if negated else 1}"

    def emit_value_all(tag_ids: List[int], negated: bool) -> str:
        if not tag_ids:
            # Requiring every tag of an empty list matches everything, so NOT of
            # it matches nothing
            return "0=1" if negated else "1=1"
        # One flag per tag, so a media can have some of the tags itself and
        # the others through one of its groups
        flags = [add_flag("", "x.tag_id = ?", [tag_id]) for tag_id in tag_ids]
        expr = " AND ".join(f"{flag} = 1" for flag in flags)
        return f"NOT ({expr})" if negated else expr

    def emit_range_exists(
        filter_id: int,  # tagset_id
        tagtype_id: int,
        lower: Optional[Union[int, float, str]],
        upper: Optional[Union[int, float, str]],
        negated: bool,
    ) -> str:
        where_parts = ["r.tagset_id = ?"]
        p: List[Any] = [filter_id]

        # Build the bound predicates. BETWEEN if both present, otherwise one-sided.
        if lower is not None and upper is not None:
            where_parts.append("r.value BETWEEN ? AND ?")
            p.extend([lower, upper])
        elif lower is not None:
            where_parts.append("r.value >= ?")
            p.append(lower)
        elif upper is not None:
            where_parts.append("r.value <= ?")
            p.append(upper)
        else:
            # No bounds means "any value in this set"; keep just the set_id check.
            pass

        flag = add_flag(
            f"JOIN {tagtype_map[tagtype_id]}_tags r ON r.id = x.tag_id",
            " AND ".join(where_parts),
            p,
        )
        return f"{flag} = {0 if negated else 1}"

    def compile_leaf(leaf: FilterLeaf) -> str:
        f = leaf.filter
        fid = f.id
        tagtype = f.tagtype_id
        constraint = f.constraint
        neg = leaf.not_

        # Value constraint
        if isinstance(constraint, DBValueConstraint):
            ids = constraint.value_ids
            op = constraint.operator
            if op == "OR":
                return emit_value_any(ids, neg)
            elif op == "AND":
                return emit_value_all(ids, neg)
            else:
                raise ValueError(f"Unknown value operator: {op}")

        # Range constraint
        if isinstance(constraint, DBRangeConstraint):
            return emit_range_exists(
                fid, tagtype, constraint.lower_bound, constraint.upper_bound, neg
            )
        raise ValueError("Unrecognized constraint shape")

    def compile_group(group: FilterGroup) -> str:
        op = group.operator  # "AND" | "OR"
        neg = group.not_
        child_sqls: List[str] = []

        for child in group.children:
            child_sqls.append(f"({compile_expr(child)})")

        if not child_sqls:
            # An empty AND matches everything, and an empty OR matches nothing
            base = "1=1" if op == "AND" else "0=1"
        else:
            joiner = f" {op} "
            base = joiner.join(child_sqls)

        if neg:
            base = f"NOT ({base})"
        return base

    def compile_expr(node: FilterExpr) -> str:
        if node.kind == "leaf":
            return compile_leaf(node)
        elif node.kind == "group":
            return compile_group(node)
        else:
            raise ValueError(f"Unknown node kind: {node}")

    # --- build final SQL query text ---

    expr_sql = compile_expr(active.root)

    if not flag_selects:
        # No flags, e.g. every leaf has an empty tag list, so the tree is a constant
        return f"SELECT m.id FROM medias m WHERE {expr_sql}", []

    flag_numbers = range(len(flag_selects))
    own = ", ".join(
        f"MAX(CASE WHEN flag = {k} THEN 1 ELSE 0 END) AS l{k}" for k in flag_numbers
    )
    combined = ", ".join(f"MAX(f.l{k}) AS l{k}" for k in flag_numbers)
    sql = f"""
        WITH RECURSIVE
        f AS (
            SELECT media_id, {own}
            FROM ({" UNION ALL ".join(flag_selects)})
            GROUP BY media_id
        ),
        down (media_id, source_id, depth) AS (
            SELECT media_id, media_id, 0 FROM f
            UNION ALL
            SELECT ch.id, d.source_id, d.depth + 1
            FROM down d
            JOIN medias ch ON ch.group_id = d.media_id
            WHERE d.depth < 4
        ),
        flags AS (
            SELECT d.media_id, {combined}
            FROM down d
            JOIN f ON f.media_id = d.source_id
            GROUP BY d.media_id
        )
        SELECT m.id
        FROM medias m
        LEFT JOIN flags fl ON fl.media_id = m.id
        WHERE {expr_sql}
        """
    return sql, flag_params
