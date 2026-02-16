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

The compiler walks the tree recursively, emitting SQL fragments and
collecting bind parameters, then wraps the result in a ``SELECT ... GROUP BY
... HAVING`` query over the ``taggings`` table.

Example generated SQL::

    SELECT tgs.media_id
    FROM taggings AS tgs
    GROUP BY tgs.media_id
    HAVING (SUM(CASE WHEN tgs.tag_id IN (?,?) THEN 1 ELSE 0 END) > 0)
"""

from typing import Any, Dict, List, Tuple, Union, Optional

from app.schemas import ActiveFilters, DBRangeConstraint, DBValueConstraint, FilterExpr, FilterGroup, FilterLeaf

def placeholders(n: int) -> str:
    """Return a comma-separated string of ``n`` SQL ``?`` placeholders."""
    return ",".join("?" for _ in range(n))

def compile_active_filters(active: ActiveFilters, tagtype_map: Dict[int, str]) -> Tuple[str, List[Any]]:
    """
    Args:
        active: selected filters rrepresented as an ActiveFiltersDB instance (dict) with shape {"root": FilterExpr}
    Returns (sql, params) for querying media_ids matching the filter expression
    """

    # --- helper functions for compiling constraints ---
    def emit_value_any(tag_ids: List[int], negated: bool) -> Tuple[str, List[Any]]:
        if not tag_ids:
            # ANY with empty list: normally false; negated => true
            return ("1=1" if negated else "0=1", [])
        ph = placeholders(len(tag_ids))
        expr = f"SUM(CASE WHEN tgs.tag_id IN ({ph}) THEN 1 ELSE 0 END) {'=' if negated else '>'} 0"
        return expr, tag_ids

    def emit_value_all(tag_ids: List[int], negated: bool) -> Tuple[str, List[Any]]:
        if not tag_ids:
            # ALL with empty list: vacuously true; negated => false
            return ("0=1" if negated else "1=1", [])
        ph = placeholders(len(tag_ids))
        needed = len(tag_ids)
        op = "<" if negated else "="
        expr = (
            f"COUNT(DISTINCT CASE WHEN tgs.tag_id IN ({ph}) THEN tgs.tag_id END) {op} {needed}"
        )
        return expr, tag_ids

    def emit_range_exists(
        filter_id: int, # tagset_id
        tagtype_id: int,
        lower: Optional[Union[int, float, str]],
        upper: Optional[Union[int, float, str]],
        negated: bool,
    ) -> Tuple[str, List[Any]]:
        where_parts = [f"r.tagset_id = ?"]
        p: List[Any] = [filter_id]

        # Build the bound predicates. BETWEEN if both present, otherwise one-sided.
        if lower is not None and upper is not None:
            where_parts.append(f"r.value BETWEEN ? AND ?")
            p.extend([lower, upper])
        elif lower is not None:
            where_parts.append(f"r.value >= ?")
            p.append(lower)
        elif upper is not None:
            where_parts.append(f"r.value <= ?")
            p.append(upper)
        else:
            # No bounds means "any value in this set"; keep just the set_id check.
            pass

        sub = (
            f"""
            EXISTS (
                SELECT 1
                FROM taggings x
                JOIN {tagtype_map[tagtype_id]}_tags r ON r.id = x.tag_id
                WHERE x.media_id = tgs.media_id AND {" AND ".join(where_parts)}
            )
            """
        )
        if negated:
            sub = f"NOT ({sub})"
        return sub, p

    def compile_leaf(leaf: FilterLeaf) -> Tuple[str, List[Any]]:
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
            return emit_range_exists(fid, tagtype, constraint.lower_bound, 
                                     constraint.upper_bound, neg)
        raise ValueError("Unrecognized constraint shape")

    def compile_group(group: FilterGroup) -> Tuple[str, List[Any]]:
        op = group.operator  # "AND" | "OR"
        neg = group.not_
        child_sqls: List[str] = []
        child_params: List[Any] = []

        for child in group.children:
            sql_i, params_i = compile_expr(child)
            child_sqls.append(f"({sql_i})")
            child_params.extend(params_i)

        if not child_sqls:
            # No children: identity element (AND -> TRUE, OR -> FALSE)
            base = "1=1" if op == "AND" else "0=1"
        else:
            joiner = f" {op} "
            base = joiner.join(child_sqls)

        if neg:
            base = f"NOT ({base})"
        return base, child_params

    def compile_expr(node: FilterExpr) -> Tuple[str, List[Any]]:
        if node.kind == "leaf":
            return compile_leaf(node)
        elif node.kind == "group":
            return compile_group(node)
        else:
            raise ValueError(f"Unknown node kind: {node}")


    # --- build final SQL query text ---

    expr_sql, expr_params = compile_expr(active.root)
    params: List[Any] = []
    params.extend(expr_params)

    sql = (
        "SELECT tgs.media_id\n"
        "FROM taggings AS tgs\n"
        "GROUP BY tgs.media_id\n"
        f"HAVING {expr_sql}"
    )
    return sql, params
