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


"""Data-access layer (repository pattern) for the Live Services Engine.

This package isolates all direct I/O with persistent stores so that the
service and strategy layers remain storage-agnostic.

Modules
-------
database_repository
    `DatabaseRepository` -- manages per-collection SQLite connections,
    item metadata queries, filter evaluation, and the bidirectional
    mapping between media IDs and vector-index positions.
index_repository
    `IndexRepository` -- manages FAISS / Zarr index loading, CLIP vector
    search, and access to Zarr embedding arrays for relevance feedback.
db_helper
    Pure functions that compile a tree of `ActiveFilters` (AND/OR/NOT
    groups with value or range constraints) into a parameterised SQL
    query executed by `DatabaseRepository.get_filtered_media_ids`.
"""
