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
