"""Pluggable search strategy implementations (Strategy pattern).

Each strategy implements one of the abstract bases defined in `base` and is
registered with `SearchService` at startup.

Modules
-------
base
    Abstract base classes: `SearchStrategy`, `TextSearchStrategy`,
    `RFSearchStrategy`, `FacetedSearchStrategy`.
clip_search
    `CLIPSearchStrategy` -- encodes a text query with the CLIP text encoder
    and retrieves the most similar media items from the vector index.
rf_search
    `RFSearchStrategy` -- trains a linear SVM on positive/negative example
    embeddings (with optional pseudo-RF from a CLIP text query) and uses
    the learned hyperplane as the search vector.
faceted_search
    `FacetedSearchStrategy` -- pure filter-based retrieval with no vector
    similarity; compiles filter expressions to SQL and returns matching IDs.
"""
