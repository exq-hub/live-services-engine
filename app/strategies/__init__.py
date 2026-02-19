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
