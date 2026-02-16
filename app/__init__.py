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


"""Live Services Engine (LSE) -- main application package.

The Live Services Engine is a FastAPI-based backend that powers the
Exquisitor multimedia search system. It provides multiple search strategies
over media collections with comprehensive audit logging.

Subpackages
-----------
api
    FastAPI route definitions, dependency injection, and request handling.
core
    Configuration management, ML model lifecycle, custom exceptions, and
    vector index implementations (FAISS, Zarr).
repositories
    Data-access layer for SQLite metadata databases and vector indices.
schemas
    Pydantic request/response models shared across the API and service layers.
services
    Business-logic layer coordinating search execution, item retrieval, and
    audit logging.
strategies
    Pluggable search algorithm implementations (CLIP text search, relevance
    feedback via SVM, faceted/filter-based search).

Modules
-------
utils
    Shared utility functions for timestamping and MessagePack log serialization.
search_utils
    Legacy utilities for IVF index search, active-filter validation, and
    temporal shot-overlap mapping for video content.
"""
