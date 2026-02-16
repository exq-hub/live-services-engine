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


"""Core package -- foundational infrastructure for the Live Services Engine.

This package provides the low-level building blocks that every other layer
of the application depends on:

Modules
-------
config
    INI-based configuration loading and Pydantic validation for server
    settings, per-collection paths, and device selection.
exceptions
    Hierarchical exception classes (`LSEException` and its subclasses)
    used throughout the application for structured error propagation.
models
    `ModelManager` for PyTorch/CLIP model lifecycle and device management,
    and `ApplicationContainer` -- the central dependency-injection container
    that wires together configuration, models, repositories, and services.
indexes
    Abstract `BaseIndex` interface with concrete `FaissIndex` and `ZarrIndex`
    implementations for nearest-neighbour vector search.
"""
