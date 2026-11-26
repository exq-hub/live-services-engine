# SPDX-FileCopyrightText: 2026 Ujjwal Sharma and Omar Shahbaz Khan
# SPDX-License-Identifier: AGPL-3.0-or-later


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
    `CLIPModelManager`/`TextModelManager` for PyTorch model lifecycle and
    device management, and `ApplicationContainer` -- the central
    dependency-injection container that wires together configuration,
    models, repositories, and services.
indexes
    Abstract `BaseIndex` interface with concrete `FaissIndex` and `ZarrIndex`
    implementations for nearest-neighbour vector search.
"""
