# SPDX-FileCopyrightText: 2026 Ujjwal Sharma and Omar Shahbaz Khan
# SPDX-License-Identifier: AGPL-3.0-or-later


"""API route handlers for the Live Services Engine.

Each module defines a FastAPI `APIRouter` that is mounted in `main.py`.
All routers share the ``/exq/`` path prefix and use background tasks for
non-blocking audit logging via MessagePack serialization.

Modules
-------
search
    ``POST /exq/search/clip``, ``POST /exq/search/text``,
    ``POST /exq/search/rf``, ``POST /exq/search/temporal``,
    ``POST /exq/search/faceted`` -- search endpoints dispatching to
    `SearchService`.
items
    ``POST /exq/item/info``, ``POST /exq/item/info/batch``,
    ``POST /exq/item/base`` (deprecated alias),
    ``POST /exq/item/details`` (deprecated),
    ``POST /exq/item/related``, ``POST /exq/item/excluded`` -- item
    metadata and relationship endpoints dispatching to `ItemService`.
admin
    Session initialization, collection info (total items, filters,
    filter values, indexes), model lifecycle logging, and client-event
    logging.
"""
