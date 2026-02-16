"""API route handlers for the Live Services Engine.

Each module defines a FastAPI `APIRouter` that is mounted in `main.py`.
All routers share the ``/exq/`` path prefix and use background tasks for
non-blocking audit logging via MessagePack serialization.

Modules
-------
search
    ``POST /exq/search/clip``, ``POST /exq/search/rf``,
    ``POST /exq/search/faceted`` -- search endpoints dispatching to
    `SearchService`.
items
    ``POST /exq/item/base``, ``POST /exq/item/details``,
    ``POST /exq/item/related``, ``POST /exq/item/excluded`` -- item
    metadata and relationship endpoints dispatching to `ItemService`.
admin
    Session initialization, collection info (total items, filters,
    filter values), model lifecycle logging, and client-event logging.
"""
