"""Web API layer for the Live Services Engine.

This package contains all FastAPI-specific code: route handlers,
dependency-injection functions, and request/response processing.

Modules
-------
dependencies
    FastAPI ``Depends`` callables that resolve the `ApplicationContainer`
    singletons into per-request service instances (`SearchService`,
    `ItemService`, `LoggingService`).

Subpackages
-----------
routes
    Endpoint modules grouped by domain: ``search``, ``items``, ``admin``.
"""
