"""Business-logic service layer for the Live Services Engine.

Services sit between the API routes and the repositories / strategies,
orchestrating complex operations and enforcing business rules.

Modules
-------
search_service
    `SearchService` -- selects and invokes the appropriate search strategy
    (CLIP, relevance feedback, or faceted) based on the incoming request,
    tracks execution time, and returns structured results.
item_service
    `ItemService` -- handles item metadata retrieval, related-item lookups,
    and exclusion checks by delegating to `DatabaseRepository`.
logging_service
    `AuditLogger` and `LoggingService` -- async, queue-based audit logging
    with MessagePack serialization and file-lock concurrency control.
"""
