"""Session context schemas shared by all API requests.

Every API request in the LSE carries a `SessionInfo` object that identifies
the user session, the target collection, and the active model. This context
is threaded through services, strategies, and logging.
"""

from __future__ import annotations
from pydantic import BaseModel


class SessionInfo(BaseModel):
    """Session information containing context for search operations.

    Attributes:
        session: Unique identifier for the search session
        collection: Name of the data collection being searched
        modelId: Identifier for the model associated with this session
    """

    session: str
    """Unique identifier for the search session."""

    collection: str
    """Name of the data collection being searched."""

    modelId: int
    """Identifier for the model associated with this session."""


class AddOrRemoveModelRequest(BaseModel):
    """Request model for adding or removing a model from a session.

    Attributes:
        session: Unique identifier for the search session
        collection: Name of the data collection to operate on
        modelId: Unique identifier for the model being managed
    """

    session: str
    """Unique identifier for the search session."""

    collection: str
    """Name of the data collection to operate on."""

    modelId: int
    """Unique identifier for the model being managed."""
