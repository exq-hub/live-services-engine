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
    collection: str
    modelId: int


class AddOrRemoveModelRequest(BaseModel):
    """Request model for adding or removing a model from a session.

    Attributes:
        session: Unique identifier for the search session
        collection: Name of the data collection to operate on
        modelId: Unique identifier for the model being managed
    """

    session: str
    collection: str
    modelId: int
