from __future__ import annotations
from typing import Optional, Union
from pydantic import BaseModel


class ClientEventRequest(BaseModel):
    """Request model for logging client-side events.

    Attributes:
        ts: Timestamp of the event
        action: Description of the action taken by the user
        element_id: Optional stable identifier for the UI element involved
        data: Optional additional data related to the event
        route: Optional route or page where the event occurred
        session: Unique identifier for the search session
    """

    ts: float
    action: str
    element_id: Optional[str] = None
    data: Optional[Union[str, float, bool]] = None
    route: Optional[str] = None
    session: str
