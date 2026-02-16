"""Logging-related request schemas for client-side event capture.

The ``POST /exq/log/clientEvent`` endpoint accepts a batch of these events,
enabling the frontend to report UI interactions (clicks, navigation, etc.)
to the audit log for user-study analysis.
"""

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
    """Timestamp of the event (Unix epoch seconds)."""

    action: str
    """Description of the action taken by the user (e.g. ``'click'``, ``'scroll'``)."""

    element_id: Optional[str] = None
    """Stable identifier for the UI element involved, if any."""

    data: Optional[Union[str, float, bool]] = None
    """Optional payload associated with the event."""

    route: Optional[str] = None
    """Client-side route or page where the event occurred."""

    session: str
    """Session identifier linking this event to a search session."""
