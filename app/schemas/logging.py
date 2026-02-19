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
