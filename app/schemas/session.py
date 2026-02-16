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
