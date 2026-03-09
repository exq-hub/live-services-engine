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
