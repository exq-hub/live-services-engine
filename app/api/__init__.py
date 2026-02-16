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
