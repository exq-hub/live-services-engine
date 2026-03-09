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
