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

"""Tests for the logging routes, run without the app's startup.

The TestClient is used without its context manager, so the lifespan
never runs, and app.state.logging_service is a stub that records calls.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def audit_log(monkeypatch):
    log = AsyncMock()
    monkeypatch.setattr(
        main.app.state,
        "logging_service",
        SimpleNamespace(audit_logger=SimpleNamespace(log=log)),
        raising=False,
    )
    return log


class TestClientEvent:
    def test_logs_each_events_client_timestamp(self, audit_log):
        events = [
            {"ts": 1762858800123, "action": "click", "session": "s1"},
            {"ts": 1762858800456, "action": "scroll", "session": "s1"},
        ]

        response = TestClient(main.app).post("/exq/log/clientEvent", json=events)

        assert response.status_code == 200
        logged = [call.kwargs["data"]["ts"] for call in audit_log.await_args_list]
        assert logged == [1762858800123, 1762858800456]
