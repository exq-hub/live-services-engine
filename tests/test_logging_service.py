# SPDX-FileCopyrightText: 2026 Ujjwal Sharma and Omar Shahbaz Khan
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for what LoggingService hands to the audit logger."""

import asyncio
from unittest.mock import AsyncMock

from app.services.logging_service import LoggingService


def _logged_data(item_id):
    audit_logger = AsyncMock()
    service = LoggingService(audit_logger)

    asyncio.run(
        service.log_item_request(
            "Get Info For Item", "s1", 0, "testcol", item_id, {"mediaType": 2}
        )
    )

    return audit_logger.log.call_args.kwargs["data"]


class TestLogItemRequest:
    def test_logs_exactly_the_request_fields_and_extra_data(self):
        assert _logged_data(7) == {
            "session": "s1",
            "modelId": 0,
            "collection": "testcol",
            "item": 7,
            "mediaType": 2,
        }

    def test_logs_a_batch_as_a_list_of_item_ids(self):
        assert _logged_data([7, 8])["item"] == [7, 8]
