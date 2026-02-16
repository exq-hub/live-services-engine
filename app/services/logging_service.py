"""Centralized logging service for audit trails and system logs.

This module provides two classes that work together to produce a durable,
structured audit trail of every user interaction with the LSE:

`AuditLogger`
    Low-level async writer.  Log entries are pushed onto an `asyncio.Queue`
    and consumed by a background worker task that serializes each entry as
    MessagePack and appends it to a log file under a `FileLock` for safe
    concurrent access.  A synchronous ``log_sync`` method is also available
    for non-async contexts.

`LoggingService`
    High-level facade consumed by route handlers and services.  It provides
    semantic methods (``log_search_request``, ``log_item_request``,
    ``log_session_init``, ``log_error``, etc.) that format the audit entry
    and delegate to `AuditLogger`.  It also mirrors errors to the standard
    Python ``logging`` system for operational monitoring.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import msgpack
from filelock import FileLock

from ..core.exceptions import LSEException


class AuditLogger:
    """Handles audit logging with structured format and async processing."""

    def __init__(self, log_file: str):
        self.log_file: Path = Path(log_file)
        """Path to the MessagePack-encoded audit log file."""

        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.lockfile: str = f"{self.log_file}.lock"
        """Path to the file-level lock used for concurrent write safety."""

        self._queue: asyncio.Queue = asyncio.Queue()
        """Async queue buffering log entries for the background writer."""

        self._worker_task: Optional[asyncio.Task] = None
        """Handle to the background asyncio task consuming the log queue."""

    async def start(self):
        """Start the async logging worker."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._log_worker())

    async def stop(self):
        """Stop the async logging worker."""
        if self._worker_task:
            await self._queue.put(None)  # Sentinel to stop worker
            await self._worker_task
            self._worker_task = None

    async def _log_worker(self):
        """Background worker that processes log messages."""
        while True:
            try:
                log_entry = await self._queue.get()
                if log_entry is None:  # Stop sentinel
                    break
                self._write_log_sync(log_entry)
                self._queue.task_done()
            except Exception as e:
                # Log to system logger if audit logging fails
                logging.error(f"Failed to write audit log: {e}")

    def _write_log_sync(self, log_entry: Dict[str, Any]):
        """Synchronously write log entry to file."""
        lock = FileLock(self.lockfile)
        with lock:
            with open(self.log_file, "ab") as f:
                packer = msgpack.Packer()
                packed_data = packer.pack(log_entry)
                f.write(packed_data)

    async def log(
        self,
        action: str,
        session: str,
        data: Dict[str, Any],
        request_timestamp: Optional[int] = None,
        completion_time: Optional[int] = None,
    ):
        """Log an audit entry asynchronously."""
        timestamp = request_timestamp or int(time.time())

        log_entry = {
            "timestamp": timestamp,
            "session": session,
            "action": action,
            "data": data,
        }

        if completion_time:
            log_entry["completion_time"] = completion_time

        await self._queue.put(log_entry)

    def log_sync(
        self,
        action: str,
        session: str,
        data: Dict[str, Any],
        request_timestamp: Optional[int] = None,
        completion_time: Optional[int] = None,
    ):
        """Log an audit entry synchronously (for non-async contexts)."""
        timestamp = request_timestamp or int(time.time())

        log_entry = {
            "timestamp": timestamp,
            "session": session,
            "action": action,
            "data": data,
        }

        if completion_time:
            log_entry["completion_time"] = completion_time

        self._write_log_sync(log_entry)


class LoggingService:
    """Service for managing all logging operations."""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger: AuditLogger = audit_logger
        """Low-level audit logger that serializes entries to disk."""

        self.system_logger: logging.Logger = logging.getLogger(__name__)
        """Standard Python logger for operational monitoring output."""

    async def log_session_init(self, session: str, collections: list):
        """Log session initialization."""
        await self.audit_logger.log(
            action="Initialize Exquisitor LSE Session",
            session=session,
            data={"session": session, "collections": collections},
        )

    async def log_total_items_request(
        self, session: str, collection: str, total_items: int
    ):
        """Log total items request."""
        await self.audit_logger.log(
            action="Get Total Items",
            session=session,
            data={
                "collection": collection,
                "total_items": total_items,
            },
        )

    async def log_search_request(
        self,
        action: str,
        session: str,
        model_id: int,
        collection: str,
        query_data: Dict[str, Any],
        suggestions: list,
        request_timestamp: int,
        completion_time: int,
    ):
        """Log search requests with performance metrics."""
        await self.audit_logger.log(
            action=action,
            session=session,
            data={
                "session": session,
                "modelId": model_id,
                "collection": collection,
                "suggestions": suggestions,
                **query_data,
            },
            request_timestamp=request_timestamp,
            completion_time=completion_time,
        )

    async def log_model_operation(
        self, action: str, session: str, model_id: int, collection: str, body_json: str
    ):
        """Log model add/remove operations."""
        await self.audit_logger.log(
            action=action,
            session=session,
            data={
                "session": session,
                "modelId": model_id,
                "collection": collection,
                "body": body_json,
            },
        )

    async def log_filter_operation(
        self,
        action: str,
        session: str,
        model_id: int,
        collection: str,
        filter_data: Dict[str, Any],
    ):
        """Log filter operations."""
        await self.audit_logger.log(
            action=action,
            session=session,
            data={
                "session": session,
                "modelId": model_id,
                "collection": collection,
                **filter_data,
            },
        )

    async def log_item_request(
        self,
        action: str,
        session: str,
        model_id: int,
        collection: str,
        item_id: int,
        item_name: str,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Log item-related requests."""
        data = {
            "session": session,
            "modelId": model_id,
            "collection": collection,
            "item": item_id,
            "mediaName": item_name,
        }
        if additional_data:
            data.update(additional_data)

        await self.audit_logger.log(action=action, session=session, data=data)

    async def log_error(
        self, error: LSEException, session: str, context: Dict[str, Any]
    ):
        """Log application errors."""
        await self.audit_logger.log(
            action="Error",
            session=session,
            data={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_details": getattr(error, "details", {}),
                "context": context,
            },
        )

        # Also log to system logger
        self.system_logger.error(
            f"{type(error).__name__}: {error}",
            extra={
                "session": session,
                "context": context,
                "error_details": getattr(error, "details", {}),
            },
        )

    async def start(self):
        """Start the logging service."""
        await self.audit_logger.start()

    async def stop(self):
        """Stop the logging service."""
        await self.audit_logger.stop()
