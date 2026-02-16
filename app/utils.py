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


"""Shared utility functions for logging and legacy compatibility.

Provides:

- `get_current_timestamp` -- integer epoch seconds.
- `dump_log_msgpack` -- appends a MessagePack-encoded dict to a log file
  with directory auto-creation and file-level locking (`FileLock`).
- `get_shared_resources` -- legacy shim that returns the global
  `ApplicationContainer` instance.
"""

import time
from enum import Enum
from pathlib import Path

import msgpack
from filelock import FileLock

from .core.exceptions import LSEException


def get_current_timestamp() -> int:
    """Returns the current timestamp in seconds."""
    return int(time.time())


def dump_log_msgpack(log: dict, logfile: str) -> None:
    """Writes MessagePack-encoded log messages to a file using Packer for streaming.

    Args:
        log: Dictionary containing log data
        logfile: Path to the log file

    Raises:
        LSEException: If logging fails
    """
    try:
        # Ensure log directory exists
        log_path = Path(logfile)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        lockfile = f"{logfile}.lock"
        lock = FileLock(lockfile)

        with lock:  # Ensure exclusive access to the file
            with open(logfile, "ab") as f:
                packer = msgpack.Packer()
                packed_data = packer.pack(log)
                f.write(packed_data)

    except Exception as e:
        raise LSEException(
            f"Failed to write log to {logfile}: {e}",
            {
                "logfile": logfile,
                "log_keys": list(log.keys()) if isinstance(log, dict) else "not_dict",
            },
        )


# For backward compatibility with existing code
def get_shared_resources():
    """Legacy function for compatibility - now returns container."""
    from .core.models import container

    return container
