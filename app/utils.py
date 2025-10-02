"""Updated utilities with better error handling and logging."""

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
        raise LSEException(f"Failed to write log to {logfile}: {e}", {
            "logfile": logfile,
            "log_keys": list(log.keys()) if isinstance(log, dict) else "not_dict"
        })


# For backward compatibility with existing code
def get_shared_resources():
    """Legacy function for compatibility - now returns container."""
    from .core.models import container
    return container