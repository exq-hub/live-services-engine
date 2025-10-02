"""Utility script for reading and decoding MessagePack-encoded log files.

This script provides functionality to read and decode log files created by
the Live Services Engine logging system, which uses MessagePack format
for efficient log storage.
"""

import argparse

import msgpack


def read_logs_msgpack(logfile: str):
    """Reads and decodes MessagePack-encoded log messages from a file."""
    logs = []
    with open(logfile, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for log in unpacker:
            logs.append(log)
    return logs


# Use argparse to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read and decode MessagePack-encoded log messages."
    )
    parser.add_argument("logfile", type=str, help="Path to the log file")

    args = parser.parse_args()

    logs = read_logs_msgpack(args.logfile)
    for log in logs:
        print(log)
