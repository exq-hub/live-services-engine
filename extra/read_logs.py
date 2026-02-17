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


"""Utility script for reading and decoding MessagePack-encoded log files.

This script provides functionality to read and decode log files created by
the Live Services Engine logging system, which uses MessagePack format
for efficient log storage.
"""

import argparse
from ast import arg

import msgpack

import json


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
    parser.add_argument(
        "--write_json_file", type=str, help="Path to output JSON file", default=None
    )
    parser.add_argument(
        "--ordered_simplified",
        action="store_true",
        help="Use ordered and simplified output format",
    )
    parser.add_argument(
        "--print_logs", action="store_true", help="Print all logs to console"
    )

    args = parser.parse_args()

    logs = read_logs_msgpack(args.logfile)
    if args.print_logs:
        for log in logs:
            print(log)

    if args.write_json_file:
        with open(args.write_json_file, "w", encoding="utf-8") as json_file:
            json.dump(logs, json_file)

    if args.ordered_simplified:
        simplified_logs = []
        for log in logs:
            simplified_log = {
                "timestamp": log.get("timestamp"),
                "action": log.get("action"),
                "session": log.get("session"),
                "details": log.get("display_attrs", log.get("data", {})),
            }
            simplified_logs.append(simplified_log)

        simplified_logs.sort(key=lambda x: x["timestamp"])

        print("\nSimplified Logs:")
        if args.print_logs:
            for slog in simplified_logs:
                print(slog)
        if args.write_json_file:
            simplified_json_file = args.write_json_file.replace(
                ".json", "_simplified.json"
            )
            with open(simplified_json_file, "w", encoding="utf-8") as json_file:
                json.dump(simplified_logs, json_file)
