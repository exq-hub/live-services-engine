#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Ujjwal Sharma and Omar Shahbaz Khan
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

uv run python -m pdoc --output-directory ./docs/api app main tasks
echo "Documentation generated in ./docs/api/"
