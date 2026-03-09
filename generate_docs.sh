#!/usr/bin/env bash
set -euo pipefail

uv run python -m pdoc --output-directory ./docs/api app main tasks
echo "Documentation generated in ./docs/api/"
