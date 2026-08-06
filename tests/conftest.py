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

"""Shared fixtures for config tests.

These helpers build valid TOML config text and the on-disk files the
config validators expect to exist (database, embeddings, index files),
without hardcoding the full TOML schema in every test.
"""

from pathlib import Path
from typing import Optional

import pytest


@pytest.fixture
def dummy_files(tmp_path: Path) -> dict:
    """Create placeholder files for the paths CollectionConfig/IndexConfig validate."""
    database_file = tmp_path / "collection.db"
    embeddings_file = tmp_path / "embeddings.zarr.zip"
    index_file = tmp_path / "index.faiss"

    database_file.touch()
    embeddings_file.touch()
    index_file.touch()

    return {
        "database_file": str(database_file),
        "embeddings_file": str(embeddings_file),
        "index_file": str(index_file),
    }


def index_toml(
    name: str,
    index_type: str,
    embeddings_file: str,
    index_file: Optional[str] = None,
    model_name: Optional[str] = None,
    embedding_type: Optional[str] = None,
    default: Optional[bool] = None,
) -> str:
    """Render a `[[collections.indexes]]` table.

    `model_name`, `embedding_type`, and `default` are omitted unless
    given, exercising the schema's own defaults.
    """
    lines = [
        "  [[collections.indexes]]",
        f'  name = "{name}"',
        f'  index_type = "{index_type}"',
    ]
    if index_file is not None:
        lines.append(f'  index_file = "{index_file}"')
    lines.append(f'  embeddings_file = "{embeddings_file}"')
    if model_name is not None:
        lines.append(f'  model_name = "{model_name}"')
    if embedding_type is not None:
        lines.append(f'  embedding_type = "{embedding_type}"')
    if default is not None:
        lines.append(f"  default = {str(default).lower()}")
    return "\n".join(lines)


def collection_toml(
    name: str,
    database_file: str,
    thumbnail_media_url: str,
    original_media_url: str,
    indexes: str,
    enabled: bool = True,
    log_directory: Optional[str] = None,
) -> str:
    """Render a `[[collections]]` table, with one or more nested index tables."""
    lines = [
        "[[collections]]",
        f'name = "{name}"',
        f"enabled = {str(enabled).lower()}",
        f'database_file = "{database_file}"',
        f'thumbnail_media_url = "{thumbnail_media_url}"',
        f'original_media_url = "{original_media_url}"',
    ]
    if log_directory is not None:
        lines.append(f'log_directory = "{log_directory}"')
    lines.append("")
    lines.append(indexes)
    return "\n".join(lines)


@pytest.fixture
def minimal_collection_toml(dummy_files: dict) -> str:
    """A single zarr collection with one index -- the smallest valid collection."""
    indexes = index_toml(
        name="primary",
        index_type="zarr",
        embeddings_file=dummy_files["embeddings_file"],
    )
    return collection_toml(
        name="testcol",
        database_file=dummy_files["database_file"],
        thumbnail_media_url="https://localhost:5000/testcol",
        original_media_url="https://localhost:5000/testcol",
        indexes=indexes,
    )


@pytest.fixture
def write_config(tmp_path: Path):
    """Write config text to a file in tmp_path and return its Path."""

    def _write(text: str, filename: str = "config.toml") -> Path:
        path = tmp_path / filename
        path.write_text(text)
        return path

    return _write
