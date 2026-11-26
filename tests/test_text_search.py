# SPDX-FileCopyrightText: 2026 Ujjwal Sharma and Omar Shahbaz Khan
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for TextEmbeddingSearchStrategy's index resolution.

Mirrors test_clip_search.py's TestResolveIndex -- the resolution logic
is shared (EmbeddingSearchStrategy._resolve_index), only the family
differs.
"""

from unittest.mock import MagicMock

import pytest

from app.core.config import ConfigManager
from app.core.models import TextModelManager
from app.strategies.text_search import TextEmbeddingSearchStrategy

from .conftest import collection_toml, index_toml


class TestResolveIndex:
    def test_falls_back_to_the_text_type_index_when_overall_default_is_clip(
        self, write_config, dummy_files
    ):
        indexes = "\n".join(
            [
                index_toml(
                    name="clip_default",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    model_name="clip-model",
                    default=True,
                ),
                index_toml(
                    name="text_idx",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    embedding_type="Text",
                    model_name="text-model",
                ),
            ]
        )
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=indexes,
        )
        path = write_config(collection)
        config = ConfigManager(str(path)).load_config()

        model_manager = TextModelManager(config)
        strategy = TextEmbeddingSearchStrategy(model_manager, MagicMock(), MagicMock())

        resolved = strategy._resolve_index("testcol")
        assert resolved.name == "text_idx"
        assert resolved.model_name == "text-model"

    def test_raises_when_collection_has_no_text_type_index(
        self, write_config, dummy_files
    ):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="clip_only",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
        )
        path = write_config(collection)
        config = ConfigManager(str(path)).load_config()

        model_manager = TextModelManager(config)
        strategy = TextEmbeddingSearchStrategy(model_manager, MagicMock(), MagicMock())

        with pytest.raises(ValueError):
            strategy._resolve_index("testcol")

    def test_get_strategy_name(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="text_idx",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                embedding_type="Text",
            ),
        )
        path = write_config(collection)
        config = ConfigManager(str(path)).load_config()

        strategy = TextEmbeddingSearchStrategy(
            TextModelManager(config), MagicMock(), MagicMock()
        )
        assert strategy.get_strategy_name() == "Text Search"
