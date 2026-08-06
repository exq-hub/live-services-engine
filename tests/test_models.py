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

"""Tests for CLIPModelManager and TextModelManager.

Both subclass the ModelManager ABC (shared device resolution and
model-name-keyed lookup). Each loads one model per distinct model_name
among the indexes belonging to its own embedding_type ('CLIP' or
'Text'), ignoring indexes that belong to the other family.
open_clip/sentence_transformers/torch calls are mocked throughout --
these tests are about the model-name bookkeeping and family filtering,
not about actually downloading or running a model.
"""

from unittest.mock import MagicMock

import pytest

from app.core.config import ConfigManager
from app.core.exceptions import ModelLoadError
from app.core.models import CLIPModelManager, TextModelManager

from .conftest import collection_toml, index_toml


@pytest.fixture
def multi_model_config(write_config, dummy_files):
    """Two CLIP indexes with distinct model_names, plus one Text index."""
    collection_a = collection_toml(
        name="collection_a",
        database_file=dummy_files["database_file"],
        thumbnail_media_url="https://localhost:5000/a",
        original_media_url="https://localhost:5000/a",
        indexes=index_toml(
            name="CLIP",
            index_type="zarr",
            embeddings_file=dummy_files["embeddings_file"],
        ),
    )
    collection_b_indexes = "\n".join(
        [
            index_toml(
                name="clip_idx",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                model_name="model-one",
                default=True,
            ),
            index_toml(
                name="text_idx",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                model_name="model-two",
                embedding_type="Text",
            ),
        ]
    )
    collection_b = collection_toml(
        name="collection_b",
        database_file=dummy_files["database_file"],
        thumbnail_media_url="https://localhost:5000/b",
        original_media_url="https://localhost:5000/b",
        indexes=collection_b_indexes,
    )
    path = write_config(collection_a + "\n\n" + collection_b)
    return ConfigManager(str(path)).load_config()


class TestCLIPModelManagerInitialize:
    def test_loads_one_model_per_distinct_clip_model_name(
        self, multi_model_config, monkeypatch
    ):
        create_model_calls = []

        def fake_create_model(model_name, **kwargs):
            create_model_calls.append(model_name)
            fake = MagicMock()
            fake.text = MagicMock()
            return fake

        monkeypatch.setattr("app.core.models.open_clip.create_model", fake_create_model)
        monkeypatch.setattr(
            "app.core.models.open_clip.get_tokenizer", lambda name: MagicMock()
        )
        monkeypatch.setattr("app.core.models.torch.save", lambda *a, **kw: None)

        manager = CLIPModelManager(multi_model_config)
        manager.initialize_models()

        assert set(create_model_calls) == {"ViT-SO400M-14-SigLIP-384", "model-one"}
        for name in create_model_calls:
            assert manager.get_text_model(name) is not None
            assert manager.get_text_tokenizer(name) is not None

    def test_ignores_text_family_indexes(self, multi_model_config, monkeypatch):
        monkeypatch.setattr(
            "app.core.models.open_clip.create_model",
            lambda name, **kw: MagicMock(text=MagicMock()),
        )
        monkeypatch.setattr(
            "app.core.models.open_clip.get_tokenizer", lambda name: MagicMock()
        )
        monkeypatch.setattr("app.core.models.torch.save", lambda *a, **kw: None)

        manager = CLIPModelManager(multi_model_config)
        manager.initialize_models()

        with pytest.raises(ModelLoadError):
            manager.get_text_model("model-two")

    def test_loads_from_cache_when_present(self, multi_model_config, monkeypatch):
        create_model_calls = []
        monkeypatch.setattr(
            "app.core.models.open_clip.create_model",
            lambda name, **kw: create_model_calls.append(name),
        )
        monkeypatch.setattr(
            "app.core.models.open_clip.get_tokenizer", lambda name: MagicMock()
        )

        cached_model = MagicMock()
        cached_model.to.return_value = cached_model
        monkeypatch.setattr("app.core.models.torch.load", lambda *a, **kw: cached_model)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

        manager = CLIPModelManager(multi_model_config)
        manager.initialize_models()

        assert create_model_calls == []


class TestCLIPModelManagerLookup:
    def test_get_text_model_raises_for_unknown_model(self, multi_model_config):
        manager = CLIPModelManager(multi_model_config)
        with pytest.raises(ModelLoadError):
            manager.get_text_model("never-loaded")

    def test_get_text_tokenizer_raises_for_unknown_model(self, multi_model_config):
        manager = CLIPModelManager(multi_model_config)
        with pytest.raises(ModelLoadError):
            manager.get_text_tokenizer("never-loaded")


class TestCLIPModelManagerCachePath:
    def test_cache_path_is_unique_per_model_name(self):
        path_a = CLIPModelManager._text_model_cache_path("model-one")
        path_b = CLIPModelManager._text_model_cache_path("model-two")
        assert path_a != path_b

    def test_cache_path_sanitizes_slashes(self):
        path = CLIPModelManager._text_model_cache_path("org/model-name")
        assert "/" not in path.name


class TestTextModelManagerInitialize:
    def test_loads_one_model_per_distinct_text_model_name(
        self, multi_model_config, monkeypatch
    ):
        create_calls = []

        def fake_sentence_transformer(model_name, **kwargs):
            create_calls.append(model_name)
            return MagicMock()

        monkeypatch.setattr(
            "app.core.models.SentenceTransformer", fake_sentence_transformer
        )

        manager = TextModelManager(multi_model_config)
        manager.initialize_models()

        assert create_calls == ["model-two"]
        assert manager.get_text_model("model-two") is not None

    def test_ignores_clip_family_indexes(self, multi_model_config, monkeypatch):
        monkeypatch.setattr(
            "app.core.models.SentenceTransformer", lambda name, **kw: MagicMock()
        )

        manager = TextModelManager(multi_model_config)
        manager.initialize_models()

        with pytest.raises(ModelLoadError):
            manager.get_text_model("model-one")
        with pytest.raises(ModelLoadError):
            manager.get_text_model("ViT-SO400M-14-SigLIP-384")


class TestTextModelManagerLookup:
    def test_get_text_model_raises_for_unknown_model(self, multi_model_config):
        manager = TextModelManager(multi_model_config)
        with pytest.raises(ModelLoadError):
            manager.get_text_model("never-loaded")
