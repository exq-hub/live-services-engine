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
model-name-keyed lookup). `initialize_models()` eagerly preloads only
each collection's *default* index's model, plus every index in a
`preload_all_indexes` collection -- filtered to this manager's own
embedding_type ('CLIP' or 'Text'). Anything else loads lazily the first
time `get_text_encoder`/`get_tokenizer` is called for it.

open_clip/sentence_transformers/torch calls are mocked throughout --
these tests are about the model-name bookkeeping, family filtering, and
eager-vs-lazy scoping, not about actually downloading or running a model.
"""

from unittest.mock import MagicMock

import pytest

from app.core.config import ConfigManager
from app.core.models import CLIPModelManager, TextModelManager

from .conftest import collection_toml, index_toml


@pytest.fixture
def multi_model_config(write_config, dummy_files):
    """Three collections exercising default-only, cross-family, and preload_all scoping.

    - collection_a: single CLIP index (implicitly default) -> "ViT-SO400M-14-SigLIP-384".
    - collection_b: clip_idx (CLIP, "model-one", default), clip_extra_idx
      (CLIP, "model-extra", not default), text_idx (Text, "model-two", not
      default). No preload_all -- only "model-one" is eager.
    - collection_c: preload_all_indexes=true, with clip_default (CLIP,
      "model-preload-a", default), clip_extra (CLIP, "model-preload-b"),
      text_extra (Text, "model-preload-c"). All three are eager.
    """
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
                name="clip_extra_idx",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                model_name="model-extra",
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
    collection_c_indexes = "\n".join(
        [
            index_toml(
                name="clip_default",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                model_name="model-preload-a",
                default=True,
            ),
            index_toml(
                name="clip_extra",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                model_name="model-preload-b",
            ),
            index_toml(
                name="text_extra",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                model_name="model-preload-c",
                embedding_type="Text",
            ),
        ]
    )
    collection_c = collection_toml(
        name="collection_c",
        database_file=dummy_files["database_file"],
        thumbnail_media_url="https://localhost:5000/c",
        original_media_url="https://localhost:5000/c",
        indexes=collection_c_indexes,
        preload_all_indexes=True,
    )
    path = write_config(collection_a + "\n\n" + collection_b + "\n\n" + collection_c)
    return ConfigManager(str(path)).load_config()


@pytest.fixture
def mocked_clip_loading(monkeypatch):
    create_model_calls = []

    def fake_create_model(model_name, **kwargs):
        create_model_calls.append(model_name)
        return MagicMock(text=MagicMock())

    monkeypatch.setattr("app.core.models.open_clip.create_model", fake_create_model)
    monkeypatch.setattr(
        "app.core.models.open_clip.get_tokenizer", lambda name: MagicMock()
    )
    monkeypatch.setattr("app.core.models.torch.save", lambda *a, **kw: None)
    return create_model_calls


@pytest.fixture
def mocked_text_loading(monkeypatch):
    create_calls = []

    def fake_sentence_transformer(model_name, **kwargs):
        create_calls.append(model_name)
        return MagicMock()

    monkeypatch.setattr(
        "app.core.models.SentenceTransformer", fake_sentence_transformer
    )
    return create_calls


class TestCLIPModelManagerInitialize:
    def test_preloads_only_default_and_preload_all_indexes(
        self, multi_model_config, mocked_clip_loading
    ):
        manager = CLIPModelManager(multi_model_config)
        manager.initialize_models()

        assert set(mocked_clip_loading) == {
            "ViT-SO400M-14-SigLIP-384",  # collection_a's sole (default) index
            "model-one",  # collection_b's default index
            "model-preload-a",  # collection_c's default index
            "model-preload-b",  # collection_c's non-default, but preload_all
        }

    def test_does_not_preload_non_default_index_without_preload_all(
        self, multi_model_config, mocked_clip_loading
    ):
        manager = CLIPModelManager(multi_model_config)
        manager.initialize_models()

        assert "model-extra" not in mocked_clip_loading

    def test_ignores_text_family_indexes(self, multi_model_config, mocked_clip_loading):
        manager = CLIPModelManager(multi_model_config)
        manager.initialize_models()

        assert "model-two" not in mocked_clip_loading
        assert "model-preload-c" not in mocked_clip_loading

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


class TestCLIPModelManagerLazyLoad:
    def test_get_text_model_lazily_loads_a_non_preloaded_model(
        self, multi_model_config, mocked_clip_loading
    ):
        manager = CLIPModelManager(multi_model_config)
        manager.initialize_models()
        assert "model-extra" not in mocked_clip_loading

        model = manager.get_text_encoder("model-extra")

        assert model is not None
        assert "model-extra" in mocked_clip_loading

    def test_get_text_model_caches_after_lazy_load(
        self, multi_model_config, mocked_clip_loading
    ):
        manager = CLIPModelManager(multi_model_config)

        first = manager.get_text_encoder("model-extra")
        second = manager.get_text_encoder("model-extra")

        assert first is second
        assert mocked_clip_loading.count("model-extra") == 1

    def test_get_text_tokenizer_lazily_loads_alongside_the_model(
        self, multi_model_config, mocked_clip_loading
    ):
        manager = CLIPModelManager(multi_model_config)

        tokenizer = manager.get_tokenizer("model-extra")

        assert tokenizer is not None
        assert "model-extra" in mocked_clip_loading


class TestCLIPModelManagerCachePath:
    def test_cache_path_is_unique_per_model_name(self):
        path_a = CLIPModelManager._text_encoder_cache_path("model-one")
        path_b = CLIPModelManager._text_encoder_cache_path("model-two")
        assert path_a != path_b

    def test_cache_path_sanitizes_slashes(self):
        path = CLIPModelManager._text_encoder_cache_path("org/model-name")
        assert "/" not in path.name


class TestTextModelManagerInitialize:
    def test_preloads_only_preload_all_indexes(
        self, multi_model_config, mocked_text_loading
    ):
        manager = TextModelManager(multi_model_config)
        manager.initialize_models()

        assert mocked_text_loading == ["model-preload-c"]

    def test_does_not_preload_non_default_text_index_in_non_preload_all_collection(
        self, multi_model_config, mocked_text_loading
    ):
        manager = TextModelManager(multi_model_config)
        manager.initialize_models()

        assert "model-two" not in mocked_text_loading

    def test_ignores_clip_family_indexes(self, multi_model_config, mocked_text_loading):
        manager = TextModelManager(multi_model_config)
        manager.initialize_models()

        assert "model-one" not in mocked_text_loading
        assert "ViT-SO400M-14-SigLIP-384" not in mocked_text_loading
        assert "model-preload-a" not in mocked_text_loading
        assert "model-preload-b" not in mocked_text_loading


class TestTextModelManagerLazyLoad:
    def test_get_text_model_lazily_loads_a_non_preloaded_model(
        self, multi_model_config, mocked_text_loading
    ):
        manager = TextModelManager(multi_model_config)

        model = manager.get_text_encoder("model-two")

        assert model is not None
        assert mocked_text_loading == ["model-two"]

    def test_get_text_model_caches_after_lazy_load(
        self, multi_model_config, mocked_text_loading
    ):
        manager = TextModelManager(multi_model_config)

        first = manager.get_text_encoder("model-two")
        second = manager.get_text_encoder("model-two")

        assert first is second
        assert mocked_text_loading.count("model-two") == 1
