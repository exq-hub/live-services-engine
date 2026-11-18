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

import json
import re
from unittest.mock import MagicMock

import open_clip
import pytest
import torch

from app.core.config import ConfigManager
from app.core.exceptions import ModelLoadError
from app.core.models import CLIPModelManager, OpenCLIPTextEncoder, TextModelManager

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
def single_pretrained_tag(monkeypatch):
    """Give every made-up model name one open_clip pretrained tag, like SigLIP."""
    monkeypatch.setattr(
        "app.core.models.open_clip.list_pretrained_tags_by_model",
        lambda name: ["fake-tag"],
    )


@pytest.fixture
def mocked_clip_loading(monkeypatch, single_pretrained_tag):
    create_model_calls = []

    def fake_create_model(model_name, **kwargs):
        create_model_calls.append(model_name)
        return MagicMock(text=MagicMock())

    monkeypatch.setattr("app.core.models.open_clip.create_model", fake_create_model)
    monkeypatch.setattr(
        "app.core.models.open_clip.get_tokenizer", lambda name: MagicMock()
    )
    monkeypatch.setattr("app.core.models.torch.save", lambda *a, **kw: None)
    # Force the cache-miss branches of _load_text_encoder and _load_tokenizer
    # regardless of what's actually on disk in the repo's real ./data/models/
    # and ./data/tokenizers/ dirs (e.g. a previously cached default model).
    # Otherwise a real cache hit silently bypasses create_model, and the
    # corresponding model_name never lands here.
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    # Keep the cache-miss branches from creating those dirs for real
    monkeypatch.setattr("pathlib.Path.mkdir", lambda self, **kw: None)
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

    def test_loads_from_cache_when_present(
        self, multi_model_config, monkeypatch, single_pretrained_tag
    ):
        create_model_calls = []
        get_tokenizer_calls = []
        monkeypatch.setattr(
            "app.core.models.open_clip.create_model",
            lambda name, **kw: create_model_calls.append(name),
        )
        monkeypatch.setattr(
            "app.core.models.open_clip.get_tokenizer",
            lambda name: get_tokenizer_calls.append(name),
        )

        cached_model = MagicMock()
        cached_model.to.return_value = cached_model
        monkeypatch.setattr("app.core.models.torch.load", lambda *a, **kw: cached_model)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

        manager = CLIPModelManager(multi_model_config)
        manager.initialize_models()

        assert create_model_calls == []
        assert get_tokenizer_calls == []


class TestCLIPModelManagerLazyLoad:
    def test_get_text_model_lazily_loads_a_non_preloaded_model(
        self, multi_model_config, mocked_clip_loading
    ):
        manager = CLIPModelManager(multi_model_config)
        manager.initialize_models()
        assert "model-extra" not in mocked_clip_loading

        manager.get_text_encoder("model-extra")

        assert "model-extra" in mocked_clip_loading

    def test_get_text_model_caches_after_lazy_load(
        self, multi_model_config, mocked_clip_loading
    ):
        manager = CLIPModelManager(multi_model_config)

        first = manager.get_text_encoder("model-extra")
        second = manager.get_text_encoder("model-extra")

        assert first is second
        assert mocked_clip_loading.count("model-extra") == 1

    def test_loaded_model_names_lists_preloaded_then_lazily_loaded_models(
        self, multi_model_config, mocked_clip_loading
    ):
        manager = CLIPModelManager(multi_model_config)
        assert manager.loaded_model_names() == []

        manager.initialize_models()
        assert manager.loaded_model_names() == [
            "ViT-SO400M-14-SigLIP-384",
            "model-one",
            "model-preload-a",
            "model-preload-b",
        ]

        manager.get_text_encoder("model-extra")
        assert "model-extra" in manager.loaded_model_names()

    def test_get_text_tokenizer_lazily_loads_alongside_the_model(
        self, multi_model_config, mocked_clip_loading
    ):
        manager = CLIPModelManager(multi_model_config)

        manager.get_tokenizer("model-extra")

        assert "model-extra" in mocked_clip_loading


class TestCLIPModelManagerRequiresPretrainedWeights:
    def test_local_dir_model_without_weights_fails_and_caches_nothing(
        self, multi_model_config, monkeypatch, tmp_path
    ):
        """Real open_clip, offline: a local-dir: model whose folder has a config
        but no weights file would otherwise load with random weights."""
        model_dir = tmp_path / "tiny-clip"
        model_dir.mkdir()
        # Smallest config open_clip accepts; custom_text gives the model the
        # .text tower the LSE extracts
        (model_dir / "open_clip_config.json").write_text(
            json.dumps(
                {
                    "model_cfg": {
                        "embed_dim": 8,
                        "custom_text": True,
                        "vision_cfg": {
                            "image_size": 32,
                            "layers": 1,
                            "width": 32,
                            "head_width": 32,
                            "patch_size": 16,
                        },
                        "text_cfg": {
                            "context_length": 8,
                            "vocab_size": 64,
                            "width": 32,
                            "heads": 1,
                            "layers": 1,
                        },
                    }
                }
            )
        )
        # Keep both caches in tmp_path, even if the load wrongly succeeds
        cache_path = tmp_path / "cache" / "tiny-clip_text.pth"
        monkeypatch.setattr(
            CLIPModelManager,
            "_text_encoder_cache_path",
            staticmethod(lambda _name: cache_path),
        )
        monkeypatch.setattr(
            CLIPModelManager,
            "_tokenizer_cache_path",
            staticmethod(lambda _name: tmp_path / "cache" / "tiny-clip_tokenizer.pth"),
        )
        model_name = f"local-dir:{model_dir}"
        manager = CLIPModelManager(multi_model_config)

        with pytest.raises(ModelLoadError, match="^Failed to load CLIP text model"):
            manager.get_text_encoder(model_name)

        assert not cache_path.exists()
        assert manager.loaded_model_names() == []


class TestCLIPModelManagerPretrainedTag:
    """Uses open_clip's built-in list of pretrained weights, which needs no network."""

    @pytest.mark.parametrize(
        "model_name", ["hf-hub:timm/ViT-B-16-SigLIP2-512", "local-dir:/models/clip"]
    )
    def test_hub_and_local_names_need_no_tag(self, model_name):
        assert CLIPModelManager._pretrained_tag(model_name) is None

    def test_plain_name_with_one_set_of_weights_uses_it(self):
        assert CLIPModelManager._pretrained_tag("ViT-B-16-SigLIP2-512") == "webli"

    def test_plain_name_with_several_sets_of_weights_lists_hub_names(self):
        with pytest.raises(ModelLoadError, match="several pretrained weights") as e:
            CLIPModelManager._pretrained_tag("ViT-B-32")

        assert "hf-hub:timm/vit_base_patch32_clip_224.openai" in str(e.value)
        assert "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K" in str(e.value)

    def test_unknown_plain_name_fails(self):
        with pytest.raises(ModelLoadError, match="no pretrained weights"):
            CLIPModelManager._pretrained_tag("not-a-model")

    def test_startup_checks_lazily_loaded_indexes_too(
        self, multi_model_config, mocked_clip_loading, monkeypatch
    ):
        """model-extra isn't preloaded, but its bad name still fails at startup."""
        monkeypatch.setattr(
            "app.core.models.open_clip.list_pretrained_tags_by_model",
            lambda name: ["a", "b"] if name == "model-extra" else ["fake-tag"],
        )
        monkeypatch.setattr(
            "app.core.models.open_clip.get_pretrained_cfg",
            lambda name, tag: {"hf_hub": f"org/{name}-{tag}/"},
        )
        manager = CLIPModelManager(multi_model_config)

        with pytest.raises(ModelLoadError, match="hf-hub:org/model-extra-a"):
            manager.initialize_models()

        assert mocked_clip_loading == []


TINY_VISION_CFG = {
    "image_size": 32, "layers": 1, "width": 32, "head_width": 32, "patch_size": 16
}
TINY_TEXT_CFG = {
    "context_length": 8, "vocab_size": 64, "width": 32, "heads": 1, "layers": 1
}


class TestOpenCLIPTextEncoder:
    """Tiny untrained open_clip models of both kinds, built offline."""

    @pytest.mark.parametrize(
        "model_class",
        [open_clip.CLIP, open_clip.CustomTextCLIP],
        ids=["classic", "custom-text"],
    )
    def test_matches_encode_text_and_drops_the_image_tower(self, model_class):
        torch.manual_seed(0)
        model = model_class(
            embed_dim=8, vision_cfg=TINY_VISION_CFG, text_cfg=TINY_TEXT_CFG
        ).eval()
        tokens = torch.randint(0, 64, (2, 8))
        expected = model.encode_text(tokens)

        encoder = OpenCLIPTextEncoder(model).eval()

        assert torch.equal(encoder(tokens), expected)
        assert encoder.model.visual is None

    def test_survives_the_disk_cache_round_trip(self, tmp_path):
        model = open_clip.CLIP(
            embed_dim=8, vision_cfg=TINY_VISION_CFG, text_cfg=TINY_TEXT_CFG
        ).eval()
        encoder = OpenCLIPTextEncoder(model).eval()
        tokens = torch.randint(0, 64, (2, 8))
        path = tmp_path / "encoder.pth"

        torch.save(encoder, path)
        loaded = torch.load(path, weights_only=False)

        assert torch.equal(loaded(tokens), encoder(tokens))


class TestCLIPModelManagerCachePath:
    def test_cache_path_is_unique_per_model_name(self):
        path_a = CLIPModelManager._text_encoder_cache_path("model-one")
        path_b = CLIPModelManager._text_encoder_cache_path("model-two")
        assert path_a != path_b

    def test_cache_path_sanitizes_slashes(self):
        path = CLIPModelManager._text_encoder_cache_path("org/model-name")
        assert "/" not in path.name

    def test_tokenizer_cache_path_is_unique_per_model_name(self):
        path_a = CLIPModelManager._tokenizer_cache_path("model-one")
        path_b = CLIPModelManager._tokenizer_cache_path("model-two")
        assert path_a != path_b

    def test_tokenizer_cache_path_sanitizes_slashes(self):
        path = CLIPModelManager._tokenizer_cache_path("org/model-name")
        assert "/" not in path.name

    CACHE_PATHS = [
        CLIPModelManager._text_encoder_cache_path,
        CLIPModelManager._tokenizer_cache_path,
    ]

    @pytest.mark.parametrize("cache_path", CACHE_PATHS)
    @pytest.mark.parametrize(
        "model_name",
        [
            "hf-hub:timm/vit_base_patch32_clip_224.openai",
            "local-dir:C:\\models\\my-clip",
            "local-dir:/home/user/models/my clip",
        ],
    )
    def test_cache_file_name_is_valid_on_windows(self, cache_path, model_name):
        """Windows file names can't hold : \\ / * ? " < > | or spaces we'd trip over."""
        assert re.fullmatch(r"[A-Za-z0-9._-]+", cache_path(model_name).name)

    @pytest.mark.parametrize("cache_path", CACHE_PATHS)
    def test_names_that_clean_up_the_same_get_different_files(self, cache_path):
        assert cache_path("org/model") != cache_path("org_model")

    @pytest.mark.parametrize("cache_path", CACHE_PATHS)
    def test_long_local_dir_names_give_short_file_names(self, cache_path):
        model_name = "local-dir:C:\\" + "\\".join(["a-long-folder-name"] * 20)
        assert len(cache_path(model_name).name) <= 120

    def test_text_encoder_and_tokenizer_cache_paths_differ(self):
        text_path = CLIPModelManager._text_encoder_cache_path("model-one")
        tokenizer_path = CLIPModelManager._tokenizer_cache_path("model-one")
        assert text_path != tokenizer_path

    def test_cache_paths_change_with_the_open_clip_version(self, monkeypatch):
        monkeypatch.setattr("app.core.models.open_clip.__version__", "3.3.0")
        old = (
            CLIPModelManager._text_encoder_cache_path("model-one"),
            CLIPModelManager._tokenizer_cache_path("model-one"),
        )
        monkeypatch.setattr("app.core.models.open_clip.__version__", "4.2.2")
        new = (
            CLIPModelManager._text_encoder_cache_path("model-one"),
            CLIPModelManager._tokenizer_cache_path("model-one"),
        )

        assert old[0] != new[0]
        assert old[1] != new[1]


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

        manager.get_text_encoder("model-two")

        assert mocked_text_loading == ["model-two"]

    def test_get_text_model_caches_after_lazy_load(
        self, multi_model_config, mocked_text_loading
    ):
        manager = TextModelManager(multi_model_config)

        first = manager.get_text_encoder("model-two")
        second = manager.get_text_encoder("model-two")

        assert first is second
        assert mocked_text_loading.count("model-two") == 1
