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

"""Tests for the TOML-based ConfigManager.

Each collection holds a list of index definitions (`IndexConfig`)
instead of flat index fields. Only one index per collection is
exercised end-to-end elsewhere in the app for now, but the schema
already stores indexes as a list so that follow-up work adding
multi-index support doesn't need a schema migration.
"""

import pytest

from app.core.config import ConfigManager
from app.core.exceptions import ConfigurationError

from .conftest import collection_toml, index_toml


class TestMinimalConfig:
    def test_loads_single_collection_with_one_index(
        self, write_config, minimal_collection_toml, dummy_files
    ):
        path = write_config(minimal_collection_toml)

        config = ConfigManager(str(path)).load_config()

        assert config.collections == ["testcol"]
        collection = config.collection_configs["testcol"]
        assert collection.database_file == dummy_files["database_file"]
        assert collection.thumbnail_media_url == "https://localhost:5000/testcol"
        assert collection.original_media_url == "https://localhost:5000/testcol"
        assert len(collection.indexes) == 1
        assert collection.indexes[0].name == "primary"
        assert collection.indexes[0].index_type == "zarr"
        assert collection.indexes[0].embeddings_file == dummy_files["embeddings_file"]
        assert collection.indexes[0].index_file is None

    def test_applies_server_logging_and_device_defaults_when_omitted(
        self, write_config, minimal_collection_toml
    ):
        path = write_config(minimal_collection_toml)

        config = ConfigManager(str(path)).load_config()

        assert config.model_device == "auto"
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.reload is True
        assert config.log_level == "INFO"

    def test_honors_server_and_logging_overrides(
        self, write_config, minimal_collection_toml
    ):
        text = (
            "[server]\n"
            'host = "0.0.0.0"\n'
            "port = 9001\n"
            "reload = false\n"
            "\n"
            "[logging]\n"
            'level = "DEBUG"\n'
            "\n" + minimal_collection_toml
        )
        path = write_config(text)

        config = ConfigManager(str(path)).load_config()

        assert config.host == "0.0.0.0"
        assert config.port == 9001
        assert config.reload is False
        assert config.log_level == "DEBUG"


class TestCollectionDiscovery:
    def test_disabled_collection_is_excluded(
        self, write_config, dummy_files, minimal_collection_toml
    ):
        disabled = collection_toml(
            name="disabledcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/disabled",
            original_media_url="https://localhost:5000/disabled",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
            enabled=False,
        )
        path = write_config(minimal_collection_toml + "\n\n" + disabled)

        config = ConfigManager(str(path)).load_config()

        assert config.collections == ["testcol"]
        assert "disabledcol" not in config.collection_configs

    def test_multiple_enabled_collections_are_all_loaded(
        self, write_config, dummy_files, minimal_collection_toml
    ):
        second = collection_toml(
            name="othercol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/other",
            original_media_url="https://localhost:5000/other",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
        )
        path = write_config(minimal_collection_toml + "\n\n" + second)

        config = ConfigManager(str(path)).load_config()

        assert set(config.collections) == {"testcol", "othercol"}


class TestIndexValidation:
    def test_faiss_index_requires_index_file(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="faiss",
                embeddings_file=dummy_files["embeddings_file"],
            ),
        )
        path = write_config(collection)

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()

    def test_faiss_index_file_must_exist_on_disk(
        self, write_config, dummy_files, tmp_path
    ):
        missing_index_file = str(tmp_path / "does_not_exist.faiss")
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="faiss",
                embeddings_file=dummy_files["embeddings_file"],
                index_file=missing_index_file,
            ),
        )
        path = write_config(collection)

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()

    def test_zarr_index_forbids_index_file(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                index_file=dummy_files["index_file"],
            ),
        )
        path = write_config(collection)

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()

    def test_embeddings_file_must_exist_on_disk(
        self, write_config, dummy_files, tmp_path
    ):
        missing_embeddings_file = str(tmp_path / "missing.zarr.zip")
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=missing_embeddings_file,
            ),
        )
        path = write_config(collection)

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()

    def test_database_file_must_exist_on_disk(
        self, write_config, dummy_files, tmp_path
    ):
        missing_db_file = str(tmp_path / "missing.db")
        collection = collection_toml(
            name="testcol",
            database_file=missing_db_file,
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
        )
        path = write_config(collection)

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()

    def test_collection_requires_at_least_one_index(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes="",
        )
        path = write_config(collection)

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()

    def test_index_names_must_be_unique_within_a_collection(
        self, write_config, dummy_files
    ):
        indexes = "\n".join(
            [
                index_toml(
                    name="primary",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                ),
                index_toml(
                    name="primary",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
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

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()


class TestMultipleIndexesPerCollection:
    def test_collection_can_declare_a_faiss_and_a_zarr_index_together(
        self, write_config, dummy_files
    ):
        indexes = "\n".join(
            [
                index_toml(
                    name="faiss_idx",
                    index_type="faiss",
                    embeddings_file=dummy_files["embeddings_file"],
                    index_file=dummy_files["index_file"],
                    default=True,
                ),
                index_toml(
                    name="zarr_idx",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
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

        loaded = config.collection_configs["testcol"].indexes
        assert [idx.name for idx in loaded] == ["faiss_idx", "zarr_idx"]
        assert [idx.index_type for idx in loaded] == ["faiss", "zarr"]


class TestModelNameAndDefaultIndex:
    def test_model_name_defaults_to_the_hardcoded_clip_model(
        self, write_config, dummy_files
    ):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                model_name=None,
            ),
        )
        path = write_config(collection)

        config = ConfigManager(str(path)).load_config()

        assert (
            config.collection_configs["testcol"].indexes[0].model_name
            == "ViT-SO400M-14-SigLIP-384"
        )

    def test_model_name_can_be_overridden(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                model_name="a-different-embedding-model",
            ),
        )
        path = write_config(collection)

        config = ConfigManager(str(path)).load_config()

        assert (
            config.collection_configs["testcol"].indexes[0].model_name
            == "a-different-embedding-model"
        )

    def test_single_index_is_the_default_without_the_flag(
        self, write_config, minimal_collection_toml
    ):
        path = write_config(minimal_collection_toml)

        config = ConfigManager(str(path)).load_config()

        collection = config.collection_configs["testcol"]
        assert collection.default_index is collection.indexes[0]

    def test_multi_index_default_resolves_to_the_flagged_index(
        self, write_config, dummy_files
    ):
        indexes = "\n".join(
            [
                index_toml(
                    name="faiss_idx",
                    index_type="faiss",
                    embeddings_file=dummy_files["embeddings_file"],
                    index_file=dummy_files["index_file"],
                ),
                index_toml(
                    name="zarr_idx",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    default=True,
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

        collection_config = config.collection_configs["testcol"]
        assert collection_config.default_index.name == "zarr_idx"

    def test_multi_index_with_no_default_raises(self, write_config, dummy_files):
        indexes = "\n".join(
            [
                index_toml(
                    name="faiss_idx",
                    index_type="faiss",
                    embeddings_file=dummy_files["embeddings_file"],
                    index_file=dummy_files["index_file"],
                ),
                index_toml(
                    name="zarr_idx",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
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

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()

    def test_multi_index_with_two_defaults_raises(self, write_config, dummy_files):
        indexes = "\n".join(
            [
                index_toml(
                    name="faiss_idx",
                    index_type="faiss",
                    embeddings_file=dummy_files["embeddings_file"],
                    index_file=dummy_files["index_file"],
                    default=True,
                ),
                index_toml(
                    name="zarr_idx",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    default=True,
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

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()

    def test_embedding_type_defaults_to_clip(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
        )
        path = write_config(collection)

        config = ConfigManager(str(path)).load_config()

        assert config.collection_configs["testcol"].indexes[0].embedding_type == "CLIP"

    def test_embedding_type_can_be_text(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="transcripts",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                embedding_type="Text",
            ),
        )
        path = write_config(collection)

        config = ConfigManager(str(path)).load_config()

        assert config.collection_configs["testcol"].indexes[0].embedding_type == "Text"

    def test_embedding_type_rejects_unknown_value(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                embedding_type="Audio",
            ),
        )
        path = write_config(collection)

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()

    def test_preload_all_indexes_defaults_to_false(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
        )
        path = write_config(collection)

        config = ConfigManager(str(path)).load_config()

        assert config.collection_configs["testcol"].preload_all_indexes is False

    def test_preload_all_indexes_can_be_enabled(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
            preload_all_indexes=True,
        )
        path = write_config(collection)

        config = ConfigManager(str(path)).load_config()

        assert config.collection_configs["testcol"].preload_all_indexes is True

    def test_source_type_defaults_to_image(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
        )
        path = write_config(collection)

        config = ConfigManager(str(path)).load_config()

        assert config.collection_configs["testcol"].indexes[0].source_type == "Image"

    def test_source_type_can_be_set_to_a_known_value(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="transcripts",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                source_type="Text",
            ),
        )
        path = write_config(collection)

        config = ConfigManager(str(path)).load_config()

        assert config.collection_configs["testcol"].indexes[0].source_type == "Text"

    def test_source_type_rejects_unknown_value(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                source_type="Smell",
            ),
        )
        path = write_config(collection)

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()


class TestTagsetName:
    def test_tagset_name_defaults_to_name_based_on_index_name(
        self, write_config, dummy_files
    ):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="SigLIP_hnsw",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
        )
        path = write_config(collection)

        config = ConfigManager(str(path)).load_config()

        index = config.collection_configs["testcol"].indexes[0]
        assert index.tagset_name == "SigLIP_hnsw Index ID"

    def test_tagset_override_is_used_verbatim(self, write_config, dummy_files):
        collection = collection_toml(
            name="testcol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/testcol",
            original_media_url="https://localhost:5000/testcol",
            indexes=index_toml(
                name="SigLIP_hnsw",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
                tagset="SigLIP",
            ),
        )
        path = write_config(collection)

        config = ConfigManager(str(path)).load_config()

        index = config.collection_configs["testcol"].indexes[0]
        assert index.tagset_name == "SigLIP"

    def test_two_indexes_can_share_the_same_tagset(self, write_config, dummy_files):
        indexes = "\n".join(
            [
                index_toml(
                    name="SigLIP_hnsw",
                    index_type="faiss",
                    embeddings_file=dummy_files["embeddings_file"],
                    index_file=dummy_files["index_file"],
                    tagset="SigLIP",
                    default=True,
                ),
                index_toml(
                    name="SigLIP_zarr",
                    index_type="zarr",
                    embeddings_file=dummy_files["embeddings_file"],
                    tagset="SigLIP",
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

        collection_config = config.collection_configs["testcol"]
        assert collection_config.get_index("SigLIP_hnsw").tagset_name == "SigLIP"
        assert collection_config.get_index("SigLIP_zarr").tagset_name == "SigLIP"


class TestCollectionConfigIndexResolution:
    """get_index / resolve_index / resolve_index_for_embedding_type.

    Used by the search endpoints: /clip and /text resolve by family
    (resolve_index_for_embedding_type), /rf resolves by explicit name or
    falls back to the collection's overall default (resolve_index).
    """

    def _collection(self, write_config, dummy_files):
        indexes = "\n".join(
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
                    embedding_type="Text",
                    model_name="model-two",
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
        return ConfigManager(str(path)).load_config().collection_configs["testcol"]

    def test_get_index_looks_up_by_name(self, write_config, dummy_files):
        collection = self._collection(write_config, dummy_files)
        assert collection.get_index("text_idx").name == "text_idx"

    def test_get_index_raises_for_unknown_name(self, write_config, dummy_files):
        collection = self._collection(write_config, dummy_files)
        with pytest.raises(ValueError):
            collection.get_index("does-not-exist")

    def test_resolve_index_with_no_name_returns_the_default(
        self, write_config, dummy_files
    ):
        collection = self._collection(write_config, dummy_files)
        assert collection.resolve_index() is collection.default_index

    def test_resolve_index_with_explicit_name_ignores_default(
        self, write_config, dummy_files
    ):
        collection = self._collection(write_config, dummy_files)
        assert collection.resolve_index("text_idx").name == "text_idx"

    def test_resolve_index_for_embedding_type_prefers_the_default_when_it_matches(
        self, write_config, dummy_files
    ):
        collection = self._collection(write_config, dummy_files)
        assert collection.resolve_index_for_embedding_type("CLIP") is (
            collection.default_index
        )

    def test_resolve_index_for_embedding_type_falls_back_when_default_is_other_family(
        self, write_config, dummy_files
    ):
        collection = self._collection(write_config, dummy_files)
        # The collection's overall default is clip_idx (CLIP), so resolving
        # for "Text" must fall back to the only Text-type index instead.
        assert collection.resolve_index_for_embedding_type("Text").name == "text_idx"

    def test_resolve_index_for_embedding_type_raises_when_no_index_matches(
        self, write_config, dummy_files
    ):
        collection = self._collection(write_config, dummy_files)
        with pytest.raises(ValueError):
            collection.resolve_index_for_embedding_type("CBIR")


class TestErrorHandling:
    def test_missing_config_file_raises_configuration_error(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.toml"

        with pytest.raises(ConfigurationError):
            ConfigManager(str(missing_path)).load_config()

    def test_malformed_toml_raises_configuration_error(self, write_config):
        path = write_config("this is not [valid toml")

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()

    def test_invalid_model_device_raises_configuration_error(
        self, write_config, minimal_collection_toml
    ):
        text = '[default]\nmodel_device = "quantum"\n\n' + minimal_collection_toml
        path = write_config(text)

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()


class TestUnsupportedConfigFormat:
    def test_unrecognized_extension_raises_configuration_error(
        self, write_config, minimal_collection_toml
    ):
        path = write_config(minimal_collection_toml, filename="config.yaml")

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()

    def test_ini_extension_is_no_longer_supported(
        self, write_config, minimal_collection_toml
    ):
        # Valid TOML content, but .ini support has been removed entirely --
        # the extension alone should be rejected before any parsing happens.
        path = write_config(minimal_collection_toml, filename="config.ini")

        with pytest.raises(ConfigurationError):
            ConfigManager(str(path)).load_config()


class TestConfigManagerCaching:
    def test_config_property_lazily_loads_once(self, write_config, minimal_collection_toml):
        path = write_config(minimal_collection_toml)
        manager = ConfigManager(str(path))

        first = manager.config
        second = manager.config

        assert first is second

    def test_reload_config_picks_up_file_changes(
        self, write_config, dummy_files, minimal_collection_toml
    ):
        path = write_config(minimal_collection_toml)
        manager = ConfigManager(str(path))
        assert manager.config.collections == ["testcol"]

        second_collection = collection_toml(
            name="othercol",
            database_file=dummy_files["database_file"],
            thumbnail_media_url="https://localhost:5000/other",
            original_media_url="https://localhost:5000/other",
            indexes=index_toml(
                name="primary",
                index_type="zarr",
                embeddings_file=dummy_files["embeddings_file"],
            ),
        )
        path.write_text(minimal_collection_toml + "\n\n" + second_collection)

        reloaded = manager.reload_config()

        assert set(reloaded.collections) == {"testcol", "othercol"}
