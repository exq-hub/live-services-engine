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


"""Configuration management with Pydantic validation.

Loads application settings from a TOML config file (default
``./data/config.toml``) and exposes them as validated Pydantic models.
Each collection declares an explicit list of named indexes, so a
collection can hold more than one index/embedding representation.

The configuration is split into three tiers:

1. **Global defaults** -- ``[default]`` table (e.g. ``model_device``).
2. **Server / logging** -- ``[server]`` and ``[logging]`` reserved tables.
3. **Collections** -- each ``[[collections]]`` table whose ``enabled`` flag
   is true, with its own database, media URLs, log directory, and list of
   indexes.

Each index requires an ``index_type`` and an ``embeddings_file`` (a Zarr
archive of raw embeddings, always needed for relevance feedback). When
``index_type = "faiss"`` an additional ``index_file`` must point to the
FAISS index; for ``index_type = "zarr"`` the ``embeddings_file`` is used
directly as the brute-force index, so no ``index_file`` is needed.

For ``index_type = "zarr"``, ``embeddings_file``'s extension picks which
Zarr store backend loads it (see ``ZarrIndex.load_index`` in
``app/core/indexes.py``): ``.zip``/``.zipstore`` open as a zip archive,
``.zarr`` opens directly as a Zarr directory store. Any other extension
is rejected at load time.

Example TOML layout::

    [server]
    host = "0.0.0.0"
    port = 8000

    [[collections]]
    name = "MyCollection"
    enabled = true
    database_file = "/data/db.sqlite"
    thumbnail_media_url = "https://cdn.example.com/thumbs"
    original_media_url = "https://cdn.example.com/originals"

      [[collections.indexes]]
      name = "CLIP"
      index_type = "zarr"
      embeddings_file = "/data/embeddings.zipstore"

      [[collections.indexes]]
      name = "Text"
      index_type = "faiss"
      index_file = "/data/transcripts.faiss"
      embeddings_file = "/data/transcript_embeddings.zarr"
"""

import os
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from .exceptions import ConfigurationError


class IndexConfig(BaseModel):
    """Configuration for a single index within a collection."""

    name: str = Field(
        ...,
        description=(
            "Identifies this index within its collection. Also the default "
            "DB id-tag namespace, as '<name> Index ID', unless overridden "
            "by `tagset`."
        ),
    )
    index_type: str = Field(..., description="Index backend: 'faiss' or 'zarr'")
    index_file: Optional[str] = Field(
        None,
        description="Path to the ANN index file. Required when index_type is 'faiss'; must be omitted for 'zarr'.",
    )
    embeddings_file: str = Field(
        ..., description="Path to Zarr embeddings file (always required)"
    )
    model_name: str = Field(
        "ViT-SO400M-14-SigLIP-384",
        description="Embedding model that produced this index's vectors.",
    )
    embedding_type: str = Field(
        "CLIP",
        description=(
            "Which embedding family model_name belongs to, e.g. 'CLIP' or "
            "'Text'. Determines which model manager and search strategy can "
            "serve this index -- deliberately independent of the specific "
            "library used to load model_name."
        ),
    )
    source_type: str = Field(
        "Image",
        description=(
            "Which kind of media this index's positions correspond to: "
            "'Image', 'Video', 'Audio', 'Text', or 'Other'. Matches a row in "
            "the database's source_types table, and picks both which medias "
            "this index maps ('Text' for an index over transcripts, "
            "'Image' for one over keyframes) and the tagset's expected "
            "source_type."
        ),
    )
    tagset: Optional[str] = Field(
        None,
        description=(
            "DB tagset this index's id mapping is read from, used verbatim. "
            "Defaults to '<name> Index ID' when unset. Override when several "
            "indexes share one underlying id space -- e.g. the same "
            "embeddings built as uncompressed zarr, HNSW, and HNSW+PQ -- so "
            "they all read the same tagset instead of each needing (and "
            "duplicating) their own."
        ),
    )
    default: bool = Field(
        False,
        description=(
            "Marks this as the index used when a request doesn't specify one. "
            "Required (exactly one) when a collection has multiple indexes; "
            "implied when a collection has only one."
        ),
    )

    @property
    def tagset_name(self) -> str:
        """The DB tagset this index's id mapping is read from."""
        if self.tagset is not None:
            return self.tagset
        return f"{self.name} Index ID"

    @field_validator("index_file")
    @classmethod
    def validate_index_file(cls, v: Optional[str], info: ValidationInfo):
        index_type = info.data.get("index_type")
        if index_type == "faiss":
            if v is None:
                raise ValueError("index_file is required when index_type is 'faiss'")
            if not os.path.exists(v):
                raise ValueError(f"Index file does not exist: {v}")
        elif v is not None:
            raise ValueError(
                f"index_file must not be specified when index_type is '{index_type}'"
            )
        return v

    @field_validator("embeddings_file")
    @classmethod
    def validate_embeddings_file(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"Embeddings file does not exist: {v}")
        return v

    @field_validator("embedding_type")
    @classmethod
    def validate_embedding_type(cls, v: str) -> str:
        # Only families with an actual model manager/strategy today. Extend
        # as they're built (e.g. 'CBIR'/'SIFT' for hand-crafted features).
        valid_types = {"CLIP", "Text"}
        if v not in valid_types:
            raise ValueError(
                f"Invalid embedding_type: {v!r}. Must be one of {sorted(valid_types)}"
            )
        return v

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        # Matches the database's source_types table default rows.
        valid_source_types = {"Image", "Video", "Audio", "Text", "Other"}
        if v not in valid_source_types:
            raise ValueError(
                f"Invalid source_type: {v!r}. Must be one of {sorted(valid_source_types)}"
            )
        return v


class CollectionConfig(BaseModel):
    """Configuration for a single collection."""

    database_file: str = Field(..., description="Path to database file")
    thumbnail_media_url: str = Field(..., description="Base URL for thumbnails")
    original_media_url: str = Field(..., description="Base URL for original media")
    indexes: List[IndexConfig] = Field(
        ..., min_length=1, description="Indexes available for this collection"
    )
    preload_all_indexes: bool = Field(
        False,
        description=(
            "If true, eagerly load every index (model, embeddings, ANN "
            "structure) for this collection at startup. If false, only "
            "the default index loads eagerly; the rest load lazily on "
            "first use."
        ),
    )

    # Optional: Logging
    log_directory: Optional[str] = Field(
        "./logs/", description="Directory for log files"
    )

    @field_validator("database_file")
    @classmethod
    def validate_database_file(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"Database file does not exist: {v}")
        return v

    @field_validator("indexes")
    @classmethod
    def validate_unique_index_names(cls, v: List[IndexConfig]) -> List[IndexConfig]:
        names = [index.name for index in v]
        if len(names) != len(set(names)):
            raise ValueError("Index names must be unique within a collection")
        return v

    @field_validator("indexes")
    @classmethod
    def validate_default_index(cls, v: List[IndexConfig]) -> List[IndexConfig]:
        if len(v) > 1:
            defaults = [index for index in v if index.default]
            if len(defaults) != 1:
                raise ValueError(
                    "Exactly one index must be marked default when a collection "
                    f"has multiple indexes (found {len(defaults)})"
                )
        return v

    @property
    def default_index(self) -> IndexConfig:
        """The index used when a request doesn't specify one explicitly."""
        if len(self.indexes) == 1:
            return self.indexes[0]
        return next(index for index in self.indexes if index.default)

    def get_index(self, name: str) -> IndexConfig:
        """Look up an index by name."""
        for index in self.indexes:
            if index.name == name:
                return index
        raise ValueError(f"No index named {name!r} configured for this collection")

    def resolve_index(self, index_name: Optional[str] = None) -> IndexConfig:
        """Resolve an index by explicit name, or the collection's default index."""
        if index_name is not None:
            return self.get_index(index_name)
        return self.default_index

    def resolve_index_for_embedding_type(self, embedding_type: str) -> IndexConfig:
        """Resolve the index for a family-scoped endpoint (e.g. /clip, /text).

        Prefers the collection's overall default index when it belongs to
        `embedding_type`; otherwise falls back to the first configured
        index of that type. Raises if the collection has none.
        """
        matching = [i for i in self.indexes if i.embedding_type == embedding_type]
        if not matching:
            raise ValueError(
                f"No {embedding_type!r}-type index configured for this collection"
            )
        if self.default_index.embedding_type == embedding_type:
            return self.default_index
        return matching[0]


class LSEConfig(BaseModel):
    """Main LSE configuration."""

    model_device: str = Field("auto", description="Device to run models on")
    collections: List[str] = Field(..., description="List of enabled collections")
    collection_configs: Dict[str, CollectionConfig] = Field(
        ..., description="Per-collection configurations"
    )

    # Server settings
    host: str = Field("127.0.0.1", description="Server host")
    port: int = Field(8000, description="Server port")
    reload: bool = Field(True, description="Enable auto-reload")

    # Logging settings
    log_level: str = Field("INFO", description="Log level")

    @field_validator("model_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        device_lower = v.lower()
        if not any(device_lower.startswith(valid) for valid in valid_devices):
            raise ValueError(
                f"Invalid device: {v}. Must be one of {valid_devices} or cuda:N"
            )
        return device_lower


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: str = "./data/config.toml"):
        self.config_path: Path = Path(config_path)
        """Resolved path to the TOML config file."""

        self._config: Optional[LSEConfig] = None
        """Cached parsed configuration, populated by `load_config`."""

    def load_config(self) -> LSEConfig:
        """Load and validate configuration."""
        if not self.config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}"
            )

        suffix = self.config_path.suffix.lower()
        try:
            if suffix != ".toml":
                raise ConfigurationError(
                    f"Unsupported configuration file extension '{suffix}' for "
                    f"{self.config_path}. Expected '.toml'."
                )

            config_dict = self._parse_toml()
            self._config = LSEConfig(**config_dict)
            return self._config

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}") from e

    def _parse_toml(self) -> dict:
        """Parse the current TOML config format into an `LSEConfig` kwargs dict."""
        with open(self.config_path, "rb") as f:
            data = tomllib.load(f)

        collection_configs: Dict[str, CollectionConfig] = {}
        for collection_data in data.get("collections", []):
            if not collection_data.get("enabled", False):
                continue

            indexes = []
            for index_data in collection_data.get("indexes", []):
                index_kwargs = {
                    "name": index_data["name"],
                    "index_type": index_data["index_type"],
                    "index_file": index_data.get("index_file"),
                    "embeddings_file": index_data["embeddings_file"],
                    "default": index_data.get("default", False),
                }
                # Only pass model_name/embedding_type/source_type when set,
                # so the IndexConfig field defaults apply rather than being
                # overridden with None.
                if "model_name" in index_data:
                    index_kwargs["model_name"] = index_data["model_name"]
                if "embedding_type" in index_data:
                    index_kwargs["embedding_type"] = index_data["embedding_type"]
                if "source_type" in index_data:
                    index_kwargs["source_type"] = index_data["source_type"]
                if "tagset" in index_data:
                    index_kwargs["tagset"] = index_data["tagset"]
                indexes.append(IndexConfig(**index_kwargs))

            collection_configs[collection_data["name"]] = CollectionConfig(
                database_file=collection_data["database_file"],
                thumbnail_media_url=collection_data["thumbnail_media_url"],
                original_media_url=collection_data["original_media_url"],
                log_directory=collection_data.get("log_directory", "./logs/"),
                preload_all_indexes=collection_data.get("preload_all_indexes", False),
                indexes=indexes,
            )

        config_dict = {
            "model_device": data.get("default", {}).get("model_device", "auto"),
            "collections": list(collection_configs.keys()),
            "collection_configs": collection_configs,
        }

        server_section = data.get("server")
        if server_section:
            config_dict.update(
                {
                    "host": server_section.get("host", "127.0.0.1"),
                    "port": int(server_section.get("port", 8000)),
                    "reload": bool(server_section.get("reload", True)),
                }
            )

        logging_section = data.get("logging")
        if logging_section:
            config_dict["log_level"] = logging_section.get("level", "INFO")

        return config_dict

    @property
    def config(self) -> LSEConfig:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            self.load_config()
        return self._config

    def reload_config(self) -> LSEConfig:
        """Reload configuration from file."""
        self._config = None
        return self.load_config()
