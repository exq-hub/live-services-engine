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

Loads application settings from an INI file (default ``./data/config.ini``)
and exposes them as validated Pydantic models. The configuration is split
into three tiers:

1. **Global defaults** -- ``[DEFAULT]`` section (e.g. ``ModelDevice``).
2. **Server / logging** -- ``[SERVER]`` and ``[LOGGING]`` reserved sections.
3. **Collections** -- every other section whose ``Enabled`` flag is ``True``
   defines a media collection with its own CLIP index, database, media URLs,
   embeddings path, and log directory.

Example INI layout::

    [DEFAULT]
    ModelDevice = auto

    [SERVER]
    Host = 0.0.0.0
    Port = 8000

    [MyCollection]
    Enabled = True
    CLIPIndex = /data/index.faiss
    CLIPIndexType = faiss
    DatabaseFile = /data/db.sqlite
    ThumbnailMediaURL = https://cdn.example.com/thumbs
    OriginalMediaURL = https://cdn.example.com/originals
    EmbeddingsFile = /data/embeddings.zarr
"""

import configparser
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .exceptions import ConfigurationError


class CollectionConfig(BaseModel):
    """Configuration for a single collection."""

    clip_index: str = Field(..., description="Path to CLIP index file")
    clip_index_type: str = Field(
        ..., description="Type of CLIP index (faiss, zarr, etc.)"
    )
    database_file: str = Field(..., description="Path to database file")
    thumbnail_media_url: str = Field(..., description="Base URL for thumbnails")
    original_media_url: str = Field(..., description="Base URL for original media")

    # Optional: Relevance feedback embeddings
    embeddings_file: Optional[str] = Field(
        None, description="Path to embeddings for relevance feedback"
    )

    # Optional: Logging
    log_directory: Optional[str] = Field(
        "./logs/", description="Directory for log files"
    )

    @validator("clip_index", "database_file")
    def validate_required_files(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Required file does not exist: {v}")
        return v

    @validator("embeddings_file")
    def validate_optional_files(cls, v):
        if v is not None and not os.path.exists(v):
            raise ValueError(f"Optional file specified but does not exist: {v}")
        return v


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

    @validator("model_device")
    def validate_device(cls, v):
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        device_lower = v.lower()
        if not any(device_lower.startswith(valid) for valid in valid_devices):
            raise ValueError(
                f"Invalid device: {v}. Must be one of {valid_devices} or cuda:N"
            )
        return device_lower


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: str = "./data/config.ini"):
        self.config_path: Path = Path(config_path)
        """Resolved path to the INI configuration file."""

        self._config: Optional[LSEConfig] = None
        """Cached parsed configuration, populated by `load_config`."""

    def load_config(self) -> LSEConfig:
        """Load and validate configuration."""
        if not self.config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}"
            )

        try:
            parser = configparser.ConfigParser()
            parser.read(self.config_path)

            # Parse DEFAULT section
            default_section = parser["DEFAULT"]

            # Discover collections: any section with Enabled = True
            # (skip reserved sections)
            reserved_sections = {"DEFAULT", "SERVER", "LOGGING"}
            collection_configs = {}
            for collection_name in parser.sections():
                if collection_name in reserved_sections:
                    continue

                section = parser[collection_name]
                if section.get("Enabled", "False").lower() != "true":
                    continue

                config_dict = {
                    "clip_index": section["CLIPIndex"],
                    "clip_index_type": section.get("CLIPIndexType", "faiss"),
                    "database_file": section["DatabaseFile"],
                    "thumbnail_media_url": section["ThumbnailMediaURL"],
                    "original_media_url": section["OriginalMediaURL"],
                }

                # Add optional fields
                optional_mappings = {
                    "EmbeddingsFile": "embeddings_file",
                    "LogDirectory": "log_directory",
                }

                for ini_key, pydantic_key in optional_mappings.items():
                    if ini_key in section:
                        config_dict[pydantic_key] = section[ini_key]

                collection_configs[collection_name] = CollectionConfig(**config_dict)

            enabled_collections = list(collection_configs.keys())

            config_dict = {
                "model_device": default_section.get("ModelDevice", "auto"),
                "collections": enabled_collections,
                "collection_configs": collection_configs,
            }

            # Add optional server settings if present
            if "SERVER" in parser:
                server_section = parser["SERVER"]
                config_dict.update(
                    {
                        "host": server_section.get("Host", "127.0.0.1"),
                        "port": int(server_section.get("Port", 8000)),
                        "reload": server_section.getboolean("Reload", True),
                    }
                )

            # Add optional logging settings if present
            if "LOGGING" in parser:
                logging_section = parser["LOGGING"]
                config_dict["log_level"] = logging_section.get("Level", "INFO")

            self._config = LSEConfig(**config_dict)
            return self._config

        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}") from e

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
