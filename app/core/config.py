"""Configuration management with validation."""

import configparser
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .exceptions import ConfigurationError


class CollectionConfig(BaseModel):
    """Configuration for a single collection."""

    enabled: bool = Field(..., description="Whether this collection is enabled")
    clip_index: str = Field(..., description="Path to CLIP index file")
    clip_index_type: str = Field(..., description="Type of CLIP index (faiss, zarr, etc.)")
    database_file: str = Field(..., description="Path to database file")
    thumbnail_media_url: str = Field(..., description="Base URL for thumbnails")
    original_media_url: str = Field(..., description="Base URL for original media")

    # Optional fields
    clip_manifest_file: Optional[str] = Field(None, description="Path to CLIP manifest file")
    caption_index: Optional[str] = Field(None, description="Path to caption index file")
    caption_index_type: Optional[str] = Field(None, description="Type of caption index (faiss, zarr, etc.)")
    caption_manifest_file: Optional[str] = Field(None, description="Path to caption manifest file")
    transcript_index: Optional[str] = Field(None, description="Path to transcript index file")
    transcript_index_type: Optional[str] = Field(None, description="Type of transcript index (faiss, zarr, etc.)")
    transcript_manifest_file: Optional[str] = Field(None, description="Path to transcript manifest file")
    pca_model: Optional[str] = Field(None, description="Path to PCA model file")
    std_scaler: Optional[str] = Field(None, description="Path to standard scaler file")
    pca_embeddings_file: Optional[str] = Field(
        None, description="Path to PCA embeddings"
    )
    embeddings_file: Optional[str] = Field(None, description="Path to embeddings file")
    log_directory: Optional[str] = Field(
        "./logs/", description="Directory for log files"
    )


    @validator("clip_index", "database_file")
    def validate_required_files(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Required file does not exist: {v}")
        return v

    @validator(
        "caption_index",
        "pca_model",
        "std_scaler",
        "pca_embeddings_file",
        "embeddings_file",
    )
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
        self.config_path = Path(config_path)
        self._config: Optional[LSEConfig] = None

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
            collections = [s.strip() for s in default_section["Collections"].split(",")]

            # Parse collection configurations
            collection_configs = {}
            for collection_name in collections:
                if collection_name not in parser:
                    raise ConfigurationError(
                        f"Collection '{collection_name}' not found in config"
                    )

                section = parser[collection_name]
                if section.get("Enabled", "False").lower() != "true":
                    continue

                config_dict = {
                    "enabled": True,
                    "clip_index": section["CLIPIndex"],
                    "clip_manifest_file": section.get("CLIPManifestFile"),
                    "clip_index_type": section.get("CLIPIndexType", "faiss"),
                    "database_file": section["DatabaseFile"],
                    "thumbnail_media_url": section["ThumbnailMediaURL"],
                    "original_media_url": section["OriginalMediaURL"],
                }

                # Add optional fields
                optional_mappings = {
                    "CaptionIndex": "caption_index",
                    "CaptionIndexType": "caption_index_type",
                    "CaptionManifestFile": "caption_manifest_file",
                    "CaptionShotMappingFile": "caption_shot_mapping_file",
                    "SegmentDuration": "segment_duration",
                    "TranscriptIndex": "transcript_index",
                    "TranscriptIndexType": "transcript_index_type",
                    "TranscriptManifestFile": "transcript_manifest_file",
                    "PCAModel": "pca_model",
                    "StdScaler": "std_scaler",
                    "PCAEmbeddingsFile": "pca_embeddings_file",
                    "EmbeddingsFile": "embeddings_file",
                    "LogDirectory": "log_directory",
                }

                for ini_key, pydantic_key in optional_mappings.items():
                    if ini_key in section:
                        value = section[ini_key]
                        if pydantic_key == "segment_duration":
                            value = float(value)
                        config_dict[pydantic_key] = value

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
