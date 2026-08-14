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


"""Model management and dependency injection container.

This module provides two main components:

`ModelManager`
    Abstract base for embedding-family-specific model managers. Handles
    PyTorch device selection (CPU / CUDA / MPS) and model-name-keyed
    lookup; subclasses own how their family's models are actually
    loaded. `CLIPModelManager` loads open_clip CLIP text encoders (and
    tokenizers) for indexes with ``embedding_type = "CLIP"`;
    `TextModelManager` loads sentence-transformers models for indexes
    with ``embedding_type = "Text"``. Each loads one model per distinct
    `model_name` referenced by its own family's indexes, once at
    startup, cached for the application lifetime.

`ApplicationContainer`
    The central dependency-injection container that wires together all
    major subsystems.  It owns singleton instances of `ConfigManager`,
    both model managers, `DatabaseRepository`, and `IndexRepository`,
    and orchestrates their initialization in the correct order during
    startup.

A module-level ``container`` instance is exported for use throughout the
application (imported by route dependencies, the lifespan handler, etc.).
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Set

import torch
import open_clip
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import ConfigManager, LSEConfig
from .exceptions import ModelLoadError
from ..repositories.database_repository import DatabaseRepository
from ..repositories.index_repository import IndexRepository

console = Console()
"""Rich console instance for styled terminal output."""

logger = logging.getLogger(__name__)
"""Module-level logger for model and container diagnostics."""


def timer_decorator(func):
    """Decorator to measure execution time."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        time_taken = end - start
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"{current_time} - {func.__name__} executed in {time_taken:.3f} seconds"
        )
        return result

    return wrapper


@timer_decorator
def resolve_device(model_device: str) -> torch.device:
    """Resolve a configured device string to a `torch.device`.

    Shared by every `ModelManager` subclass, since they all resolve the
    same `config.model_device` setting the same way.
    """
    device_str = model_device.lower()

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    try:
        return torch.device(device_str)
    except RuntimeError as e:
        raise ModelLoadError(f"Error setting device: {e}")


class ModelManager(ABC):
    """Base class for embedding-family-specific model managers.

    Each subclass owns one `IndexConfig.embedding_type` family (e.g.
    "CLIP" or "Text") and implements `_load_text_encoder` for however
    that family's models are actually loaded. `initialize_models`
    eagerly loads each collection's default-index encoder (plus every
    matching index in a `preload_all_indexes` collection); anything
    else loads lazily the first time `get_text_encoder` is called for it.
    """

    embedding_type: ClassVar[str]
    """Which `IndexConfig.embedding_type` this manager serves."""

    def __init__(self, config: LSEConfig):
        self.config = config
        """Validated LSE configuration snapshot."""

        self._device: Optional[torch.device] = None
        """Lazily resolved PyTorch device (CPU, CUDA, or MPS)."""

        self._text_encoders: Dict[str, Any] = {}
        """Loaded text encoders keyed by `model_name`."""

    @property
    def device(self) -> torch.device:
        """Get the configured device."""
        if self._device is None:
            self._device = resolve_device(self.config.model_device)
        return self._device

    def get_text_encoder(self, model_name: str) -> Any:
        """Get the text encoder for `model_name`, loading it on demand if not already cached."""
        if model_name not in self._text_encoders:
            logger.info(f"Loading {self.embedding_type} model: {model_name}")
            self._text_encoders[model_name] = self._load_text_encoder(model_name)
            self._after_load_text_encoder(model_name)
        return self._text_encoders[model_name]

    def initialize_models(self) -> None:
        """Eagerly load this family's encoders for indexes that should preload at startup.

        That's each collection's default index (when it belongs to this
        family), plus every matching index in a `preload_all_indexes`
        collection. Anything else loads lazily via `get_text_encoder` on
        first use.
        """
        try:
            for model_name in sorted(self._configured_model_names()):
                self.get_text_encoder(model_name)

            logger.info(f"All {self.embedding_type} models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize {self.embedding_type} models: {e}")
            raise ModelLoadError(
                f"{self.embedding_type} model initialization failed: {e}"
            ) from e

    def _configured_model_names(self) -> Set[str]:
        """model_names to eagerly preload, restricted to this manager's embedding_type."""
        names = set()
        for collection_config in self.config.collection_configs.values():
            default_index = collection_config.default_index
            if default_index.embedding_type == self.embedding_type:
                names.add(default_index.model_name)

            if collection_config.preload_all_indexes:
                names.update(
                    index.model_name
                    for index in collection_config.indexes
                    if index.embedding_type == self.embedding_type
                )
        return names

    def _after_load_text_encoder(self, model_name: str) -> None:
        """Optional hook for subclasses needing extra per-model setup (e.g. a tokenizer)."""
        pass

    @abstractmethod
    def _load_text_encoder(self, model_name: str) -> Any:
        """Load a single text encoder by name. Implemented per embedding family."""
        ...


class CLIPModelManager(ModelManager):
    """Loads open_clip CLIP text encoders and tokenizers for "CLIP"-type indexes."""

    embedding_type: ClassVar[str] = "CLIP"

    def __init__(self, config: LSEConfig):
        super().__init__(config)

        self._tokenizers: Dict[str, Any] = {}
        """Loaded tokenizers keyed by `model_name`, matching `_text_encoders`."""

    def get_tokenizer(self, model_name: str) -> Any:
        """Get the tokenizer for `model_name`, loading the encoder on demand if needed."""
        self.get_text_encoder(model_name)
        return self._tokenizers[model_name]

    def _after_load_text_encoder(self, model_name: str) -> None:
        self._tokenizers[model_name] = self._load_tokenizer(model_name)

    @timer_decorator
    def _load_text_encoder(self, model_name: str) -> torch.nn.Module:
        """Load a CLIP text encoder, caching the extracted text tower on disk."""
        try:
            cache_path = self._text_encoder_cache_path(model_name)

            if not cache_path.exists():
                logger.info(f"{model_name} not found locally, downloading...")
                model = open_clip.create_model(
                    model_name,
                    pretrained="webli",
                    precision="fp16",
                    device=self.device,
                )
                text_model = model.text
                torch.save(text_model, cache_path)
            else:
                text_model = torch.load(cache_path, weights_only=False).to(self.device)

            text_model.eval()
            return text_model

        except Exception as e:
            raise ModelLoadError(f"Failed to load CLIP text model {model_name!r}: {e}")

    @timer_decorator
    def _load_tokenizer(self, model_name: str) -> Any:
        """Load the CLIP tokenizer matching `model_name`."""
        try:
            return open_clip.get_tokenizer(model_name)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load CLIP tokenizer for {model_name!r}: {e}"
            )

    @staticmethod
    def _text_encoder_cache_path(model_name: str) -> Path:
        """On-disk cache path for the extracted text tower of `model_name`."""
        safe_name = model_name.replace("/", "_")
        return Path(f"./data/model_text_{safe_name}.pth")


class TextModelManager(ModelManager):
    """Loads sentence-transformers text embedding models for "Text"-type indexes.

    Unlike CLIP, sentence-transformers models don't need a separate
    tokenizer accessor (encoding is handled internally by `.encode()`),
    and rely on the library's own on-disk cache rather than a
    manually-managed one.
    """

    embedding_type: ClassVar[str] = "Text"

    @timer_decorator
    def _load_text_encoder(self, model_name: str) -> SentenceTransformer:
        try:
            return SentenceTransformer(model_name, device=str(self.device))
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load text embedding model {model_name!r}: {e}"
            )


class ApplicationContainer:
    """Dependency injection container for the application."""

    def __init__(self):
        self._config_manager: Optional[ConfigManager] = None
        """Singleton configuration manager, created on first access."""

        self._clip_model_manager: Optional[CLIPModelManager] = None
        """Singleton CLIP model manager, created on first access."""

        self._text_model_manager: Optional[TextModelManager] = None
        """Singleton text-embedding model manager, created on first access."""

        self._database_repo: Optional[DatabaseRepository] = None
        """Singleton database repository for all collections."""

        self._index_repo: Optional[IndexRepository] = None
        """Singleton index repository for all collections."""

        self._initialized: bool = False
        """Whether `initialize()` has completed successfully."""

    @property
    def config_manager(self) -> ConfigManager:
        """Get the configuration manager."""
        if self._config_manager is None:
            self._config_manager = ConfigManager()
        return self._config_manager

    @property
    def clip_model_manager(self) -> CLIPModelManager:
        """Get the CLIP model manager."""
        if self._clip_model_manager is None:
            self._clip_model_manager = CLIPModelManager(self.config_manager.config)
        return self._clip_model_manager

    @property
    def text_model_manager(self) -> TextModelManager:
        """Get the text-embedding model manager."""
        if self._text_model_manager is None:
            self._text_model_manager = TextModelManager(self.config_manager.config)
        return self._text_model_manager

    @property
    def database_repository(self) -> DatabaseRepository:
        """Get the database repository."""
        if self._database_repo is None:
            self._database_repo = DatabaseRepository(self.config_manager.config)
        return self._database_repo

    @property
    def index_repository(self) -> IndexRepository:
        """Get the index repository."""
        if self._index_repo is None:
            self._index_repo = IndexRepository(self.config_manager.config)
        return self._index_repo

    def initialize(self):
        """Initialize all components and load data."""
        if self._initialized:
            return

        self._display_system_info()

        config = self.config_manager.config
        clip_model_manager = self.clip_model_manager
        text_model_manager = self.text_model_manager
        database_repo = self.database_repository
        index_repo = self.index_repository

        logger.info(f"Running on Device: {clip_model_manager.device}")

        # Explicitly initialize models during startup
        logger.info("Loading ML models...")
        clip_model_manager.initialize_models()
        text_model_manager.initialize_models()

        # Load data for each collection
        for collection in config.collections:
            logger.info(f"Loading artifacts for collection: {collection}")
            collection_config = config.collection_configs[collection]

            # Load metadata
            database_repo.load_database(collection, collection_config.database_file)

            # Always preload the default index; the rest load lazily on
            # first use unless this collection opts into preload_all_indexes.
            default_index = collection_config.default_index
            index_repo.preload(collection, default_index.name)

            if collection_config.preload_all_indexes:
                for index in collection_config.indexes:
                    if index.name != default_index.name:
                        index_repo.preload(collection, index.name)

        self._initialized = True

    def _display_system_info(self):
        """Display system information."""
        system_name = Text("Live Services Engine (v0.2)", style="bold green")
        console.print(
            Panel(system_name, title="Exquisitor", border_style="bold yellow")
        )


container: ApplicationContainer = ApplicationContainer()
"""Global singleton container used throughout the application for dependency resolution."""
