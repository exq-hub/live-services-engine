"""Model management and dependency injection container."""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import open_clip
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import ConfigManager, LSEConfig
from .exceptions import ModelLoadError
from ..repositories.database_repository import DatabaseRepository
from ..repositories.metadata_repository import MetadataRepository
from ..repositories.index_repository import IndexRepository
from ..search_utils import ShotOverlapMapper

console = Console()
logger = logging.getLogger(__name__)


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


class ModelManager:
    """Manages ML models and device configuration."""

    def __init__(self, config: LSEConfig):
        self.config = config
        self._device: Optional[torch.device] = None
        self._clip_text_model: Optional[torch.nn.Module] = None
        self._clip_text_tokenizer = None
        self._caption_embedding_model: Optional[SentenceTransformer] = None
        self._shot_overlap_mappers: Dict[str, Dict] = {}
        self._caption_shot_ids_lists: Dict[str, list] = {}

    @property
    def device(self) -> torch.device:
        """Get the configured device."""
        if self._device is None:
            self._device = self._set_device()
        return self._device

    @property
    def clip_text_model(self) -> torch.nn.Module:
        """Get the CLIP text model."""
        if self._clip_text_model is None:
            self._clip_text_model = self._load_clip_text_model()
        return self._clip_text_model

    @property
    def clip_text_tokenizer(self):
        """Get the CLIP text tokenizer."""
        if self._clip_text_tokenizer is None:
            self._clip_text_tokenizer = self._load_clip_text_tokenizer()
        return self._clip_text_tokenizer

    @property
    def caption_embedding_model(self) -> SentenceTransformer:
        """Get the caption embedding model."""
        if self._caption_embedding_model is None:
            self._caption_embedding_model = self._load_caption_embedding_model()
        return self._caption_embedding_model

    def get_shot_overlap_mapper(self, collection: str) -> Optional[Dict]:
        """Get shot overlap mapper for a collection."""
        return self._shot_overlap_mappers.get(collection)

    def get_caption_shot_ids_list(self, collection: str) -> Optional[list]:
        """Get caption shot IDs list for a collection."""
        return self._caption_shot_ids_lists.get(collection)

    def initialize_models(self):
        """Explicitly initialize all models during startup."""
        try:
            logger.info("Initializing CLIP text model...")
            self._clip_text_model = self._load_clip_text_model()

            logger.info("Initializing caption embedding model...")
            self._caption_embedding_model = self._load_caption_embedding_model()

            logger.info("Initializing CLIP tokenizer...")
            self._clip_text_tokenizer = self._load_clip_text_tokenizer()

            logger.info("All models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise ModelLoadError(f"Model initialization failed: {e}") from e

    @timer_decorator
    def _set_device(self) -> torch.device:
        """Determine the device to use for models."""
        device_str = self.config.model_device.lower()

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

    @timer_decorator
    def _load_clip_text_model(self) -> torch.nn.Module:
        """Load the CLIP text model."""
        try:
            model_path = Path("./data/model_text.pth")

            if not model_path.exists():
                logger.info("CLIP model not found locally, downloading...")
                model = open_clip.create_model(
                    "ViT-SO400M-14-SigLIP-384",
                    pretrained="webli",
                    precision="fp16",
                    device=self.device,
                )
                text_model = model.text
                torch.save(text_model, model_path)
            else:
                text_model = torch.load(model_path, weights_only=False).to(self.device)

            text_model.eval()
            return text_model

        except Exception as e:
            raise ModelLoadError(f"Failed to load CLIP text model: {e}")

    @timer_decorator
    def _load_clip_text_tokenizer(self):
        """Load the CLIP text tokenizer."""
        try:
            return open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
        except Exception as e:
            raise ModelLoadError(f"Failed to load CLIP tokenizer: {e}")

    @timer_decorator
    def _load_caption_embedding_model(self) -> SentenceTransformer:
        """Load the caption embedding model."""
        try:
            model = SentenceTransformer(
                "mixedbread-ai/mxbai-embed-large-v1", device=self.device
            )
            model.eval()
            return model
        except Exception as e:
            raise ModelLoadError(f"Failed to load caption embedding model: {e}")

    def initialize_collection_models(
        self, collection: str, collection_config, metadata_repo: MetadataRepository
    ):
        """Initialize collection-specific models and mappers."""
        if (
            collection_config.caption_index
            and collection_config.caption_shot_mapping_file
            and collection_config.segment_duration
        ):
            try:
                # Initialize shot overlap mapper for this collection
                mapper, shot_ids = self._init_shot_overlap_mapper(
                    collection, collection_config, metadata_repo
                )
                self._shot_overlap_mappers[collection] = mapper
                self._caption_shot_ids_lists[collection] = shot_ids
            except Exception as e:
                logger.warning(
                    f"Failed to initialize shot overlap mapper for {collection}: {e}"
                )

    @timer_decorator
    def _init_shot_overlap_mapper(
        self, collection: str, collection_config, metadata_repo: MetadataRepository
    ):
        """Initialize shot overlap mapper for a collection."""
        metadata = metadata_repo.get_metadata(collection)
        if not metadata:
            raise ModelLoadError(f"No metadata available for collection {collection}")

        # Prepare base shot mapping
        items = metadata["items"]
        item_to_shot_mapping = {
            x["item_id"]: {
                "video_id": x["group"],
                "start_time": x["metadata"]["segment_info"]["start"],
                "end_time": x["metadata"]["segment_info"]["end"],
            }
            for x in items
        }

        # Prepare caption shot mapping
        mapping_file = Path(collection_config.caption_shot_mapping_file)
        if not mapping_file.exists():
            raise ModelLoadError(f"Caption shot mapping file not found: {mapping_file}")

        with open(mapping_file, "r") as f:
            caption_shot_mapping_lines = f.readlines()

        caption_shot_mapping_dict = {}
        caption_shot_ids_list = []

        for line in caption_shot_mapping_lines:
            group_id_str, block_str = line.strip().split()
            block = int(block_str)
            start_time = block * collection_config.segment_duration
            end_time = start_time + collection_config.segment_duration
            key = f"{group_id_str}_{start_time}_{end_time}"
            value = {
                "video_id": group_id_str,
                "start_time": start_time,
                "end_time": end_time,
            }
            caption_shot_mapping_dict[key] = value
            caption_shot_ids_list.append(key)

        # Create the overlap mapping
        shot_mapper = ShotOverlapMapper()
        result = shot_mapper.create_overlap_mapping(
            item_to_shot_mapping=item_to_shot_mapping,
            second_mapping=caption_shot_mapping_dict,
        )

        return result, caption_shot_ids_list


class ApplicationContainer:
    """Dependency injection container for the application."""

    def __init__(self):
        self._config_manager: Optional[ConfigManager] = None
        self._model_manager: Optional[ModelManager] = None
        self._metadata_repo: Optional[MetadataRepository | DatabaseRepository] = None
        self._index_repo: Optional[IndexRepository] = None
        self._initialized = False

    @property
    def config_manager(self) -> ConfigManager:
        """Get the configuration manager."""
        if self._config_manager is None:
            self._config_manager = ConfigManager()
        return self._config_manager

    @property
    def model_manager(self) -> ModelManager:
        """Get the model manager."""
        if self._model_manager is None:
            config = self.config_manager.config
            self._model_manager = ModelManager(config)
        return self._model_manager

    @property
    def metadata_repository(self) -> MetadataRepository | DatabaseRepository:
        """Get the metadata repository."""
        if self._metadata_repo is None:
            # self._metadata_repo = MetadataRepository()
            self._metadata_repo = DatabaseRepository()
        return self._metadata_repo

    @property
    def index_repository(self) -> IndexRepository:
        """Get the index repository."""
        if self._index_repo is None:
            self._index_repo = IndexRepository()
        return self._index_repo

    def initialize(self):
        """Initialize all components and load data."""
        if self._initialized:
            return

        self._display_system_info()

        config = self.config_manager.config
        model_manager = self.model_manager
        metadata_repo = self.metadata_repository
        index_repo = self.index_repository

        logger.info(f"Running on Device: {model_manager.device}")

        # Explicitly initialize models during startup
        logger.info("Loading ML models...")
        model_manager.initialize_models()

        # Load data for each collection
        for collection in config.collections:
            logger.info(f"Loading artifacts for collection: {collection}")
            collection_config = config.collection_configs[collection]

            # Load metadata
            if isinstance(metadata_repo, DatabaseRepository):
                metadata_repo.load_database(collection, collection_config.database_file)
                metadata_repo.map_manifest_to_db(collection, collection_config.clip_manifest_file)
            else:
                metadata_repo.load_metadata(collection, collection_config.metadata_file)
                metadata_repo.load_filters(collection, collection_config.filters_file)
                metadata_repo.load_related_items(
                    collection, collection_config.related_items_file
                )

            # Load indices
            index_repo.load_clip_index(collection, collection_config.clip_index)

            if collection_config.caption_index and collection_config.caption_manifest_file:
                index_repo.load_caption_index(
                    collection, collection_config.caption_index
                )
                # if isinstance(metadata_repo, DatabaseRepository):
                #     metadata_repo.map_manifest_to_db(
                #         collection, 
                #         collection_config.caption_manifest_file,
                #         index='caption'
                #     )

            # Load PCA data or embeddings
            if (
                collection_config.pca_model
                and collection_config.std_scaler
                and collection_config.pca_embeddings_file
            ):
                index_repo.load_pca_data(
                    collection,
                    collection_config.pca_model,
                    collection_config.std_scaler,
                    collection_config.pca_embeddings_file,
                )
            elif collection_config.embeddings_file:
                index_repo.set_embeddings_zarr_path(
                    collection, collection_config.embeddings_file
                )

            # Initialize collection-specific models
            model_manager.initialize_collection_models(
                collection, collection_config, metadata_repo
            )

        self._initialized = True

    def _display_system_info(self):
        """Display system information."""
        system_name = Text("Live Services Engine (v0.2)", style="bold green")
        console.print(
            Panel(system_name, title="Exquisitor", border_style="bold yellow")
        )


# Global container instance
container = ApplicationContainer()
