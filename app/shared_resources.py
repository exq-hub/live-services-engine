import configparser
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime

import compress_pickle as cp
import open_clip
import orjson
import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from sentence_transformers import SentenceTransformer

import faiss

from app.search_utils import ShotOverlapMapper

# Initialize Rich Console
console = Console()

# Access Uvicorn logger to later plug into it for logging.
logger = logging.getLogger("uvicorn")


# Timer decorator
def timer_decorator(func):
    """Decorator to measure the execution time of a function."""

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


def generate_mnemonic():
    """Generates a random mnemonic using a combination of adjectives and nouns."""
    # List of adjectives
    adjectives = [
        "happy",
        "sleepy",
        "fast",
        "slow",
        "bright",
        "dark",
        "shiny",
        "dull",
        "quiet",
        "loud",
        "warm",
        "cold",
    ]

    # List of nouns
    nouns = [
        "apple",
        "banana",
        "cherry",
        "dragon",
        "elephant",
        "flower",
        "giraffe",
        "hat",
        "island",
        "jungle",
        "kangaroo",
        "lemon",
    ]

    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    unique_id = str(uuid.uuid4())[:8]
    return f"{adjective}-{noun}-{unique_id}"


class SingletonMeta(type):
    """Metaclass to ensure a single instance of the class"""

    _instance = None

    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class SharedResources(metaclass=SingletonMeta):
    """Shared resources for the application. This class is a singleton and should be instantiated only once."""

    def __init__(self):
        # Display System Name (Header)
        def display_system_name():
            system_name = Text(
                "Live Services Engine (v0.2)",
                style="bold green",
            )
            console.print(
                Panel(system_name, title="Exquisitor", border_style="bold yellow")
            )

        @timer_decorator
        def _load_config(config_path):
            """Load the configuration file."""
            config = configparser.ConfigParser()
            config.read(config_path)
            return config

        @timer_decorator
        def _set_device(device):
            """Determine the device to use for the model."""
            device = device.lower()
            if device == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda")
                elif torch.backends.mps.is_available():
                    return torch.device("mps")
                else:
                    return torch.device("cpu")
            try:
                return torch.device(device)
            except RuntimeError as e:
                print(f"Error setting device: {e}")
                sys.exit(1)

        @timer_decorator
        def _load_clip_text_model():
            """Load the CLIP text model."""
            if not os.path.exists("./data/model_text.pth"):
                model = open_clip.create_model(
                    "ViT-SO400M-14-SigLIP-384",
                    pretrained="webli",
                    precision="fp16",
                    device=self.device,
                )
                text_model = model.text
                torch.save(text_model, "./data/model_text.pth")
            else:
                model = torch.load("./data/model_text.pth", weights_only=False).to(
                    self.device
                )
            model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
            return model

        @timer_decorator
        def _load_clip_text_tokenizer():
            """Load the CLIP text tokenizer."""
            return open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")

        @timer_decorator
        def _load_caption_embedding_model():
            """Load the caption embedding model."""
            model = SentenceTransformer(
                "mixedbread-ai/mxbai-embed-large-v1", device=self.device
            )
            model.eval()  # Ensure the model is in evaluation mode
            return model

        @timer_decorator
        def _load_faiss_ann_index(config, collection, index):
            """Load the FAISS index."""
            if index == "clip":
                return faiss.read_index(config[collection]["CLIPIndex"])
            elif index == "caption":
                return faiss.read_index(config[collection]["CaptionIndex"])
            else:
                raise ValueError(
                    f"Unknown index type: {index}. Expected 'clip' or 'caption'."
                )

        @timer_decorator
        def _load_metadata_and_paths(config, collection):
            """Load metadata and paths."""
            with open(config[collection]["MetadataFile"], "r") as f:
                metadata = orjson.loads(f.read())

                return metadata

        @timer_decorator
        def _load_filters(config, collection):
            """Load filters."""
            filters = {}
            with open(config[collection]["FiltersFile"], "r") as f:
                filters = orjson.loads(f.read())
            return filters

        @timer_decorator
        def _load_related_items(config, collection):
            """Load the related item groups from json file"""
            with open(config[collection]["RelatedItemsFile"], "r") as f:
                return orjson.loads(f.read())

        @timer_decorator
        def _init_shot_overlap_mapper(config, collection):
            # Prepare base shot mapping.
            items = self.metadata[collection]["items"]
            item_to_shot_mapping = {
                x["item_id"]: {
                    "video_id": x["group"],
                    "start_time": x["metadata"]["segment_info"]["start"],
                    "end_time": x["metadata"]["segment_info"]["end"],
                }
                for x in items
            }

            # Prepare caption shot mapping.
            with open(config[collection]["CaptionShotMappingFile"], "r") as f:
                caption_shot_mapping_lines = f.readlines()

            caption_shot_mapping_dict = {}
            caption_shot_ids_list = []

            for line in caption_shot_mapping_lines:
                # Assuming each line is formatted as "group_id block"
                group_id_str, block_str = line.strip().split()
                block = int(block_str)
                start_time = block * float(config[collection]["SegmentDuration"])
                end_time = start_time + float(config[collection]["SegmentDuration"])
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

        @timer_decorator
        def load_pca_data(config, collection):
            """Load PCA data and model"""
            with open(config[collection]["PCAModel"], "rb") as f:
                model = cp.load(f, compression="gzip")

            with open(config[collection]["StdScaler"], "rb") as f:
                scaler = cp.load(f, compression="gzip")

            return model, scaler

        display_system_name()

        self.model_code = generate_mnemonic()

        logger.info(f"Model Code: {self.model_code}")

        self.config = _load_config("./data/config.ini")

        self.collections = []
        for s in self.config["DEFAULT"]["Collections"].split(","):
            if self.config[s.strip()]["Enabled"] == "True":
                self.collections.append(s.strip())

        self.device = _set_device(device=self.config["DEFAULT"]["ModelDevice"])
        logger.info(
            f"Running on Device: {self.device}{'(auto-selected)' if self.config['DEFAULT']['ModelDevice'].lower() == 'auto' else ''}"
        )

        # Load the clip text model
        self.clip_text_model = (
            _load_clip_text_model()
        )  # This causes problems on the mac.

        self.caption_embedding_model = (
            _load_caption_embedding_model()
        )  # Load the caption embedding model

        # Load the clip text tokenizer
        self.clip_text_tokenizer = _load_clip_text_tokenizer()

        # Loading artifacts for each task type
        self.clip_ann_index = {}
        self.caption_ann_index = {}
        self.metadata = {}
        self.filters = {}
        self.related_items = {}
        self.centroids = {}
        self.clusters = {}
        self.cluster_datapoint_idxs = {}
        self.item_to_datapoint = {}
        self.total_items = {}
        self.rf_ann_index_props = {}
        self.thumbnail_path = {}
        self.original_path = {}
        self.logfile = {}
        self.pca = {}
        self.embeddings_zarr = {}
        self.shot_overlap_mapper = {}
        self.caption_shot_ids_list = {}

        for collection in self.collections:
            logger.info(f"Loading artifacts for task type: {collection}")

            # Load the FAISS index
            self.clip_ann_index[collection] = _load_faiss_ann_index(
                config=self.config,
                collection=collection,
                index="clip",
            )

            # Load metadata and paths
            self.metadata[collection] = _load_metadata_and_paths(
                self.config, collection
            )

            if "CaptionIndex" in self.config[collection].keys():
                self.caption_ann_index[collection] = _load_faiss_ann_index(
                    config=self.config,
                    collection=collection,
                    index="caption",
                )
                # Initialize shot overlap mapper
                (
                    self.shot_overlap_mapper[collection],
                    self.caption_shot_ids_list[collection],
                ) = _init_shot_overlap_mapper(self.config, collection)

            # Load filters
            self.filters[collection] = _load_filters(self.config, collection)

            # Load related items
            self.related_items[collection] = _load_related_items(
                self.config, collection
            )

            # Intialize an item_id to idx mapping
            self.item_to_datapoint[collection] = {
                item["item_id"]: idx
                for idx, item in enumerate(self.metadata[collection]["items"])
            }

            # Load IVF Index
            # (
            #     self.centroids[collection],
            #     self.clusters[collection],
            #     self.cluster_datapoint_idxs[collection],
            #     self.item_to_datapoint[collection],
            # ) = _load_ivf_index(self.config, collection)

            if "PCAEmbeddingsFile" in self.config[collection]:
                (pca_model, std_scaler) = load_pca_data(self.config, collection)
                self.pca[collection] = {}
                self.pca[collection]["embeddings_f"] = self.config[collection][
                    "PCAEmbeddingsFile"
                ]
                self.pca[collection]["model"] = pca_model
                self.pca[collection]["scaler"] = std_scaler
            elif "EmbeddingsFile" in self.config[collection]:
                self.embeddings_zarr[collection] = self.config[collection][
                    "EmbeddingsFile"
                ]

            self.total_items[collection] = len(self.metadata[collection]["items"])

            # self.rf_ann_index_props[collection] = (
            #     self.centroids[collection],
            #     self.clusters[collection],
            #     self.cluster_datapoint_idxs[collection],
            #     self.item_to_datapoint[collection],
            # )

            self.thumbnail_path[collection] = self.config[collection][
                "ThumbnailMediaURL"
            ]

            self.original_path[collection] = self.config[collection]["OriginalMediaURL"]

        self.logfile = (
            str(self.config[collection]["LogDirectory"]) + f"{self.model_code}.log"
        )
