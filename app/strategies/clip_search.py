# SPDX-FileCopyrightText: 2026 Ujjwal Sharma and Omar Shahbaz Khan
# SPDX-License-Identifier: AGPL-3.0-or-later


"""CLIP text-to-image search strategy.

Encodes a natural-language text query into the CLIP embedding space using
the collection's CLIP text encoder, then retrieves the *n* most similar
media items from the collection's vector index.

Everything but the actual encoding call lives in `EmbeddingSearchStrategy`
(shared with `TextEmbeddingSearchStrategy`): index resolution, skip-set
construction, expanding search, and ID mapping.
"""

import asyncio
import contextlib
from functools import partial
from typing import ClassVar

import torch
import numpy as np

from .embedding_search import EmbeddingSearchStrategy


class CLIPSearchStrategy(EmbeddingSearchStrategy):
    """Search strategy using CLIP text embeddings."""

    embedding_type: ClassVar[str] = "CLIP"

    def get_strategy_name(self) -> str:
        return "CLIP Search"

    def _sync_encode_text(self, model_name: str, text: str) -> np.ndarray:
        "Synchronous text encoding function to be run in a thread pool."
        # Model manager is a CLIPModelManager
        device = self.model_manager.device
        tokenizer = self.model_manager.get_tokenizer(model_name)
        text_model = self.model_manager.get_text_encoder(model_name)

        with (
            torch.inference_mode(),
            (
                torch.amp.autocast("cuda")
                if torch.cuda.is_available()
                else contextlib.nullcontext()
            ),
        ):
            tokenized_text = tokenizer([text]).to(device)
            text_features = text_model(tokenized_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.detach().cpu().numpy()

    async def _encode_text(self, model_name: str, text: str) -> np.ndarray:
        """Asynchronously encode text using CLIP by running the synchronous encoding function in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._sync_encode_text, model_name, text)
        )
